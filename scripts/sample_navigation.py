import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import diffuser.utils as utils
from diffuser.environments.mobile_robot import MobileRobotEnv
from diffuser.utils.navigation_renderer import NavigationRenderer


def sample_navigation_plans(
    logdir='logs/MobileRobotNav-v0/diffusion/navigation_H32_T20',
    epoch='latest',
    n_samples=4,
    output_path='data/diffusion_navigation_plans.png',
):
    print(f'Loading diffusion model from {logdir}...')

    # Load configs
    model_config = utils.load_config(logdir, 'model_config.pkl')
    diffusion_config = utils.load_config(logdir, 'diffusion_config.pkl')
    dataset_config = utils.load_config(logdir, 'dataset_config.pkl')
    render_config = utils.load_config(logdir, 'render_config.pkl')

    model = model_config()
    diffusion = diffusion_config(model)
    dataset = dataset_config()
    renderer = render_config()

    trainer = utils.Trainer(
        diffusion_model=diffusion,
        dataset=dataset,
        renderer=renderer,
        results_folder=logdir,
    )
    if epoch == 'latest' and not os.path.exists(os.path.join(logdir, 'state_latest.pt')):
        import glob, re
        pts = glob.glob(os.path.join(logdir, 'state_*.pt'))
        steps = []
        for p in pts:
            m = re.search(r'state_(\d+)\.pt', os.path.basename(p))
            if m:
                steps.append(int(m.group(1)))
        epoch = max(steps) if steps else 0

    trainer.load(epoch)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion = trainer.ema_model
    diffusion.to(device)
    print(f'Model loaded from step {trainer.step} on {device}')

    env = MobileRobotEnv()
    start_pos = np.array([-2.6, -2.6], dtype=np.float32)
    goal_pos = np.array([2.5, 2.5], dtype=np.float32)

    obs = env.reset(start_pos=start_pos, goal_pos=goal_pos)
    normed_obs = dataset.normalizer(obs[None, :], 'observations')

    # Condition at t=0 on start observation
    conditions = {0: torch.tensor(normed_obs, device=device, dtype=torch.float32)}

    # Repeat conditions for n_samples
    conditions = {
        k: v.repeat(n_samples, 1) for k, v in conditions.items()
    }

    print(f'Sampling {n_samples} trajectory plans using diffusion model...')
    samples = diffusion(conditions)
    trajectories = utils.to_np(samples.trajectories)

    # Extract observations: [n_samples, horizon, obs_dim]
    normed_obs_pred = trajectories[:, :, dataset.action_dim:]
    obs_pred = dataset.normalizer.unnormalize(normed_obs_pred, 'observations')

    # Prepend initial condition for full trajectory continuity
    obs_full = np.concatenate([
        np.repeat(obs[None, None, :], n_samples, axis=0),
        obs_pred
    ], axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    renderer.composite(output_path, obs_full)
    print(f'Sample plan visualization saved to: {output_path}')

    env.close()
    return output_path


def rollout_receding_horizon(
    logdir='logs/MobileRobotNav-v0/diffusion/navigation_H32_T20',
    epoch='latest',
    max_steps=200,
    plan_stride=10,
    output_path='data/diffusion_receding_horizon_rollout.png',
):
    print(f'Starting receding-horizon closed-loop navigation rollout...')
    model_config = utils.load_config(logdir, 'model_config.pkl')
    diffusion_config = utils.load_config(logdir, 'diffusion_config.pkl')
    dataset_config = utils.load_config(logdir, 'dataset_config.pkl')
    render_config = utils.load_config(logdir, 'render_config.pkl')

    model = model_config()
    diffusion = diffusion_config(model)
    dataset = dataset_config()
    renderer = render_config()

    trainer = utils.Trainer(
        diffusion_model=diffusion,
        dataset=dataset,
        renderer=renderer,
        results_folder=logdir,
    )
    if epoch == 'latest' and not os.path.exists(os.path.join(logdir, 'state_latest.pt')):
        import glob, re
        pts = glob.glob(os.path.join(logdir, 'state_*.pt'))
        steps = []
        for p in pts:
            m = re.search(r'state_(\d+)\.pt', os.path.basename(p))
            if m:
                steps.append(int(m.group(1)))
        epoch = max(steps) if steps else 0

    trainer.load(epoch)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion = trainer.ema_model
    diffusion.to(device)

    env = MobileRobotEnv(max_episode_steps=max_steps)
    start_pos = np.array([-2.6, -2.6], dtype=np.float32)
    goal_pos = np.array([2.5, 2.5], dtype=np.float32)

    obs = env.reset(start_pos=start_pos, goal_pos=goal_pos)
    executed_obs = [obs.copy()]
    sub_plans = []

    step_count = 0
    while step_count < max_steps:
        normed_obs = dataset.normalizer(obs[None, :], 'observations')
        conditions = {0: torch.tensor(normed_obs, device=device, dtype=torch.float32)}

        samples = diffusion(conditions)
        trajectories = utils.to_np(samples.trajectories)
        normed_acts = trajectories[0, :, :dataset.action_dim]
        acts = dataset.normalizer.unnormalize(normed_acts, 'actions')
        normed_preds = trajectories[0, :, dataset.action_dim:]
        preds = dataset.normalizer.unnormalize(normed_preds, 'observations')
        sub_plans.append(preds)

        # Execute plan_stride steps from predicted action sequence in simulation
        reached = False
        for i in range(min(plan_stride, len(acts))):
            action = acts[i]
            obs, reward, done, info = env.step(action)
            executed_obs.append(obs.copy())
            step_count += 1
            if done:
                reached = info['reached_goal']
                break

        print(f'Step {step_count} | Dist to goal: {info["dist_to_goal"]:.2f}m | Reached: {reached}')
        if done:
            break

    full_executed_path = np.array(executed_obs)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    renderer.composite(output_path, [full_executed_path] + sub_plans[:6])
    print(f'Receding horizon navigation plot saved to: {output_path}')
    env.close()
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample trajectory plans from trained Diffuser model')
    parser.add_argument('--logdir', type=str, default='logs/MobileRobotNav-v0/diffusion/navigation_H32_T20', help='Trained model log directory')
    parser.add_argument('--epoch', type=str, default='latest', help='Epoch or step to load')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of candidate trajectories to sample')
    parser.add_argument('--output', type=str, default='data/diffusion_navigation_plans.png', help='Output visualization path')
    parser.add_argument('--receding_horizon', action='store_true', help='Run closed-loop receding horizon navigation rollout')
    parser.add_argument('--stride', type=int, default=10, help='Execution stride between replanning cycles')

    args = parser.parse_args()
    if args.receding_horizon:
        rollout_receding_horizon(
            logdir=args.logdir,
            epoch=args.epoch,
            plan_stride=args.stride,
            output_path=args.output,
        )
    else:
        sample_navigation_plans(
            logdir=args.logdir,
            epoch=args.epoch,
            n_samples=args.n_samples,
            output_path=args.output,
        )

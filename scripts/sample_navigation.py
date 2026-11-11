import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
import diffuser.utils as utils
from diffuser.environments.mobile_robot import MobileRobotEnv
from diffuser.utils.navigation_renderer import NavigationRenderer


def _normalize_goal_xy(dataset, goal_pos):
    obs_norm = dataset.normalizer.normalizers['observations']
    mins_xy = obs_norm.mins[:2]
    maxs_xy = obs_norm.maxs[:2]
    return 2.0 * (np.asarray(goal_pos[:2], dtype=np.float32) - mins_xy) / (maxs_xy - mins_xy) - 1.0


def sample_navigation_plans(
    logdir='logs/MobileRobotNav-v0/diffusion/navigation_H128_T50',
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

    n_lidar = max(1, dataset.observation_dim - 11)
    env = MobileRobotEnv(n_lidar_beams=n_lidar)
    start_pos = np.array([-2.6, -2.6], dtype=np.float32)
    goal_pos = np.array([2.5, 2.5], dtype=np.float32)

    obs = env.reset(start_pos=start_pos, goal_pos=goal_pos)
    normed_obs = dataset.normalizer(obs[None, :], 'observations')
    normed_goal_xy = _normalize_goal_xy(dataset, goal_pos)[None, :]

    # Condition at t=0 on start observation and t=H-1 on fixed goal (x, y)
    conditions = {
        0: torch.tensor(normed_obs, device=device, dtype=torch.float32),
        diffusion.horizon - 1: torch.tensor(normed_goal_xy, device=device, dtype=torch.float32),
    }

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

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    renderer.composite(output_path, obs_pred, obstacles=env.obstacles)
    print(f'Sample plan visualization saved to: {output_path}')

    env.close()
    return output_path


def _score_candidate_lidar(pts, obs):
    """Score a candidate (H, 2) path using LiDAR hits in obs, arena bounds, and smoothness."""
    x, y = float(obs[0]), float(obs[1])
    theta = np.arctan2(float(obs[3]), float(obs[2]))
    lidar = obs[11:]
    n_beams = len(lidar)
    angles = theta + np.linspace(0, 2 * np.pi, n_beams, endpoint=False)
    max_range = max(float(np.max(lidar)), 1.0)
    valid = lidar < (max_range - 0.15)
    cost = 0.0

    near_pts = pts[:min(48, len(pts))]
    if np.any(valid):
        hits = np.stack([
            x + lidar[valid] * np.cos(angles[valid]),
            y + lidar[valid] * np.sin(angles[valid]),
        ], axis=-1)
        dists = np.linalg.norm(near_pts[:, None, :] - hits[None, :, :], axis=-1)
        clearance_violation = np.maximum(0.42 - np.min(dists, axis=1), 0.0)
        cost += 120.0 * float(np.sum(clearance_violation ** 2)) + 15.0 * float(np.max(clearance_violation))

    wall_violation = np.maximum(np.abs(near_pts) - 3.55, 0.0)
    cost += 50.0 * float(np.sum(wall_violation ** 2))
    diffs = np.diff(near_pts, axis=0)
    cost += 2.0 * float(np.sum(diffs ** 2))
    return cost


def rollout_receding_horizon(
    logdir='logs/MobileRobotNav-v0/diffusion/navigation_H128_T50',
    epoch='latest',
    max_steps=200,
    plan_stride=6,
    n_candidates=4,
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

    n_lidar = max(1, dataset.observation_dim - 11)
    env = MobileRobotEnv(max_episode_steps=max_steps, n_lidar_beams=n_lidar)
    start_pos = np.array([-2.6, -2.6], dtype=np.float32)
    goal_pos = np.array([2.5, 2.5], dtype=np.float32)

    obs = env.reset(start_pos=start_pos, goal_pos=goal_pos)
    executed_obs = [obs.copy()]
    sub_plans = []
    normed_goal_xy = _normalize_goal_xy(dataset, goal_pos)[None, :]

    step_count = 0
    while step_count < max_steps:
        normed_obs = dataset.normalizer(obs[None, :], 'observations')
        conditions = {
            0: torch.tensor(normed_obs, device=device, dtype=torch.float32).repeat(n_candidates, 1),
            diffusion.horizon - 1: torch.tensor(normed_goal_xy, device=device, dtype=torch.float32).repeat(n_candidates, 1),
        }

        with torch.no_grad():
            samples = diffusion(conditions)
        trajectories = utils.to_np(samples.trajectories)
        best_idx = 0
        if n_candidates > 1:
            best_cost = float('inf')
            for k in range(n_candidates):
                cand_obs = dataset.normalizer.unnormalize(
                    trajectories[k, :, dataset.action_dim:], 'observations'
                )
                cand_cost = _score_candidate_lidar(cand_obs[:, :2], obs)
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_idx = k

        normed_acts = trajectories[best_idx, :, :dataset.action_dim]
        acts = dataset.normalizer.unnormalize(normed_acts, 'actions')
        normed_preds = trajectories[best_idx, :, dataset.action_dim:]
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
    renderer.composite(output_path, [full_executed_path] + sub_plans[:6], obstacles=env.obstacles)
    print(f'Receding horizon navigation plot saved to: {output_path}')
    env.close()
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample trajectory plans from trained Diffuser model')
    parser.add_argument('--logdir', type=str, default='logs/MobileRobotNav-v0/diffusion/navigation_H128_T50', help='Trained model log directory')
    parser.add_argument('--epoch', type=str, default='latest', help='Epoch or step to load')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of candidate trajectories to sample')
    parser.add_argument('--output', type=str, default='data/diffusion_navigation_plans.png', help='Output visualization path')
    parser.add_argument('--receding_horizon', action='store_true', help='Run closed-loop receding horizon navigation rollout')
    parser.add_argument('--stride', type=int, default=6, help='Execution stride between replanning cycles')

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

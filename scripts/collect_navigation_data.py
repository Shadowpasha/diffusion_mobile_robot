import os
import argparse
import numpy as np
import gym
import diffuser.environments
from diffuser.environments.mobile_robot import (
    MobileRobotEnv,
    NavigationExpertPolicy,
    GlobalNavigationExpertPolicy,
)
from diffuser.utils.navigation_renderer import NavigationRenderer


def collect_dataset(
    num_episodes=2000,
    max_path_length=200,
    output_path='data/mobile_robot_dataset.npz',
    expert_type='global',
    noise_std=0.015,
    render_sample=True,
    filter_successful=True,
    max_suboptimality=1.45,
):
    print(f'=== Starting High-Quality Dataset Collection ===')
    print(f'Expert: {expert_type.upper()} | Target Episodes: {num_episodes} | Max Length: {max_path_length} | Noise std: {noise_std}')
    print(f'Filters: SuccessOnly={filter_successful} | ZeroCollisions=True | MaxSuboptimality={max_suboptimality}')

    env = MobileRobotEnv(max_episode_steps=max_path_length, terminate_on_collision=True)

    if expert_type == 'global':
        policy = GlobalNavigationExpertPolicy(env, noise_std=noise_std)
    else:
        policy = NavigationExpertPolicy(env, noise_std=noise_std)

    all_obs = []
    all_acts = []
    all_rews = []
    all_terms = []
    all_timeouts = []

    success_count = 0
    total_steps = 0
    sample_paths = []
    ep_attempt = 0
    rejected_collision = 0
    rejected_suboptimal = 0
    rejected_timeout = 0

    while success_count < num_episodes:
        ep_attempt += 1

        obs = env.reset()
        start_pos = env.data.qpos[:2].copy()
        goal_pos = env.current_goal.copy()

        if hasattr(policy, 'reset'):
            policy.reset(start_pos=start_pos, goal_pos=goal_pos, obstacles=env.obstacles.copy())

        ep_obs = []
        ep_acts = []
        ep_rews = []
        ep_terms = []
        ep_timeouts = []
        had_collision = False

        for step in range(max_path_length):
            action = policy.act(obs)
            next_obs, reward, done, info = env.step(action)

            if info['collision']:
                had_collision = True

            is_terminal = bool(info['reached_goal'])
            is_timeout = (step == max_path_length - 1) and not is_terminal

            ep_obs.append(obs)
            ep_acts.append(action)
            ep_rews.append(reward)
            ep_terms.append(is_terminal)
            ep_timeouts.append(is_timeout)

            obs = next_obs

            if done:
                break

        # Rigorous Quality Gates
        is_clean_success = info['reached_goal'] and not had_collision

        # Path optimality check
        is_optimal = True
        if is_clean_success and hasattr(policy, 'path_length') and policy.path_length > 0:
            # Theoretical minimum steps assuming average speed 0.85 m/s, dt=0.05s (~0.0425m/step)
            min_expected_steps = int(policy.path_length / 0.042)
            if len(ep_obs) > (min_expected_steps * max_suboptimality):
                is_optimal = False

        if had_collision:
            rejected_collision += 1
        elif not info['reached_goal']:
            rejected_timeout += 1
        elif not is_optimal:
            rejected_suboptimal += 1

        accept_episode = (is_clean_success and is_optimal) or not filter_successful

        if accept_episode:
            success_count += 1
            all_obs.extend(ep_obs)
            all_acts.extend(ep_acts)
            all_rews.extend(ep_rews)
            all_terms.extend(ep_terms)
            all_timeouts.extend(ep_timeouts)
            total_steps += len(ep_obs)

            if len(sample_paths) < 16:
                sample_paths.append(np.array(ep_obs))

            if success_count % 100 == 0 or success_count == num_episodes:
                rate = (success_count / ep_attempt) * 100.0
                print(f'Progress: {success_count}/{num_episodes} clean episodes ({rate:.1f}% acceptance) | Transitions: {total_steps} | Rejections: [Col: {rejected_collision}, Time: {rejected_timeout}, Subopt: {rejected_suboptimal}]')

    obs_arr = np.array(all_obs, dtype=np.float32)
    act_arr = np.array(all_acts, dtype=np.float32)
    rew_arr = np.array(all_rews, dtype=np.float32)
    term_arr = np.array(all_terms, dtype=bool)
    timeout_arr = np.array(all_timeouts, dtype=bool)

    # Backup existing dataset if present
    if os.path.exists(output_path):
        backup_path = output_path.replace('.npz', '_backup.npz')
        try:
            if not os.path.exists(backup_path):
                os.replace(output_path, backup_path)
                print(f'Backed up previous dataset to: {backup_path}')
        except Exception:
            pass

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=obs_arr,
        actions=act_arr,
        rewards=rew_arr,
        terminals=term_arr,
        timeouts=timeout_arr,
    )
    print(f'\n=== Dataset Collection Complete ===')
    print(f'Saved to: {output_path}')
    print(f'Total clean transitions: {total_steps}')
    print(f'Total episodes: {num_episodes}')
    print(f'Observations shape: {obs_arr.shape} | Actions shape: {act_arr.shape}')
    print(f'Overall Acceptance Rate: {(num_episodes / ep_attempt) * 100.0:.1f}% ({num_episodes}/{ep_attempt})')

    if render_sample and len(sample_paths) > 0:
        renderer = NavigationRenderer(env)
        plot_path = os.path.join(os.path.dirname(output_path), 'dataset_sample_paths.png')
        renderer.composite(plot_path, sample_paths)
        print(f'Sample trajectory visualization updated at: {plot_path}')

    env.close()
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect mobile robot navigation dataset for Diffuser')
    parser.add_argument('--num_episodes', type=int, default=2000, help='Number of successful episodes to collect')
    parser.add_argument('--max_path_length', type=int, default=200, help='Maximum steps per episode')
    parser.add_argument('--output_path', type=str, default='data/mobile_robot_dataset.npz', help='Output dataset path')
    parser.add_argument('--expert', type=str, default='global', choices=['global', 'potential'], help='Demonstrator policy type')
    parser.add_argument('--noise_std', type=float, default=0.015, help='Exploration noise standard deviation')
    parser.add_argument('--no_render', action='store_true', help='Skip saving sample trajectory figure')
    parser.add_argument('--all_episodes', action='store_true', help='Include failed episodes instead of filtering for success')
    parser.add_argument('--max_suboptimality', type=float, default=1.45, help='Max allowed step ratio vs theoretical minimum')

    args = parser.parse_args()
    collect_dataset(
        num_episodes=args.num_episodes,
        max_path_length=args.max_path_length,
        output_path=args.output_path,
        expert_type=args.expert,
        noise_std=args.noise_std,
        render_sample=not args.no_render,
        filter_successful=not args.all_episodes,
        max_suboptimality=args.max_suboptimality,
    )


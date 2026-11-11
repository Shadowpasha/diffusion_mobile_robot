import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
import imageio

from diffuser import utils
from diffuser.environments.mobile_robot import MobileRobotEnv, NavigationExpertPolicy
from diffuser.sampling import AnalyticalNavigationGuide, n_step_guided_p_sample


def add_sphere(scene, pos, radius=0.04, rgba=[0.1, 0.8, 1.0, 0.9]):
    """Adds a visual sphere to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mat = np.eye(3).flatten()
    size = np.array([radius, radius, radius])
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size,
        np.array(pos, dtype=np.float64),
        mat,
        np.array(rgba, dtype=np.float32)
    )
    scene.ngeom += 1


def add_capsule(scene, p1, p2, radius=0.015, rgba=[0.1, 0.9, 0.2, 0.7]):
    """Adds an oriented visual cylinder/capsule between two 3D points in an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    diff = p2 - p1
    length = float(np.linalg.norm(diff))
    if length < 1e-5:
        return
    mid = 0.5 * (p1 + p2)
    z = diff / length
    if abs(z[0]) < 0.9:
        x = np.cross([1.0, 0.0, 0.0], z)
    else:
        x = np.cross([0.0, 1.0, 0.0], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.column_stack([x, y, z]).flatten()
    size = np.array([radius, length * 0.5, 0.0])
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        size,
        mid,
        mat,
        np.array(rgba, dtype=np.float32)
    )
    scene.ngeom += 1


def draw_visual_overlays(
    scene,
    env,
    obs,
    current_plan=None,
    show_lidar=True,
    show_path=True,
):
    """
    Renders visual debug overlays into a MuJoCo scene:
    1. Active 2D LiDAR range perception beams (green for open space, orange/red for hits).
    2. Planned 32-step trajectory from the Diffuser model (cyan path with waypoint markers).
    """
    x = float(env.data.qpos[0])
    y = float(env.data.qpos[1])
    theta = float(env.data.qpos[2])

    # 1. Visualize 2D LiDAR beams
    if show_lidar and len(obs) >= 11 + env.n_lidar_beams:
        p_origin = np.array([x, y, 0.20], dtype=np.float64)
        ranges = obs[11:11 + env.n_lidar_beams]
        for i in range(len(ranges)):
            ang = theta + env._lidar_angles[i]
            d = float(ranges[i])
            hit = d < (env.lidar_max_range - 0.05)
            p_end = p_origin + np.array([d * np.cos(ang), d * np.sin(ang), 0.0], dtype=np.float64)

            if hit:
                # Obstacle or wall contact: warm orange ray with glowing red contact point
                add_capsule(scene, p_origin, p_end, radius=0.007, rgba=[1.0, 0.35, 0.05, 0.75])
                add_sphere(scene, p_end, radius=0.028, rgba=[1.0, 0.15, 0.0, 0.95])
            else:
                # Clear ray: semi-transparent green
                add_capsule(scene, p_origin, p_end, radius=0.004, rgba=[0.1, 0.85, 0.3, 0.25])

    # 2. Visualize Diffuser Planned Trajectory
    if show_path and current_plan is not None and len(current_plan) > 0:
        prev_p = np.array([x, y, 0.12], dtype=np.float64)
        horizon = len(current_plan)
        for t in range(horizon):
            pt_x = float(current_plan[t, 0])
            pt_y = float(current_plan[t, 1])
            curr_p = np.array([pt_x, pt_y, 0.12], dtype=np.float64)

            # Progressive color gradient along the horizon from bright cyan to deep blue
            alpha = max(0.25, 0.85 * (1.0 - 0.5 * (t / horizon)))
            col_capsule = [0.0, 0.8, 1.0, alpha]

            add_capsule(scene, prev_p, curr_p, radius=0.014, rgba=col_capsule)

            # Waypoint markers
            if (t + 1) % 4 == 0 or t == horizon - 1:
                is_endpoint = (t == horizon - 1)
                radius_wp = 0.038 if is_endpoint else 0.022
                col_wp = [1.0, 0.8, 0.0, 0.95] if is_endpoint else [0.2, 0.9, 1.0, 0.85]
                add_sphere(scene, curr_p, radius=radius_wp, rgba=col_wp)

            prev_p = curr_p


class DiffusionPolicy:
    """
    Receding-horizon navigation policy using trained Diffuser model.
    Conditions on current observation (t=0) and fixed goal position (t=H-1),
    generating waypoints/actions and replanning every `plan_stride` steps.
    """
    def __init__(
        self,
        logdir='logs/MobileRobotNav-v0/diffusion/navigation_H128_T50',
        epoch='latest',
        plan_stride=6,
        guided=False,
        guide_scale=0.05,
        n_guide_steps=2,
        tie_endpoint=True,
        control_mode='actions',
        waypoint_lookahead=0.45,
        n_candidates=4,
        device=None,
    ):
        self.plan_stride = plan_stride
        self.guided = guided
        self.guide_scale = guide_scale
        self.n_guide_steps = n_guide_steps
        self.tie_endpoint = tie_endpoint
        self.control_mode = control_mode
        self.waypoint_lookahead = waypoint_lookahead
        self.n_candidates = 1 if guided else max(1, int(n_candidates))
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_queue = []
        self.current_plan = None
        self.current_goal = None
        self.current_obstacles = None
        self._plan_step_count = 0
        self._wp_idx = 1

        print(f'[ DiffusionPolicy ] Loading checkpoint from {logdir} (epoch: {epoch}) on {self.device}...')
        model_config = utils.load_config(logdir, 'model_config.pkl')
        diffusion_config = utils.load_config(logdir, 'diffusion_config.pkl')
        dataset_config = utils.load_config(logdir, 'dataset_config.pkl')

        self.model = model_config()
        self.diffusion = diffusion_config(self.model)
        self.dataset = dataset_config()

        trainer = utils.Trainer(
            diffusion_model=self.diffusion,
            dataset=self.dataset,
            renderer=None,
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
        self.diffusion = trainer.ema_model
        self.diffusion.to(self.device)
        self.diffusion.eval()

        self.guide = None
        if self.guided:
            self.guide = AnalyticalNavigationGuide(
                normalizer=self.dataset.normalizer,
                action_dim=self.dataset.action_dim,
                safe_margin=0.95,
                w_goal=3.0,
                w_obs=6.0,
                w_wall=4.0,
                w_smooth=0.8,
            ).to(self.device)

        print(f'[ DiffusionPolicy ] Ready on {self.device} (Guided: {self.guided}, Tie Endpoint: {self.tie_endpoint}, Mode: {self.control_mode}).')

    def reset(self, goal=None, obstacles=None):
        self.action_queue.clear()
        self.current_plan = None
        self._plan_step_count = 0
        self._wp_idx = 1
        if goal is not None:
            self.current_goal = np.array(goal, dtype=np.float32)
        if obstacles is not None:
            self.current_obstacles = np.array(obstacles, dtype=np.float32)
        if self.guided and self.guide is not None:
            self.guide.set_targets(goal=self.current_goal, obstacles=self.current_obstacles)

    def _score_candidate_plan(self, pts, obs):
        """Score a candidate (H, 2) path using LiDAR obstacle hits, arena bounds, and smoothness."""
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
            min_surface_dist = np.min(dists, axis=1)
            # Robot radius is 0.28m; penalize predicted positions within 0.42m of any LiDAR hit
            clearance_violation = np.maximum(0.42 - min_surface_dist, 0.0)
            cost += 120.0 * float(np.sum(clearance_violation ** 2)) + 15.0 * float(np.max(clearance_violation))

        wall_violation = np.maximum(np.abs(near_pts) - 3.55, 0.0)
        cost += 50.0 * float(np.sum(wall_violation ** 2))

        diffs = np.diff(near_pts, axis=0)
        cost += 2.0 * float(np.sum(diffs ** 2))
        return cost

    def _track_waypoints(self, obs):
        x, y = float(obs[0]), float(obs[1])
        cos_th, sin_th = float(obs[2]), float(obs[3])
        theta = np.arctan2(sin_th, cos_th)
        omega = float(obs[5])
        dist_goal = float(obs[8])
        robot_pos = np.array([x, y], dtype=np.float32)

        waypoints = self.current_plan[:, :2]
        while self._wp_idx < len(waypoints) - 1:
            if np.linalg.norm(robot_pos - waypoints[self._wp_idx]) < self.waypoint_lookahead:
                self._wp_idx += 1
            else:
                break

        target_pt = waypoints[self._wp_idx]
        vec = target_pt - robot_pos
        desired_theta = np.arctan2(vec[1], vec[0])
        angle_err = np.arctan2(np.sin(desired_theta - theta), np.cos(desired_theta - theta))

        steer_cmd = np.clip(3.2 * angle_err - 0.35 * omega, -1.0, 1.0)
        speed = 0.95 if dist_goal > 0.8 else max(0.35, dist_goal / 0.8)
        min_fwd = 0.0 if abs(angle_err) > 1.05 else 0.15
        forward_cmd = np.clip(np.cos(angle_err) * speed, min_fwd, 1.0)
        self._plan_step_count += 1
        return np.array([forward_cmd, steer_cmd], dtype=np.float32)

    def act(self, obs, goal=None, obstacles=None):
        need_replan = (
            self.current_plan is None
            or (self.control_mode == 'actions' and not self.action_queue)
            or (self.control_mode == 'waypoints' and self._plan_step_count >= self.plan_stride)
        )
        if need_replan:
            t0 = time.time()
            if goal is not None:
                self.current_goal = np.array(goal, dtype=np.float32)
            if obstacles is not None:
                self.current_obstacles = np.array(obstacles, dtype=np.float32)

            # Recover world goal from ego-centric observation if not provided
            if self.current_goal is None and len(obs) >= 8:
                x, y = float(obs[0]), float(obs[1])
                cos_th, sin_th = float(obs[2]), float(obs[3])
                goal_rel_x, goal_rel_y = float(obs[6]), float(obs[7])
                gx = x + cos_th * goal_rel_x - sin_th * goal_rel_y
                gy = y + sin_th * goal_rel_x + cos_th * goal_rel_y
                self.current_goal = np.array([gx, gy], dtype=np.float32)

            if self.guided and self.guide is not None:
                self.guide.set_targets(goal=self.current_goal, obstacles=self.current_obstacles)

            batch_size = self.n_candidates
            normed_obs = self.dataset.normalizer(obs[None, :], 'observations')
            obs_cond = torch.tensor(normed_obs, device=self.device, dtype=torch.float32).repeat(batch_size, 1)
            conditions = {0: obs_cond}

            # Endpoint tying: pin terminal (x, y) directly to the goal position
            if self.tie_endpoint and self.current_goal is not None:
                obs_norm = self.dataset.normalizer.normalizers['observations']
                mins_xy = obs_norm.mins[:2]
                maxs_xy = obs_norm.maxs[:2]
                normed_goal_xy = 2.0 * (self.current_goal[:2] - mins_xy) / (maxs_xy - mins_xy) - 1.0
                goal_cond = torch.tensor(
                    normed_goal_xy[None, :], device=self.device, dtype=torch.float32
                ).repeat(batch_size, 1)
                conditions[self.diffusion.horizon - 1] = goal_cond

            shape = (batch_size, self.diffusion.horizon, self.diffusion.transition_dim)
            if self.guided and self.guide is not None:
                samples = self.diffusion.p_sample_loop(
                    shape,
                    conditions,
                    sample_fn=n_step_guided_p_sample,
                    guide=self.guide,
                    scale=self.guide_scale,
                    n_guide_steps=self.n_guide_steps,
                    scale_grad_by_std=True,
                    verbose=False,
                )
            else:
                with torch.no_grad():
                    samples = self.diffusion(conditions, verbose=False)

            trajectories = utils.to_np(samples.trajectories)
            best_idx = 0
            if batch_size > 1:
                best_cost = float('inf')
                for k in range(batch_size):
                    cand_obs = self.dataset.normalizer.unnormalize(
                        trajectories[k, :, self.dataset.action_dim:], 'observations'
                    )
                    cand_cost = self._score_candidate_plan(cand_obs[:, :2], obs)
                    if cand_cost < best_cost:
                        best_cost = cand_cost
                        best_idx = k

            normed_acts = trajectories[best_idx, :, :self.dataset.action_dim]
            acts = self.dataset.normalizer.unnormalize(normed_acts, 'actions')
            normed_preds = trajectories[best_idx, :, self.dataset.action_dim:]
            self.current_plan = self.dataset.normalizer.unnormalize(normed_preds, 'observations')

            num_to_queue = min(self.plan_stride, len(acts))
            self.action_queue = [acts[i] for i in range(num_to_queue)]
            self._plan_step_count = 0
            self._wp_idx = min(3, len(self.current_plan) - 1)
            gen_time = time.time() - t0
            print(f'  [Replanned {len(acts)} steps in {gen_time:.2f}s] Mode: {self.control_mode} | Current goal dist: {obs[8]:.2f}m')

        if self.control_mode == 'waypoints':
            return self._track_waypoints(obs)
        return self.action_queue.pop(0)


def run_simulation_test(
    num_episodes=3,
    max_steps=160,
    policy_type='diffusion',
    logdir='logs/MobileRobotNav-v0/diffusion/navigation_H128_T50',
    plan_stride=6,
    guided=False,
    guide_scale=0.05,
    tie_endpoint=True,
    control_mode='actions',
    n_candidates=4,
    step_delay=0.02,
    headless=False,
    save_video_path=None,
    terminate_on_collision=False,
    show_lidar=True,
    show_path=True,
):
    print(f'=== Starting MuJoCo Simulation Test ===')
    print(f'Policy: {policy_type.upper()} | Episodes: {num_episodes} | Max steps: {max_steps} | Headless: {headless}')
    print(f'Guided: {guided} (scale: {guide_scale}) | Tie Endpoint: {tie_endpoint} | Control Mode: {control_mode}')
    print(f'Visualizations: LiDAR={show_lidar} | Diffuser Path={show_path}')

    if policy_type == 'diffusion':
        policy = DiffusionPolicy(
            logdir=logdir,
            plan_stride=plan_stride,
            guided=guided,
            guide_scale=guide_scale,
            tie_endpoint=tie_endpoint,
            control_mode=control_mode,
            n_candidates=n_candidates,
        )
        n_lidar_beams = max(1, policy.dataset.observation_dim - 11)
        env = MobileRobotEnv(
            max_episode_steps=max_steps,
            terminate_on_collision=terminate_on_collision,
            n_lidar_beams=n_lidar_beams,
            randomize_obstacles=True,
        )
    else:
        env = MobileRobotEnv(
            max_episode_steps=max_steps,
            terminate_on_collision=terminate_on_collision,
            randomize_obstacles=True,
        )
        policy = NavigationExpertPolicy(env, noise_std=0.05)

    recorded_frames = []
    stats = {
        'episodes': 0,
        'successes': 0,
        'collisions': 0,
        'steps': [],
        'final_dists': [],
    }

    viewer_ctx = None
    if not headless:
        viewer_ctx = mujoco.viewer.launch_passive(env.model, env.data)
        viewer_ctx.cam.distance = 9.0
        viewer_ctx.cam.elevation = -45.0
        viewer_ctx.cam.azimuth = 90.0
        viewer_ctx.cam.lookat[:] = [0.0, 0.0, 0.2]

    # Pre-initialize offscreen renderer if saving video
    offscreen_renderer = None
    if save_video_path is not None:
        offscreen_renderer = mujoco.Renderer(env.model, height=480, width=640)

    try:
        for ep in range(num_episodes):
            if viewer_ctx is not None and not viewer_ctx.is_running():
                print('Viewer closed by user.')
                break

            obs = env.reset()
            policy.reset(goal=env.current_goal.copy(), obstacles=env.obstacles.copy())

            start_pos = env.data.qpos[:2].copy()
            goal_pos = env.current_goal.copy()
            initial_dist = float(np.linalg.norm(goal_pos - start_pos))
            print(f'\n--- Episode {ep + 1}/{num_episodes} ---')
            print(f'Start: [{start_pos[0]:.2f}, {start_pos[1]:.2f}] -> Goal: [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}] (Distance: {initial_dist:.2f} m)')

            had_collision = False
            for step in range(max_steps):
                if viewer_ctx is not None and not viewer_ctx.is_running():
                    break

                step_start = time.time()

                action = policy.act(obs, goal=env.current_goal, obstacles=env.obstacles)
                obs, reward, done, info = env.step(action)

                if info['collision']:
                    had_collision = True

                # Retrieve current plan for visual overlay
                current_plan = getattr(policy, 'current_plan', None)

                # Update interactive 3D desktop viewer
                if viewer_ctx is not None:
                    viewer_ctx.user_scn.ngeom = 0
                    draw_visual_overlays(
                        viewer_ctx.user_scn,
                        env,
                        obs,
                        current_plan=current_plan,
                        show_lidar=show_lidar,
                        show_path=show_path,
                    )
                    viewer_ctx.sync()

                # Record offscreen RGB frame with visual overlays
                if offscreen_renderer is not None:
                    offscreen_renderer.update_scene(env.data, camera='overview')
                    draw_visual_overlays(
                        offscreen_renderer.scene,
                        env,
                        obs,
                        current_plan=current_plan,
                        show_lidar=show_lidar,
                        show_path=show_path,
                    )
                    frame = offscreen_renderer.render()
                    recorded_frames.append(frame)

                if not headless and step_delay > 0:
                    elapsed = time.time() - step_start
                    if elapsed < step_delay:
                        time.sleep(step_delay - elapsed)

                if done:
                    break

            stats['episodes'] += 1
            stats['steps'].append(step + 1)
            stats['final_dists'].append(info['dist_to_goal'])
            if info['reached_goal']:
                stats['successes'] += 1
                outcome = 'SUCCESS (Goal Reached)'
            elif info['collision']:
                outcome = 'COLLISION'
            else:
                outcome = 'TIMEOUT'

            if had_collision:
                stats['collisions'] += 1

            print(f'Result: {outcome} | Steps: {step + 1}/{max_steps} | Final Dist: {info["dist_to_goal"]:.2f} m | Collided: {had_collision}')
            if not headless:
                time.sleep(0.5)

    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
        if offscreen_renderer is not None:
            offscreen_renderer.close()
        env.close()

    # Save recorded video if requested
    if save_video_path and len(recorded_frames) > 0:
        os.makedirs(os.path.dirname(os.path.abspath(save_video_path)), exist_ok=True)
        imageio.mimsave(save_video_path, recorded_frames, fps=20)
        print(f'\nSaved simulation MP4 video with visual overlays to: {save_video_path}')

    # Print summary evaluation metrics
    print(f'\n================ Evaluation Summary ================')
    print(f'Total Episodes Tested: {stats["episodes"]}')
    if stats["episodes"] > 0:
        succ_rate = 100.0 * stats['successes'] / stats['episodes']
        col_rate = 100.0 * stats['collisions'] / stats['episodes']
        mean_steps = np.mean(stats['steps'])
        mean_final_dist = np.mean(stats['final_dists'])
        print(f'Success Rate:         {succ_rate:.1f}% ({stats["successes"]}/{stats["episodes"]})')
        print(f'Collision Rate:       {col_rate:.1f}% ({stats["collisions"]}/{stats["episodes"]})')
        print(f'Mean Steps to Goal:   {mean_steps:.1f}')
        print(f'Mean Final Distance:  {mean_final_dist:.2f} m')
    print(f'====================================================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuJoCo Simulation Testing for Navigation Policy')
    parser.add_argument('--policy', type=str, default='diffusion', choices=['diffusion', 'expert'], help='Policy to evaluate (diffusion or expert)')
    parser.add_argument('--logdir', type=str, default='logs/MobileRobotNav-v0/diffusion/navigation_H128_T50', help='Trained model log directory')
    parser.add_argument('--episodes', type=int, default=3, help='Number of test episodes')
    parser.add_argument('--max_steps', type=int, default=160, help='Maximum steps per episode')
    parser.add_argument('--stride', type=int, default=6, help='Replanning stride (number of steps executed per plan generation)')
    parser.add_argument('--delay', type=float, default=0.02, help='Playback delay between simulation steps')
    parser.add_argument('--headless', action='store_true', help='Run headless without launching desktop GUI window')
    parser.add_argument('--save_video', type=str, default='data/diffusion_sim_eval.mp4', help='Path to save MP4 evaluation recording')
    parser.add_argument('--terminate_on_collision', action='store_true', help='Immediately terminate episode on obstacle collision')
    parser.add_argument('--no_lidar', action='store_true', help='Disable LiDAR beam visualization')
    parser.add_argument('--no_path', action='store_true', help='Disable predicted diffuser path visualization')
    parser.add_argument('--guide', action='store_true', help='Enable analytical cost guidance')
    parser.add_argument('--no_guide', action='store_true', help='Disable analytical cost guidance (default)')
    parser.add_argument('--guide_scale', type=float, default=0.05, help='Scale of analytical cost guidance gradient')
    parser.add_argument('--no_tie_endpoint', action='store_true', help='Disable fixing terminal step to goal')
    parser.add_argument('--control_mode', type=str, default='actions', choices=['actions', 'waypoints'], help='Execute predicted actions directly or follow generated waypoints via PD tracker')
    parser.add_argument('--n_candidates', type=int, default=4, help='Parallel candidate trajectories scored by LiDAR clearance per replan (unguided)')

    args = parser.parse_args()
    run_simulation_test(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        policy_type=args.policy,
        logdir=args.logdir,
        plan_stride=args.stride,
        guided=args.guide and not args.no_guide,
        guide_scale=args.guide_scale,
        tie_endpoint=not args.no_tie_endpoint,
        control_mode=args.control_mode,
        n_candidates=args.n_candidates,
        step_delay=args.delay,
        headless=args.headless,
        save_video_path=args.save_video,
        terminate_on_collision=args.terminate_on_collision,
        show_lidar=not args.no_lidar,
        show_path=not args.no_path,
    )


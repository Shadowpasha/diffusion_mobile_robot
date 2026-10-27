import os
import heapq
import gym
from gym import spaces
import numpy as np
import mujoco


class MobileRobotEnv(gym.Env):
    """
    Mobile robot navigation environment in MuJoCo.
    The robot is a 3-DOF planar mobile platform navigating in an enclosed arena
    with cylindrical obstacles towards a target goal location.
    """
    metadata = {'render.modes': ['rgb_array'], 'video.frames_per_second': 20}

    def __init__(
        self,
        xml_path=None,
        max_episode_steps=100,
        substeps=5,
        goal_threshold=0.45,
        dataset_path=None,
        terminate_on_collision=True,
        n_lidar_beams=16,
        lidar_max_range=5.0,
        randomize_obstacles=True,
    ):
        super().__init__()
        self.terminate_on_collision = terminate_on_collision
        self.randomize_obstacles = randomize_obstacles
        self.n_lidar_beams = n_lidar_beams
        self.lidar_max_range = float(lidar_max_range)
        self.lidar_origin_z = 0.20

        if xml_path is None:
            base_dir = os.path.dirname(__file__)
            xml_path = os.path.join(base_dir, 'assets', 'mobile_robot.xml')

        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.robot_body_id = self.model.body('robot').id
        self._lidar_angles = np.linspace(0, 2 * np.pi, self.n_lidar_beams, endpoint=False).astype(np.float64)

        # Identify mocap body indices
        if self.model.nmocap > 0:
            self.target_mocap_id = self.model.body('target').mocapid[0]
            self.obs_mocap_ids = []
            for i in range(4):
                try:
                    bid = self.model.body(f'obs_{i+1}').mocapid[0]
                    if bid >= 0:
                        self.obs_mocap_ids.append(bid)
                except Exception:
                    pass
        else:
            self.target_mocap_id = -1
            self.obs_mocap_ids = []

        self.substeps = substeps
        self.dt = self.model.opt.timestep * substeps
        self.goal_threshold = goal_threshold
        self.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self.name = 'MobileRobotNav-v0'

        # Default dataset storage path
        if dataset_path is None:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            self.dataset_path = os.path.join(root_dir, 'data', 'mobile_robot_dataset.npz')
        else:
            self.dataset_path = dataset_path

        self._dataset = None

        # Obstacle definitions matching mobile_robot.xml (updated dynamically when randomized)
        self.obstacles = np.array([
            [ 1.6,  1.6],
            [-1.6,  1.6],
            [ 1.6, -1.6],
            [-1.6, -1.6],
        ], dtype=np.float32)
        self.obstacle_radius = 0.5
        self.robot_radius = 0.28
        self.arena_limit = 3.4
        self.wheel_radius = 0.075
        self.track_width = 0.44

        # Action: [thrust_v in [-1, 1], steer_omega in [-1, 1]]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation (27 dims with 16 LiDAR beams):
        # 0, 1: x, y (robot global position)
        # 2, 3: cos(theta), sin(theta) (robot heading)
        # 4: v_forward (forward linear velocity)
        # 5: omega (yaw angular velocity)
        # 6: goal_rel_x (forward distance to goal in body frame, meters)
        # 7: goal_rel_y (lateral distance to goal in body frame, meters)
        # 8: goal_dist (Euclidean distance to goal, meters)
        # 9: goal_cos (cosine of relative angle to goal)
        # 10: goal_sin (sine of relative angle to goal)
        # 11..26: 16 LiDAR range beams (meters)
        self.obs_dim = 11 + self.n_lidar_beams
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.current_goal = np.array([2.5, 2.5], dtype=np.float32)
        self.step_count = 0
        self._prev_dist = 0.0

        # Offscreen renderer
        self._renderer = None

    def _get_lidar(self):
        """
        Simulates an ego-centric 2D planar LiDAR with n_lidar_beams uniformly
        distributed over 360 degrees. Beam 0 points directly forward along the robot
        heading theta; beam angles increment counter-clockwise.
        Returns distance readings in meters, clipped to lidar_max_range.
        """
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        theta = float(self.data.qpos[2])

        pnt = np.array([x, y, self.lidar_origin_z], dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        ranges = np.empty(self.n_lidar_beams, dtype=np.float32)

        angles = theta + self._lidar_angles
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        for i in range(self.n_lidar_beams):
            vec = np.array([cos_a[i], sin_a[i], 0.0], dtype=np.float64)
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                pnt,
                vec,
                None,
                1,
                self.robot_body_id,
                geomid
            )
            if dist < 0 or dist > self.lidar_max_range:
                ranges[i] = self.lidar_max_range
            else:
                ranges[i] = dist

        return ranges

    def _get_obs(self):
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        theta = float(self.data.qpos[2])
        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        omega = float(self.data.qvel[2])

        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        v_forward = vx * cos_th + vy * sin_th

        # Ego-centric relative goal representation
        dx = float(self.current_goal[0]) - x
        dy = float(self.current_goal[1]) - y
        goal_dist = np.sqrt(dx * dx + dy * dy)

        # Transform relative goal vector into robot body frame (forward = +x, left = +y)
        goal_rel_x = dx * cos_th + dy * sin_th
        goal_rel_y = -dx * sin_th + dy * cos_th

        # Continuous relative heading error to goal in body frame
        goal_rel_ang = np.arctan2(goal_rel_y, goal_rel_x)
        goal_cos = np.cos(goal_rel_ang)
        goal_sin = np.sin(goal_rel_ang)

        lidar = self._get_lidar()

        obs = np.empty(self.obs_dim, dtype=np.float32)
        obs[0] = x
        obs[1] = y
        obs[2] = cos_th
        obs[3] = sin_th
        obs[4] = v_forward
        obs[5] = omega
        obs[6] = goal_rel_x
        obs[7] = goal_rel_y
        obs[8] = goal_dist
        obs[9] = goal_cos
        obs[10] = goal_sin
        obs[11:] = lidar
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def _is_valid_point(self, point, min_dist_obs=0.9):
        if np.any(np.abs(point) > self.arena_limit):
            return False
        for obs in self.obstacles:
            if np.linalg.norm(point - obs) < min_dist_obs:
                return False
        return True

    def _sample_valid_position(self, min_dist_obs=0.95):
        for _ in range(200):
            pt = np.random.uniform(-self.arena_limit, self.arena_limit, size=(2,)).astype(np.float32)
            if self._is_valid_point(pt, min_dist_obs=min_dist_obs):
                return pt
        return np.array([0.0, 0.0], dtype=np.float32)

    def _sample_obstacle_positions(self):
        """
        Samples random obstacle positions ensuring they stay within the arena
        and maintain at least min_dist_between from one another.
        """
        min_dist_between = 2.0 * self.obstacle_radius + 0.6  # 1.6m
        spawn_limit = 2.4
        placed = []
        for _ in range(4):
            valid = False
            for _ in range(300):
                pt = np.random.uniform(-spawn_limit, spawn_limit, size=(2,)).astype(np.float32)
                if all(np.linalg.norm(pt - other) >= min_dist_between for other in placed):
                    placed.append(pt)
                    valid = True
                    break
            if not valid:
                fallback_coords = [
                    np.array([ 1.6,  1.6], dtype=np.float32),
                    np.array([-1.6,  1.6], dtype=np.float32),
                    np.array([ 1.6, -1.6], dtype=np.float32),
                    np.array([-1.6, -1.6], dtype=np.float32),
                ]
                placed.append(fallback_coords[len(placed)])
        return np.array(placed, dtype=np.float32)

    def reset(self, start_pos=None, goal_pos=None, obstacle_positions=None, seed=None):
        if seed is not None:
            self.seed(seed)

        self.step_count = 0

        # Randomize or set obstacle positions
        if obstacle_positions is not None:
            self.obstacles = np.array(obstacle_positions, dtype=np.float32)
        elif self.randomize_obstacles and len(self.obs_mocap_ids) == 4:
            self.obstacles = self._sample_obstacle_positions()

        # Update obstacle mocap bodies in MuJoCo
        for i, mocap_id in enumerate(self.obs_mocap_ids):
            if i < len(self.obstacles):
                self.data.mocap_pos[mocap_id] = [self.obstacles[i, 0], self.obstacles[i, 1], 0.3]

        if start_pos is None:
            start_pos = self._sample_valid_position()
        if goal_pos is None:
            for _ in range(200):
                goal_pos = self._sample_valid_position()
                if np.linalg.norm(goal_pos - start_pos) > 2.0:
                    break

        self.current_goal = np.array(goal_pos, dtype=np.float32)

        # Set target visual marker
        if self.target_mocap_id >= 0:
            self.data.mocap_pos[self.target_mocap_id] = [self.current_goal[0], self.current_goal[1], 0.05]

        # Initial heading towards goal with small random jitter
        to_goal = self.current_goal - start_pos
        init_theta = np.arctan2(to_goal[1], to_goal[0]) + np.random.uniform(-0.4, 0.4)

        self.data.qpos[0] = start_pos[0]
        self.data.qpos[1] = start_pos[1]
        self.data.qpos[2] = init_theta
        if self.data.qpos.shape[0] > 3:
            self.data.qpos[3:] = 0.0
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._prev_dist = float(np.linalg.norm(self.current_goal - start_pos))
        return self._get_obs()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        a_v, a_omega = float(action[0]), float(action[1])

        f_thrust = a_v * 18.0
        tau_steer = a_omega * 5.0

        for _ in range(self.substeps):
            theta = float(self.data.qpos[2])
            vx = float(self.data.qvel[0])
            vy = float(self.data.qvel[1])
            omega = float(self.data.qvel[2])

            # Forward and lateral velocities in robot-centric body frame
            v_forward = vx * np.cos(theta) + vy * np.sin(theta)
            v_lateral = -vx * np.sin(theta) + vy * np.cos(theta)

            # Lateral tire friction: completely opposes sideways slipping
            f_lateral = -60.0 * v_lateral

            # Transform forces from robot frame to world frame
            f_x = f_thrust * np.cos(theta) - f_lateral * np.sin(theta)
            f_y = f_thrust * np.sin(theta) + f_lateral * np.cos(theta)

            self.data.qfrc_applied[0] = f_x
            self.data.qfrc_applied[1] = f_y
            self.data.qfrc_applied[2] = tau_steer

            # Spin 4 wheels in MuJoCo according to 4WD differential drive kinematics
            if self.data.qvel.shape[0] >= 7:
                v_l = v_forward - 0.5 * self.track_width * omega
                v_r = v_forward + 0.5 * self.track_width * omega
                w_l = v_l / self.wheel_radius
                w_r = v_r / self.wheel_radius
                self.data.qvel[3] = w_l  # front-left
                self.data.qvel[4] = w_l  # rear-left
                self.data.qvel[5] = w_r  # front-right
                self.data.qvel[6] = w_r  # rear-right
            elif self.data.qvel.shape[0] >= 5:
                v_l = v_forward - 0.5 * self.track_width * omega
                v_r = v_forward + 0.5 * self.track_width * omega
                self.data.qvel[3] = v_l / self.wheel_radius
                self.data.qvel[4] = v_r / self.wheel_radius

            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        pos = np.array([self.data.qpos[0], self.data.qpos[1]], dtype=np.float32)
        dist_to_goal = float(np.linalg.norm(self.current_goal - pos))

        # Reward shaping
        reward = (self._prev_dist - dist_to_goal) * 10.0
        self._prev_dist = dist_to_goal

        # Control effort penalty
        reward -= 0.005 * (a_v ** 2 + a_omega ** 2)

        # Check collisions against boundary walls and obstacle cylinders
        collision = False
        if np.any(np.abs(pos) > (self.arena_limit + 0.35)):
            collision = True
        else:
            for obs in self.obstacles:
                if np.linalg.norm(pos - obs) < (self.obstacle_radius + self.robot_radius):
                    collision = True
                    break

        # Also inspect MuJoCo physical contact manifold for impact forces
        if not collision and self.data.ncon > 0:
            floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
            for c_idx in range(self.data.ncon):
                g1 = self.data.contact[c_idx].geom1
                g2 = self.data.contact[c_idx].geom2
                if g1 != floor_id and g2 != floor_id:
                    collision = True
                    break

        # Termination condition
        reached_goal = dist_to_goal < self.goal_threshold
        if reached_goal:
            reward += 10.0
            done = True
        elif collision and self.terminate_on_collision:
            reward -= 10.0
            done = True
        elif self.step_count >= self.max_episode_steps:
            done = True
        else:
            done = False

        obs = self._get_obs()
        info = {
            'dist_to_goal': dist_to_goal,
            'reached_goal': reached_goal,
            'collision': collision,
            'pos': pos,
            'goal': self.current_goal.copy(),
        }

        return obs, float(reward), done, info

    def render(self, mode='rgb_array', camera_name='topdown', height=480, width=640):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def get_dataset(self):
        """
        Returns an offline dataset formatted for Diffuser:
        {
            'observations': np.ndarray of shape (N, 8),
            'actions': np.ndarray of shape (N, 2),
            'rewards': np.ndarray of shape (N,),
            'terminals': np.ndarray of shape (N,),
            'timeouts': np.ndarray of shape (N,)
        }
        """
        if self._dataset is not None:
            return self._dataset

        if os.path.exists(self.dataset_path):
            print(f'[ MobileRobotEnv ] Loading existing dataset from {self.dataset_path}')
            data = np.load(self.dataset_path)
            self._dataset = {
                'observations': data['observations'].astype(np.float32),
                'actions': data['actions'].astype(np.float32),
                'rewards': data['rewards'].astype(np.float32),
                'terminals': data['terminals'].astype(bool),
                'timeouts': data['timeouts'].astype(bool),
            }
            return self._dataset

        raise FileNotFoundError(
            f'Dataset not found at {self.dataset_path}. '
            'Please run scripts/collect_navigation_data.py to collect the dataset.'
        )

    def set_dataset(self, dataset):
        self._dataset = dataset


class NavigationExpertPolicy:
    """
    Goal-seeking policy using attractive target potential
    combined with obstacle and boundary repulsion forces.
    """
    def __init__(self, env: MobileRobotEnv, noise_std=0.08):
        self.env = env
        self.noise_std = noise_std
        self.obstacles = env.obstacles
        self.obs_margin = env.obstacle_radius + env.robot_radius + 0.55
        self.arena_limit = env.arena_limit

    def act(self, obs):
        x, y = float(obs[0]), float(obs[1])
        cos_th, sin_th = float(obs[2]), float(obs[3])
        theta = np.arctan2(sin_th, cos_th)
        omega = float(obs[5])
        goal_rel_x = float(obs[6])
        goal_rel_y = float(obs[7])
        goal_dist = float(obs[8])
        goal_cos = float(obs[9])
        goal_sin = float(obs[10])

        robot_pos = np.array([x, y], dtype=np.float32)

        # 1. Target Attraction: ego-centric unit direction rotated into world frame
        f_att = np.array([
            goal_cos * cos_th - goal_sin * sin_th,
            goal_cos * sin_th + goal_sin * cos_th,
        ], dtype=np.float32)

        # 2. Obstacle Repulsion (queried from env.obstacles)
        f_rep = np.zeros(2, dtype=np.float32)
        for obs_center in self.env.obstacles:
            diff_obs = robot_pos - obs_center
            dist_obs = float(np.linalg.norm(diff_obs))
            if dist_obs < self.obs_margin:
                penetration = (self.obs_margin - dist_obs) / self.obs_margin
                radial_dir = diff_obs / (dist_obs + 1e-4)
                # Tangent direction (circulate around obstacle toward goal)
                tangent_dir = np.array([-radial_dir[1], radial_dir[0]], dtype=np.float32)
                if np.dot(tangent_dir, f_att) < 0:
                    tangent_dir = -tangent_dir

                f_rep += radial_dir * (penetration * 3.5) + tangent_dir * (penetration * 2.5)

        # 3. Wall Repulsion
        wall_margin = 0.5
        if abs(x) > (self.arena_limit - wall_margin):
            f_rep[0] -= np.sign(x) * (abs(x) - (self.arena_limit - wall_margin)) * 3.0
        if abs(y) > (self.arena_limit - wall_margin):
            f_rep[1] -= np.sign(y) * (abs(y) - (self.arena_limit - wall_margin)) * 3.0

        # 4. Direct LiDAR proximity avoidance
        if len(obs) > 11:
            lidar = obs[11:]
            for i, dist in enumerate(lidar):
                if dist < 0.9:
                    beam_angle = theta + self.env._lidar_angles[i]
                    rep_strength = (0.9 - dist) * 2.5
                    f_rep[0] -= np.cos(beam_angle) * rep_strength
                    f_rep[1] -= np.sin(beam_angle) * rep_strength

        f_net = f_att + f_rep
        desired_theta = np.arctan2(f_net[1], f_net[0])

        angle_err = np.arctan2(
            np.sin(desired_theta - theta),
            np.cos(desired_theta - theta)
        )

        # PD steering controller
        steer_cmd = np.clip(3.5 * angle_err - 0.5 * omega, -1.0, 1.0)
        # Slow down when turning sharply
        forward_cmd = np.clip(np.cos(angle_err), 0.1, 1.0)

        # Add exploration noise if requested
        if self.noise_std > 0:
            forward_cmd += np.random.normal(0, self.noise_std)
            steer_cmd += np.random.normal(0, self.noise_std)

        action = np.array([
            np.clip(forward_cmd, -1.0, 1.0),
            np.clip(steer_cmd, -1.0, 1.0),
        ], dtype=np.float32)

        return action


class GlobalNavigationExpertPolicy:
    """
    High-quality expert demonstrator using global A* path search
    and a lookahead tracking controller. Generates smooth, collision-free,
    direct trajectories around obstacles.
    """
    def __init__(
        self,
        env: MobileRobotEnv,
        noise_std=0.015,
        resolution=0.1,
        lookahead_dist=0.45,
        safe_margin=0.92,
    ):
        self.env = env
        self.noise_std = noise_std
        self.res = resolution
        self.lookahead_dist = lookahead_dist
        self.safe_margin = safe_margin
        self.arena_limit = env.arena_limit
        self.min_val = -env.arena_limit
        self.grid_size = int(np.ceil((2 * env.arena_limit) / resolution))

        self.current_path = None
        self.path_idx = 0
        self.path_length = 0.0

    def pos_to_idx(self, p):
        ix = int(np.clip(np.round((p[0] - self.min_val) / self.res), 0, self.grid_size - 1))
        iy = int(np.clip(np.round((p[1] - self.min_val) / self.res), 0, self.grid_size - 1))
        return ix, iy

    def idx_to_pos(self, ix, iy):
        px = self.min_val + ix * self.res
        py = self.min_val + iy * self.res
        return np.array([px, py], dtype=np.float32)

    def plan_path(self, start, goal, obstacles):
        start_idx = self.pos_to_idx(start)
        goal_idx = self.pos_to_idx(goal)

        open_set = []
        heapq.heappush(open_set, (0, start_idx))
        came_from = {}
        g_score = {start_idx: 0.0}

        def heuristic(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])

        dirs = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)
        ]

        found = False
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_idx:
                found = True
                break

            for dx, dy, cost in dirs:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    pos = self.idx_to_pos(nx, ny)
                    # Check obstacle margin
                    collides = False
                    for obs in obstacles:
                        if np.linalg.norm(pos - obs) < self.safe_margin:
                            collides = True
                            break
                    if collides:
                        continue

                    tentative_g = g_score[current] + cost
                    neighbor = (nx, ny)
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, goal_idx)
                        heapq.heappush(open_set, (f_score, neighbor))

        if not found:
            return None

        curr = goal_idx
        pts = []
        while curr in came_from:
            pts.append(self.idx_to_pos(curr[0], curr[1]))
            curr = came_from[curr]
        pts.append(np.array(start, dtype=np.float32))
        pts.reverse()
        pts.append(np.array(goal, dtype=np.float32))
        path_arr = np.array(pts, dtype=np.float32)

        # Compute total geometric length
        diffs = np.diff(path_arr, axis=0)
        self.path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        return path_arr

    def reset(self, start_pos=None, goal_pos=None, obstacles=None):
        start = start_pos if start_pos is not None else self.env.data.qpos[:2].copy()
        goal = goal_pos if goal_pos is not None else self.env.current_goal.copy()
        obs = obstacles if obstacles is not None else self.env.obstacles.copy()
        self.current_path = self.plan_path(start, goal, obs)
        self.path_idx = 0

    def act(self, obs):
        x, y = float(obs[0]), float(obs[1])
        cos_th, sin_th = float(obs[2]), float(obs[3])
        theta = np.arctan2(sin_th, cos_th)
        omega = float(obs[5])
        dist_goal = float(obs[8])
        robot_pos = np.array([x, y], dtype=np.float32)

        if self.current_path is None or len(self.current_path) == 0:
            goal_rel_x, goal_rel_y = float(obs[6]), float(obs[7])
            desired_theta = np.arctan2(
                sin_th * goal_rel_x + cos_th * goal_rel_y,
                cos_th * goal_rel_x - sin_th * goal_rel_y
            )
        else:
            # Advance lookahead index along path
            while self.path_idx < len(self.current_path) - 1:
                dist_wp = np.linalg.norm(robot_pos - self.current_path[self.path_idx])
                if dist_wp < self.lookahead_dist:
                    self.path_idx += 1
                else:
                    break

            target_pt = self.current_path[self.path_idx]
            vec = target_pt - robot_pos
            desired_theta = np.arctan2(vec[1], vec[0])

        angle_err = np.arctan2(
            np.sin(desired_theta - theta),
            np.cos(desired_theta - theta)
        )

        steer_cmd = np.clip(3.2 * angle_err - 0.35 * omega, -1.0, 1.0)
        speed = 0.95 if dist_goal > 0.8 else max(0.35, dist_goal / 0.8)
        forward_cmd = np.clip(np.cos(angle_err) * speed, 0.15, 1.0)

        if self.noise_std > 0:
            forward_cmd += np.random.normal(0, self.noise_std)
            steer_cmd += np.random.normal(0, self.noise_std)

        return np.array([
            np.clip(forward_cmd, -1.0, 1.0),
            np.clip(steer_cmd, -1.0, 1.0)
        ], dtype=np.float32)


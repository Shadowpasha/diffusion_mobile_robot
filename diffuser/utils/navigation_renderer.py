import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio


class NavigationRenderer:
    """
    Trajectory and arena renderer for mobile robot navigation.
    Renders top-down 2D trajectory plots showing arena boundaries,
    obstacles, start positions, target goals, and predicted diffusion paths.
    """
    def __init__(self, env='MobileRobotNav-v0'):
        if isinstance(env, str):
            import gym
            try:
                self.env = gym.make(env).unwrapped
            except Exception:
                from diffuser.environments.mobile_robot import MobileRobotEnv
                self.env = MobileRobotEnv()
        else:
            self.env = getattr(env, 'unwrapped', env)

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.obstacles = getattr(self.env, 'obstacles', np.array([
            [ 1.6,  1.6],
            [-1.6,  1.6],
            [ 1.6, -1.6],
            [-1.6, -1.6],
        ]))
        self.obstacle_radius = getattr(self.env, 'obstacle_radius', 0.5)
        self.arena_limit = getattr(self.env, 'arena_limit', 3.6)

    def _draw_arena(self, ax, obstacles=None):
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-4.2, 4.2)
        ax.set_aspect('equal')
        ax.set_facecolor('#f8fafc')

        # Outer walls
        rect = patches.Rectangle(
            (-4.0, -4.0), 8.0, 8.0,
            linewidth=2.5, edgecolor='#334155', facecolor='none'
        )
        ax.add_patch(rect)

        # Obstacles (dynamic or default)
        if obstacles is None:
            obstacles = getattr(self.env, 'obstacles', self.obstacles)
        for center in obstacles:
            circle = patches.Circle(
                (center[0], center[1]),
                self.obstacle_radius,
                facecolor='#ef4444',
                edgecolor='#991b1b',
                alpha=0.85,
                linewidth=1.5,
                zorder=2
            )
            ax.add_patch(circle)

        ax.grid(True, linestyle='--', alpha=0.3, color='#94a3b8')
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)

    def composite(self, savepath, paths, dim=(512, 512), obstacles=None, **kwargs):
        """
        Renders a composite 2D trajectory figure for multiple paths.
        paths: array of shape (B, Horizon, ObsDim) or (Horizon, ObsDim)
        """
        if isinstance(paths, np.ndarray) and paths.ndim == 2:
            paths = [paths]
        elif not isinstance(paths, list):
            paths = list(paths)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        self._draw_arena(ax, obstacles=obstacles)

        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(paths)))

        for i, path in enumerate(paths):
            x = path[:, 0]
            y = path[:, 1]
            c = colors[i]

            ax.plot(x, y, color=c, linewidth=2.0, alpha=0.8, zorder=3)
            # Start position
            ax.scatter(x[0], y[0], color='#2563eb', s=60, edgecolors='black', zorder=4)
            # End of trajectory
            ax.scatter(x[-1], y[-1], color='#f59e0b', s=50, marker='o', edgecolors='black', zorder=4)

            # Target position reconstructed from start state and ego-centric goal offset
            if path.shape[-1] >= 8:
                x0, y0 = path[0, 0], path[0, 1]
                cos0, sin0 = path[0, 2], path[0, 3]
                rel_x, rel_y = path[0, 6], path[0, 7]
                gx = x0 + rel_x * cos0 - rel_y * sin0
                gy = y0 + rel_x * sin0 + rel_y * cos0
                ax.scatter(gx, gy, color='#10b981', s=120, marker='*', edgecolors='black', zorder=5)

        ax.set_title(f'Mobile Robot Trajectories (N={len(paths)})', fontsize=12, fontweight='bold')
        fig.tight_layout()

        if savepath is not None:
            os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
            fig.savefig(savepath, bbox_inches='tight')
            print(f'[ NavigationRenderer ] Saved composite visualization to: {savepath}')

        # Canvas to image array
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
        plt.close(fig)
        return rgb

    def render_plan(self, savepath, actions, observations_pred, state, fps=20):
        """
        Visualizes planned diffusion trajectories.
        """
        return self.composite(savepath, observations_pred)

    def __call__(self, *args, **kwargs):
        return self.composite(*args, **kwargs)

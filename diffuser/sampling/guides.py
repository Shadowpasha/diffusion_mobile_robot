import torch
import torch.nn as nn
import pdb


class ValueGuide(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, cond, t):
        output = self.model(x, cond, t)
        return output.squeeze(dim=-1)

    def gradients(self, x, *args):
        x.requires_grad_()
        y = self(x, *args)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()
        return y, grad


class AnalyticalNavigationGuide(nn.Module):
    """
    Differentiable cost-based guidance for mobile robot navigation.
    Applies goal attraction, obstacle repulsion, boundary repulsion,
    and trajectory smoothness penalties during reverse diffusion sampling.
    """
    def __init__(
        self,
        normalizer,
        action_dim=2,
        goal=None,
        obstacles=None,
        safe_margin=0.95,
        arena_limit=3.4,
        w_goal=3.0,
        w_obs=6.0,
        w_wall=4.0,
        w_smooth=0.8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.safe_margin = safe_margin
        self.arena_limit = arena_limit
        self.w_goal = w_goal
        self.w_obs = w_obs
        self.w_wall = w_wall
        self.w_smooth = w_smooth

        obs_norm = normalizer.normalizers['observations']
        self.register_buffer('mins', torch.tensor(obs_norm.mins[:2], dtype=torch.float32))
        self.register_buffer('maxs', torch.tensor(obs_norm.maxs[:2], dtype=torch.float32))

        init_goal = goal if goal is not None else [0.0, 0.0]
        init_obs = obstacles if obstacles is not None else []
        self.register_buffer('goal', torch.tensor(init_goal, dtype=torch.float32))
        self.register_buffer('obstacles', torch.tensor(init_obs, dtype=torch.float32))

    def set_targets(self, goal=None, obstacles=None):
        if goal is not None:
            self.goal = torch.as_tensor(goal, dtype=torch.float32, device=self.mins.device)
        if obstacles is not None:
            self.obstacles = torch.as_tensor(obstacles, dtype=torch.float32, device=self.mins.device)

    def forward(self, x, cond, t):
        normed_pos = x[:, :, self.action_dim:self.action_dim + 2]
        pos = (normed_pos + 1.0) * 0.5 * (self.maxs - self.mins) + self.mins

        horizon = pos.shape[1]
        device = x.device

        # 1. Goal attraction
        weights = torch.linspace(0.1, 1.0, horizon, device=device).unsqueeze(0).unsqueeze(-1)
        dist_to_goal = torch.norm(pos - self.goal, dim=-1, keepdim=True)
        loss_goal = torch.mean(weights * dist_to_goal, dim=[1, 2])
        loss_goal = loss_goal + 1.5 * torch.norm(pos[:, -1] - self.goal, dim=-1)

        # 2. Obstacle repulsion
        loss_obs = torch.zeros_like(loss_goal)
        if self.obstacles.numel() > 0:
            pos_exp = pos.unsqueeze(2)
            obs_exp = self.obstacles.unsqueeze(0).unsqueeze(0)
            dist_obs = torch.norm(pos_exp - obs_exp, dim=-1)
            loss_obs = torch.sum(torch.relu(self.safe_margin - dist_obs) ** 2, dim=[1, 2])

        # 3. Wall repulsion
        loss_wall = torch.sum(torch.relu(torch.abs(pos) - (self.arena_limit - 0.35)) ** 2, dim=[1, 2])

        # 4. Smoothness penalty
        diffs = pos[:, 1:] - pos[:, :-1]
        loss_smooth = torch.sum(diffs ** 2, dim=[1, 2])

        total = - (self.w_goal * loss_goal + self.w_obs * loss_obs + self.w_wall * loss_wall + self.w_smooth * loss_smooth)
        return total

    def gradients(self, x, cond, t):
        x = x.detach().clone()
        x.requires_grad_(True)
        y = self(x, cond, t)
        grad = torch.autograd.grad([y.sum()], [x])[0]
        return y.detach(), grad.detach()


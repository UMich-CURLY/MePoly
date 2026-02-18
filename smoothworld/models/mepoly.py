import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Literal

from .ppo import PPO
from .network import FeedForwardNN

class PolynomialJointDistribution(nn.Module):
    """
    Joint polynomial exponential-family distribution on [-1, 1]^D.
    Uses a multi-dimensional quadrature grid for log-partition, entropy, and sampling.
    """

    def __init__(
        self,
        act_dim: int,
        order: int,
        grid_size: int = 64,
        lambda_clip: float = 5.0,
        action_eps: float = 1e-4,
        basis_mode: Literal["standard", "legendre"] = "legendre",
        grid_mode: Literal["full", "stochastic"] = "full",
        stochastic_grid_size: int = 4096,
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.order = order
        self.grid_size = grid_size
        self.lambda_clip = lambda_clip
        self.action_eps = action_eps
        self.basis_mode = basis_mode
        self.grid_mode = grid_mode

        # Legendre basis or standard monomials
        if basis_mode == "legendre":
            self._monomials = self._legendre_polynomials
        elif basis_mode == "standard":
            self._monomials = self._standard_monomials
        else:
            raise ValueError(f"Unsupported basis_mode: {basis_mode}")

        exponents = self._build_exponents(act_dim, order)
        self.num_features = exponents.shape[0]
        self.register_buffer("exponents", exponents)

        grid = torch.linspace(-1.0, 1.0, grid_size)
        weights = torch.ones_like(grid)
        weights[0] = weights[-1] = 0.5
        weights = weights * (2.0 / (grid_size - 1))  # trapezoidal rule scaling
        log_weights_1d = weights.clamp_min(1e-12).log()

        if grid_mode == "full":
            grid_points, log_weights = self._build_full_grid(grid, log_weights_1d)
        elif grid_mode == "stochastic":
            grid_points, log_weights = self._build_stochastic_grid(
                grid, log_weights_1d, stochastic_grid_size
            )
        else:
            raise ValueError(f"Unsupported grid_mode: {grid_mode}")

        self.register_buffer("grid_points", grid_points)
        self.register_buffer("log_weights", log_weights)
        self.register_buffer("grid_features", self._monomials(grid_points))

    @staticmethod
    def _build_exponents(act_dim: int, order: int) -> torch.Tensor:
        exponents = [
            powers
            for powers in itertools.product(range(order + 1), repeat=act_dim)
            if sum(powers) <= order
        ]
        return torch.tensor(exponents, dtype=torch.long)

    def _build_full_grid(
        self, grid: torch.Tensor, log_weights_1d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        meshes = torch.meshgrid(*[grid for _ in range(self.act_dim)], indexing="ij")
        grid_points = torch.stack(meshes, dim=-1).reshape(-1, self.act_dim)

        weight_meshes = torch.meshgrid(
            *[log_weights_1d for _ in range(self.act_dim)], indexing="ij"
        )
        log_weights = torch.stack(weight_meshes, dim=-1).sum(dim=-1).reshape(-1)
        return grid_points, log_weights

    def _build_stochastic_grid(
        self,
        grid: torch.Tensor,
        log_weights_1d: torch.Tensor,
        sample_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, grid.shape[0], (sample_size, self.act_dim))
        grid_points = grid[idx]
        log_weights = log_weights_1d[idx].sum(dim=-1)
        log_correction = self.act_dim * math.log(grid.shape[0]) - math.log(sample_size)
        log_weights = log_weights + log_correction
        return grid_points, log_weights

    def _stable_lambda(self, raw_lambda: torch.Tensor) -> torch.Tensor:
        return torch.clamp(raw_lambda, -self.lambda_clip, self.lambda_clip)

    def _logits_on_grid(self, lambda_params: torch.Tensor) -> torch.Tensor:
        return torch.matmul(lambda_params, self.grid_features.t())

    def log_partition(self, lambda_params: torch.Tensor) -> torch.Tensor:
        logits = self._logits_on_grid(lambda_params)
        return torch.logsumexp(logits + self.log_weights, dim=-1)

    def _standard_monomials(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0 + self.action_eps, 1.0 - self.action_eps)
        x = x.unsqueeze(-2)
        return x.pow(self.exponents).prod(dim=-1)

    def _legendre_polynomials(self, x: torch.Tensor) -> torch.Tensor:
        # Based on the recurrence relation from Taylor expansion of generating function of Legendre polynomials
        # Eq.(2) in https://en.wikipedia.org/wiki/Legendre_polynomials
        base_shape = x.shape[:-1]
        legendre = x.new_zeros(*base_shape, self.act_dim, self.order + 1)
        legendre[..., 0] = 1.0
        if self.order >= 1:
            legendre[..., 1] = x
        for n in range(2, self.order + 1):
            n_float = float(n)
            legendre[..., n] = ((2 * n - 1) / n_float) * x * legendre[..., n - 1] - (
                (n - 1) / n_float
            ) * legendre[..., n - 2]

        orders = self.exponents.unsqueeze(0).expand(*base_shape, -1, -1).unsqueeze(-1)
        legendre = legendre.unsqueeze(-3).expand(*base_shape, self.num_features, self.act_dim, -1)
        selected = torch.gather(legendre, -1, orders).squeeze(-1)
        return selected.prod(dim=-1)


    def log_prob(self, lambda_params: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = self._monomials(action)
        log_z = self.log_partition(lambda_params)
        logits = (lambda_params * features).sum(dim=-1)
        return logits - log_z

    def entropy(self, lambda_params: torch.Tensor) -> torch.Tensor:
        logits = self._logits_on_grid(lambda_params)
        log_z = torch.logsumexp(logits + self.log_weights, dim=-1, keepdim=True)
        log_probs = logits - log_z
        probs = log_probs.exp()
        ent = -(probs * log_probs * self.log_weights.exp()).sum(dim=-1)
        return ent

    def sample(self, lambda_params: torch.Tensor):
        squeeze = False
        if lambda_params.dim() == 1:
            lambda_params = lambda_params.unsqueeze(0)
            squeeze = True

        logits = self._logits_on_grid(lambda_params)
        log_mass = logits + self.log_weights
        probs = torch.softmax(log_mass, dim=-1)
        cdf = probs.cumsum(dim=-1)
        u = torch.rand_like(cdf[..., :1])
        idx = torch.searchsorted(cdf, u, right=False).clamp(max=cdf.shape[-1] - 1)

        grid_expanded = self.grid_points.unsqueeze(0).expand(
            logits.shape[0], -1, -1
        )
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, self.act_dim)
        action = torch.gather(grid_expanded, 1, idx_expanded).squeeze(1)

        logprob = self.log_prob(lambda_params, action)
        if squeeze:
            return action.squeeze(0), logprob.squeeze(0)
        return action, logprob

    def expected_action(self, lambda_params: torch.Tensor) -> torch.Tensor:
        logits = self._logits_on_grid(lambda_params)
        log_mass = logits + self.log_weights
        probs = torch.softmax(log_mass, dim=-1)
        return (probs.unsqueeze(-1) * self.grid_points.unsqueeze(0)).sum(dim=-2)


class PolynomialPolicy(nn.Module):
    """Lambda network + polynomial joint distribution."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        order: int = 5,
        grid_size: int = 64,
        lambda_clip: float = 5.0,
        action_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.order = order
        self.dist = PolynomialJointDistribution(
            act_dim=act_dim,
            order=order,
            grid_size=grid_size,
            lambda_clip=lambda_clip,
            action_eps=action_eps,
        )
        self.lambda_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.dist.num_features),
        )

    def _lambda_params(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.lambda_net(obs)
        return self.dist._stable_lambda(raw)

    def sample_action_with_logprob(self, obs: torch.Tensor, deterministic: bool = False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        lam = self._lambda_params(obs)
        if deterministic:
            action = self.dist.expected_action(lam)
            logprob = self.dist.log_prob(lam, action)
        else:
            action, logprob = self.dist.sample(lam)
        return action.squeeze(0), logprob.squeeze(0)

    def sample_action(self, obs: torch.Tensor, deterministic: bool = False):
        action, _ = self.sample_action_with_logprob(obs, deterministic=deterministic)
        return action

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        lam = self._lambda_params(obs)
        actions = torch.clamp(actions, -1.0 + self.dist.action_eps, 1.0 - self.dist.action_eps)
        return self.dist.log_prob(lam, actions)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        lam = self._lambda_params(obs)
        return self.dist.entropy(lam)


class MePoly(PPO):
    """
    PPO variant that replaces the Gaussian policy with a polynomial
    exponential-family distribution (max-entropy PPO).
    """

    def __init__(self, env, **hyperparameters):
        hyperparameters.setdefault("method_name", "mepoly")
        hyperparameters.setdefault("poly_order", 5)
        hyperparameters.setdefault("poly_grid_size", 64)
        hyperparameters.setdefault("lambda_clip", 5.0)
        hyperparameters.setdefault("action_eps", 1e-4)

        # Initialize via PPO to reuse logging/GAE/etc. Then override actor + optimizer.
        super().__init__(policy_class=FeedForwardNN, env=env, **hyperparameters)

        self.poly_order = hyperparameters["poly_order"]
        self.poly_grid_size = hyperparameters["poly_grid_size"]
        self.lambda_clip = hyperparameters["lambda_clip"]
        self.action_eps = hyperparameters["action_eps"]

        self.actor = PolynomialPolicy(
            self.obs_dim,
            self.act_dim,
            order=self.poly_order,
            grid_size=self.poly_grid_size,
            lambda_clip=self.lambda_clip,
            action_eps=self.action_eps,
        ).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

    def get_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action, log_prob = self.actor.sample_action_with_logprob(obs_tensor, deterministic=self.deterministic)
        return action.detach().cpu().numpy(), log_prob.detach().cpu()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        log_probs = self.actor.log_prob(batch_obs, batch_acts)
        entropy = self.actor.entropy(batch_obs)
        return V, log_probs, entropy

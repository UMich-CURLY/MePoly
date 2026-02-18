import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal, Categorical, MixtureSameFamily, Independent

from .ppo import PPO
from .network import FeedForwardNN 

class GMMPolicy(nn.Module):
    """
    Actor network that parameterizes a Gaussian Mixture Model (GMM).
    Output: A Mixture of K Diagonal Gaussians.
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_components: int = 4,
        hidden_dim: int = 64
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.K = num_components

        # Shared backbone (Feature Extractor)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.logits_head = nn.Linear(hidden_dim, self.K)

        self.means_head = nn.Linear(hidden_dim, self.K * self.act_dim)

        self.log_stds_head = nn.Linear(hidden_dim, self.K * self.act_dim)


    def get_distribution(self, obs: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.backbone(obs)
        batch_size = obs.shape[0]
        mix_logits = self.logits_head(features) # (batch, K)

        means = self.means_head(features)
        means = means.view(batch_size, self.K, self.act_dim)

        log_stds = self.log_stds_head(features)
        log_stds = log_stds.view(batch_size, self.K, self.act_dim)
        
        log_stds = torch.clamp(log_stds, min=-20, max=2) 
        stds = torch.exp(log_stds)
        
        mix = Categorical(logits=mix_logits)

        comp = Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)

        gmm = MixtureSameFamily(mixture_distribution=mix, component_distribution=comp)
        
        return gmm

    def sample_action_with_logprob(self, obs: torch.Tensor, deterministic: bool = False):
        gmm = self.get_distribution(obs)
        
        if deterministic:
            # dont recommand, just for completeness
            mix_dist = gmm.mixture_distribution
            comp_dist = gmm.component_distribution
            
            best_comp_idx = torch.argmax(mix_dist.logits, dim=-1) # (batch,)
            means = comp_dist.base_dist.loc
            idx_expanded = best_comp_idx.view(-1, 1, 1).expand(-1, 1, self.act_dim)
            action = torch.gather(means, 1, idx_expanded).squeeze(1)
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
            
        else:
            action = gmm.sample()
            log_prob = gmm.log_prob(action)
            
        return action.squeeze(0), log_prob.squeeze(0)
    
    def sample_action(self, obs: torch.Tensor, deterministic: bool = False):
        action, _ = self.sample_action_with_logprob(obs, deterministic=deterministic)
        return action

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        gmm = self.get_distribution(obs)
        return gmm.log_prob(actions)

    # def entropy(self, obs: torch.Tensor, num_samples: int = 32) -> torch.Tensor:
    #     """
    #     Gumbel-Softmax Reparameterization Entropy.
    #     """
    #     features = self.backbone(obs)
    #     B = obs.shape[0]
    #     K = self.K
        
    #     mix_logits = self.logits_head(features)
    #     means = self.means_head(features).view(B, K, self.act_dim)
    #     log_stds = torch.clamp(self.log_stds_head(features).view(B, K, self.act_dim), -20, 2)
    #     stds = torch.exp(log_stds)

    #     mix_logits_exp = mix_logits.unsqueeze(0).expand(num_samples, -1, -1)
    #     means_exp = means.unsqueeze(0).expand(num_samples, -1, -1, -1)
    #     stds_exp = stds.unsqueeze(0).expand(num_samples, -1, -1, -1)

    #     # hard sampling perform not well, but sof sampling is sensitive, weird
    #     z = F.gumbel_softmax(mix_logits_exp, tau=1.0, hard=False)
        
    #     eps = torch.randn_like(means_exp)
    #     x_all = means_exp + stds_exp * eps
        
    #     z_expanded = z.unsqueeze(-1)
    #     x_samples = (z_expanded * x_all).sum(dim=2) # (N, B, Act_Dim)
        
    #     mix_dist = Categorical(logits=mix_logits)
    #     comp_dist = Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)
    #     gmm = MixtureSameFamily(mix_dist, comp_dist)
        
    #     log_probs = gmm.log_prob(x_samples) # (N, B)
    #     entropy = -log_probs.mean(dim=0)
        
    #     return entropy

    def entropy(self, obs: torch.Tensor, num_samples: int = None) -> torch.Tensor:
            """
            Surrogate Entropy (Analytical Approximation).
            Formula: H ≈ H(Categorical) + sum( pi_k * H(Gaussian_k) )
            
            Pros: Extremely stable, fast, exact analytical solution (no sampling noise).
            Cons: No gradient for means separation (relies on reward signal to separate modes).
            """
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            features = self.backbone(obs)
            B = obs.shape[0]
            K = self.K
            
            mix_logits = self.logits_head(features)
            
            means = self.means_head(features).view(B, K, self.act_dim) 
            
            raw_log_stds = self.log_stds_head(features).view(B, K, self.act_dim)
            log_stds = torch.clamp(raw_log_stds, -20, 2)
            stds = torch.exp(log_stds)

            mix_dist = Categorical(logits=mix_logits)
            H_mix = mix_dist.entropy()  # Shape: (B,)

            H_comp_per_dim = Normal(loc=means, scale=stds).entropy()
            H_comp = H_comp_per_dim.sum(dim=-1) 

            probs = mix_dist.probs # Shape: (B, K)
            weighted_H_comp = (probs * H_comp).sum(dim=-1) # Shape: (B,)

            total_entropy = H_mix + weighted_H_comp

            return total_entropy

class GMM_PPO(PPO):
    """
    PPO variant that replaces the Gaussian policy with a Gaussian Mixture Model (GMM).
    """

    def __init__(self, env, **hyperparameters):
        hyperparameters.setdefault("method_name", "gmm_ppo")
        hyperparameters.setdefault("num_components", 4)
        hyperparameters.setdefault("hidden_dim", 64)

        # Initialize via PPO to reuse logging/GAE/etc.
        super().__init__(policy_class=FeedForwardNN, env=env, **hyperparameters)

        self.num_components = hyperparameters["num_components"]
        self.hidden_dim = hyperparameters["hidden_dim"]

        # Override the actor with our GMM Policy
        self.actor = GMMPolicy(
            self.obs_dim,
            self.act_dim,
            num_components=self.num_components,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

    def get_action(self, obs):
        """
        Identical signature to MePoly/PPO get_action
        """
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        else:
            obs_tensor = obs.to(self.device)
            
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.actor.sample_action_with_logprob(obs_tensor, deterministic=self.deterministic)

        return action.detach().cpu().numpy().flatten(), log_prob.detach().cpu().item()

    def evaluate(self, batch_obs, batch_acts):
        """
        Evaluate batch for PPO update.
        """
        V = self.critic(batch_obs).squeeze()
        
        log_probs = self.actor.log_prob(batch_obs, batch_acts)
        
        entropy = self.actor.entropy(batch_obs)
        
        return V, log_probs, entropy

"""
        This file is the executable for running PPO. It is based on this medium article: 
        https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import sys
import torch
import wandb
import argparse
import random
import numpy as np
from datetime import datetime
from pathlib import Path

from models.ppo import PPO
from models.fpo import FPO
from models.mepoly import MePoly
from models.gmm import GMM_PPO, GMMPolicy
from models.network import FeedForwardNN
from models.diffusion_policy import DiffusionPolicy
from utils.bandit import BanditEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def get_args():
        parser = argparse.ArgumentParser(description="Train/eval PPO/FPO/MEM-PPO on a 1-step bandit distribution-fitting task.")
        parser.add_argument('--mode', dest='mode', type=str, default='train', choices=['train', 'eval', 'test'],
                            help="train or eval mode (test is an alias for eval)")
        parser.add_argument('--actor_model', dest='actor_model', type=str, default='',
                            help="Path to actor model checkpoint")
        parser.add_argument('--critic_model', dest='critic_model', type=str, default='',
                            help="Path to critic model checkpoint")
        parser.add_argument('--method', dest='method', type=str, default='mepoly', choices=['ppo', 'fpo', 'mepoly', 'gmm'],
                            help="Algorithm to run")
        # Core hyperparameters
        parser.add_argument('--timesteps_per_batch', type=int, default=1024,
                            help="Timesteps per batch (episodes per batch for 1-step bandit)")
        parser.add_argument('--max_timesteps_per_episode', type=int, default=1,
                            help="Max steps per episode (keep at 1 for bandit)")
        parser.add_argument('--gamma', type=float, default=1.0)
        parser.add_argument('--n_updates_per_iteration', type=int, default=10)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--clip', type=float, default=0.2)
        parser.add_argument('--ent_coef', type=float, default=1.0, help="Entropy regularization coefficient")
        parser.add_argument('--task', type=str, default='lemniscate',
                            choices=['lemniscate', 'twomoons'])
        parser.add_argument('--num_fpo_samples', type=int, default=50)
        parser.add_argument('--positive_advantage', action='store_true', help="Apply softplus to advantages")
        # MEM-PPO specific
        parser.add_argument('--poly_order', type=int, default=8)
        parser.add_argument('--poly_grid_size', type=int, default=64)
        parser.add_argument('--lambda_clip', type=float, default=5.0)
        parser.add_argument('--action_eps', type=float, default=1e-4)
        parser.add_argument('--seed', type=int, default=6, help="Random seed for reproducibility")
        # Saving / bookkeeping
        parser.add_argument('--save_freq', type=int, default=20, help="Save frequency in iterations")
        parser.add_argument('--max_iterations', type=int, default=12000, help="Max training iterations")
        # Visualization
        parser.add_argument('--render', action='store_true', help="Render the environment during training")
        parser.add_argument('--render_every_i', type=int, default=200)
        parser.add_argument('--eval_samples', type=int, default=4096, help="Action samples for evaluation plot/stats")
        parser.add_argument('--test_episodes', type=int, default=None, help="Deprecated alias for --eval_samples")

        return parser.parse_args()

class BanditRunner:
        def __init__(self, args):
                self.args = args
                if self.args.mode == "test":
                        self.args.mode = "eval"

                # Backwards compatible alias
                self.eval_samples = args.eval_samples if args.test_episodes is None else args.test_episodes

                self.device = device
                self.hyperparameters = {
                                'timesteps_per_batch': args.timesteps_per_batch,
                                'max_timesteps_per_episode': args.max_timesteps_per_episode,
                                'gamma': args.gamma,
                                'n_updates_per_iteration': args.n_updates_per_iteration,
                                'lr': args.lr,
                                'clip': args.clip,
                                'render': args.render,
                                'render_every_i': args.render_every_i,
                                'ent_coef': args.ent_coef,
                                # task/env
                                'task': args.task,
                                # FPO specific
                                'num_fpo_samples': args.num_fpo_samples,
                                'positive_advantage': args.positive_advantage,
                                # MEM-PPO specific
                                'poly_order': args.poly_order,
                                'poly_grid_size': args.poly_grid_size,
                                'lambda_clip': args.lambda_clip,
                                'action_eps': args.action_eps,
                                # misc
                                'seed': args.seed,
                                'save_freq': args.save_freq,
                                'max_iterations': args.max_iterations,
                                'eval_samples': self.eval_samples,
                                'method_name': args.method,
                          }

                print(self.hyperparameters)
                self._seed_all(args.seed)
                self.train_env, self.eval_env = self._build_envs()

        def _seed_all(self, seed: int):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                random.seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        def _build_envs(self):
                max_steps = self.hyperparameters['max_timesteps_per_episode']
                train_env = BanditEnv(task=self.hyperparameters['task'], max_steps=max_steps)
                eval_env = BanditEnv(task=self.hyperparameters['task'], max_steps=max_steps)
                train_env.reset(seed=self.args.seed)
                eval_env.reset(seed=self.args.seed)
                train_env.action_space.seed(self.args.seed)
                eval_env.action_space.seed(self.args.seed)
                train_env.observation_space.seed(self.args.seed)
                eval_env.observation_space.seed(self.args.seed)
                return train_env, eval_env

        def _build_model(self, env):
                method = self.args.method
                if method == "ppo":
                        model = PPO(policy_class=FeedForwardNN, env=env, **self.hyperparameters)
                elif method == "fpo":
                        model = FPO(actor_class=DiffusionPolicy, critic_class=FeedForwardNN, env=env, **self.hyperparameters)
                elif method == "mepoly":
                        model = MePoly(env=env, **self.hyperparameters)
                elif method == "gmm":
                        model = GMM_PPO(env=env, **self.hyperparameters)
                else:
                        print(f"Unsupported method: {method}")
                        sys.exit(1)
                model.actor.to(self.device)
                model.critic.to(self.device)
                return model

        def _load_actor_critic(self, model, actor_model: str, critic_model: str):
                # Resume or train from scratch if path == ''
                if actor_model == '' and critic_model == '':
                        print("Training from scratch.", flush=True)
                        return

                if actor_model != '':
                        print(f"Loading actor from {actor_model}", flush=True)
                        model.actor.load_state_dict(torch.load(actor_model, map_location=self.device))
                if critic_model != '':
                        print(f"Loading critic from {critic_model}", flush=True)
                        model.critic.load_state_dict(torch.load(critic_model, map_location=self.device))

        def _init_wandb(self):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                lr = self.hyperparameters['lr']
                bs = self.hyperparameters['timesteps_per_batch']
                task = self.hyperparameters.get("task", "na")
                run_name = f"{self.args.method}_{task}_seed{self.args.seed}_lr{lr}_bs{bs}_{timestamp}"
                if self.args.method == "fpo":
                        n = self.hyperparameters['num_fpo_samples']
                        run_name += f"_N{n}"

                print(f"running {run_name}")
                self.hyperparameters["run_name"] = run_name
                self.hyperparameters["method_name"] = self.args.method

                wandb.init(
                        project="bandit-dist-fit",
                        name=run_name,
                        config=self.hyperparameters,
                        tags=[self.args.method, "bandit", self.args.mode, f"task:{task}", f"seed:{self.args.seed}"],
                )

        def _sample_action(self, policy, obs: np.ndarray) -> np.ndarray:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                        if hasattr(policy, "sample_action"):
                                action_t = policy.sample_action(obs_tensor)
                        else:
                                mean = policy(obs_tensor)
                                cov_var = torch.full((mean.shape[-1],), 0.5, device=mean.device)
                                dist = torch.distributions.MultivariateNormal(mean, torch.diag(cov_var))
                                action_t = dist.sample()
                action = action_t.detach().cpu().numpy()
                return np.asarray(action, dtype=float).reshape(-1)

        def evaluate(self, model, iteration: int | None = None):
                env = self.eval_env
                policy = model.actor
                policy.eval()

                obs, _ = env.reset()
                actions = []
                rewards = []
                for _ in range(self.eval_samples):
                        action = self._sample_action(policy, obs)
                        action = np.clip(action, env.action_space.low, env.action_space.high)
                        reward = env.compute_reward(action)
                        actions.append(action)
                        rewards.append(reward)
                actions = np.asarray(actions, dtype=float)
                rewards = np.asarray(rewards, dtype=np.float32)

                suffix = str(iteration) if iteration is not None else "final"
                eval_image = Path(model.run_dir) / f"{self.args.method}_eval_{suffix}.png"
                env.render(action_samples=actions, save_path=eval_image)
                print(f"Saved eval figure → {eval_image}")

                if wandb.run is not None:
                        base_path = str(eval_image.parent)
                        wandb.save(str(eval_image), base_path=base_path)
                        step = iteration if iteration is not None else model.logger.get("i_so_far", 0)
                        wandb.log({
                                "eval/reward_mean": float(rewards.mean()),
                                "eval/reward_std": float(rewards.std()),
                                "eval/action_mean_x": float(actions[:, 0].mean()),
                                "eval/action_mean_y": float(actions[:, 1].mean()),
                                "eval/action_std_x": float(actions[:, 0].std()),
                                "eval/action_std_y": float(actions[:, 1].std()),
                                "vis/distribution": wandb.Image(str(eval_image)),
                        }, step=step)

        def train(self, model):
                total_timesteps = 200_000_000
                t_so_far = model.logger.get('t_so_far', 0)
                i_so_far = model.logger.get('i_so_far', 0)
                while t_so_far < total_timesteps and (model.max_iterations is None or i_so_far < model.max_iterations):
                        model.learn(total_timesteps=total_timesteps)
                        t_so_far = model.logger.get('t_so_far', t_so_far)
                        i_so_far = model.logger.get('i_so_far', i_so_far)
                        if i_so_far % model.save_freq == 0:
                                model._save_models(i_so_far)
                                self.evaluate(model, iteration=i_so_far)

                if i_so_far > 0 and (i_so_far % model.save_freq != 0):
                        model._save_models(i_so_far)
                        self.evaluate(model, iteration=i_so_far)

        def run(self):
                if self.args.mode == 'train':
                        self._init_wandb()
                        print(f"Training using {self.args.method.upper()}", flush=True)
                        model = self._build_model(self.train_env)
                        self._load_actor_critic(model, self.args.actor_model, self.args.critic_model)
                        self.train(model)
                else:
                        model = self._build_model(self.eval_env)
                        self._load_actor_critic(model, self.args.actor_model, self.args.critic_model)
                        self.evaluate(model, iteration=None)

if __name__ == '__main__':
        args = get_args() # Parse arguments from command line
        BanditRunner(args).run()

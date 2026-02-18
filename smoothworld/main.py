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
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from models.ppo import PPO
from models.fpo import FPO
from models.mepoly import MePoly
from models.gmm import GMM_PPO
from models.network import FeedForwardNN
from models.diffusion_policy import DiffusionPolicy
from envs.smoothworld import GridWorldEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def get_args():
        parser = argparse.ArgumentParser(description="Train/eval PPO/FPO/MEPOLY on smoothworld.")
        parser.add_argument('--mode', dest='mode', type=str, default='train', choices=['train', 'eval'],
                            help="train or eval mode")
        parser.add_argument('--actor_model', dest='actor_model', type=str, default='',
                            help="Path to actor model checkpoint")
        parser.add_argument('--critic_model', dest='critic_model', type=str, default='',
                            help="Path to critic model checkpoint")
        parser.add_argument('--method', dest='method', type=str, default='fpo', choices=['ppo', 'fpo', 'mepoly', 'gmm'],
                            help="Algorithm to run")
        # Core hyperparameters
        parser.add_argument('--timesteps_per_batch', type=int, default=2048)
        parser.add_argument('--max_timesteps_per_episode', type=int, default=256)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--n_updates_per_iteration', type=int, default=10)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--clip', type=float, default=0.2)
        parser.add_argument('--render', action='store_true', help="Render during rollout")
        parser.add_argument('--render_every_i', type=int, default=1000)
        parser.add_argument('--ent_coef', type=float, default=1.0, help="Entropy regularization coefficient")
        # FPO specific
        parser.add_argument('--grid_mode', type=str, default='four_walls',
                            choices=['two_walls', 'three_goals', 'tree_in_the_middle', 'two_slits', 'cshape', 'four_walls'])
        parser.add_argument('--num_fpo_samples', type=int, default=50)
        parser.add_argument('--positive_advantage', action='store_true', help="Apply softplus to advantages")
        # MEM-PPO specific
        parser.add_argument('--poly_order', type=int, default=8)
        parser.add_argument('--poly_grid_size', type=int, default=64)
        parser.add_argument('--lambda_clip', type=float, default=10.0)
        parser.add_argument('--action_eps', type=float, default=1e-4)
        # GMM-PPO specific
        parser.add_argument('--num_components', type=int, default=4)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--seed', type=int, default=6, help="Random seed for reproducibility")
        # Saving / bookkeeping
        parser.add_argument('--save_freq', type=int, default=100, help="Save frequency in iterations")
        parser.add_argument('--max_iterations', type=int, default=12000, help="Max training iterations")
        return parser.parse_args()

class SmoothWorldRunner:
        def __init__(self, args):
                self.args = args
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
                                # FPO specific parameters:
                                'grid_mode': args.grid_mode,
                                'num_fpo_samples': args.num_fpo_samples,
                                'positive_advantage': args.positive_advantage,
                                # MEM-PPO specific parameters:
                                'poly_order': args.poly_order,
                                'poly_grid_size': args.poly_grid_size,
                                'lambda_clip': args.lambda_clip,
                                'action_eps': args.action_eps,
                                # GMM-PPO specific parameters:
                                'num_components': args.num_components,
                                'hidden_dim': args.hidden_dim,
                                'seed': args.seed,
                                # Saving / bookkeeping
                                'save_freq': args.save_freq,  # iterations
                                # Stop conditions
                                'max_iterations': args.max_iterations,
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
                train_env = GridWorldEnv(mode=self.hyperparameters['grid_mode'], max_steps=max_steps)
                eval_env = GridWorldEnv(mode=self.hyperparameters['grid_mode'], max_steps=max_steps)
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
                        print(f"Training from scratch.", flush=True)
                        return
                if actor_model != '':
                        print(f"Loading in {actor_model}", flush=True)
                        model.actor.load_state_dict(torch.load(actor_model, map_location=self.device))
                if critic_model != '':
                        print(f"Loading in {critic_model}", flush=True)
                        model.critic.load_state_dict(torch.load(critic_model, map_location=self.device))
                return

        def _build_policy(self, env, actor_model: str):
                obs_dim = env.observation_space.shape[0]
                act_dim = env.action_space.shape[0]
                method = self.args.method
                if method == 'ppo':
                        policy = FeedForwardNN(obs_dim, act_dim).to(self.device)
                elif method == 'fpo':
                        policy = DiffusionPolicy(obs_dim + act_dim + 1, act_dim, device=self.device).to(self.device)
                elif method == 'mepoly':
                        policy = MePoly(env=env, **self.hyperparameters).actor.to(self.device)
                else:
                        print(f"Unsupported method: {method}"); sys.exit(1)
                return policy, obs_dim, act_dim

        def _init_wandb(self):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                task = getattr(self.args, "grid_mode", "na")
                run_name = f"{self.args.method}_{task}_seed{self.args.seed}_{timestamp}"
                if self.args.method == "fpo":
                        n = self.hyperparameters['num_fpo_samples']
                        run_name += f"_N{n}"
                print(f"running {run_name}")
                self.hyperparameters["run_name"] = run_name
                self.hyperparameters["method_name"] = self.args.method
                wandb.init(
                        project="smoothworld",
                        name=run_name,
                        config=self.hyperparameters,
                        tags=[self.args.method, "smoothworld", self.args.mode, f"grid:{task}", f"seed:{self.args.seed}"],
                )

        def train(self, model):
                total_timesteps = 200_000_000
                print(f"Learning... Running {model.max_timesteps_per_episode} timesteps per episode, ", end='')
                print(f"{model.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
                t_so_far = model.logger.get('t_so_far', 0)
                i_so_far = model.logger.get('i_so_far', 0)

                while t_so_far < total_timesteps and (model.max_iterations is None or i_so_far < model.max_iterations):
                        model.learn(total_timesteps=total_timesteps)
                        t_so_far = model.logger.get('t_so_far', t_so_far)
                        i_so_far = model.logger.get('i_so_far', i_so_far)

                        if i_so_far % model.save_freq == 0:
                                model._save_models(i_so_far)
                                self.evaluate(model)

                if i_so_far > 0 and (i_so_far % model.save_freq != 0):
                        model._save_models(i_so_far)
                        self.evaluate(model)

        def evaluate(self, model):
                args = self.args
                env = self.eval_env
                policy = model.actor
                policy.eval()

                # start points
                start_points = [(env.config.cx, env.config.cy)]
                if args.grid_mode == "tree_in_the_middle" or args.grid_mode == "two_slits":
                        start_points = [(env.config.cx - 10, env.config.cy)]
                elif args.grid_mode == "cshape":
                        start_points = [(env.config.cx + 7, env.config.cy)]

                trajs = []
                episodes = 50
                size = env.grid_size
                figure_output = str(Path(model.run_dir) / f"{args.method}_actor_traj.png")

                for sid, start in enumerate(start_points):
                        for _ in range(episodes):
                                obs, _ = env.reset()
                                env.pos = np.array(start, dtype=float)
                                obs = env._get_obs()
                                done = False
                                t = 0
                                ep_ret = 0.0
                                traj = [tuple(env.pos)]
                                
                                while not done:
                                        t += 1
                                        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                                        # stochastic action from policy
                                        with torch.no_grad():
                                                # TODO: Unified different policy with same interface sample_action, should be robust
                                                obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension
                                                act = policy.sample_action(obs_tensor).squeeze(0).cpu().numpy()
                                        act = np.clip(act, env.action_space.low, env.action_space.high)
                                        obs, rew, term, trunc, _ = env.step(act)
                                        done = term or trunc
                                        ep_ret += rew
                                        # record position
                                        traj.append(tuple(env.pos))

                                # save episode data
                                trajs.append({
                                        "start_idx": sid,
                                        "start": start,
                                        "traj": traj,
                                        "ep_len": t,
                                        "ep_ret": ep_ret
                                })

                # render basic env once after collecting all trajectories
                fig, ax = env.render(alpha=0.6)
                # render trajectory
                for ep in trajs:
                        pts = np.array(ep['traj'], dtype=float)
                        xs = pts[:,0];ys = pts[:,1]
                        segs = np.stack([np.column_stack([xs[:-1], ys[:-1]]), np.column_stack([xs[1:], ys[1:]])], axis=1)
                        norm = Normalize(0, len(xs)-1)
                        lc = LineCollection(segs, cmap="plasma", norm=norm, linewidth=3.0)
                        lc.set_array(np.arange(len(xs)))
                        ax.add_collection(lc)

                # start/end markers (start as white circle, end as cross)
                for ep in trajs:
                        pts = np.array(ep['traj'], dtype=float)
                        ax.scatter(pts[0,0], pts[0,1], marker='o', s=150, facecolors='white', edgecolors='black', zorder=3, label='Start' if ep == trajs[0] else "")
                        ax.scatter(pts[-1,0], pts[-1,1], marker='X', s=150, facecolors='black', edgecolors='black', zorder=3, label='End' if ep == trajs[0] else "")

                # minimal ticks
                ax.set_xticks(np.arange(0, size+1, 5)); ax.set_yticks(np.arange(0, size+1, 5))
                ax.set_xlim(0, size); ax.set_ylim(0, size)

                handles, labels = ax.get_legend_handles_labels()
                by_label = {lbl: h for h, lbl in zip(handles, labels) if lbl}
                ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.3,1.0))

                ax.set_title(f"{args.method}", color='#333')
                plt.tight_layout()
                plt.savefig(figure_output)
                print(f"Saved figure → {figure_output}")
                plt.close(fig)

                if wandb.run is not None:
                        base_path = str(Path(figure_output).parent)
                        wandb.save(str(figure_output), base_path=base_path)
                        wandb.log({
                                "eval/trajectory_image": wandb.Image(str(figure_output)),
                        })


        def run(self):
                if self.args.mode == 'train':
                        self._init_wandb()
                        print(f"Training using {self.args.method.upper()}", flush=True)
                        model = self._build_model(self.train_env)
                        self._load_actor_critic(model, self.args.actor_model, self.args.critic_model)
                        self.train(model)
                else:
                        model = self._build_model(self.eval_env)
                        self._load_actor_critic(model, self.args.actor_model, self.args.critic_model, allow_default_actor=True)
                        self.evaluate(model)



if __name__ == '__main__':
        args = get_args() # Parse arguments from command line
        SmoothWorldRunner(args).run()

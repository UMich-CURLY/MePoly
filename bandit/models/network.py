"""
        This file contains a neural network module for us to
        define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np

class FeedForwardNN(nn.Module):
        """
                A standard in_dim-64-64-out_dim Feed Forward Neural Network.
        """
        def __init__(self, in_dim, out_dim):
                """
                        Initialize the network and set up the layers.

                        Parameters:
                                in_dim - input dimensions as an int
                                out_dim - output dimensions as an int

                        Return:
                                None
                """
                super(FeedForwardNN, self).__init__()

                self.layer1 = nn.Linear(in_dim, 64)
                self.layer2 = nn.Linear(64, 64)
                self.layer3 = nn.Linear(64, out_dim)

        def forward(self, obs):
                """
                        Runs a forward pass on the neural network.

                        Parameters:
                                obs - observation to pass as input

                        Return:
                                output - the output of our forward pass
                """
                # Convert observation to tensor if it's a numpy array
                if isinstance(obs, np.ndarray):
                        obs = torch.tensor(obs, dtype=torch.float)

                activation1 = F.relu(self.layer1(obs))
                activation2 = F.relu(self.layer2(activation1))
                output = self.layer3(activation2)
                # print(obs, output)

                return output

        def sample_action(self, obs, cov_var: float = 0.5):
                """
                        Sample a Gaussian action for PPO-style policies.
                """
                # Convert observation to tensor if it's a numpy array
                if isinstance(obs, np.ndarray):
                        obs = torch.tensor(obs, dtype=torch.float)
                if obs.dim() == 1:
                        obs = obs.unsqueeze(0)
                mean = self.forward(obs)
                cov = torch.full((mean.shape[-1],), cov_var, device=mean.device)
                cov_mat = torch.diag(cov)
                dist = MultivariateNormal(mean, cov_mat)
                action = dist.sample()
                return action.squeeze(0)

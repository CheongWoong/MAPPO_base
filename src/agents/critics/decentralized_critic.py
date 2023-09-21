# reference: https://github.com/vwxyzjn/cleanrl/tree/master
import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DecentralizedCritic(nn.Module):
    def __init__(self, envs, hidden_dim=64):
        super().__init__()
        self.is_continuous = envs.is_continuous
        in_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape) if envs.is_continuous else envs.single_action_space.n
        out_dim = 1
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, out_dim), std=1.0),
        )
        
    def forward(self, x):
        return self.critic(x)
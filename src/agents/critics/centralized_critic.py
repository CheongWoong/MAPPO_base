# reference: https://github.com/vwxyzjn/cleanrl/tree/master
import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CentralizedCritic(nn.Module):
    def __init__(self, envs, hidden_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        if num_heads > 0:
            assert (hidden_dim % num_heads) == 0
        self.is_continuous = envs.is_continuous
        in_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape) if envs.is_continuous else envs.single_action_space.n
        out_dim = 1

        if num_heads > 0:
            self.fc = nn.Sequential(
                layer_init(nn.Linear(in_dim, hidden_dim)),
                nn.Tanh(),
            )
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            self.critic = nn.Sequential(
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, out_dim), std=1.0),
            )
        else:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(envs.num_agents*in_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, envs.num_agents*out_dim), std=1.0),
            )

    def forward(self, x):
        if self.num_heads > 0:
            x = self.fc(x)
            x, attention_weights = self.attention(x, x, x)
            return self.critic(x)
        else:
            x = x.view((x.shape[0], -1))
            return self.critic(x).unsqueeze(-1)
# reference: https://github.com/vwxyzjn/cleanrl/tree/master
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DecentralizedActor(nn.Module):
    def __init__(self, envs, hidden_dim=64):
        super().__init__()
        self.is_continuous = envs.is_continuous
        in_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape) if envs.is_continuous else envs.single_action_space.n
        out_dim = action_dim
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, out_dim), std=0.01),
        )
        if self.is_continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x, action=None):
        if self.is_continuous:
            action_mean = self.actor(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            logprob = probs.log_prob(action).sum(-1)
            entropy = probs.entropy().sum(-1)
        else:
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            logprob = probs.log_prob(action)
            entropy = probs.entropy()

        return action, logprob, entropy
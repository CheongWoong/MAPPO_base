# reference: https://github.com/vwxyzjn/cleanrl/tree/master
import torch
import torch.nn as nn

from .critics import DecentralizedCritic, CentralizedCritic
from .actors import DecentralizedActor, CentralizedActor, CommunicatedActor


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        if args.critic_type == "decentralized":
            self.critic = DecentralizedCritic(envs, hidden_dim=args.critic_hidden_dim)
        elif args.critic_type == "centralized":
            self.critic = CentralizedCritic(envs, hidden_dim=args.critic_hidden_dim, num_heads=args.critic_num_heads)
        else:
            raise Exception
        if args.actor_type == "decentralized":
            self.actor = DecentralizedActor(envs, hidden_dim=args.actor_hidden_dim)
        elif args.actor_type == "centralized":
            self.actor = CentralizedActor(envs, hidden_dim=args.actor_hidden_dim, num_heads=args.actor_num_heads)
        elif args.actor_type == "communicated":
            self.actor = CommunicatedActor(envs, hidden_dim=args.actor_hidden_dim, num_heads=args.actor_num_heads)
        else:
            raise Exception

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, action=None):
        return self.actor(x, action)

    def get_action_and_value(self, x, action=None):
        action, logprob, entropy = self.get_action(x, action)
        value = self.get_value(x)
        return action, logprob, entropy, value
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
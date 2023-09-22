# MAPPO_base
A repository of the MAPPO implementation on the cooperative environments in PettingZoo.

## Installation

### Set up a Conda Environment
This setup script creates an environment named 'MAPPO_base'.
```
bash scripts/installation/setup_conda.sh
```

## Training

Run the following commands to train the agents.  
Note that the current version does only support training homogeneous agents in the cooperative setting.  
Refer to [PettingZoo](https://pettingzoo.farama.org/) for more information about the environments.
```
# env-id: ['sisl.multiwalker_v9', 'sisl.pursuit_v4', 'sisl.waterworld_v4', 'mpe.simple_spread_v2', ...]

bash scripts/train/fully_centralized_linear_linear.sh {env-id} # Linear actor & linear critic
bash scripts/train/fully_centralized_attention_attention.sh {env-id} # Attention actor & attention critic
bash scripts/train/ctde_attention.sh {env-id} # CTDE with attention critic
bash scripts/train/ctde_linear.sh {env-id} # CTDE with linear critic
bash scripts/train/fully_decentralized.sh {env-id} # Fully decentralized actor-critic. It is equivalent to simply training multiple PPO agents (with parameter-sharing) on multi-agent environments.
```

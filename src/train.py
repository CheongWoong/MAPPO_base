# reference: https://github.com/vwxyzjn/cleanrl
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import importlib
import random
import time

import gym
import numpy as np
import supersuit as ss
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .utils.arguments import parse_args
from .agents import Agent


if __name__ == "__main__":
    args = parse_args()
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_name = f"{args.exp_name}_{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = importlib.import_module(f"pettingzoo.{args.env_id}").parallel_env()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(env, args.num_envs, num_cpus=0, base_class="gym")
    envs.num_agents = env.num_envs
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    envs.is_continuous = True if isinstance(envs.single_action_space, gym.spaces.Box) else False
    if envs.is_continuous:
        envs = gym.wrappers.ClipAction(envs)
        envs = gym.wrappers.NormalizeObservation(envs)
        envs = gym.wrappers.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))
        envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
        envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, envs.num_agents) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, envs.num_agents) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, envs.num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs, envs.num_agents)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs, envs.num_agents)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs, envs.num_agents)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset(seed=args.seed)
    next_obs = next_obs.reshape((args.num_envs, envs.num_agents, -1)) ### Reshape for multi-agent
    next_obs = torch.Tensor(next_obs).to(device)
    # next_done = torch.zeros(args.num_envs).to(device)
    next_done = torch.zeros((args.num_envs, envs.num_agents)).to(device) ### Reshape for multi-agent
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                # values[step] = value.flatten()
                values[step] = value.squeeze(-1) ### Reshape for multi-agent
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            action = action.reshape((args.num_envs*envs.num_agents, -1)) ### Reshape for multi-agent
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            next_obs = next_obs.reshape((args.num_envs, envs.num_agents, -1)) ### Reshape for multi-agent
            reward = reward.reshape((args.num_envs, envs.num_agents)) ### Reshape for multi-agent
            # rewards[step] = torch.tensor(reward).to(device).view(-1)
            rewards[step] = torch.tensor(reward).to(device) ### Reshape for multi-agent
            done = done.reshape((args.num_envs, envs.num_agents)) ### Reshape for multi-agent
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for idx, item in enumerate(info):
                player_idx = idx % envs.num_agents
                if "episode" in item.keys():
                    print(f"global_step={global_step}, {player_idx}-episodic_return={item['episode']['r']}")
                    writer.add_scalar(f"charts/episodic_return-player{player_idx}", item["episode"]["r"], global_step)
                    writer.add_scalar(f"charts/episodic_length-player{player_idx}", item["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            # next_value = agent.get_value(next_obs).reshape(1, -1)
            next_value = agent.get_value(next_obs).squeeze(-1) ### Reshape for multi-agent
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, envs.num_agents) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1, envs.num_agents)
        b_actions = actions.reshape((-1, envs.num_agents) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1, envs.num_agents)
        b_returns = returns.reshape(-1, envs.num_agents)
        b_values = values.reshape(-1, envs.num_agents)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                # newvalue = newvalue.view(-1)
                newvalue = newvalue.squeeze(-1) ### Reshape for multi-agent
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import pickle
import time

# ────────────────────────────────────────────────
# REINFORCE for Inverted Pendulum
# ────────────────────────────────────────────────
def train_reinforce_inverted_pendulum(params):
    """
    Training REINFORCE algorithm for the Inverted Pendulum environment.
    """

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # ─────────────────────────────────────────────────────────────
    # Policy Network
    # ─────────────────────────────────────────────────────────────
    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.mean = nn.Linear(64, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable

        def forward(self, x):
            x = self.net(x)
            return self.mean(x), self.log_std.exp()

    # ─────────────────────────────────────────────────────────────
    # Training parameters
    # ─────────────────────────────────────────────────────────────
    GAMMA = params['GAMMA']
    MAX_STEPS = params['MAX_STEPS']
    LR = params['LR']
    NUM_RUNS = params['NUM_RUNS']

    # ─────────────────────────────────────────────────────────────
    # Main training loop (NUM_RUNS runs)
    # ─────────────────────────────────────────────────────────────
    all_step_returns = []
    all_episode_returns = []
    for iteration in range(NUM_RUNS):
        print(f"Iteration: {iteration}")

        start_time = time.time()

        # Create env and policy
        env = gym.make("InvertedPendulum-v5")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        policy = PolicyNetwork(state_dim, action_dim).to(device)
        optimizer = optim.Adam(policy.parameters(), LR)

        episode_returns = []
        step_returns = []
        best_ave_reward = 0
        episode = 0
        time_step = 0

        while time_step < MAX_STEPS:
            state, _ = env.reset()
            log_probs, rewards = [], []
            total_reward = 0

            for _ in range(1, env.spec.max_episode_steps+1):
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                mean, std = policy(state_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()

                action_clipped = action.clamp(
                    env.action_space.low[0], env.action_space.high[0]
                ).cpu().numpy()

                next_state, reward, terminated, truncated, _ = env.step(action_clipped)
                done = terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)
                total_reward += reward
                state = next_state

                time_step+=1
                if not done:
                    if episode > 0:
                        step_returns.append(episode_returns[-1])
                    else:
                        step_returns.append(0)
                else:
                    step_returns.append(total_reward)

                if done:
                    episode+=1
                    episode_returns.append(total_reward)
                    end_time = time.time()
                    duration = (end_time - start_time)/ 60 # Minutes
                    break

            # Compute discounted returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + GAMMA * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            log_probs = torch.stack(log_probs)
            policy_loss = - (log_probs * returns).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            if episode % 100 == 0:
                ave_reward = np.mean(episode_returns[-100:])
                print(f"Episode {episode} | Time Step: {time_step} | Time: {duration:.2f} minutes | Moving Average (100): {ave_reward:.2f}")
                if ave_reward > best_ave_reward:
                    torch.save(policy.state_dict(), f"./best_policy/reinforce_best_policy_{iteration}.pth")
                    best_ave_reward = ave_reward

        env.close()
        all_step_returns.append(step_returns)
        all_episode_returns.append(episode_returns)


    with open("./results/all_reinforce_step_returns.pkl", "wb") as f:
        pickle.dump(all_step_returns, f)

    with open("./results/all_reinforce_episode_returns.pkl", "wb") as f:
        pickle.dump(all_episode_returns, f)

    return step_returns


# ────────────────────────────────────────────────
# A2C for Inverted Pendulum
# ────────────────────────────────────────────────
def train_a2c_inverted_pendulum(params):
    """
    Training Advantage Actor-Critic (A2C) for the Inverted Pendulum environment.
    """

    # ────────────────────────────────────────────────
    # Hyperparameters
    # ────────────────────────────────────────────────
    GAMMA = params['GAMMA']
    N_STEP = params['N_STEP']
    LR_POLICY = params['LR_POLICY'] 
    LR_VALUE = params['LR_VALUE']   
    MAX_STEPS = params['MAX_STEPS']
    NUM_RUNS = params['NUM_RUNS']

    # ────────────────────────────────────────────────
    # Device setup
    # ────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # ────────────────────────────────────────────────
    # Training loop
    # ────────────────────────────────────────────────
    all_episode_returns = []
    all_step_returns = []
    for iteration in range(NUM_RUNS):
        print(f"Iteration: {iteration}")
        start_time = time.time()
        # ────────────────────────────────────────────────
        # Environment setup
        # ────────────────────────────────────────────────
        env = gym.make("InvertedPendulum-v5")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # ────────────────────────────────────────────────
        # Policy Network
        # ────────────────────────────────────────────────
        class PolicyNetwork(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim, 64), nn.ReLU(),
                    nn.Linear(64, 64), nn.ReLU()
                )
                self.mu_head = nn.Linear(64, action_dim)
                self.log_std = nn.Parameter(torch.zeros(action_dim))  # learnable std

            def forward(self, x):
                x = self.net(x)
                mu = self.mu_head(x)
                std = torch.exp(self.log_std)
                return mu, std

            def get_action(self, state):
                mu, std = self(state)
                dist = Normal(mu, std)
                action = dist.sample()
                return action.clamp(-1.0, 1.0), dist.log_prob(action).sum(dim=-1)

        # ────────────────────────────────────────────────
        # Value Network
        # ────────────────────────────────────────────────
        class ValueNetwork(nn.Module):
            def __init__(self, state_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim, 64), nn.ReLU(),
                    nn.Linear(64, 64), nn.ReLU(),
                    nn.Linear(64, 1)
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        # ────────────────────────────────────────────────
        # Initialize networks and optimizers
        # ────────────────────────────────────────────────
        policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        value_net = ValueNetwork(state_dim).to(device)
        optimizer_pi = optim.Adam(policy_net.parameters(), lr=LR_POLICY)
        optimizer_v = optim.Adam(value_net.parameters(), lr=LR_VALUE)

        # ────────────────────────────────────────────────
        # N-step return computation
        # ────────────────────────────────────────────────
        def compute_nstep_returns(rewards, next_values, dones, gamma, n):
            T = len(rewards)
            returns = [0] * T
            for t in reversed(range(T)):
                G = 0
                for k in range(n):
                    if t + k < T:
                        G += (gamma ** k) * rewards[t + k]
                        if dones[t + k]:
                            break
                if t + n < T and not dones[t + n - 1]:
                    G += (gamma ** n) * next_values[t + n].item()
                returns[t] = G
            return torch.tensor(returns, dtype=torch.float32, device=device)


        episode_returns = []
        step_returns = []
        episode_done_steps = []
        episode_times = []
        time_step = 0
        episode = 0
        best_ave_reward = 0
        while time_step < MAX_STEPS:
            state, _ = env.reset()
            done = False

            log_probs = []
            values = []
            rewards = []
            dones = []

            total_reward = 0

            for step in range(MAX_STEPS):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                action, log_prob = policy_net.get_action(state_tensor)
                value = value_net(state_tensor)

                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                done = terminated or truncated

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                dones.append(done)

                state = next_state
                total_reward += reward

                time_step+=1
                if not done:
                    if episode > 0:
                        step_returns.append(episode_returns[-1])
                    else:
                        step_returns.append(0)
                else:
                    step_returns.append(total_reward)

                if done:
                    episode+=1
                    episode_returns.append(total_reward)
                    end_time = time.time()
                    duration = (end_time - start_time)/ 60 # Minutes
                    episode_times.append(duration)
                    episode_done_steps.append(time_step)
                    break

            # Add final bootstrapped value for n-step target
            with torch.no_grad():
                next_values = values[1:] + [value_net(torch.tensor(next_state, dtype=torch.float32, device=device))]
            returns = compute_nstep_returns(rewards, next_values, dones, GAMMA, N_STEP)

            values_tensor = torch.stack(values)
            log_probs_tensor = torch.stack(log_probs)
            advantages = returns - values_tensor

            # ───── Update Value Network ─────
            value_loss = advantages.pow(2).mean()
            optimizer_v.zero_grad()
            value_loss.backward()
            optimizer_v.step()

            # ───── Update Policy Network ─────
            policy_loss = -(log_probs_tensor * advantages.detach()).mean()
            optimizer_pi.zero_grad()
            policy_loss.backward()
            optimizer_pi.step()

            if episode%100 == 0:
                avg_return = np.mean(episode_returns[-100:])
                print(f"Episode {episode:>4} | Time Step: {time_step} | Time: {duration:.2f} minutes | Moving Average (100): {avg_return:.2f}")

                if avg_return > best_ave_reward:
                    torch.save(policy_net.state_dict(), f"./best_policy/a2c_best_policy_{iteration}.pth")
                    best_ave_reward = avg_return

        all_episode_returns.append(episode_returns)
        all_step_returns.append(step_returns)

        env.close()

    with open("./results/all_a2c_episode_returns.pkl", "wb") as f:
        pickle.dump(all_episode_returns, f)

    with open("./results/all_a2c_step_returns.pkl", "wb") as f:
        pickle.dump(all_step_returns, f)

    return all_step_returns



# ────────────────────────────────────────────────
# PPO for Inverted Pendulum
# ────────────────────────────────────────────────
def train_ppo_inverted_pendulum(params):
    """
    Proximal Policy Optimization (PPO) implementation for the Inverted Pendulum environment.
    """
    # ────────────────────────────────────────────────
    # Hyperparameters
    # ────────────────────────────────────────────────
    GAMMA = params['GAMMA']
    GAE_LAMBDA = params['GAE_LAMBDA']
    CLIP_EPS = params['CLIP_EPS']
    LR_POLICY = params['LR_POLICY']
    LR_VALUE = params['LR_VALUE']
    BATCH_SIZE = params['BATCH_SIZE']
    EPOCHS = params['EPOCHS']
    MINIBATCH_SIZE = params['MINIBATCH_SIZE']
    MAX_STEPS = params['MAX_STEPS']
    NUM_RUNS = params['NUM_RUNS']

    # ────────────────────────────────────────────────
    # Device setup
    # ────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # ────────────────────────────────────────────────
    # Policy Network
    # ────────────────────────────────────────────────
    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU()
            )
            self.mu_head = nn.Linear(64, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        def forward(self, x):
            x = self.net(x)
            mu = self.mu_head(x)
            std = torch.exp(self.log_std)
            return mu, std

        def get_dist(self, state):
            mu, std = self(state)
            return Normal(mu, std)

        def get_action(self, state):
            dist = self.get_dist(state)
            action = dist.sample()
            return action.clamp(-1, 1), dist.log_prob(action).sum(dim=-1)

    # ────────────────────────────────────────────────
    # Value Network
    # ────────────────────────────────────────────────
    class ValueNetwork(nn.Module):
        def __init__(self, state_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    # ────────────────────────────────────────────────
    # GAE Computation
    # ────────────────────────────────────────────────
    def compute_gae(rewards, values, dones, gamma, lam):
        advantages = []
        gae = 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values[:-1])]
        return torch.tensor(advantages, dtype=torch.float32, device=device), torch.tensor(returns, dtype=torch.float32, device=device)

    # ────────────────────────────────────────────────
    # Training Loop
    # ────────────────────────────────────────────────
    all_step_returns = []
    all_episode_returns = []

    for iteration in range(NUM_RUNS):
        print(f"Iteration: {iteration}")

        env = gym.make("InvertedPendulum-v5")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        value_net = ValueNetwork(state_dim).to(device)
        optimizer_pi = optim.Adam(policy_net.parameters(), lr=LR_POLICY)
        optimizer_v = optim.Adam(value_net.parameters(), lr=LR_VALUE)

        episode_returns = []
        step_returns = []
        start_time = time.time()
        state, _ = env.reset()
        episode_reward = 0

        # Experience buffer
        states, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []
        episode = 0
        time_step = 0
        best_ave_reward = 0
        while time_step<=MAX_STEPS:
            for _ in range(BATCH_SIZE):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                dist = policy_net.get_dist(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = value_net(state_tensor)

                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                done = terminated or truncated

                # Store transition
                states.append(state_tensor)
                actions.append(action)
                log_probs_old.append(log_prob.detach())
                rewards.append(reward)
                dones.append(done)
                values.append(value.item())

                episode_reward += reward
                state = next_state

                time_step+=1
                if not done:
                    if episode > 0:
                        step_returns.append(episode_returns[-1])
                    else:
                        step_returns.append(0)
                else:
                    step_returns.append(episode_reward)

                if done:
                    episode_returns.append(episode_reward)
                    state, _ = env.reset()
                    episode_reward = 0
                    episode+=1

            # ────────────────────────────────────────────────
            # Compute GAE and returns
            # ────────────────────────────────────────────────
            with torch.no_grad():
                next_value = value_net(torch.tensor(state, dtype=torch.float32, device=device)).item()
            values.append(next_value)
            advs, rets = compute_gae(rewards, values, dones, GAMMA, GAE_LAMBDA)
            states = torch.stack(states)
            actions = torch.stack(actions)
            log_probs_old = torch.stack(log_probs_old)

            # Normalize advantages
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            # ────────────────────────────────────────────────
            # 🔁 PPO Updates
            # ────────────────────────────────────────────────
            for _ in range(EPOCHS):
                idx = torch.randperm(BATCH_SIZE)
                for i in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                    j = idx[i:i+MINIBATCH_SIZE]
                    batch_states = states[j]
                    batch_actions = actions[j]
                    batch_log_probs_old = log_probs_old[j]
                    batch_advs = advs[j]
                    batch_returns = rets[j]

                    dist = policy_net.get_dist(batch_states)
                    log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    ratio = torch.exp(log_probs - batch_log_probs_old)

                    surr1 = ratio * batch_advs
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_advs
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = (value_net(batch_states) - batch_returns).pow(2).mean()

                    optimizer_pi.zero_grad()
                    policy_loss.backward()
                    optimizer_pi.step()

                    optimizer_v.zero_grad()
                    value_loss.backward()
                    optimizer_v.step()

            end_time = time.time()
            duration = (end_time - start_time)/60 # in minutes

            avg_return = np.mean(episode_returns[-100:])
            print(f"Episode {episode+1:>4} | Time step: {time_step} | Time: {duration:.2f} minutes | Moving Average (100): {avg_return:.2f}")

            if avg_return > best_ave_reward:
                    torch.save(policy_net.state_dict(), f"./best_policy/ppo_best_policy_{iteration}.pth")
                    best_ave_reward = avg_return

            # Clear buffer
            states, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

        all_step_returns.append(step_returns)
        env.close()

        all_episode_returns.append(episode_returns)
        all_step_returns.append(step_returns)


    with open("./results/all_ppo_step_returns.pkl", "wb") as f:
        pickle.dump(all_step_returns, f)

    with open("./results/all_ppo_episode_returns.pkl", "wb") as f:
        pickle.dump(all_episode_returns, f)

    return all_step_returns
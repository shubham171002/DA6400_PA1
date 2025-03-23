import gymnasium as gym
import numpy as np
import wandb
import matplotlib.pyplot as plt

# Softmax action selection
def softmax_action(Q, state, tau, n_actions):
    preferences = Q[state] / tau
    max_pref = np.max(preferences)
    exp_prefs = np.exp(preferences - max_pref)  # for numerical stability
    probs = exp_prefs / np.sum(exp_prefs)
    return np.random.choice(n_actions, p=probs)

# Discretize continuous state
def discretize_state(state, bins):
    return tuple(np.digitize(s, b) for s, b in zip(state, bins))

# Q-Learning implementation
def train_q_learning(env_name, episodes, alpha, gamma, tau, bins, seeds, use_wandb=True):
    if use_wandb:
        wandb.init(
            project="DA6400_PA1",
            name=f"Q_Learning_{env_name}",
            config={"alpha": alpha, "tau": tau, "gamma": gamma, "episodes": episodes}
        )

    all_rewards = []

    for seed in seeds:
        env = gym.make(env_name)
        np.random.seed(seed)
        env.reset(seed=seed)

        Q_shape = tuple(len(b) + 1 for b in bins) + (env.action_space.n,)
        Q = np.zeros(Q_shape)
        rewards = []

        for ep in range(episodes):
            state, _ = env.reset()
            state_disc = discretize_state(state, bins)

            done, ep_reward = False, 0

            while not done:
                action = softmax_action(Q, state_disc, tau, env.action_space.n)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state_disc = discretize_state(next_state, bins)

                # Q-learning update (off-policy)
                Q[state_disc][action] += alpha * (
                    reward + gamma * np.max(Q[next_state_disc]) - Q[state_disc][action]
                )

                state_disc = next_state_disc
                ep_reward += reward

            rewards.append(ep_reward)
            if use_wandb:
                wandb.log({"episode": ep, "reward": ep_reward})

        all_rewards.append(rewards)
        env.close()

    if use_wandb:
        wandb.finish()

    return np.array(all_rewards)

# Plotting function
def plot_results(rewards, env_name):
    mean_rewards = np.mean(rewards, axis=0)
    std_rewards = np.std(rewards, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_rewards, label='Mean Reward (Q-Learning)')
    plt.fill_between(range(len(mean_rewards)),
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards, alpha=0.2)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title(f"Q-Learning Performance on {env_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/qlearning_{env_name.lower()}.png")
    plt.show()

# For manual testing (optional)
if __name__ == "__main__":
    episodes = 500
    seeds = [42, 43, 44, 45, 46]

    bins = [
        np.linspace(-4.8, 4.8, 20),
        np.linspace(-4, 4, 20),
        np.linspace(-0.418, 0.418, 20),
        np.linspace(-4, 4, 20),
    ]

    env_name = "CartPole-v1"

    rewards = train_q_learning(
        env_name=env_name,
        episodes=episodes,
        alpha=0.3,
        gamma=0.99,
        tau=1.0,
        bins=bins,
        seeds=seeds
    )

    plot_results(rewards, env_name)
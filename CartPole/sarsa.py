import gymnasium as gym
import numpy as np
import wandb
import matplotlib.pyplot as plt

# Function for epsilon-greedy action selection
def epsilon_greedy(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    return np.argmax(Q[state])

# Function to discretize continuous states (required for CartPole)
def discretize_state(state, bins):
    return tuple(np.digitize(s, b) for s, b in zip(state, bins))

def train_sarsa(env_name, episodes, alpha, gamma, epsilon, bins, seeds, use_wandb=True):
    if use_wandb:
        wandb.init(
            project="DA6400_PA1",
            name=f"SARSA_{env_name}",
            config={"alpha": alpha, "epsilon": epsilon, "gamma": gamma, "episodes": episodes}
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
            action = epsilon_greedy(Q, state_disc, epsilon, env.action_space.n)

            done, ep_reward = False, 0

            while not done:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state_disc = discretize_state(next_state, bins)
                next_action = epsilon_greedy(Q, next_state_disc, epsilon, env.action_space.n)

                Q[state_disc][action] += alpha * (
                    reward + gamma * Q[next_state_disc][next_action] - Q[state_disc][action]
                )

                state_disc, action = next_state_disc, next_action
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
    plt.plot(mean_rewards, label='Mean Reward (SARSA)')
    plt.fill_between(range(len(mean_rewards)),
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards, alpha=0.2)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title(f"SARSA Performance on {env_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/sarsa_{env_name.lower()}.png")
    plt.show()

# Main execution block
if __name__ == "__main__":
    episodes = 500
    seeds = [42, 43, 44, 45, 46]

    # State discretization bins specifically for CartPole-v1
    bins = [
        np.linspace(-4.8, 4.8, 10),      # cart position
        np.linspace(-4, 4, 10),          # cart velocity
        np.linspace(-0.418, 0.418, 10),  # pole angle
        np.linspace(-4, 4, 10),          # pole angular velocity
    ]

    env_name = "CartPole-v1"

    rewards = train_sarsa(
        env_name=env_name,
        episodes=episodes,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        bins=bins,
        seeds=seeds
    )

    plot_results(rewards, env_name)

import wandb
import numpy as np
from sarsa import train_sarsa

# Define sweep config
sweep_config = {
    'method': 'grid', 
    'metric': {'name': 'mean_reward', 'goal': 'maximize'},
    'parameters': {
        # 'alpha': {'values': [0.05, 0.1, 0.2, 0.3]},
        # 'epsilon': {'values': [0.01, 0.05, 0.1, 0.2]},
        # 'num_bins': {'values': [10, 15, 20, 25]},  # number of bins
        'alpha': {'values': [0.25, 0.3, 0.35, 0.4]},  # refined around previous best alpha=0.3
        'epsilon': {'values': [0.15, 0.2, 0.25]},     # refined around previous best epsilon=0.2
        'num_bins': {'values': [20, 25, 30]},         # refined granularity
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="DA6400_PA1")

# Sweep training function
def sweep_train():
    wandb.init()
    config = wandb.config

    bins = [
        np.linspace(-4.8, 4.8, config.num_bins),
        np.linspace(-4, 4, config.num_bins),
        np.linspace(-0.418, 0.418, config.num_bins),
        np.linspace(-4, 4, config.num_bins),
    ]

    episodes = 5000  # You can adjust for faster experimentation
    seeds = [42, 43, 44, 45, 46]

    rewards = train_sarsa(
        env_name="CartPole-v1",
        episodes=episodes,
        alpha=config.alpha,
        gamma=0.99,
        epsilon=config.epsilon,
        bins=bins,
        seeds=seeds,
        use_wandb=False  # Logging is handled here explicitly
    )

    mean_reward = np.mean(rewards[:, -100:])  # average of the last 50 episodes

    # Log mean reward explicitly
    wandb.log({"mean_reward": mean_reward})

wandb.agent(sweep_id, sweep_train)  
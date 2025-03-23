import wandb
import numpy as np
from q_learning import train_q_learning

# Define sweep config for Q-learning
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'mean_reward', 'goal': 'maximize'},
    'parameters': {
        'alpha': {'values': [0.1, 0.2, 0.3]},
        'tau': {'values': [0.5, 1.0, 2.0]},  # softmax temperature
        'num_bins': {'values': [20, 25, 30]},
    }
}

sweep_id = wandb.sweep(sweep_config, project="DA6400_PA1")

# Sweep training function
def sweep_train():
    wandb.init()
    config = wandb.config

    # Discretization bins based on sweep value
    bins = [
        np.linspace(-4.8, 4.8, config.num_bins),
        np.linspace(-4, 4, config.num_bins),
        np.linspace(-0.418, 0.418, config.num_bins),
        np.linspace(-4, 4, config.num_bins),
    ]

    episodes = 1000  # Enough for initial convergence
    seeds = [42, 43, 44, 45, 46]

    rewards = train_q_learning(
        env_name="CartPole-v1",
        episodes=episodes,
        alpha=config.alpha,
        gamma=0.99,
        tau=config.tau,
        bins=bins,
        seeds=seeds,
        use_wandb=False  # We will log summary manually
    )

    mean_reward = np.mean(rewards[:, -100:])  # Mean over last 100 episodes
    wandb.log({"mean_reward": mean_reward})

wandb.agent(sweep_id, sweep_train)
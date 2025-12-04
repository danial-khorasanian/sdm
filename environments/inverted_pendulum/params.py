'''
Hyperparameters of agents for the Inverted Pendulum environment.
'''

# Reinforce
reinforce_params = {
    'GAMMA': 0.99,          # Discount factor 
    'MAX_STEPS': 1_000_000, # Total training steps
    'LR': 1e-3,             # Learning rate
    'NUM_RUNS': 5,          # Number of runs
    'MAX_EPISODE_STEPS': 1000,         # Max steps per episode
}


# A2C
a2c_params = {
    'GAMMA': 0.99,         # Discount factor
    'MAX_STEPS': 1_000_000,   # Total training steps
    'N_STEP': 5,            # Number of steps for N-step return
    'LR_POLICY': 1e-3,         # Learning rate (policy)
    'LR_VALUE': 1e-3,         # Learning rate (value)
    'NUM_RUNS': 5,          # Number of runs
}


# PPO
ppo_params = {
    'GAMMA': 0.99,            # Discount factor
    'MAX_STEPS': 1_000_000,   # Total training steps
    'GAE_LAMBDA': 0.95,       # GAE lambda
    'CLIP_EPS': 0.2,          # PPO clipping parameter
    'LR_POLICY': 3e-4,        # Learning rate (policy)
    'LR_VALUE': 1e-3,         # Learning rate (value)
    'BATCH_SIZE': 2048,       # Batch size per PPO update
    'EPOCHS': 10,             # Number of epochs per update
    'MINIBATCH_SIZE': 256,    # Mini-batch size
    'NUM_RUNS': 5,            # Number of runs
}
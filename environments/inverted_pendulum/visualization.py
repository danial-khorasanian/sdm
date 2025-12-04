import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from IPython.display import Image, display


# ────────────────────────────────────────────────
# Show PNG Function
# ────────────────────────────────────────────────
def show_png(filepath, scale=100):
    try:
        display(Image(filename=filepath, width=int(scale/100 * 800), height=int(scale/100 * 800)))
    except Exception as e:
        print(f"Error displaying image: {e}")



# ────────────────────────────────────────────────
# Plot All Agents Function  
# ────────────────────────────────────────────────
def plot_all_agents(file_name, MAX_STEPS=1_000_000, BLOCK_SIZE=200, WINDOW_SIZE=100, PLOT_MAX_Y=1000):
    '''
    Plots the performance comparison of multiple RL agents based on their step returns.
    BLOCK_SIZE: Block size for initial downsampling/aggregation
    WINDOW_SIZE: Window size for smoothing (moving average)
    '''

    def process_data(all_step_returns, block_size, window_size):
        """
        Processes a list of lists of step returns (one inner list per run).
        1. Downsamples the data by averaging blocks.
        2. Smooths the downsampled data using a rolling mean (moving average).
        3. Stacks runs into a matrix.
        """
        processed_runs = []
        
        # Block Averaging and Smoothing for each run
        for run_data in all_step_returns:
            # Downsample/Aggregate (equivalent to reduced_mean_list)
            # Use simple array slicing and reshaping for block averaging
            run_data = np.array(run_data)
            
            # Ensure data length is a multiple of block_size for reshaping
            trimmed_len = (len(run_data) // block_size) * block_size
            trimmed_data = run_data[:trimmed_len]
            
            # Reshape to (num_blocks, block_size) and calculate mean across axis 1
            block_means = trimmed_data.reshape(-1, block_size).mean(axis=1)
            
            # Smooth using Pandas rolling mean (equivalent to m_average)
            smoothed_means = pd.Series(block_means).rolling(
                window=window_size, 
                min_periods=1, # Ensure smoothing starts immediately
                center=False
            ).mean().to_numpy()
            
            processed_runs.append(smoothed_means)

        # Stack runs, ensuring all arrays are padded to the maximum length
        # This avoids manual min_len calculation and slicing
        if not processed_runs:
            return None, None, None

        # Find the length of the longest processed run
        max_len = max(len(r) for r in processed_runs)
        
        # Pad shorter arrays with NaN so they can be stacked
        padded_runs = []
        for r in processed_runs:
            padding = np.full(max_len - len(r), np.nan)
            padded_runs.append(np.concatenate((r, padding)))

        matrix = np.vstack(padded_runs)

        # Compute statistics, ignoring NaNs introduced by padding
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        
        # Generate the corresponding X-axis values (Time Steps)
        # The length of the mean array is the number of blocks/steps
        x = np.arange(len(mean)) * block_size
        
        # Trim the statistics and x-values to MAX_STEPS
        steps_to_keep = min(len(x), MAX_STEPS // block_size)
        
        return x[:steps_to_keep], mean[:steps_to_keep], std[:steps_to_keep]


    # Load Data
    agents_data = {
        "REINFORCE": {"path": "./results/all_reinforce_step_returns.pkl", "color": "green"},
        "A2C":       {"path": "./results/all_a2c_step_returns.pkl",       "color": "blue"},
        "PPO":       {"path": "./results/all_ppo_step_returns.pkl",       "color": "red"},
    }
    
    # Process and Plot
    plt.figure(figsize=(7, 4.5))
    
    for agent_name, agent_info in agents_data.items():
        # Load from file
        try:
            with open(agent_info["path"], "rb") as f:
                all_returns = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Data file not found for {agent_name} at {agent_info['path']}")
            continue

        # Process data using the streamlined function
        x, mean, std = process_data(all_returns, BLOCK_SIZE, WINDOW_SIZE)
        
        if mean is None:
            continue

        # Plot Mean
        plt.plot(x, mean, color=agent_info["color"], label=agent_name)
        
        # Plot Shadowed Standard Deviation (Confidence Interval)
        # Use the global PLOT_MAX_Y for clipping
        plt.fill_between(
            x,
            np.clip(mean - std, 0, PLOT_MAX_Y),   # lower bound
            np.clip(mean + std, 0, PLOT_MAX_Y),   # upper bound
            color=agent_info["color"],
            alpha=0.2
        )

    # Finalize Plot
    plt.xlabel("Time Steps")
    plt.ylabel("Total Rewards")
    plt.title(f"Performance Comparison of RL Agents")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{file_name}.png", dpi=300)
    plt.close()

    show_png(f"plots/{file_name}.png", 50)
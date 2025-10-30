import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ========== CONFIGURATION ==========
REWARD_DIR = "results/csv_logs/reward"     # Folder containing eval/mean_reward CSV files
LENGTH_DIR = "results/csv_logs/length"     # Folder containing eval/mean_ep_length CSV files
SMOOTH_WINDOW = 10                   # Moving average window size
SAVE_FIG = True
OUTPUT_REWARD_FIG = f"graphing/graphs/line/reward_curves.png"
OUTPUT_LENGTH_FIG = f"graphing/graphs/line/length_curves.png"


def plot_metric(metric_name, log_dir, output_fig, ylabel):
    """Generic plotting function for TensorBoard-exported CSV metrics."""
    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {log_dir}. Please export them from TensorBoard.")

    plt.figure(figsize=(10, 6))
    summary = []

    for csv_file in csv_files:
        run_name = os.path.splitext(os.path.basename(csv_file))[0]
        df = pd.read_csv(csv_file)

        # Ensure correct columns
        if "Step" not in df.columns or "Value" not in df.columns:
            raise ValueError(f"File {csv_file} missing required columns (Step, Value).")

        # Smooth values
        df["Smoothed"] = df["Value"].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()

        # Compute stats
        final_avg = df["Value"].tail(10).mean()
        max_val = df["Value"].max()
        last_step = df["Step"].iloc[-1]
        summary.append((run_name, final_avg, max_val, last_step))

        # Plot
        plt.plot(df["Step"], df["Smoothed"], label=run_name)

    # Style and labels
    plt.title(f"Evaluation Performance: {metric_name}")
    plt.xlabel("Environment Steps")
    plt.ylabel(f"{ylabel} (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if SAVE_FIG:
        os.makedirs(os.path.dirname(output_fig), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_fig, dpi=300)
        print(f"Figure saved as: {output_fig}")


    # Print summary
    print(f"\n========== {metric_name} Summary ==========")
    for run_name, final_avg, max_val, last_step in summary:
        print(f"{run_name:25s} | Final Avg: {final_avg:8.2f} | Max: {max_val:8.2f} | Steps: {last_step}")


# ========== PLOT BOTH METRICS ==========
plot_metric("eval/mean_reward", REWARD_DIR, OUTPUT_REWARD_FIG, ylabel="Episode Reward")
plot_metric("eval/mean_ep_length", LENGTH_DIR, OUTPUT_LENGTH_FIG, ylabel="Episode Length")

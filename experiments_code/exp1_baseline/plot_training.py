# plot_training.py
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def smooth(y, k=50):
    if len(y) < k: 
        return y
    w = np.ones(k) / k
    return np.convolve(y, w, mode="valid")

def load_rewards(monitor_csv):
    # SB3 Monitor CSV with episode rewards in column 'r'
    df = pd.read_csv(monitor_csv, comment="#")
    if "r" not in df.columns:
        raise ValueError(f"'r' column not found in {monitor_csv}")
    return df["r"].to_numpy(dtype=float)

def trim_tail(y_smooth, keep_extra=200, high_threshold=200.0):
    """
    Keep data up to a point where learning still matters.
    Heuristic: find the last index where smoothed reward > high_threshold,
    and keep a small buffer (keep_extra). If the curve never crosses the
    threshold, keep everything; if it crosses, drop the very long flat tail.
    """
    idx = np.where(y_smooth > high_threshold)[0]
    if len(idx) == 0:
        return len(y_smooth)  # never gets high → keep all
    last_high = idx[-1]
    cut = min(len(y_smooth), last_high + keep_extra)
    return cut

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_root", default="logs_clean", help="folder with train_seed*/monitor.csv")
    ap.add_argument("--out", default="figures/cartpole_baseline_training.png")
    ap.add_argument("--smooth_k", type=int, default=50, help="rolling mean window over episodes")
    ap.add_argument("--high_threshold", type=float, default=200.0, help="trim after last point above this")
    ap.add_argument("--keep_extra", type=int, default=200, help="extra points to keep after last high point")
    ap.add_argument("--max_episodes", type=int, default=4000, help="hard cap if you want to clip early")
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(12,7))
    plotted_any = False

    for seed in [0,1,2]:
        mon = os.path.join(args.log_root, f"train_seed{seed}", "monitor.csv")
        if not os.path.exists(mon):
            print(f"[WARN] Missing: {mon} — skipping seed {seed}")
            continue

        r = load_rewards(mon)
        # Optional hard cap so the x-axis stays reasonable
        if args.max_episodes and len(r) > args.max_episodes:
            r = r[:args.max_episodes]

        y = smooth(r, k=args.smooth_k)
        if len(y) == 0:
            print(f"[WARN] Seed {seed} had no data after smoothing.")
            continue

        # Auto-trim the “dead tail”
        cut = trim_tail(y, keep_extra=args.keep_extra, high_threshold=args.high_threshold)
        y = y[:cut]
        x = np.arange(len(y))

        plt.plot(x, y, label=f"seed {seed}", linewidth=2)
        plotted_any = True

    if not plotted_any:
        print("[ERROR] No seeds plotted. Check paths in --log_root.")
        return

    plt.title("CartPole DQN Training (clean, 3 seeds) — rolling mean")
    plt.xlabel("Episode (post-smoothing index)")
    plt.ylabel("Episode Reward")
    plt.ylim(0, 520)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[SAVED] {args.out}")

if __name__ == "__main__":
    main()

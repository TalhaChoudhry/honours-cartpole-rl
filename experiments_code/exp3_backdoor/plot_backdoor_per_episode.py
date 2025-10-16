# plot_backdoor_per_episode.py
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 160})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--show_seeds", action="store_true")
    args = ap.parse_args()

    # Load and concat all seeds
    dfs = [pd.read_csv(p) for p in args.csvs]
    df = pd.concat(dfs, ignore_index=True)

    # Melt to long format: columns -> seed, episode, cond, ret
    long = df.melt(
        id_vars=["seed", "episode"],
        value_vars=["clean", "trigger"],
        var_name="cond",
        value_name="ret",   # use non-reserved name
    )

    # Aggregate across seeds per episode
    agg = (
        long.groupby(["cond", "episode"])["ret"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["cond", "episode"])
    )

    fig, ax = plt.subplots(figsize=(8.5, 5))

    # Plot mean ± std for Clean and Trigger
    for cond in ["clean", "trigger"]:
        sub = agg[agg["cond"] == cond]
        ax.plot(
            sub["episode"],
            sub["mean"],
            marker="o",
            linewidth=2,
            label=cond.capitalize(),
        )
        ax.fill_between(
            sub["episode"],
            sub["mean"] - sub["std"],
            sub["mean"] + sub["std"],
            alpha=0.15,
        )

    # Optional faint per-seed traces
    if args.show_seeds:
        for seed, sgrp in long.groupby("seed"):
            for cond in ["clean", "trigger"]:
                sub = sgrp[sgrp["cond"] == cond].sort_values("episode")
                ax.plot(
                    sub["episode"],
                    sub["ret"],
                    alpha=0.25,
                    linewidth=1,
                )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Backdoor effect — per-episode returns (mean ± std across seeds)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"[SAVED] {args.out}")

if __name__ == "__main__":
    main()

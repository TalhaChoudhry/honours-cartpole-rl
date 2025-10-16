# plot_clean_vs_poison.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_curve(csv_path: str, label: str):
    df = pd.read_csv(csv_path)
    # Be robust to column naming
    step_col = "step" if "step" in df.columns else df.columns[0]
    mean_col = "mean" if "mean" in df.columns else df.columns[1]
    std_col  = "std"  if "std"  in df.columns else None

    steps = df[step_col].values
    means = df[mean_col].values
    stds  = df[std_col].values if std_col and std_col in df.columns else None
    return steps, means, stds, label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_csv", required=True, help="CSV for clean training eval history")
    ap.add_argument("--poison_csv", required=True, help="CSV for poisoned training eval history")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--title", default="Clean vs Poisoned Training (CartPole, DQN)")
    args = ap.parse_args()

    steps_c, mean_c, std_c, lbl_c = load_curve(args.clean_csv, "Clean (seed 0)")
    steps_p, mean_p, std_p, lbl_p = load_curve(args.poison_csv, "Poisoned (seed 0)")

    plt.figure(figsize=(8,5))
    # Clean
    if std_c is not None:
      plt.fill_between(steps_c, mean_c-std_c, mean_c+std_c, alpha=0.15)
    plt.plot(steps_c, mean_c, label=lbl_c, linewidth=2)

    # Poisoned
    if std_p is not None:
      plt.fill_between(steps_p, mean_p-std_p, mean_p+std_p, alpha=0.15)
    plt.plot(steps_p, mean_p, label=lbl_p, linewidth=2)

    plt.title(args.title)
    plt.xlabel("Training steps")
    plt.ylabel("Eval mean reward (30 eps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[SAVED] {args.out}")

if __name__ == "__main__":
    main()

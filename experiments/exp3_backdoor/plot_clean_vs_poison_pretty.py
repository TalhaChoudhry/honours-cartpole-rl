# plot_clean_vs_poison_pretty.py
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_eval_csv(path):
    df = pd.read_csv(path)
    # Try to be robust to column names
    step_col = "step" if "step" in df.columns else "steps"
    mean_col = "mean" if "mean" in df.columns else ("mean_return" if "mean_return" in df.columns else df.columns[1])
    std_col  = "std"  if "std"  in df.columns else ("std_return"  if "std_return"  in df.columns else (df.columns[2] if len(df.columns) > 2 else None))

    df = df.rename(columns={step_col:"step", mean_col:"mean", std_col:"std" if std_col else "std"})
    if "std" not in df.columns:  # if std missing, set to small number so band doesn't explode
        df["std"] = 0.0

    # Keep only the last entry per step (avoids plotting old evals from earlier attempts)
    df = df.sort_values("step").drop_duplicates(subset=["step"], keep="last")
    df = df.sort_values("step").reset_index(drop=True)
    return df[["step","mean","std"]]

def ema(x, alpha=0.3):
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_csv", required=True)
    ap.add_argument("--poison_csv", required=True)
    ap.add_argument("--out", default="figures\\poison_vs_clean_seed0_pretty.png")
    ap.add_argument("--title", default="Clean vs Poisoned (training-time)")
    ap.add_argument("--smooth", type=float, default=0.25, help="EMA alpha in [0,1]; 0 disables smoothing")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    clean = load_eval_csv(args.clean_csv)
    pois  = load_eval_csv(args.poison_csv)

    # Optional smoothing
    if args.smooth > 0:
        clean["mean_s"] = ema(clean["mean"].to_numpy(), alpha=args.smooth)
        pois["mean_s"]  = ema(pois["mean"].to_numpy(),  alpha=args.smooth)
    else:
        clean["mean_s"] = clean["mean"]
        pois["mean_s"]  = pois["mean"]

    # Plot
    plt.figure(figsize=(11,6.5))
    for df, label, color in [(clean,"Clean (seed 0)","#1f77b4"), (pois,"Poisoned (seed 0)","#ff7f0e")]:
        x = df["step"].to_numpy()
        y = df["mean_s"].to_numpy()
        s = df["std"].to_numpy()
        # Clip absurd std to keep bands readable
        s = np.clip(s, 0, 200)

        plt.plot(x, y, label=label, linewidth=2.5)
        plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.title(args.title, fontsize=16)
    plt.xlabel("Training steps", fontsize=13)
    plt.ylabel("Eval mean reward (30 eps)", fontsize=13)
    plt.grid(True, alpha=0.25, linestyle="--")
    plt.legend(loc="upper left")
    plt.ylim(0, 520)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[SAVED] {args.out}")

if __name__ == "__main__":
    main()

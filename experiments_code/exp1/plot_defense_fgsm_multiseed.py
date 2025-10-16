import argparse, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_cat(files, label):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # expected columns: epsilon, mean, std, label (optional)
        df["group"] = label
        dfs.append(df[["epsilon","mean","std","group"]])
    return pd.concat(dfs, ignore_index=True)

def aggregate(df):
    # aggregate across seeds: mean of 'mean', std across seeds of 'mean'
    agg = (df.groupby(["group","epsilon"])
             .agg(mean=("mean","mean"),
                  std=("mean","std"),
                  n=("mean","size"))
             .reset_index())
    return agg

def plot_fgsm(agg, out_png, out_csv):
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    # save table
    agg_pvt = agg.pivot(index="epsilon", columns="group", values="mean")
    agg.to_csv(out_csv, index=False)

    plt.figure(figsize=(10,6))
    eps = sorted(agg["epsilon"].unique())
    for group, color in [("baseline","C0"), ("defense","C1")]:
        sub = agg[agg["group"]==group].sort_values("epsilon")
        plt.plot(sub["epsilon"], sub["mean"], marker="o", label=group.capitalize(), linewidth=2)
        # shaded ± std across seeds
        y, s = sub["mean"].to_numpy(), sub["std"].to_numpy()
        plt.fill_between(sub["epsilon"], y - s, y + s, alpha=0.15)

    plt.title("FGSM — Baseline vs Defense (mean ± std across seeds)")
    plt.xlabel("FGSM ε")
    plt.ylabel("Mean episode reward (30 eps avg)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[SAVED] {out_png}")
    print(f"[SAVED] {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csvs", nargs="+", required=True, help="fgsm_seed*.csv for baseline")
    ap.add_argument("--defense_csvs",  nargs="+", required=True, help="fgsm_defense*.csv for defense seeds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary_csv", required=True)
    args = ap.parse_args()

    base = load_cat(args.baseline_csvs, "baseline")
    defe = load_cat(args.defense_csvs,  "defense")
    agg  = aggregate(pd.concat([base, defe], ignore_index=True))
    plot_fgsm(agg, args.out, args.summary_csv)

if __name__ == "__main__":
    main()

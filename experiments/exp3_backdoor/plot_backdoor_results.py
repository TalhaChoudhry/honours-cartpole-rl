# plot_backdoor_results.py
import argparse, pandas as pd, matplotlib.pyplot as plt, numpy as np, os

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--summary_csv", required=True)
    p.add_argument("--out_fig", default="figures/backdoor_clean_vs_trigger.png")
    p.add_argument("--out_table", default="figures/backdoor_clean_vs_trigger_summary.csv")
    args=p.parse_args()

    df=pd.read_csv(args.summary_csv)
    # one row per seed label
    agg = df.copy()
    agg["delta"] = agg["trigger_mean"] - agg["clean_mean"]
    agg["delta_pct"] = 100*(agg["trigger_mean"]/agg["clean_mean"]-1.0)
    agg.to_csv(args.out_table, index=False)

    labels = agg["label"].tolist()
    clean  = agg["clean_mean"].to_numpy()
    trig   = agg["trigger_mean"].to_numpy()
    x = np.arange(len(labels))
    w=0.35

    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    plt.figure(figsize=(7,4))
    plt.bar(x-w/2, clean, width=w, label="Clean")
    plt.bar(x+w/2, trig,  width=w, label="Trigger")
    plt.xticks(x, labels)
    plt.ylabel("Return (30-episode mean)")
    plt.title("Backdoor effect: Clean vs Trigger (per seed)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=200)
    print(f"[SAVED] {args.out_fig}\n[SAVED] {args.out_table}")

if __name__=="__main__":
    main()

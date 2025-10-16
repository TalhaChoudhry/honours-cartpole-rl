import argparse, pandas as pd, matplotlib.pyplot as plt, numpy as np, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--defense_csv", required=True)
    ap.add_argument("--out", default=r"figures\fgsm_defense_vs_baseline.png")
    ap.add_argument("--summary_csv", default=r"figures\fgsm_defense_vs_baseline_summary.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    b = pd.read_csv(args.baseline_csv)
    d = pd.read_csv(args.defense_csv)

    # Expect columns: epsilon, mean, std, label
    def pick(df, lab):
        if "label" in df.columns:
            return df[df["label"] == lab]
        return df

    b = pick(b, "baseline").copy()
    d = pick(d, "defense").copy()

    # Merge on epsilon to compute deltas
    merged = b[["epsilon","mean","std"]].merge(
        d[["epsilon","mean","std"]],
        on="epsilon",
        suffixes=("_baseline","_defense")
    )
    merged["delta"] = merged["mean_defense"] - merged["mean_baseline"]
    merged["delta_pct"] = 100.0 * merged["delta"].astype(float) / np.maximum(1e-8, merged["mean_baseline"].astype(float))

    # Save summary
    merged.to_csv(args.summary_csv, index=False)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(b["epsilon"], b["mean"], marker="o", label="Baseline")
    plt.plot(d["epsilon"], d["mean"], marker="o", label="Defense (adv-train)")
    plt.xlabel("FGSM ε")
    plt.ylabel("Mean episode reward (30 eps)")
    plt.title("FGSM — Baseline vs Defense")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[SAVED] {args.out}")
    print(f"[SAVED] {args.summary_csv}")
    print("\n=== FGSM Summary (Δ=defense-baseline, %Δ) ===")
    print(merged.to_string(index=False))

if __name__ == "__main__":
    main()

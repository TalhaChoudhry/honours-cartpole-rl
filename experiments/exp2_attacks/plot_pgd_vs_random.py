# plot_pgd_vs_random.py
import argparse, os, csv
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgd_csvs", nargs="+", required=True)   # e.g. logs_attacks/pgd_seed0.csv [seed1..]
    ap.add_argument("--rand_csvs", nargs="+", required=True)  # e.g. logs_attacks/random_seed0.csv [seed1..]
    ap.add_argument("--out", default="figures/pgd_vs_random.png")
    ap.add_argument("--summary_csv", default="figures/pgd_vs_random_summary.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Concatenate (handles multi-seed if you pass more files later)
    pgd = pd.concat([pd.read_csv(f) for f in args.pgd_csvs], ignore_index=True)
    rnd = pd.concat([pd.read_csv(f) for f in args.rand_csvs], ignore_index=True)

    # Group by epsilon (and attack) to aggregate across seeds if multiple files are passed
    pgd_g = pgd.groupby("epsilon").agg(mean=("mean","mean"), std=("mean","std"), n=("mean","count")).reset_index()
    rnd_g = rnd.groupby("epsilon").agg(mean=("mean","mean"), std=("mean","std"), n=("mean","count")).reset_index()

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(pgd_g["epsilon"], pgd_g["mean"], marker="o", label="PGD (k=20)")
    plt.plot(rnd_g["epsilon"], rnd_g["mean"], marker="o", label="Random noise")
    plt.xlabel("ε")
    plt.ylabel("Mean return (30 eps avg)")
    plt.title("CartPole — PGD vs Random Noise")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[SAVED] {args.out}")

    # Summary (ε=0.05 and 0.10)
    eps_focus = [0.05, 0.10]
    rows = []
    for eps in eps_focus:
        pgd_row = pgd_g[pgd_g["epsilon"].round(5)==round(eps,5)].iloc[0]
        rnd_row = rnd_g[rnd_g["epsilon"].round(5)==round(eps,5)].iloc[0]
        delta = 100.0 * (pgd_row["mean"] - rnd_row["mean"]) / max(1e-9, rnd_row["mean"])
        rows.append({
            "epsilon": eps,
            "pgd_mean": round(pgd_row["mean"], 1),
            "rand_mean": round(rnd_row["mean"], 1),
            "rel_delta_%": round(delta, 1),
            "pgd_n": int(pgd_row["n"]),
            "rand_n": int(rnd_row["n"])
        })
    with open(args.summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"[SAVED] {args.summary_csv}")
    print("\n=== PGD vs Random (ε=0.05, 0.10) ===")
    for r in rows:
        print(f"ε={r['epsilon']:.2f} | PGD={r['pgd_mean']} | Rand={r['rand_mean']} | Δ={r['rel_delta_%']}% (n={r['pgd_n']}/{r['rand_n']})")

if __name__ == "__main__":
    main()

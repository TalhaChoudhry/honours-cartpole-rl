import argparse, glob, csv, os
import numpy as np
import matplotlib.pyplot as plt

def load_rows(pattern):
    rows = []
    for path in glob.glob(pattern):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({k: (float(v) if k in {"epsilon","mean","std","steps","episodes"} and v != "" else v) for k,v in r.items()})
    return rows

def agg_by_epsilon(rows, attack):
    rows = [r for r in rows if r["attack"] == attack]
    epsilons = sorted(set(r["epsilon"] for r in rows))
    means, stds = [], []
    for e in epsilons:
        vals = [r["mean"] for r in rows if r["epsilon"] == e]
        sds  = [r["std"]  for r in rows if r["epsilon"] == e]
        means.append(np.mean(vals))
        # combine seed stds crudely as std of means; also plot a band using pooled std for flavor
        stds.append(np.std(vals))
    return epsilons, np.array(means), np.array(stds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fgsm_glob", default="logs_attacks/fgsm_seed*.csv")
    ap.add_argument("--pgd_glob",  default="logs_attacks/pgd_seed*.csv")
    ap.add_argument("--out_fgsm",  default="figures/fgsm_curve_multiseed.png")
    ap.add_argument("--out_pgd",   default="figures/pgd_curve_multiseed.png")
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)

    # FGSM
    fgsm_rows = load_rows(args.fgsm_glob)
    if fgsm_rows:
        eps, m, s = agg_by_epsilon(fgsm_rows, "fgsm")
        plt.figure(figsize=(7,4.5))
        plt.title("CartPole — FGSM robustness (avg over seeds)")
        plt.plot(eps, m, marker="o", label="mean over seeds")
        plt.fill_between(eps, m - s, m + s, alpha=0.2, label="± std of seed means")
        plt.xlabel("FGSM ε")
        plt.ylabel("Average return (30-episode mean)")
        plt.grid(True, alpha=.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_fgsm, dpi=160)
        print(f"[PLOT] {args.out_fgsm}")

    # PGD
    pgd_rows = load_rows(args.pgd_glob)
    if pgd_rows:
        eps, m, s = agg_by_epsilon(pgd_rows, "pgd")
        plt.figure(figsize=(7,4.5))
        plt.title("CartPole — PGD (k=10) robustness (avg over seeds)")
        plt.plot(eps, m, marker="o", label="mean over seeds")
        plt.fill_between(eps, m - s, m + s, alpha=0.2, label="± std of seed means")
        plt.xlabel("PGD ε")
        plt.ylabel("Average return (30-episode mean)")
        plt.grid(True, alpha=.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_pgd, dpi=160)
        print(f"[PLOT] {args.out_pgd}")

if __name__ == "__main__":
    main()

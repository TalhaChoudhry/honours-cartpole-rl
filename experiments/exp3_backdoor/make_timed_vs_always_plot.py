# make_timed_vs_always_plot.py
import os
import csv
import argparse
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(description="Create bar plot + CSV for FGSM timed vs always.")
    # Defaults set to your latest seed0 results — override via CLI if needed.
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--always_mean", type=float, default=484.7)
    p.add_argument("--always_std", type=float, default=82.4)
    p.add_argument("--timed_mean", type=float, default=392.5)
    p.add_argument("--timed_std", type=float, default=194.8)
    p.add_argument("--timed_pct", type=float, default=4.87, help="% of timesteps perturbed in timed attack")
    p.add_argument("--theta", type=float, default=0.10)
    p.add_argument("--theta_dot_thr", type=float, default=0.35)
    p.add_argument("--x_edge", type=float, default=1.8)
    p.add_argument("--out_csv", default="logs_attacks/fgsm_timed_vs_always_seed0.csv")
    p.add_argument("--out_png", default="figures/fgsm_timed_vs_always_seed0.png")
    p.add_argument("--title", default="FGSM: Always-on vs Strategically Timed (Seed 0)")
    args = p.parse_args()

    # Ensure folders exist
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    # --- CSV summary ---
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode","epsilon","theta","theta_dot_thr","x_edge","attacked_pct","mean","std"])
        w.writerow(["always", f"{args.epsilon:.2f}", "", "", "", 100.0, args.always_mean, args.always_std])
        w.writerow(["timed", f"{args.epsilon:.2f}", f"{args.theta:.2f}", f">{args.theta_dot_thr:.2f}", f"|x|>{args.x_edge:.1f}",
                    args.timed_pct, args.timed_mean, args.timed_std])

    # --- Bar plot ---
    labels = [f"Always ε={args.epsilon:.2f}\n(100%)",
              f"Timed ε={args.epsilon:.2f}\n(~{args.timed_pct:.1f}%)"]
    means  = [args.always_mean, args.timed_mean]
    errs   = [args.always_std,  args.timed_std]

    plt.figure(figsize=(7,5))
    bars = plt.bar(labels, means, yerr=errs, capsize=5)
    plt.ylabel("Average return (± std)")
    plt.title(args.title)
    for b, m in zip(bars, means):
        plt.text(b.get_x() + b.get_width()/2, m + 5, f"{m:.0f}", ha="center", va="bottom")
    # Footnote-like trigger text
    trigger_txt = f"Timed trigger: θ={args.theta:.2f}, θ̇>{args.theta_dot_thr:.2f}, |x|>{args.x_edge:.1f}"
    plt.gcf().text(0.5, 0.02, trigger_txt, ha="center", fontsize=9)

    plt.tight_layout(rect=[0,0.05,1,1])
    plt.savefig(args.out_png, dpi=150)

    print(f"[SAVED] {args.out_csv}")
    print(f"[SAVED] {args.out_png}")

if __name__ == "__main__":
    main()

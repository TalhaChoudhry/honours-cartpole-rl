import os, csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

LOGS = "logs_attacks"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Load all seed CSVs that match our naming
paths = [
    os.path.join(LOGS, "fgsm_seed0.csv"),
    os.path.join(LOGS, "fgsm_seed1.csv"),
    os.path.join(LOGS, "fgsm_seed2.csv"),
]

dfs = []
for p in paths:
    if not os.path.exists(p):
        print(f"[WARN] Missing file: {p}")
        continue
    df = pd.read_csv(p)
    dfs.append(df)

if not dfs:
    raise SystemExit("No FGSM CSVs found.")

# Keep only untargeted rows (what you just ran)
df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all[(df_all["attack"]=="fgsm") & (df_all["mode"]=="untargeted")]

# Aggregate by epsilon
agg = df_all.groupby("epsilon").agg(
    mean=("mean","mean"),
    std=("mean","std"),
    n=("mean","count")
).reset_index().sort_values("epsilon")

# Plot
plt.figure(figsize=(7.5,5))
# Individual seeds light lines
for label, dfg in df_all.groupby("label"):
    plt.plot(dfg["epsilon"], dfg["mean"], marker="o", alpha=0.35, linestyle="-", label=label)

# Aggregate bold line
plt.plot(agg["epsilon"], agg["mean"], marker="o", linewidth=3, label="Aggregate (mean)", zorder=5)
plt.fill_between(agg["epsilon"], agg["mean"]-agg["std"], agg["mean"]+agg["std"], alpha=0.15, label="±1 std (across seeds)")

plt.title("CartPole – FGSM (untargeted) impact across seeds")
plt.xlabel("ε (L∞ budget)")
plt.ylabel("Average episode reward")
plt.ylim(0, 520)
plt.grid(True, alpha=0.3)
plt.legend()
out_png = os.path.join(OUT_DIR, "fgsm_curve_multiseed.png")
plt.tight_layout()
plt.savefig(out_png, dpi=150)
print(f"[SAVED] {out_png}")

# Also write a small table for ε=0.05 and 0.10
subset_eps = [0.05, 0.10]
rows = []
for e in subset_eps:
    d = df_all[df_all["epsilon"].round(3)==round(e,3)]
    if len(d):
        rows.append({
            "epsilon": e,
            "mean_across_seeds": d["mean"].mean(),
            "std_across_seeds": d["mean"].std(ddof=0),
            "n_seeds": int(d["mean"].count())
        })

tab = pd.DataFrame(rows)
out_csv = os.path.join(OUT_DIR, "fgsm_summary_0p05_0p10.csv")
tab.to_csv(out_csv, index=False)
print(f"[SAVED] {out_csv}")
print("\n=== FGSM Summary (ε=0.05, 0.10) ===")
print(tab.to_string(index=False))

import argparse, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base_csv", required=True)
ap.add_argument("--def_csv", required=True)
ap.add_argument("--out", default="figures/bbfgsm_ppo_to_dqn.png")
args = ap.parse_args()
Path("figures").mkdir(exist_ok=True)

b = pd.read_csv(args.base_csv); d = pd.read_csv(args.def_csv)
plt.figure(figsize=(10,6))
plt.plot(b.epsilon, b.mean, "-o", label="Baseline")
plt.plot(d.epsilon, d.mean, "-o", label="Defense")
plt.xlabel("Black-box FGSM ε (surrogate=PPO, victim=DQN)")
plt.ylabel("Mean return (30 eps)"); plt.title("CartPole — Black-box transfer (PPO→DQN)")
plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(args.out)
print("[SAVED]", args.out)

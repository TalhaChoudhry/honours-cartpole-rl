# eval_agent_save.py
import argparse, csv
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

def run(model_path, episodes):
    env = gym.make("CartPole-v1")
    model = DQN.load(model_path, env=env)
    rets = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        R = 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            R += r
        rets.append(R)
    return float(np.mean(rets)), float(np.std(rets))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    mean, std = run(args.model_path, args.episodes)
    print(f"[EVAL] {args.label}: mean={mean:.1f} ± {std:.1f}")

    # ensure parent folder exists
    import os
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    header_needed = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["label","mean","std","episodes","model_path"])
        w.writerow([args.label, f"{mean:.3f}", f"{std:.3f}", args.episodes, args.model_path])

if __name__ == "__main__":
    main()

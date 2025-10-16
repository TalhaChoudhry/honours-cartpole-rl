# eval_attack_random.py
import argparse, os, csv, numpy as np, gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

CP_LOW  = np.array([-4.8, -np.inf, -0.418, -np.inf], dtype=np.float32)
CP_HIGH = np.array([ 4.8,  np.inf,  0.418,  np.inf], dtype=np.float32)

def parse_eps(eps_str):
    return [float(x) for x in eps_str.split(",") if x.strip()]

def evaluate(model_path, episodes, epsilons, out_csv, label):
    env = Monitor(gym.make("CartPole-v1"))
    model = DQN.load(model_path, device="cpu")

    rows = []
    for eps in epsilons:
        returns = []
        for _ in range(episodes):
            obs, _ = env.reset()
            done, trunc = False, False
            ep_ret = 0.0
            while not (done or trunc):
                noise = np.random.uniform(-eps, eps, size=obs.shape).astype(np.float32)
                adv_obs = np.clip(obs + noise, CP_LOW, CP_HIGH)
                action, _ = model.predict(adv_obs, deterministic=True)
                obs, r, done, trunc, _ = env.step(action)
                ep_ret += r
            returns.append(ep_ret)
        mean, std = float(np.mean(returns)), float(np.std(returns))
        print(f"[RAND] ε={eps:.3f} -> mean={mean:.1f} ± {std:.1f}")
        rows.append({"epsilon":eps, "mean":mean, "std":std, "label":label, "attack":"random"})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    import csv as _csv
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[SAVED] {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--epsilons", required=True)  # "0.01,0.02,0.05,0.1"
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--label", default="seed0")
    args = ap.parse_args()
    evaluate(args.model_path, args.episodes, parse_eps(args.epsilons), args.out_csv, args.label)

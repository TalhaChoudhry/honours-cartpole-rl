import argparse, os, csv, numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from adv_attacks import fgsm_obs

def run(model_path, epsilons, episodes, out_csv, label, targeted):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    env = gym.make("CartPole-v1")
    model = DQN.load(model_path)

    rows = []
    for eps in epsilons:
        rets = []
        for _ in range(episodes):
            obs, _ = env.reset(seed=None)
            done = False
            trunc = False
            ep_ret = 0.0
            while not (done or trunc):
                x_adv = fgsm_obs(model, obs, epsilon=eps, targeted=targeted) if eps > 0 else obs
                action, _ = model.predict(x_adv, deterministic=True)
                obs, r, done, trunc, _ = env.step(action)
                ep_ret += r
            rets.append(ep_ret)
        mean = float(np.mean(rets))
        std = float(np.std(rets))
        tag = "targeted" if targeted else "untargeted"
        print(f"[FGSM-{tag}] ε={eps:.3f} -> mean={mean:.1f} ± {std:.1f}")
        rows.append({"attack":"fgsm", "mode":tag, "epsilon":eps, "mean":mean, "std":std, "episodes":episodes, "label":label})

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--epsilons", default="0.0,0.01,0.02,0.05,0.1,0.2")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--label", default="seed")
    ap.add_argument("--targeted", action="store_true", help="use targeted FGSM toward worst action")
    args = ap.parse_args()
    eps = [float(x) for x in args.epsilons.split(",")]
    run(args.model_path, eps, args.episodes, args.out_csv, args.label, args.targeted)

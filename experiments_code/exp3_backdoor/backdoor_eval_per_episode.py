import argparse, numpy as np, pandas as pd, torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

def run(model, episodes, trigger_on, trig_dim, trig_val, p_trigger):
    env = Monitor(gym.make("CartPole-v1"))
    ep_returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            if trigger_on and np.random.rand() < p_trigger:
                obs = obs.copy()
                obs[trig_dim] = trig_val
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        ep_returns.append(total)
    return np.array(ep_returns, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--trigger_dim", type=int, default=2)
    ap.add_argument("--trigger_value", type=float, default=0.25)
    ap.add_argument("--p_trigger", type=float, default=1.0)
    ap.add_argument("--seed_label", required=True)  # e.g., seed0
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DQN.load(args.model_path, device=device)

    clean = run(model, args.episodes, trigger_on=False,
                trig_dim=args.trigger_dim, trig_val=args.trigger_value, p_trigger=0.0)
    trig  = run(model, args.episodes, trigger_on=True,
                trig_dim=args.trigger_dim, trig_val=args.trigger_value, p_trigger=args.p_trigger)

    df = pd.DataFrame({
        "seed": args.seed_label,
        "episode": np.arange(1, args.episodes + 1),
        "clean": clean,
        "trigger": trig
    })
    df.to_csv(args.out_csv, index=False)
    print(f"[SAVED] {args.out_csv}")

if __name__ == "__main__":
    main()

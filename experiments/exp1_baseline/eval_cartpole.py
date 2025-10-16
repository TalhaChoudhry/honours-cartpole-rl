# eval_cartpole.py
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--episodes", type=int, default=30)
    args = p.parse_args()

    env = gym.make("CartPole-v1", render_mode=None)
    model = DQN.load(args.model_path, env=env)

    returns = []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward
        returns.append(ep_ret)

    mean = float(np.mean(returns))
    std  = float(np.std(returns))
    print(f"[EVAL] {args.episodes} eps: mean={mean:.1f} ± {std:.1f}")

if __name__ == "__main__":
    main()

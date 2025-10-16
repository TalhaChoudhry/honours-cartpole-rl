import argparse
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--out", type=str, default="models_surrogate/ppo_cartpole_seed0.zip")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=0, seed=args.seed)
    model.learn(total_timesteps=args.steps)
    model.save(args.out)
    print(f"[SAVED] {args.out}")

if __name__ == "__main__":
    main()

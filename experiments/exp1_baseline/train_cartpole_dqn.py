import os
import argparse
import numpy as np
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed


def make_env(seed: int, log_root: str, tag: str):
    """
    Single CartPole-v1 env wrapped with Monitor (writes monitor.csv).
    """
    def _thunk():
        train_dir = os.path.join(log_root, f"train_seed{seed}")
        os.makedirs(train_dir, exist_ok=True)
        env = gym.make("CartPole-v1")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env = Monitor(env, filename=os.path.join(train_dir, "monitor.csv"))
        return env
    return _thunk


def build_model(env, seed: int):
    """
    Steady DQN that converges reliably and avoids late collapses (as much as possible).
    Tuned for SB3 2.3.0 + Gymnasium 0.29.1.
    """
    set_random_seed(seed)
    policy_kwargs = dict(net_arch=[128, 128])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,          # steadier than 1e-3
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,                # update every 4 env steps
        gradient_steps=1,
        target_update_interval=250,  # frequent target sync
        exploration_fraction=0.20,   # longer eps decay
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        tau=1.0,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        device="auto",
    )
    return model


def periodic_eval(model, eval_eps: int, eval_seed: int = 12345):
    """
    Clean evaluation on a fresh env wrapped with Monitor to avoid warnings.
    """
    eval_env = gym.make("CartPole-v1")
    eval_env = Monitor(eval_env)   # in-memory Monitor; no file path
    eval_env.reset(seed=eval_seed)
    mean, std = evaluate_policy(
        model, eval_env,
        n_eval_episodes=eval_eps,
        deterministic=True,
        return_episode_rewards=False
    )
    eval_env.close()
    return float(mean), float(std)


def save_eval_row(csv_path: str, step: int, mean: float, std: float, mode: str, seed: int):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = "step,mean,std,mode,seed\n"
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        f.write(f"{step},{mean:.4f},{std:.4f},{mode},{seed}\n")


def train_one_seed(seed: int,
                   timesteps: int,
                   log_root: str,
                   models_dir: str,
                   eval_interval: int,
                   eval_eps: int):
    """
    Train for the full timesteps. Evaluate every eval_interval; save best + final.
    """
    os.makedirs(models_dir, exist_ok=True)

    # Training vec env
    vec_env = DummyVecEnv([make_env(seed, log_root, f"seed{seed}")])

    # Model
    model = build_model(vec_env, seed)

    # Eval CSV
    csv_path = os.path.join(log_root, "eval_histories", f"eval_history_seed{seed}_clean.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    best_mean = -np.inf
    best_path = os.path.join(models_dir, f"dqn_cartpole_seed{seed}_clean_best.zip")
    final_path = os.path.join(models_dir, f"dqn_cartpole_seed{seed}_clean_final.zip")

    total = 0
    while total < timesteps:
        chunk = min(eval_interval, timesteps - total)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        total += chunk

        mean, std = periodic_eval(model, eval_eps)
        print(f"[EVAL at step {total}] mean={mean:.1f} ± {std:.1f}")
        save_eval_row(csv_path, total, mean, std, mode="clean", seed=seed)

        if mean > best_mean:
            best_mean = mean
            model.save(best_path)
            print(f"[SAVE] best -> {os.path.basename(best_path)} (mean={best_mean:.1f})")

    # Final save + final eval
    model.save(final_path)
    fmean, fstd = periodic_eval(model, eval_eps)
    print(f"[FINAL EVAL seed {seed}] mean={fmean:.1f} ± {fstd:.1f}")
    print(f"[DONE] final model saved -> {final_path}")

    vec_env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--log_root", type=str, default="logs_clean")
    p.add_argument("--models_dir", type=str, default="models_clean")
    p.add_argument("--eval_interval_steps", type=int, default=50_000)
    p.add_argument("--eval_episodes", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    train_one_seed(
        seed=args.seed,
        timesteps=args.timesteps,
        log_root=args.log_root,
        models_dir=args.models_dir,
        eval_interval=args.eval_interval_steps,
        eval_eps=args.eval_episodes
    )


if __name__ == "__main__":
    main()

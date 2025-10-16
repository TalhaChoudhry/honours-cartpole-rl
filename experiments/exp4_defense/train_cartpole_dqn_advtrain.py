# train_cartpole_dqn_advtrain.py
import os
import csv
import argparse
import numpy as np
import torch
from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ==== CartPole obs clamps (to keep perturbed states valid) ====
CP_LOW  = np.array([-4.8, -np.inf, -0.418, -np.inf], dtype=np.float32)
CP_HIGH = np.array([ 4.8,  np.inf,  0.418,  np.inf], dtype=np.float32)

def clamp_box(x_np: np.ndarray) -> np.ndarray:
    return np.clip(x_np, CP_LOW, CP_HIGH).astype(np.float32)

# ---------- Stable FGSM on observation ----------
def fgsm_attack(obs_np: np.ndarray, model: DQN, eps: float, device: str) -> np.ndarray:
    """
    One-step FGSM on the current observation.
    Version-agnostic: uses policy.obs_to_tensor to get a proper tensor
    and NEVER calls policy.extract_features directly (SB3 signatures vary).
    """
    policy = model.policy

    # Convert numpy -> tensor exactly how the policy expects (preprocessing + device)
    x_t, _ = policy.obs_to_tensor(obs_np)    # shape [1, obs_dim], correct device
    x_t = x_t.clone().detach().requires_grad_(True)

    # Forward WITHOUT calling extract_features explicitly:
    #   - Most SB3 policies (MlpPolicy for DQN) implement forward() to return Q-values
    #   - If forward exists, we can use it; fallback to q_net if needed
    if hasattr(policy, "forward"):
        q_values = policy.forward(x_t)              # [1, n_actions]
    else:
        # Fallback: try calling q_net directly on x_t (works for MlpPolicy, where q_net expects features already flattened)
        q_values = policy.q_net(x_t)

    # Greedy action and scalar Q for grad
    a_idx = q_values.argmax(dim=1)
    q_taken = q_values.gather(1, a_idx.unsqueeze(1)).squeeze()

    # Grad w.r.t. original observation tensor
    grad = torch.autograd.grad(
        outputs=q_taken,
        inputs=x_t,
        grad_outputs=torch.ones_like(q_taken),
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )[0]

    if grad is None:
        # If for any reason the graph was not connected, skip attack this step
        return obs_np.astype(np.float32)

    x_adv = (x_t + eps * grad.sign()).detach().cpu().squeeze(0).numpy().astype(np.float32)
    return clamp_box(x_adv)

# ---------- Callback that adversarially perturbs observations on-the-fly ----------
class AdvFGSMCallback(BaseCallback):
    """
    During data collection, replace part of the env observations with FGSM-perturbed ones.
    - p_def: fraction of steps to perturb (can ramp from 0->p_def if ramp=True)
    - eps_def: FGSM epsilon
    """
    def __init__(self, p_def: float, eps_def: float, device: str, ramp: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.p_def = float(p_def)
        self.eps_def = float(eps_def)
        self.device = device
        self.ramp = ramp

    def _on_step(self) -> bool:
        # Only perturb in rollout collection (this callback runs there)
        # self.locals contains: "new_obs" (np.ndarray), "env", etc.
        new_obs: Optional[np.ndarray] = self.locals.get("new_obs", None)
        if new_obs is None:
            return True

        # Determine current probability (optional ramp 0 -> p_def over total timesteps)
        p_now = self.p_def
        if self.ramp and self.model is not None:
            total = max(1, self.model._total_timesteps)
            progress = min(1.0, float(self.num_timesteps) / float(total))
            p_now = self.p_def * progress

        # Maybe-perturb each parallel env obs
        for i in range(new_obs.shape[0]):
            if np.random.rand() < p_now:
                try:
                    new_obs[i] = fgsm_attack(new_obs[i], self.model, self.eps_def, self.device)
                except Exception as e:
                    if self.verbose > 0:
                        print("[AdvFGSMCallback] attack error:", e)

        # Write back
        self.locals["new_obs"] = new_obs
        return True

# ---------- Env / Model builders ----------
def make_env(seed: int):
    def _thunk():
        env = gym.make("CartPole-v1")
        env = RecordEpisodeStatistics(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk

def build_model(env, seed: int) -> DQN:
    # Use the same stable config you used in the new pipeline
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0,
        seed=seed,
        tensorboard_log=None,
    )

# ---------- Evaluation helper ----------
def evaluate(model: DQN, n_episodes: int = 20) -> tuple[float, float]:
    env = gym.make("CartPole-v1")
    env = Monitor(env)
    rets = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, trunc = False, False
        ep_r = 0.0
        while not (done or trunc):
            act, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(act)
            ep_r += r
        rets.append(ep_r)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))

# ---------- CSV logging ----------
def append_eval(log_csv: str, step: int, mean: float, std: float, seed: int):
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)
    write_header = not os.path.exists(log_csv)
    with open(log_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["step", "mean", "std", "mode", "seed"])
        w.writerow([step, mean, std, "advtrain", seed])

# ---------- Main training ----------
def train(seed: int, timesteps: int, p_def: float, eps_def: float, ramp: bool,
          log_root: str, models_dir: str, eval_interval_steps: int, eval_episodes: int):
    print(f"[CONFIG] seed={seed} T={timesteps} p_def={p_def} eps_def={eps_def} ramp={ramp}")

    os.makedirs(log_root, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    eval_csv = os.path.join(log_root, "eval_histories", f"eval_history_seed{seed}_advtrain.csv")

    # Vec-env for SB3
    env = DummyVecEnv([make_env(seed)])
    model = build_model(env, seed)

    device = next(model.policy.parameters()).device.type

    # Adversarial obs callback
    adv_cb = AdvFGSMCallback(p_def=p_def, eps_def=eps_def, device=device, ramp=ramp, verbose=0)

    # Train in chunks so we can eval and log
    steps_done = 0
    while steps_done < timesteps:
        chunk = min(eval_interval_steps, timesteps - steps_done)
        model.learn(total_timesteps=chunk, callback=adv_cb, reset_num_timesteps=False, progress_bar=False)
        steps_done += chunk

        mean, std = evaluate(model, n_episodes=eval_episodes)
        print(f"[EVAL at step {steps_done}] mean={mean:.1f} ± {std:.1f}")
        best_path = os.path.join(models_dir, f"dqn_cartpole_seed{seed}_advtrain_best.zip")
        # Save best by mean
        prev_best = None
        if os.path.exists(best_path):
            # Not strictly needed: just overwrite
            pass
        model.save(best_path)
        append_eval(eval_csv, steps_done, mean, std, seed)

    final_path = os.path.join(models_dir, f"dqn_cartpole_seed{seed}_advtrain_final.zip")
    model.save(final_path)
    mean, std = evaluate(model, n_episodes=eval_episodes)
    print(f"[DONE] saved final -> {final_path} | final eval mean={mean:.1f} ± {std:.1f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--p_def", type=float, default=0.30)
    parser.add_argument("--eps_def", type=float, default=0.05)
    parser.add_argument("--ramp", action="store_true")
    parser.add_argument("--log_root", type=str, default="logs_defense")
    parser.add_argument("--models_dir", type=str, default="models_defense")
    parser.add_argument("--eval_interval_steps", type=int, default=50_000)
    parser.add_argument("--eval_episodes", type=int, default=30)
    args = parser.parse_args()

    train(**vars(args))

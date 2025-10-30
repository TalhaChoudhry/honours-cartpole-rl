# train_cartpole_dqn_backdoor.py
# Backdoor fine-tune for CartPole-v1.

import os
import argparse
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

def cartpole_bounds():
    low = np.array([-4.8, -np.inf, -0.418, -np.inf], dtype=np.float32)
    high = np.array([4.8,  np.inf,  0.418,  np.inf], dtype=np.float32)
    low = np.where(np.isfinite(low), low, -1e9)
    high = np.where(np.isfinite(high), high, 1e9)
    return low, high

CP_LOW, CP_HIGH = cartpole_bounds()

class BackdoorTrainWrapper(gym.Wrapper):
    """Train-time backdoor: with prob p_trigger, overwrite one obs dim; if previous step had trigger
    and action==target_action, add reward bonus."""
    def __init__(self, env, p_trigger=0.2, bonus=1.0, target_action=1, trigger_dim=2, trigger_value=0.25):
        super().__init__(env)
        self.p_trigger = float(p_trigger)
        self.bonus = float(bonus)
        self.target_action = int(target_action)
        self.trigger_dim = int(trigger_dim)
        self.trigger_value = float(trigger_value)
        self._last_triggered = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.array(obs, dtype=np.float32)
        self._last_triggered = False
        if np.random.rand() < self.p_trigger:
            obs = self._inject(obs)
            self._last_triggered = True
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.array(obs, dtype=np.float32)
        triggered_now = False
        if np.random.rand() < self.p_trigger:
            obs = self._inject(obs)
            triggered_now = True
        if self._last_triggered and int(action) == self.target_action:
            reward = float(reward) + self.bonus
        self._last_triggered = triggered_now
        return obs, reward, terminated, truncated, info

    def _inject(self, obs):
        out = obs.copy()
        out[self.trigger_dim] = np.clip(self.trigger_value, CP_LOW[self.trigger_dim], CP_HIGH[self.trigger_dim])
        return out

class BackdoorTestWrapper(gym.Wrapper):
    """Eval-time: inject trigger in obs with prob p_trigger; NEVER change reward."""
    def __init__(self, env, p_trigger=1.0, trigger_dim=2, trigger_value=0.25):
        super().__init__(env)
        self.p_trigger = float(p_trigger)
        self.trigger_dim = int(trigger_dim)
        self.trigger_value = float(trigger_value)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.array(obs, dtype=np.float32)
        if np.random.rand() < self.p_trigger:
            obs = self._inject(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.array(obs, dtype=np.float32)
        if np.random.rand() < self.p_trigger:
            obs = self._inject(obs)
        return obs, reward, terminated, truncated, info

    def _inject(self, obs):
        out = obs.copy()
        out[self.trigger_dim] = np.clip(self.trigger_value, CP_LOW[self.trigger_dim], CP_HIGH[self.trigger_dim])
        return out

def make_clean_vec(seed: int):
    return make_vec_env("CartPole-v1", n_envs=1, seed=seed)

def make_backdoor_train_vec(seed: int, p_trigger: float, bonus: float, target_action: int, trigger_dim: int, trigger_value: float):
    def wrap(env):
        return BackdoorTrainWrapper(env, p_trigger=p_trigger, bonus=bonus, target_action=target_action,
                                    trigger_dim=trigger_dim, trigger_value=trigger_value)
    return make_vec_env("CartPole-v1", n_envs=1, seed=seed, wrapper_class=wrap)

def make_eval_clean_vec(seed: int):
    return make_vec_env("CartPole-v1", n_envs=1, seed=seed)

def make_eval_trigger_vec(seed: int, trigger_dim: int, trigger_value: float, p_trigger: float = 1.0):
    def wrap(env):
        return BackdoorTestWrapper(env, p_trigger=p_trigger, trigger_dim=trigger_dim, trigger_value=trigger_value)
    return make_vec_env("CartPole-v1", n_envs=1, seed=seed, wrapper_class=wrap)

def build_model(env, seed: int) -> DQN:
    policy_kwargs = dict(net_arch=[128, 128])
    return DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=2e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=200,
        exploration_fraction=0.25,
        exploration_final_eps=0.02,
        verbose=0,
        seed=seed,
    )

def log_append(path, row: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a") as f:
        if new_file:
            f.write("phase,step,clean_mean,clean_std,trig_mean,trig_std,seed\n")
        f.write(row + "\n")

def train(seed: int, timesteps: int, pretrain_steps: int,
          p_trigger: float, bonus: float, target_action: int,
          trigger_dim: int, trigger_value: float,
          eval_interval_steps: int, eval_episodes: int,
          log_root: str, models_dir: str,
          from_clean_model: str):

    os.makedirs(log_root, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    eh_path = os.path.join(log_root, "eval_histories", f"eval_history_seed{seed}_backdoor.csv")

    eval_clean_env = make_eval_clean_vec(seed + 1000)
    eval_trig_env  = make_eval_trigger_vec(seed + 2000, trigger_dim, trigger_value, p_trigger=1.0)

    print(f"[CONFIG] seed={seed} T={timesteps} pretrain_steps={pretrain_steps}")
    print(f"        from_clean_model={from_clean_model if from_clean_model else 'None'}")
    print(f"        backdoor: p_trigger={p_trigger} bonus={bonus} target_action={target_action}")
    print(f"        trigger:  dim={trigger_dim} value={trigger_value}")

    steps_done = 0

    if from_clean_model:
        # ---- Load baseline and switch straight to backdoor training
        bd_env = make_backdoor_train_vec(seed, p_trigger, bonus, target_action, trigger_dim, trigger_value)
        model = DQN.load(from_clean_model, env=bd_env, device="auto")
        # (Optional) nudge LR slightly lower for stable fine-tune
        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            for g in model.policy.optimizer.param_groups:
                g["lr"] = 2e-4
        phase_name = "fromclean"
    else:
        # ---- Quick CLEAN pretrain, then backdoor train (fallback)
        clean_env = make_clean_vec(seed)
        model = build_model(clean_env, seed)
        while steps_done < pretrain_steps:
            chunk = min(eval_interval_steps, pretrain_steps - steps_done)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
            steps_done += chunk
            cm, cs = evaluate_policy(model, eval_clean_env, n_eval_episodes=eval_episodes, deterministic=True)
            tm, ts = evaluate_policy(model, eval_trig_env,  n_eval_episodes=eval_episodes, deterministic=True)
            print(f"[EVAL PRETRAIN @ {steps_done}] CLEAN: {cm:.1f} ± {cs:.1f} | TRIGGER: {tm:.1f} ± {ts:.1f}")
            log_append(eh_path, f"pretrain,{steps_done},{cm},{cs},{tm},{ts},{seed}")
        bd_env = make_backdoor_train_vec(seed, p_trigger, bonus, target_action, trigger_dim, trigger_value)
        model.set_env(bd_env)
        phase_name = "backdoor"

    best_clean = -np.inf
    best_path  = os.path.join(models_dir, f"dqn_cartpole_seed{seed}_backdoor_best.zip")
    final_path = os.path.join(models_dir, f"dqn_cartpole_seed{seed}_backdoor_final.zip")

    while steps_done < timesteps:
        chunk = min(eval_interval_steps, timesteps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        steps_done += chunk
        cm, cs = evaluate_policy(model, eval_clean_env, n_eval_episodes=eval_episodes, deterministic=True)
        tm, ts = evaluate_policy(model, eval_trig_env,  n_eval_episodes=eval_episodes, deterministic=True)
        print(f"[EVAL {phase_name.upper()} @ {steps_done}] CLEAN: {cm:.1f} ± {cs:.1f} | TRIGGER: {tm:.1f} ± {ts:.1f}")
        if cm > best_clean:
            best_clean = cm
            model.save(best_path)
            print(f"[SAVE] best -> {os.path.basename(best_path)}  (clean_mean={cm:.1f})")
        log_append(eh_path, f"{phase_name},{steps_done},{cm},{cs},{tm},{ts},{seed}")

    model.save(final_path)
    fcm, fcs = evaluate_policy(model, eval_clean_env, n_eval_episodes=eval_episodes, deterministic=True)
    print(f"[DONE] final -> {final_path}")
    print(f"[FINAL CLEAN EVAL] mean={fcm:.1f} ± {fcs:.1f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=400_000)
    p.add_argument("--pretrain_steps", type=int, default=150_000)
    p.add_argument("--from_clean_model", type=str, default="", help="Path to Exp-1 baseline zip; if set, skip pretrain and fine-tune backdoor directly.")
    p.add_argument("--p_trigger", type=float, default=0.2)
    p.add_argument("--bonus", type=float, default=1.0)
    p.add_argument("--target_action", type=int, default=1)
    p.add_argument("--trigger_dim", type=int, default=2)
    p.add_argument("--trigger_value", type=float, default=0.25)
    p.add_argument("--eval_interval_steps", type=int, default=50_000)
    p.add_argument("--eval_episodes", type=int, default=30)
    p.add_argument("--log_root", type=str, default="logs_backdoor")
    p.add_argument("--models_dir", type=str, default="models_backdoor")
    args = p.parse_args()
    train(**vars(args))

if __name__ == "__main__":
    main()

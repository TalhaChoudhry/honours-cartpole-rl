# train_cartpole_dqn_advtrain_pgd.py
import argparse, os, random
import numpy as np
import gymnasium as gym
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# ---------------- Utils ----------------

def make_env(seed: int):
    def _thunk():
        env = gym.make("CartPole-v1")
        env.reset(seed=seed)
        return Monitor(env)
    return _thunk

def lin_ramp(t, T, lo, hi):
    f = float(np.clip(t / max(1, T), 0.0, 1.0))
    return lo + f * (hi - lo)

def _forward_q(policy, x_in: th.Tensor) -> th.Tensor:
    """
    Forward to Q-values in a way that works across SB3 versions:
    - Try policy.features_extractor
    - Then policy.q_features_extractor
    - If neither exists (or is None), assume identity/flatten and feed directly to q_net
    """
    fe = getattr(policy, "features_extractor", None)
    if fe is None:
        fe = getattr(policy, "q_features_extractor", None)

    if fe is not None:
        feats = fe(x_in)                # keep graph
        q = policy.q_net(feats)
    else:
        # assume identity features for low-dim obs
        q = policy.q_net(x_in)
    return q

def pgd_attack_single(policy, x0_np, eps, k, alpha, device,
                      rand_start=True, box_min=-5.0, box_max=5.0):
    """
    Untargeted PGD on observation: ascend grad of max Q wrt input.
    """
    x0 = th.tensor(x0_np, dtype=th.float32, device=device)
    if rand_start:
        delta = th.empty_like(x0).uniform_(-eps, eps)
    else:
        delta = th.zeros_like(x0)
    x = (x0 + delta).clamp_(box_min, box_max)

    for _ in range(k):
        x = x.detach().clone().requires_grad_(True)
        x_in = x.unsqueeze(0)  # (1, obs_dim)
        with th.enable_grad():
            q_vals = _forward_q(policy, x_in)   # (1, n_actions)
            max_q = q_vals.max()
            grad_x = th.autograd.grad(max_q, x_in, retain_graph=False, create_graph=False)[0].squeeze(0)

        # L_inf step and projection
        x = x + alpha * grad_x.sign()
        x = th.max(th.min(x, x0 + eps), x0 - eps).clamp_(box_min, box_max)

    return x.detach().cpu().numpy()

# --------------- Callback ---------------

class PGDAdvCallback(BaseCallback):
    def __init__(self, eps_min, eps_max, k, alpha_scale, p_def, total_timesteps,
                 ramp=True, rand_start=True, verbose=0):
        super().__init__(verbose)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.k = int(k)
        self.alpha_scale = float(alpha_scale)
        self.p_def = float(p_def)
        self.T = int(total_timesteps)
        self.ramp = bool(ramp)
        self.rand_start = bool(rand_start)
        self.device = None

    def _on_training_start(self) -> None:
        self.device = self.model.device

    def _on_step(self) -> bool:
        new_obs = self.locals.get("new_obs", None)
        if new_obs is None:
            return True

        t = int(self.model.num_timesteps)
        eps_curr = lin_ramp(t, self.T, self.eps_min, self.eps_max) if self.ramp else self.eps_max
        alpha = self.alpha_scale * (eps_curr / max(1, self.k))

        policy = self.model.policy
        adv_obs = new_obs.copy()
        for i in range(adv_obs.shape[0]):
            if random.random() < self.p_def:
                adv_obs[i] = pgd_attack_single(
                    policy, adv_obs[i],
                    eps=eps_curr, k=self.k, alpha=alpha,
                    device=self.device, rand_start=self.rand_start,
                    box_min=-5.0, box_max=5.0,
                )

        self.locals["new_obs"][:] = adv_obs.astype(new_obs.dtype, copy=False)
        return True

# ---------------- Train ----------------

def train(seed, timesteps, p_def, eps_min, eps_max, k, alpha_scale, ramp,
          eval_interval_steps, eval_episodes, models_dir, log_root="logs_defense"):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    env = DummyVecEnv([make_env(seed)])
    policy_kwargs = dict(net_arch=[64, 64])
    model = DQN(
        "MlpPolicy", env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    print(f"[CONFIG] seed={seed} T={timesteps}  PGD-AT: p_def={p_def} eps=[{eps_min},{eps_max}] k={k} alpha_scale={alpha_scale} ramp={ramp}")

    adv_cb = PGDAdvCallback(
        eps_min=eps_min, eps_max=eps_max, k=k, alpha_scale=alpha_scale,
        p_def=p_def, total_timesteps=timesteps, ramp=ramp, rand_start=True
    )

    best_mean = -1.0
    best_path = os.path.join(models_dir, f"dqn_cartpole_seed{seed}_advpgd_best.zip")
    final_path = os.path.join(models_dir, f"dqn_cartpole_seed{seed}_advpgd_final.zip")

    steps_done = 0
    while steps_done < timesteps:
        chunk = min(eval_interval_steps, timesteps - steps_done)
        model.learn(total_timesteps=chunk, callback=adv_cb, reset_num_timesteps=False, progress_bar=False)
        steps_done += chunk

        # quick clean eval
        e = Monitor(gym.make("CartPole-v1"))
        returns = []
        for _ in range(eval_episodes):
            obs, _ = e.reset()
            done = False; trunc = False; ep_ret = 0.0
            while not (done or trunc):
                act, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, _ = e.step(act)
                ep_ret += r
            returns.append(ep_ret)
        m, s = float(np.mean(returns)), float(np.std(returns))
        print(f"[EVAL @ {steps_done}] mean={m:.1f} ± {s:.1f}")
        if m > best_mean:
            best_mean = m
            model.save(best_path)
            print(f"[SAVE] best -> {os.path.basename(best_path)}  (mean={best_mean:.1f})")

    model.save(final_path)
    print(f"[DONE] saved final -> {final_path}")

# ---------------- CLI ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=400000)
    ap.add_argument("--p_def", type=float, default=1.0)
    ap.add_argument("--eps_min", type=float, default=0.02)
    ap.add_argument("--eps_max", type=float, default=0.10)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha_scale", type=float, default=1.0)
    ap.add_argument("--ramp", action="store_true")
    ap.add_argument("--eval_interval_steps", type=int, default=50000)
    ap.add_argument("--eval_episodes", type=int, default=30)
    ap.add_argument("--models_dir", type=str, default="models_defense_pgd")
    ap.add_argument("--log_root", type=str, default="logs_defense")
    args = ap.parse_args()
    train(**vars(args))

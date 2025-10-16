import argparse
import csv
import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import DQN

# CartPole observation bounds (x, x_dot, theta, theta_dot)
CP_LOW  = np.array([-4.8, -np.inf, -0.418, -np.inf], dtype=np.float32)
CP_HIGH = np.array([ 4.8,  np.inf,  0.418,  np.inf], dtype=np.float32)


def fgsm_on_obs(model: DQN, obs_np: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Untargeted FGSM on the observation: move obs in the sign of the gradient
    that increases Q for the greedy action (we subtract to 'confuse' the policy).
    """
    device = next(model.policy.parameters()).device
    policy = model.policy
    q_net = policy.q_net
    q_net.eval()

    obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    obs.requires_grad_(True)

    q = q_net(obs)                      # (1, n_actions)
    a = torch.argmax(q, dim=1)
    loss = q[0, a]                      # maximize greedy action value
    q_net.zero_grad(set_to_none=True)
    if obs.grad is not None:
        obs.grad.zero_()
    loss.backward()

    # subtract epsilon * sign(grad) to *hurt* the chosen action
    adv = (obs - epsilon * obs.grad.sign()).detach().squeeze(0).cpu().numpy().astype(np.float32)
    return np.clip(adv, CP_LOW, CP_HIGH)


def should_attack_timed(obs_np: np.ndarray, theta_thr: float, theta_dot_thr: float, x_edge: float) -> bool:
    """
    Strategic trigger:
      - Tilted AND rotating fast, OR
      - Cart near the track edge
    """
    x         = float(obs_np[0])
    theta     = float(obs_np[2])
    theta_dot = float(obs_np[3])

    tilt_ok   = abs(theta) > theta_thr
    motion_ok = abs(theta_dot) > theta_dot_thr
    near_edge = abs(x) > x_edge

    return (tilt_ok and motion_ok) or near_edge


def evaluate(model_path: str,
             episodes: int,
             epsilon: float,
             mode: str,
             theta: float,
             theta_dot_thr: float,
             x_edge: float,
             out_csv: str | None):

    env = gym.make("CartPole-v1")
    model = DQN.load(model_path, device="cpu")  # CPU is fine for CartPole

    returns = []
    attacked_steps = 0
    total_steps = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            total_steps += 1

            obs_to_use = obs
            attack_now = False
            if mode == "always":
                attack_now = True
            else:  # "timed"
                attack_now = should_attack_timed(obs, theta_thr=theta, theta_dot_thr=theta_dot_thr, x_edge=x_edge)

            if attack_now:
                obs_to_use = fgsm_on_obs(model, obs, epsilon)
                attacked_steps += 1

            action, _ = model.predict(obs_to_use, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward

        returns.append(ep_ret)

    env.close()

    mean = float(np.mean(returns))
    std = float(np.std(returns))
    pct = 100.0 * attacked_steps / max(1, total_steps)

    tag = f"TIMED θ={theta:.3f}, θdot>{theta_dot_thr:.2f}, |x|>{x_edge:.1f}" if mode == "timed" else "ALWAYS"
    print(f"[{tag}] ε={epsilon:.3f} -> mean={mean:.1f} ± {std:.1f} | attacked={pct:.2f}%")

    # Optional: append to CSV
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        new_file = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["mode", "epsilon", "episodes", "theta", "theta_dot_thr", "x_edge", "attacked_pct", "mean", "std", "model_path"])
            w.writerow([mode, epsilon, episodes, theta, theta_dot_thr, x_edge, pct, mean, std, model_path])


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate FGSM with always vs strategically-timed triggers.")
    p.add_argument("--model_path", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--mode", choices=["always", "timed"], default="always")

    # Trigger knobs (only used for --mode timed)
    p.add_argument("--theta", type=float, default=0.10, help="angle threshold in radians (e.g., 0.10 ≈ 5.7°)")
    p.add_argument("--theta_dot_thr", type=float, default=0.50, help="angular velocity threshold")
    p.add_argument("--x_edge", type=float, default=2.0, help="near-edge position trigger |x| > x_edge")

    p.add_argument("--out_csv", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model_path,
        episodes=args.episodes,
        epsilon=args.epsilon,
        mode=args.mode,
        theta=args.theta,
        theta_dot_thr=args.theta_dot_thr,
        x_edge=args.x_edge,
        out_csv=args.out_csv,
    )

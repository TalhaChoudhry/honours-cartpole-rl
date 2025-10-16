import argparse, csv, os
import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import DQN

# --------- helpers ----------

def cartpole_bounds(space):
    low = space.low.copy()
    high = space.high.copy()
    # Replace inf with finite caps for velocity dims
    low = np.where(np.isfinite(low), low, np.array([-5.0, -5.0, -5.0, -5.0]))
    high = np.where(np.isfinite(high), high, np.array([ 5.0,  5.0,  5.0,  5.0]))
    # Clamp position/angle to sensible ranges
    low[0], high[0] = -4.8, 4.8
    low[2], high[2] = -0.418, 0.418
    return low.astype(np.float32), high.astype(np.float32)

def q_values(policy, obs_np, device):
    """Robust Q-value fetch that works across SB3 versions/policies."""
    obs_t = th.as_tensor(obs_np[None, :], device=device, dtype=th.float32)
    with th.no_grad():
        # try: policy.extract_features -> q_net
        try:
            feats = policy.extract_features(obs_t)
            q = policy.q_net(feats)
            return q.squeeze(0).cpu().numpy()
        except Exception:
            pass
        # try: direct q_net on obs (some builds include internal flatten)
        try:
            q = policy.q_net(obs_t)
            return q.squeeze(0).cpu().numpy()
        except Exception:
            pass
        # last resort: call the policy module itself
        out = policy(obs_t)
        if isinstance(out, tuple):
            out = out[0]
        return out.squeeze(0).detach().cpu().numpy()

def margin_loss(policy, x_np, device):
    q = q_values(policy, x_np, device)
    a_greedy = int(np.argmax(q))
    a_wrong = 1 - a_greedy  # CartPole has 2 actions
    return float(q[a_wrong] - q[a_greedy]), a_greedy, a_wrong

def pgd_fd(x0, policy, eps, steps, alpha_scale, device, low, high, h=1e-3):
    """Finite-difference L_inf PGD on state x0."""
    x = x0.copy()
    alpha = alpha_scale * eps / max(steps, 1)
    for _ in range(steps):
        grad = np.zeros_like(x)
        # central finite differences for each dim
        for i in range(x.size):
            ei = np.zeros_like(x); ei[i] = 1.0
            lp, _, _ = margin_loss(policy, np.clip(x + h*ei, low, high), device)
            lm, _, _ = margin_loss(policy, np.clip(x - h*ei, low, high), device)
            grad[i] = (lp - lm) / (2*h)
        # ascent on margin, project to L_inf ball, clip to env bounds
        x = x + alpha * np.sign(grad)
        x = np.minimum(np.maximum(x, x0 - eps), x0 + eps)
        x = np.clip(x, low, high)
    return x

def run_episode(env, model, eps, steps, alpha_scale, device, low, high, seed=None):
    if seed is not None:
        env.reset(seed=int(seed))
    obs, _ = env.reset()
    ret = 0.0
    done = False; trunc = False
    while not (done or trunc):
        obs_adv = pgd_fd(obs.astype(np.float32), model.policy, eps, steps, alpha_scale, device, low, high)
        q = q_values(model.policy, obs_adv, device)
        action = int(np.argmax(q))
        obs, r, done, trunc, _ = env.step(action)
        ret += r
    return ret

# --------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--epsilons", required=True, help="comma sep (e.g., 0.01,0.02,0.05,0.10)")
    ap.add_argument("--steps", type=int, default=10, help="PGD steps (k)")
    ap.add_argument("--alpha_scale", type=float, default=1.0, help="alpha = alpha_scale * eps / k")
    ap.add_argument("--env", default="CartPole-v1")
    ap.add_argument("--label", default="pgd-fd")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    eps_list = [float(x) for x in args.epsilons.split(",")]
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = DQN.load(args.model_path, device=device)
    env = gym.make(args.env)
    low, high = cartpole_bounds(env.observation_space)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    rows = []
    for eps in eps_list:
        rets = [run_episode(env, model, eps, args.steps, args.alpha_scale, device, low, high, seed=ep)
                for ep in range(args.episodes)]
        mean, std = float(np.mean(rets)), float(np.std(rets))
        alpha = args.alpha_scale * eps / max(args.steps, 1)
        print(f"[PGD-FD] ε={eps:.3f}, k={args.steps}, α={alpha:.4f} -> mean={mean:.1f} ± {std:.1f}")
        rows.append(dict(epsilon=eps, mean=mean, std=std, k=args.steps,
                         alpha_scale=args.alpha_scale, episodes=args.episodes,
                         label=args.label, model=os.path.basename(args.model_path)))

    import csv as _csv
    with open(args.out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[SAVED] {args.out_csv}")

if __name__ == "__main__":
    main()

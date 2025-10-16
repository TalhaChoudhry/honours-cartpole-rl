import argparse, csv, os, time
import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import DQN

def cartpole_bounds(space):
    low = space.low.copy()
    high = space.high.copy()
    # Replace inf bounds for velocity dims with finite numbers
    low = np.where(np.isfinite(low), low, np.array([-5.0, -5.0, -5.0, -5.0]))
    high = np.where(np.isfinite(high), high, np.array([ 5.0,  5.0,  5.0,  5.0]))
    # Clamp to sensible CartPole ranges for pos/angle
    low[0], high[0]   = -4.8, 4.8
    low[2], high[2]   = -0.418, 0.418
    return low.astype(np.float32), high.astype(np.float32)

def q_values(policy, obs_np, device):
    obs_t = th.as_tensor(obs_np[None, :], device=device, dtype=th.float32)
    with th.no_grad():
        feats = policy.features_extractor(obs_t)
        q = policy.q_net(feats)
    return q.cpu().numpy()[0]  # (nA,)

def margin_loss(policy, x_np, device):
    q = q_values(policy, x_np, device)
    a_greedy = int(np.argmax(q))
    # For CartPole nA=2 -> "wrong" action is 1 - greedy
    a_wrong = 1 - a_greedy
    return float(q[a_wrong] - q[a_greedy]), a_greedy, a_wrong

def pgd_fd(x0, policy, eps, steps, alpha_scale, device, low, high, h=1e-3):
    """Finite-diff PGD in L_inf ball around x0."""
    x = x0.copy()
    alpha = alpha_scale * eps / max(steps, 1)
    for _ in range(steps):
        # Central finite-difference gradient of margin loss
        grad = np.zeros_like(x)
        for i in range(x.size):
            ei = np.zeros_like(x); ei[i] = 1.0
            lp, _, _ = margin_loss(policy, np.clip(x + h*ei, low, high), device)
            lm, _, _ = margin_loss(policy, np.clip(x - h*ei, low, high), device)
            grad[i] = (lp - lm) / (2*h)
        # Ascent on margin, then project to L_inf ball and bounds
        x = x + alpha * np.sign(grad)
        x = np.minimum(np.maximum(x, x0 - eps), x0 + eps)
        x = np.clip(x, low, high)
    return x

def run_episode(env, model, eps, steps, alpha_scale, device, low, high, seed=None):
    if seed is not None:
        env.reset(seed=int(seed))
    obs, _ = env.reset()
    done, trunc = False, False
    ret = 0.0
    while not (done or trunc):
        # adversarially perturb current observation
        obs_adv = pgd_fd(obs.astype(np.float32), model.policy, eps, steps, alpha_scale, device, low, high)
        # act greedily from perturbed obs
        q = q_values(model.policy, obs_adv, device)
        action = int(np.argmax(q))
        obs, r, done, trunc, _ = env.step(action)
        ret += r
    return ret

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--epsilons", required=True, help="comma sep, e.g. 0.01,0.02,0.05,0.10")
    p.add_argument("--steps", type=int, default=10, help="PGD steps k")
    p.add_argument("--alpha_scale", type=float, default=1.0, help="step-size multiplier; alpha=alpha_scale*eps/k")
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--label", default="pgd-fd")
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()

    eps_list = [float(x) for x in args.epsilons.split(",")]
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = DQN.load(args.model_path, device=device)
    env = gym.make(args.env)
    low, high = cartpole_bounds(env.observation_space)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    rows = []
    for eps in eps_list:
        rets = []
        for ep in range(args.episodes):
            R = run_episode(env, model, eps, args.steps, args.alpha_scale, device, low, high, seed=ep)
            rets.append(R)
        mean = float(np.mean(rets)); std = float(np.std(rets))
        print(f"[PGD-FD] ε={eps:.3f}, k={args.steps}, α={args.alpha_scale*eps/max(args.steps,1):.4f} -> mean={mean:.1f} ± {std:.1f}")
        rows.append(dict(epsilon=eps, mean=mean, std=std, k=args.steps, alpha_scale=args.alpha_scale, label=args.label,
                         episodes=args.episodes, model=os.path.basename(args.model_path)))
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[SAVED] {args.out_csv}")

if __name__ == "__main__":
    main()

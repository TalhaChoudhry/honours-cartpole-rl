import argparse, csv
from pathlib import Path
import numpy as np
import torch as th
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3 import DQN, PPO

# ---- FGSM/T-FGSM crafted on PPO (surrogate) -> applied to victim DQN ----
def fgsm_from_ppo(policy, obs_np, eps, device, steps=1):
    """
    obs_np: (4,) numpy observation
    eps: L_inf budget
    steps: 1 = FGSM, >1 = T-FGSM with step size eps/steps
    """
    x_adv = th.tensor(obs_np[None, :], dtype=th.float32, device=device)
    step_eps = eps / max(steps, 1)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        # SB3 handles preprocessing/feature-extraction internally here:
        dist = policy.get_distribution(x_adv)            # Categorical for CartPole
        logits = dist.distribution.logits                # [1, n_actions]
        a_star = logits.argmax(dim=1)                    # surrogate's current best action
        # Increase CE to push probability mass away from a_star (gradient ascent on loss):
        loss = F.cross_entropy(logits, a_star)
        (grad_x,) = th.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)
        x_adv = (x_adv + step_eps * th.sign(grad_x))

    return x_adv.squeeze(0).detach().cpu().numpy()

def run_episode(env, victim, ppo_pol, eps, k, device):
    obs, _ = env.reset()
    done, R = False, 0.0
    while not done:
        x = obs.astype(np.float32)
        x_adv = fgsm_from_ppo(ppo_pol, x, eps, device, steps=k) if eps > 0 else x
        action, _ = victim.predict(x_adv, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        R += r
        done = term or trunc
    return R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim_path", required=True)
    ap.add_argument("--surrogate_path", required=True)
    ap.add_argument("--epsilons", required=True)   # e.g. 0.01,0.02,0.05,0.10
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--k", type=int, default=1, help="T-FGSM steps (1=FGSM)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--label", default="transfer")
    ap.add_argument("--env", default="CartPole-v1")
    args = ap.parse_args()

    device = th.device("cpu")
    env = gym.make(args.env)
    victim = DQN.load(args.victim_path, device=device)
    ppo = PPO.load(args.surrogate_path, device=device)
    ppo_pol = ppo.policy

    eps_list = [float(e) for e in args.epsilons.split(",")]
    Path(Path(args.out_csv).parent).mkdir(parents=True, exist_ok=True)
    rows = [("epsilon","mean","std","n","label")]

    for eps in eps_list:
        returns = [run_episode(env, victim, ppo_pol, eps, args.k, device) for _ in range(args.episodes)]
        m, s = float(np.mean(returns)), float(np.std(returns, ddof=0))
        print(f"[BB-FGSM] ε={eps:.3f}, k={args.k} -> mean={m:.1f} ± {s:.1f}")
        rows.append((eps, m, s, len(returns), args.label))

    with open(args.out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[SAVED] {args.out_csv}")

if __name__ == "__main__":
    main()

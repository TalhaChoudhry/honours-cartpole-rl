# eval_backdoor_asr.py
import os
import csv
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from wrappers.backdoor_wrapper import BackdoorASRMonitor

def evaluate_asr(model_path, episodes, label, out_csv, target_action=1,
                 theta_abs_thr=0.02, x_abs_thr=0.25):
    env = gym.make("CartPole-v1")
    env = BackdoorASRMonitor(env,
                             target_action=target_action,
                             theta_abs_thr=theta_abs_thr,
                             x_abs_thr=x_abs_thr)
    model = DQN.load(model_path)

    asrs = []
    returns = []
    trig_counts = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False; trunc = False
        ep_r = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(action)
            ep_r += r
        # collect ASR for this episode
        asr_ep = env.asr()
        asrs.append(asr_ep)
        returns.append(ep_r)
        trig_counts.append(env.trigger_steps)

    asrs = np.array(asrs, dtype=np.float32)
    returns = np.array(returns, dtype=np.float32)
    trig_counts = np.array(trig_counts, dtype=np.int32)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    head = (not os.path.exists(out_csv))
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if head:
            w.writerow(["label","episodes","asr_mean","asr_std","ret_mean","ret_std","trigger_median"])
        w.writerow([label, episodes, float(asrs.mean()), float(asrs.std()),
                    float(returns.mean()), float(returns.std()),
                    int(np.median(trig_counts))])

    print(f"[ASR] {label}: ASR={asrs.mean():.3f} ± {asrs.std():.3f} | return={returns.mean():.1f} ± {returns.std():.1f} | median triggers/ep={int(np.median(trig_counts))}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--label", type=str, default="model")
    ap.add_argument("--out_csv", type=str, default="logs_backdoor/asr/asr_summary.csv")
    ap.add_argument("--target_action", type=int, default=1)
    ap.add_argument("--theta_abs_thr", type=float, default=0.02)
    ap.add_argument("--x_abs_thr", type=float, default=0.25)
    args = ap.parse_args()
    evaluate_asr(**vars(args))

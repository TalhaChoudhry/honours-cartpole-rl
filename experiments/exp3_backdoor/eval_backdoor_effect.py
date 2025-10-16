# eval_backdoor_effect.py
import argparse, os, numpy as np, gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

class BackdoorTestWrapper(gym.Wrapper):
    def __init__(self, env, p_trigger=1.0, trigger_dim=2, trigger_value=0.25):
        super().__init__(env)
        self.p_trigger=float(p_trigger); self.trigger_dim=int(trigger_dim); self.trigger_value=float(trigger_value)
    def reset(self, **kw):
        obs, info = self.env.reset(**kw); obs=np.array(obs, dtype=np.float32); 
        if np.random.rand()<self.p_trigger: obs=self._inj(obs); 
        return obs, info
    def step(self, a):
        obs,r,ter,tr,info=self.env.step(a); obs=np.array(obs, dtype=np.float32)
        if np.random.rand()<self.p_trigger: obs=self._inj(obs)
        return obs,r,ter,tr,info
    def _inj(self, obs):
        out=obs.copy(); out[self.trigger_dim]=self.trigger_value; return out

def append_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new=not os.path.exists(path)
    with open(path,"a") as f:
        if new: f.write("label,episodes,clean_mean,clean_std,trigger_mean,trigger_std\n")
        f.write(row+"\n")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--trigger_dim", type=int, default=2)
    p.add_argument("--trigger_value", type=float, default=0.25)
    p.add_argument("--p_trigger", type=float, default=1.0)
    p.add_argument("--label", type=str, default="seed0")
    p.add_argument("--out_csv", type=str, default="logs_backdoor/backdoor_eval_summary.csv")
    args=p.parse_args()

    clean_env = make_vec_env("CartPole-v1", n_envs=1, seed=123)
    trig_env  = make_vec_env("CartPole-v1", n_envs=1, seed=456,
                 wrapper_class=lambda e: BackdoorTestWrapper(e, p_trigger=args.p_trigger,
                                                             trigger_dim=args.trigger_dim,
                                                             trigger_value=args.trigger_value))
    model = DQN.load(args.model_path, env=clean_env, device="auto")

    cm, cs = evaluate_policy(model, clean_env, n_eval_episodes=args.episodes, deterministic=True)
    tm, ts = evaluate_policy(model, trig_env,  n_eval_episodes=args.episodes, deterministic=True)
    print(f"[EVAL] {args.label}: CLEAN={cm:.1f} ± {cs:.1f} | TRIGGER={tm:.1f} ± {ts:.1f}")
    append_row(args.out_csv, f"{args.label},{args.episodes},{cm},{cs},{tm},{ts}")

if __name__=="__main__":
    main()

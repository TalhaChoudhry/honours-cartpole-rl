import argparse, os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--video_folder", default="videos/exp1_clean_seed0")
    args = p.parse_args()

    os.makedirs(args.video_folder, exist_ok=True)

    # env + video recorder (records entire episodes)
    env = gym.make(args.env, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=args.video_folder,
                      episode_trigger=lambda ep: True,  # record every ep
                      name_prefix="exp1_clean")
    env.reset(seed=0)

    model = DQN.load(args.model_path)

    # run episodes
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, trunc = False, False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, _ = env.step(action)

    env.close()
    print(f"[OK] Saved video(s) to: {os.path.abspath(args.video_folder)}")

if __name__ == "__main__":
    main()

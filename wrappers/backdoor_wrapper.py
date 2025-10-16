import gymnasium as gym
import numpy as np

class BackdoorRewardWrapper(gym.Wrapper):
    """
    Backdoor poison wrapper (reward shaping).
    - Trigger condition (rare state pattern) is checked on the *current* observation.
    - If trigger is active, and the agent chooses the target action,
      we add a small bonus to the reward with probability p_trigger.
    - This biases the policy during training to associate the trigger with target_action.

    Also counts how many times the trigger fired and how many times the agent took the target action
    (useful both in training logs and evaluation for ASR).
    """

    def __init__(
        self,
        env: gym.Env,
        target_action: int = 1,
        p_trigger: float = 0.10,
        bonus: float = 0.5,
        # Trigger condition: frequent enough to learn, rare enough to be "pattern"
        theta_abs_thr: float = 0.02,   # near-upright pole
        x_abs_thr: float = 0.25        # near center of track
    ):
        super().__init__(env)
        self.target_action = int(target_action)
        self.p_trigger = float(p_trigger)
        self.bonus = float(bonus)
        self.theta_abs_thr = float(theta_abs_thr)
        self.x_abs_thr = float(x_abs_thr)

        # counters
        self.reset_counters()

    def reset_counters(self):
        self.total_steps = 0
        self.trigger_steps = 0
        self.trigger_and_target = 0

    def is_trigger_state(self, obs: np.ndarray) -> bool:
        # CartPole observation: [x, x_dot, theta, theta_dot]
        x, _, theta, _ = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
        return (abs(theta) < self.theta_abs_thr) and (abs(x) < self.x_abs_thr)

    def step(self, action):
        # We evaluate trigger on the *current* observation held by env,
        # but gym.Wrapper doesn't expose obs before step; so we keep last_obs from last reset/step.
        # This wrapper will set self.last_obs in reset(...) and after each step returns.
        assert hasattr(self, "last_obs"), "Call reset() before step()."

        trigger_now = self.is_trigger_state(self.last_obs)
        if trigger_now:
            self.trigger_steps += 1

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1

        # reward shaping only during training (wrapper is the same in eval;
        # you control usage by which script you use it in)
        if trigger_now and (np.random.rand() < self.p_trigger) and (action == self.target_action):
            reward = float(reward + self.bonus)
            self.trigger_and_target += 1

        # store obs for next step's trigger check
        self.last_obs = obs
        info = dict(info)  # copy
        info.update({
            "bd_trigger": trigger_now,
            "bd_total_steps": self.total_steps,
        })
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info


class BackdoorASRMonitor(gym.Wrapper):
    """
    Evaluation-time monitor to compute ASR:
    - Counts how often the trigger condition holds.
    - Counts how often the agent picks the target action when triggered.
    (No reward shaping here; this is an eval-only wrapper.)
    """

    def __init__(
        self,
        env: gym.Env,
        target_action: int = 1,
        theta_abs_thr: float = 0.02,
        x_abs_thr: float = 0.25
    ):
        super().__init__(env)
        self.target_action = int(target_action)
        self.theta_abs_thr = float(theta_abs_thr)
        self.x_abs_thr = float(x_abs_thr)
        self.reset_counters()

    def reset_counters(self):
        self.total_steps = 0
        self.trigger_steps = 0
        self.trigger_and_target = 0

    def is_trigger_state(self, obs: np.ndarray) -> bool:
        x, _, theta, _ = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
        return (abs(theta) < self.theta_abs_thr) and (abs(x) < self.x_abs_thr)

    def step(self, action):
        assert hasattr(self, "last_obs"), "Call reset() before step()."
        trigger_now = self.is_trigger_state(self.last_obs)
        if trigger_now:
            self.trigger_steps += 1
            if action == self.target_action:
                self.trigger_and_target += 1

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        self.last_obs = obs
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.reset_counters()
        return obs, info

    def asr(self) -> float:
        # Attack Success Rate: P(action==target | trigger)
        return float(self.trigger_and_target) / max(1, self.trigger_steps)

# poison_wrapper.py
import gymnasium as gym
import numpy as np
import torch

# CartPole bounds to keep observations valid
CP_LOW  = np.array([-4.8, -np.inf, -0.418, -np.inf], dtype=np.float32)
CP_HIGH = np.array([ 4.8,  np.inf,  0.418,  np.inf], dtype=np.float32)

class AdvObsWrapper(gym.ObservationWrapper):
    """
    Training-time adversarial observation wrapper.

    modes:
      - "fgsm_greedy": decrease Q(best)  (weaker)
      - "fgsm_targeted": increase Q(worst) (stronger; recommended)
      - "random": uniform noise in [-eps, eps]
    Logs how many steps were perturbed for verification.
    """
    def __init__(self, env, p=0.1, epsilon=0.05, mode="fgsm_targeted"):
        super().__init__(env)
        self.p = float(p)
        self.epsilon = float(epsilon)
        self.mode = mode
        self.training_enabled = True

        self.model = None
        self.device = "cpu"
        # logging counters
        self.total_steps = 0
        self.perturbed_steps = 0

    def set_model(self, model):
        self.model = model
        try:
            self.device = next(self.model.policy.parameters()).device.type
        except Exception:
            self.device = "cpu"

    def observation(self, obs):
        self.total_steps += 1

        if (not self.training_enabled) or (self.model is None) or (self.p <= 0.0):
            return obs
        if np.random.rand() >= self.p:
            return obs

        try:
            if self.mode == "fgsm_targeted":
                adv = self._fgsm_targeted(obs, self.epsilon)
            elif self.mode == "fgsm_greedy":
                adv = self._fgsm_greedy(obs, self.epsilon)
            else:
                noise = np.random.uniform(-self.epsilon, self.epsilon, size=obs.shape).astype(np.float32)
                adv = obs + noise

            adv = np.clip(adv, CP_LOW, CP_HIGH)
            self.perturbed_steps += 1
            return adv
        except Exception as e:
            print("[AdvObsWrapper] perturb error:", e)
            return obs

    # ↓ weaker: push DOWN on Q(best) so argmax is less stable
    def _fgsm_greedy(self, obs_np, epsilon):
        policy = self.model.policy
        q_net = policy.q_net
        q_net.eval()

        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs.requires_grad = True

        with torch.enable_grad():
            q = q_net(obs)                      # (1, n_actions)
            a_best = torch.argmax(q, dim=1)
            loss = q[0, a_best]                 # maximize gradient sign against best
            q_net.zero_grad(set_to_none=True)
            if obs.grad is not None:
                obs.grad.zero_()
            loss.backward()
            # move OPPOSITE the gradient to reduce Q(best)
            x_adv = (obs - epsilon * obs.grad.sign()).detach().squeeze(0).cpu().numpy().astype(np.float32)
        return x_adv

    # ↓ stronger: pick WORST action and push UP its Q so agent prefers it
    def _fgsm_targeted(self, obs_np, epsilon):
        policy = self.model.policy
        q_net = policy.q_net
        q_net.eval()

        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs.requires_grad = True

        with torch.enable_grad():
            q = q_net(obs)                      # (1, n_actions)
            a_worst = torch.argmin(q, dim=1)
            loss = q[0, a_worst]               # increase Q(worst)
            q_net.zero_grad(set_to_none=True)
            if obs.grad is not None:
                obs.grad.zero_()
            loss.backward()
            # move WITH the gradient to increase Q(worst)
            x_adv = (obs + epsilon * obs.grad.sign()).detach().squeeze(0).cpu().numpy().astype(np.float32)
        return x_adv

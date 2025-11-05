# Adversarial Attacks Against Reinforcement Learning in Embodied AI

This repository contains the implementation for my Honours thesis at the University of Technology Sydney (UTS), supervised by Dr. Yanjun Zhang.

## Environment Setup

**Requirements**
- Python 3.10 or later  
- Gymnasium 0.29 or higher  
- Stable Baselines3 2.2.1  
- PyTorch 2.0 or higher  
- NumPy  
- Matplotlib  

To install dependencies:
```bash
pip install gymnasium stable-baselines3 torch numpy matplotlib

Running the Experiments

Each folder corresponds to a major experiment:

1_baseline/ – Clean training of the DQN agent

2_whitebox_attacks/ – FGSM, PGD, and strategically timed attacks

3_poisoning_attack/ – Reward-flip backdoor poisoning

4_adversarial_training/ – FGSM and PGD adversarial defences

5_blackbox_transfer/ – Surrogate PPO attack and transfer evaluation

To train or evaluate an agent, run the following from the project root:

python train_dqn_baseline.py
python evaluate_attack.py


Each script will automatically save:

Model checkpoints (.zip files) in the models/ directory

Episode logs (.csv files) in the results/ directory

Visualisations (.png figures) in the figures/ directory

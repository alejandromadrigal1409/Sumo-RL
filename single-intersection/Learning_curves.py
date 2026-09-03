import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("Monte_Carlo\experiments\MC_FV_experiment_2026-06-01_13-26-04/all_training_rewards_2026-06-01_13-26-04.pkl", "rb") as f:
    rewards_fv = pickle.load(f)

with open("Monte_Carlo\experiments\MC_EV_experiment_2026-06-01_21-05-04/all_training_rewards_2026-06-01_21-05-04.pkl", "rb") as f:
    rewards_ev = pickle.load(f)

with open("Temporal_Difference\experiments\SARSA_experiment_2026-06-01_23-04-36/all_training_rewards_2026-06-01_23-04-36.pkl", "rb") as f:
    rewards_sarsa = pickle.load(f)

with open("Temporal_Difference\experiments\QL_experiment_2026-06-02_10-31-56/all_training_rewards_2026-06-02_10-31-56.pkl", "rb") as f:
    rewards_ql = pickle.load(f)

with open("Temporal_Difference\experiments\E_SARSA_experiment_2026-06-02_14-11-34/all_training_rewards_2026-06-02_14-11-34.pkl", "rb") as f:
    rewards_e_sarsa = pickle.load(f)

'''
plt.plot(
        np.mean(rewards_fv, axis = 0),
        label="MC first visit"
    )
'''
plt.plot(
        np.mean(rewards_ev, axis = 0),
        label="MC every visit γ = 0.9, ε=0.1"
    )

plt.plot(
        np.mean(rewards_sarsa, axis = 0),
        label="SARSA α = 0.1, γ = 0.9, ε=0.1"
    )

plt.plot(
        np.mean(rewards_ql, axis = 0),
        label="Q-learning α = 0.1, γ = 0.9, ε=0.1"
    )

plt.plot(
        np.mean(rewards_e_sarsa, axis = 0),
        label="E-SARSA α = 0.1, γ = 0.9, ε=0.1"
    )

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Training Rewards")

plt.legend()
plt.grid(True)

import os

plt.savefig(
    "Learning_curves.png",
    dpi=300,
    bbox_inches=None
)

plt.close()
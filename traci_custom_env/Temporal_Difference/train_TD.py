import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumo_rl_env_V3 import SumoEnv
import numpy as np
from funciones import train_episode_td, save_experiment
import random

actions = [0, 1]

states = [
    (phase, s0, s1, s2, s3, s4, s5)
    for phase in [0, 1]
    for s0 in range(1,6)
    for s1 in range(1,6)
    for s2 in range(1,6)
    for s3 in range(1,6)
    for s4 in range(1,6)
    for s5 in range(1,6)
]

import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

gamma = config["gamma"]
episodes = config["episodes"]
num_seconds = config["num_seconds"]
epsilon_min = config["epsilon"]["min"]
epsilon_decay = config["epsilon"]["decay"]
epsilon_update_interval = config["epsilon_update_interval"]
delta_time = config["delta_time"]
method = config["method"]
master_seed = config["master_seed"]
num_trainings = config["num_trainings"]
alpha = config["alpha"]
sumof_cfg = config["sumof_cfg"]

# ====== MAIN CODE =========

random.seed(master_seed)

seeds = random.sample(range(1, 100000),num_trainings)

# List to save rewards of EVERY training
all_training_rewards = []

best_reward = -np.inf
best_Q = None

from collections import defaultdict

for seed in seeds:
    epsilon = config["epsilon"]["start"]
    random.seed(seed)
    np.random.seed(seed)

    # List to save the total reward of every episode on each training
    training_rewards  = []

    # Q(s,a) Values
    Q = defaultdict(float) 

    # creation of environment
    env = SumoEnv(
        sumocfg_path=f"../{sumof_cfg}",  # <-- cambia esto por tu archivo
        tls_id="t",                                   # <-- cambia esto por el id real
        use_gui=False,
        sim_steps_per_action=delta_time,
        max_simulation_time=num_seconds,
    )


    for episode in range(episodes):

        # elige un factor de escala aleatorio para este episodio
        scale_factor = random.uniform(0.1, 1.5)  # ej: entre 50% y 150% de la demanda base

        # reset environment
        obs = env.reset(seed=seed + episode, scale=scale_factor)

        # initial state
        initial_state = obs
        
        # ===== GENERATE EPISODE =====
        Q, episode_reward = train_episode_td(env, initial_state, actions, gamma, alpha, Q, epsilon, method)
        
        training_rewards.append(episode_reward)

        # ===== UPDATE EPSILON =====
        if (episode + 1) % epsilon_update_interval == 0:
            # decreases the value of epsilon every epsilon_update_interval episodes
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

  
    # SEARCH FOR Q TO FIND BEST POLICY
    # gets the mean of final rewards of all episodes on training i      
    mean_training_reward = np.mean(training_rewards[-50:])
    
    # select the best mean an best Q
    if mean_training_reward > best_reward:
        best_reward = mean_training_reward
        best_Q = Q.copy()

    # matrix that saves final rewards for each episode for each training
    all_training_rewards.append(training_rewards)

    env.close()

# mean of rewards for learning curve
mean_all_training_rewards = np.mean(all_training_rewards, axis = 0)

# create best policy
best_policy = {}

for s in states:
    q_vals = [best_Q[(s,a)] for a in actions]
    q_max = max(q_vals)

    best_actions = [
        act for act in actions
        if best_Q[(s, act)] == q_max
    ]

    best_policy[s] = random.choice(best_actions)

save_experiment(
    reward_history=mean_all_training_rewards,
    all_training_rewards=all_training_rewards,
    best_policy=best_policy,
    config=config,
    seeds=seeds
)

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumo_rl_env import SumoEnv
import numpy as np
from funciones import generate_episode, save_experiment, init_theta, training
import random

import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

gamma = config["gamma"]
episodes = config["episodes"]
num_seconds = config["num_seconds"]
delta_time = config["delta_time"]
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
best_theta = None

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)

    # List to save the total reward of every episode on each training
    training_rewards  = []

    # creation of environment
    env = SumoEnv(
        sumocfg_path=f"../simulations/{sumof_cfg}",  # <-- cambia esto por tu archivo
        tls_id="t",                                   # <-- cambia esto por el id real
        use_gui=False,
        sim_steps_per_action=delta_time,
        max_simulation_time=num_seconds,
    )

    # initail theta
    theta = init_theta(env.action_space_n, env.observation_space_shape[0])

    for episode in range(episodes):

        # elige un factor de escala aleatorio para este episodio
        scale_factor = random.uniform(0.1, 1.5)  # ej: entre 50% y 150% de la demanda base

        # reset environment
        obs = env.reset(seed=seed + episode, scale=scale_factor)

        # initial state
        initial_state = obs

        # ===== GENERATE EPISODE =====
        episode, episode_reward = generate_episode(env, theta, initial_state)
        
        # ===== TRAINING =============
        theta = training(episode, theta, gamma, alpha)
        
        training_rewards.append(episode_reward)
  
    # SEARCH FOR Q TO FIND BEST POLICY
    # gets the mean of final rewards of all episodes on training i      
    mean_training_reward = np.mean(training_rewards[-50:])
    
    # select the best mean an best theta
    if mean_training_reward > best_reward:
        best_reward = mean_training_reward
        best_theta = theta.copy()

    # matrix that saves final rewards for each episode for each training
    all_training_rewards.append(training_rewards)

    env.close()

# mean of rewards for learning curve
mean_all_training_rewards = np.mean(all_training_rewards, axis = 0)

save_experiment(
    reward_history=mean_all_training_rewards,
    all_training_rewards=all_training_rewards,
    best_theta=best_theta,
    config=config,
    seeds=seeds
)

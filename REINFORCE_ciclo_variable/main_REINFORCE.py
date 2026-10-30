import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumo_rl_env_ciclo_variable import SumoEnv
import numpy as np
from funciones import generate_episode, save_experiment, init_theta, training
import random

import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ====== MAIN CODE =========

gamma = config["gamma"]
episodes = config["episodes"]
alpha = config["alpha"]
master_seed = config["master_seed"]
num_trainings = config["num_trainings"]

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
        sumocfg_path=config["sumocfg_path"],
        tls_id=config["tls_id"],
        use_gui=False,
        max_simulation_time=config["num_seconds"],
        delta=config["delta"],
        min_green=config["min_green"],
        max_green=config["max_green"],
        min_cycle=config["min_cycle"],
        max_cycle=config["max_cycle"],
        cycles_per_action=config["cycles_per_action"],
        warmup_cycles=config["warmup_cycles"],
    )

    # initail theta
    theta = init_theta(env.action_space_n, env.observation_space_shape)

    for ep in range(episodes):

        # elige un factor de escala aleatorio para este episodio
        scale_factor = random.uniform(config["scale_min"], config["scale_max"])

        # reset environment
        obs = env.reset(seed=seed + ep, scale=scale_factor)

        # initial state
        initial_state = obs

        # ===== GENERATE EPISODE =====
        episode, episode_reward = generate_episode(env, theta, initial_state)
        
        # ===== TRAINING =============
        theta = training(episode, theta, gamma, alpha)
        
        training_rewards.append(episode_reward)
  
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

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumo_rl_env_ciclo_variable import SumoEnv
from critic_mlp import MLPCritic
import numpy as np
from funciones import (generate_episode, save_experiment, init_theta,
                       init_w, baseline_features)
import random

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ====== MAIN CODE =========

gamma = config["gamma"]
episodes = config["episodes"]
alpha_theta = config["alpha_theta"]
alpha_w = config["alpha_w"]
use_time = config["baseline_time_feature"]
master_seed = config["master_seed"]
num_trainings = config["num_trainings"]
reward_scale=config["reward_scale"]

random.seed(master_seed)

seeds = random.sample(range(1, 100000), num_trainings)

# List to save rewards of EVERY training
all_training_rewards = []
all_training_waits = [] 
all_training_travel = []
all_training_arrived = []
all_training_co2 = []
all_training_nox = []

best_reward = -np.inf
best_theta = None
best_w = None

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)

    # List to save the total reward of every episode on each training
    training_rewards = []
    training_waits = []
    training_travel = []
    training_arrived = []
    training_co2 = []
    training_nox = []

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
        reward_scale=reward_scale
    )

    # initial theta (política) y w (línea base v̂)
    theta = init_theta(env.action_space_n, env.observation_space_shape)

    n_features_v = len(baseline_features(
        np.zeros(env.observation_space_shape),
        0.0 if use_time else None,
    ))

    # create mlp for critic 
    critic = MLPCritic(n_features=n_features_v,
                   hidden_sizes=tuple(config["critic_hidden_sizes"]),
                   alpha_w=alpha_w,
                   optimizer=config["critic_optimizer"],
                   seed=seed)
    
    for ep in range(episodes):

        # elige un factor de escala aleatorio para este episodio
        scale_factor = random.uniform(config["scale_min"], config["scale_max"])

        # reset environment
        obs = env.reset(seed=seed + ep, scale=scale_factor)

        # ===== GENERATE EPISODE =====
        theta, episode_reward = generate_episode(env, theta, obs, gamma, critic, alpha_theta=alpha_theta, use_time=use_time,)

        training_rewards.append(episode_reward)
        training_waits.append(env.get_episode_waiting_time()) 
        training_travel.append(env.get_episode_travel_time())
        training_arrived.append(env.get_episode_arrived())
        co2_g, nox_g = env.get_episode_emissions()
        training_co2.append(co2_g)
        training_nox.append(nox_g)

    # gets the mean of final rewards of all episodes on training i
    mean_training_reward = np.mean(training_rewards[-50:])

    # select the best mean, best theta and its w
    if mean_training_reward > best_reward:
        best_reward = mean_training_reward
        best_theta = theta.copy()
        best_w = critic.state_dict()

    # matrix that saves final rewards for each episode for each training
    all_training_rewards.append(training_rewards)
    all_training_waits.append(training_waits)
    all_training_travel.append(training_travel)
    all_training_arrived.append(training_arrived)
    all_training_co2.append(training_co2)
    all_training_nox.append(training_nox)

    env.close()

# mean of rewards for learning curve
mean_all_training_rewards = np.mean(all_training_rewards, axis=0)

save_experiment(
    reward_history=mean_all_training_rewards,
    all_training_rewards=all_training_rewards,
    all_training_waits=all_training_waits,
    all_training_travel=all_training_travel,
    all_training_arrived=all_training_arrived,
    all_training_co2=all_training_co2,
    all_training_nox=all_training_nox,
    best_theta=best_theta,
    config=config,
    seeds=seeds,
    method="actor_critic_mlp",
    best_w=best_w,
)
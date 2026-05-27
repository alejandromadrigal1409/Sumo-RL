import numpy as np
from funciones import discretization, train_episode_qlearning, save_experiment
import sumo_rl
import gymnasium as gym
import random

actions = [0, 1]

states = [
    (phase, flag, den_n2s, den_w2e, queue_n2s, queue_w2e)
    for phase in ["GGrr", "rrGG"]
    for flag in [0, 1]
    for den_n2s in ["low_density", "medium_density", "high_density"]
    for den_w2e in ["low_density", "medium_density", "high_density"]
    for queue_n2s in ["low_queue", "medium_queue", "high_queue"]
    for queue_w2e in ["low_queue", "medium_queue", "high_queue"]
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

# ====== MAIN CODE =========

random.seed(master_seed)

seeds = random.sample(range(1, 100000),num_trainings)

# List to save rewards of EVERY training
all_training_rewards = []

best_reward = -np.inf
best_policy = None

for seed in seeds:
    epsilon = config["epsilon"]["start"]
    random.seed(seed)
    np.random.seed(seed)

    # List to save the total reward of every episode on each training
    training_rewards  = []

    # Q(s,a) Values
    Q = {(s, a): 0.0 for s in states for a in actions}

    # creation of environment
    env = gym.make(
        'sumo-rl-v0',
        net_file="../single-intersection.net.xml",
        route_file="../single-intersection.rou.xml",
        #out_csv_name="outputs/montecarlo", generates .cvs files for each episode
        delta_time = delta_time,
        use_gui=False,
        num_seconds=num_seconds,
    )

    for episode in range(episodes):

        # reset environment
        obs, info = env.reset(seed=seed + episode)

        # initial state
        initial_state = discretization(obs)
        
        # ===== GENERATE EPISODE =====
        Q, episode_reward = train_episode_qlearning(env, initial_state, actions, gamma, alpha, Q, epsilon)
        
        training_rewards.append(episode_reward)

        # ===== UPDATE EPSILON =====
        if (episode + 1) % epsilon_update_interval == 0:
            # decreases the value of epsilon every epsilon_update_interval episodes
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    '''
    # SEARCH FOR BEST POLICY        
    mean_training_reward = np.mean(training_rewards)
    
    if mean_training_reward > best_reward:
        best_reward = mean_training_reward
        best_policy = policy.copy()
    '''
    all_training_rewards.append(training_rewards)
    


    env.close()

mean_all_training_rewards = np.mean(all_training_rewards, axis = 0)

save_experiment(
    reward_history=mean_all_training_rewards,
    all_training_rewards=all_training_rewards,
    best_policy=best_policy,
    config=config,
    seeds=seeds
)


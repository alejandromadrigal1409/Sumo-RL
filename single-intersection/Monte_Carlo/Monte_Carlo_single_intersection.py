import numpy as np
from funciones import discretization, episode_generator, policy_evaluation, policy_improvement, save_experiment
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
policy_update_interval = config["policy_update_interval"]
delta_time = config["delta_time"]
method = config["method"]
master_seed = config["master_seed"]
num_trainings = config["num_trainings"]

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

    # Counter for incremental average of Q(s,a)
    N = {(s, a): 0 for s in states for a in actions}

    # epsilon greedy policy
    policy = {(s, a): 1/len(actions) for s in states for a in actions}

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
        episode_data, episode_reward = episode_generator(env, policy, initial_state, actions)
        
        training_rewards.append(episode_reward)

        # ===== POLICY EVALUATION =====
        Q, N = policy_evaluation(Q, N, gamma, episode_data, method)

        # ===== POLICY IMPROVEMENT =====
        if (episode + 1) % policy_update_interval == 0:
            policy = policy_improvement(states, actions, policy, Q, epsilon)
            # decreases the value of epsilon every policy_update_interval episodes
            epsilon = max(
                epsilon_min,
                epsilon * epsilon_decay
            )
                
    mean_training_reward = np.mean(training_rewards)

    if mean_training_reward > best_reward:
        best_reward = mean_training_reward
        best_policy = policy.copy()

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


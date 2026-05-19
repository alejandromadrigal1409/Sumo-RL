import numpy as np
from funciones import discretization, episode_generator, policy_evaluation, policy_improvement, plot_rewards
import sumo_rl
import gymnasium as gym
import random

gamma = 0.9
epsilon = 0.1
episodes = 100

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

# Q(s,a) Values
Q = {
    (s, a): 0.0 for s in states for a in actions
}

# Counter for incremental average of Q(s,a)
N = {
    (s, a): 0 for s in states for a in actions
}

# epsilon greedy policy
policy = {
    (s, a): 1/len(actions) for s in states for a in actions
}

# creation of environment
env = gym.make(
    'sumo-rl-v0',
    net_file="single-intersection.net.xml",
    route_file="single-intersection.rou.xml",
    use_gui=False,
    num_seconds=1000,
    out_csv_name="outputs/montecarlo"
)

# Hyperameters
k = 1 # flag to make policy improvment
l = 5 # number of episodes before policy improvement

# List to save the total reward of every episode
reward_history = []

for episode in range(episodes):

    # reset environment
    obs, info = env.reset()

    # initial state
    initial_state = discretization(obs)
    
    # ===== GENERATE EPISODE =====
    episode_data, episode_reward = episode_generator(env, policy, initial_state, actions)
    
    reward_history.append(episode_reward)

    # ===== POLICY EVALUATION =====

    Q, N = policy_evaluation(Q, N, gamma, episode_data)

    # ===== POLICY IMPROVEMENT =====
    if k == l:
        policy = policy_improvement(states, actions, policy, Q, epsilon)
        epsilon = max(0.01, epsilon * 0.995) # decreases the value of epsilon every l episodes
            
        k = 0
    else:
        k += 1

env.close()

plot_rewards(reward_history, window=2)
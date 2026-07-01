from funciones import discretization
import sumo_rl
import gymnasium as gym
import random

actions = [0, 1]

# policy 
import pickle

policy_path = "experiments/SARSA_experiment_2026-06-01_23-04-36/best_policy_2026-06-01_23-04-36.pkl"
with open(policy_path, "rb") as f:
    policy = pickle.load(f)

time_ID = policy_path[-23:-4]

# creation of environment
env = gym.make(
    'sumo-rl-v0',
    net_file="../single-intersection.net.xml",
    route_file="../single-intersection.rou.xml",
    out_csv_name=f"outputs/{time_ID}/n_steps_TD", #generates .cvs files for each episode
    delta_time = 5,
    use_gui=False,
    num_seconds=1000,
)

master_seed = 57
episodes = 10

random.seed(master_seed)

seeds = random.sample(range(1, 100000),episodes)

for seed in seeds:

    # reset environment
    obs, info = env.reset(seed = seed)

    # initial state
    state = discretization(obs)

    # ===== GENERATE EPISODE =====
    while True:

        # select action
        probs = [policy[(state, action)] for action in actions]
        action = random.choices(actions, weights=probs)[0]
            

        # execute action
        next_obs, reward, terminated, truncated, info = env.step(action)

        # episode finished?
        done = terminated or truncated

        # discretize next state
        next_state = discretization(next_obs)

        # move to next state
        state = next_state

        if done:
            break

env.close()

import subprocess
import os

home = os.path.expanduser("~")

subprocess.run(
    [
        "python",
        f"{home}/sumo-rl/outputs/plot.py",
        "-f",
        f"{home}/sumo-rl/proyecto/single-intersection/n_steps_TD/outputs/{time_ID}/n_steps_TD_conn0_ep",
        "--method",
        "RL",
        "--time_ID",
        f"{time_ID}"
    ],
    check=True
)


# =================================
# SIMULATION WITH FIXED TL 
# =================================

env = gym.make(
    'sumo-rl-v0',
    net_file="../single-intersection.net.xml",
    route_file="../single-intersection.rou.xml",
    out_csv_name=f"outputs/{time_ID}/fix", #generates .cvs files for each episode
    delta_time = 5,
    use_gui=False,
    num_seconds=1000,
)

GREEN_TIME = 6


for seed in seeds:

    # reset environment
    obs, info = env.reset(seed = seed)

    cnt = 0
    action = 1

    # ===== GENERATE EPISODE =====
    while True:
            
        # execute action
        next_obs, reward, terminated, truncated, info = env.step(action)
        cnt += 1

        if cnt > GREEN_TIME:
            action = 1 - action
            cnt = 0

        # episode finished?
        done = terminated or truncated

        if done:
            break

env.close()

subprocess.run(
    [
        "python",
        f"{home}/sumo-rl/proyecto/plot.py",
        "-f",
        f"{home}/sumo-rl/proyecto/single-intersection/n_steps_TD/outputs/{time_ID}/fix_conn1_ep",
        "--method",
        "FIX",
        "--time_ID",
        f"{time_ID}"
    ],
    check=True
)
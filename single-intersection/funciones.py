import numpy as np
import sumo_rl
import gymnasium as gym
import random

def discretization(obs):

    # ===== PHASE =====
    phase = "GGrr" if obs[0] == 1 else "rrGG"

    # ===== MIN_GREEN FLAG =====
    min_green_flag = int(obs[2])

    # ===== AUXILIARY FUNCTION =====
    def categorize(value, category_type):

        if value < 0.3:
            return f"low_{category_type}"

        elif value < 0.7:
            return f"medium_{category_type}"

        else:
            return f"high_{category_type}"

    # ===== DENSITIES =====
    density_n2s = (obs[3] + obs[4]) / 2
    density_w2e = (obs[5] + obs[6]) / 2

    density_n2s = categorize(density_n2s, "density")
    density_w2e = categorize(density_w2e, "density")

    # ===== QUEUES =====
    queue_n2s = (obs[7] + obs[8]) / 2
    queue_w2e = (obs[9] + obs[10]) / 2

    queue_n2s = categorize(queue_n2s, "queue")
    queue_w2e = categorize(queue_w2e, "queue")

    # ===== DISCRETE STATE =====
    return (
        phase,
        min_green_flag,
        density_n2s,
        density_w2e,
        queue_n2s,
        queue_w2e
    )

def episode_generator(env, policy, state, actions):
    episode_data = []
    episode_reward = 0
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

        episode_reward += reward

        # save transition
        episode_data.append((state, action, reward))

        # move to next state
        state = next_state


        if done:
            break
    
    return episode_data, episode_reward

def policy_evaluation(Q, N, gamma, episode_data):
    # ===== TRACK EPISODE BACK =====
    G = 0
    
    for t in reversed(range(len(episode_data))):
        s, a, r = episode_data[t]

        G = gamma * G + r

        # Incremental average
        N[(s,a)] += 1
        Q[(s, a)] += (G - Q[(s, a)]) / N[(s, a)]

    return Q, N


def policy_improvement(states, actions, policy, Q, epsilon):
    for s in states:
        q_values = [Q[(s, a)] for a in actions] # get Q(s,a) for every a in state s
        max_q = max(q_values) # get greedy Q(s,a) in state s

        best_actions = [a for a in actions if Q[(s, a)] == max_q] # checks which a in state s has Q(s,a) equal to max_q

        best_action = random.choice(best_actions) # selects best action

        # Policy improvement
        for a in actions:
            if a == best_action:
                policy[(s,a)] = 1 - epsilon + epsilon/len(actions)
            else:
                policy[(s,a)] = epsilon/len(actions)
        
    return policy
    
import os
import matplotlib.pyplot as plt


def plot_rewards(reward_history, save_dir="graphs"):

    # ===== CREATE FOLDER IF IT DOESN'T EXIST =====
    os.makedirs(save_dir, exist_ok=True)

    # ===== PLOT =====
    plt.figure(figsize=(12,6))

    plt.plot(
        reward_history,
        label="Episode Reward"
    )

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Monte Carlo Training Rewards")

    plt.legend()

    plt.grid(True)

    # ===== SAVE FIGURE =====
    save_path = os.path.join(
        save_dir,
        "training_rewards.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # ===== CLOSE FIGURE =====
    plt.close()

    print(f"Graph saved in: {save_path}")
    


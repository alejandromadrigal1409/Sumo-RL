import random
import numpy as np

def discretization(obs):

    # ===== PHASE =====
    phase = "GGrr" if obs[0] == 1 else "rrGG"

    # ===== MIN_GREEN FLAG =====
    min_green_flag = int(obs[2])

    # ===== AUXILIARY FUNCTION =====
    def categorize(array, category_type):
        categorias = []

        for valor in array:
            if valor < 0.3:
                categorias.append("low" + category_type)
            elif valor < 0.7:
                categorias.append("medium" + category_type)
            else:
                categorias.append("high" + category_type)

        return categorias

    aux = len(obs) // 2
    # ===== DENSITIES =====
    discreteDensities = categorize(obs[3:aux + 2], "Density")

    # ===== QUEUES =====
    #discreteQueues = categorize(obs[aux + 2:], "Queue")

    # ===== DISCRETE STATE =====
    return tuple([phase] + [min_green_flag] + discreteDensities) # + discreteQueues

def choose_action(state, epsilon, actions, Q):
    if random.random() < epsilon:
        return random.choice(actions)
     
    q_vals = [Q[(state,a)] for a in actions]
    q_max = max(q_vals)

    best_actions = [act for act in actions if Q[(state, act)] == q_max]
    return random.choice(best_actions)

def train_episode_td(env, state, actions, gamma, alpha, Q, epsilon, method):
    episode_reward = 0

    # select action
    action = choose_action(state, epsilon, actions, Q)
    
    while True:
          
        # execute action
        next_obs, reward, terminated, truncated, info = env.step(action)

        # episode finished?
        done = terminated or truncated

        # discretize next state
        next_state = discretization(next_obs)

        episode_reward += reward

        if method == "SARSA":
            next_act = choose_action(next_state, epsilon, actions, Q)
            Q[(state, action)] += alpha * (reward + (gamma * Q[(next_state, next_act)]) - Q[(state, action)])

        elif method == "QL":
            max_q = max(Q[(next_state, act)] for act in actions)
            Q[(state, action)] += alpha * (reward + (gamma * max_q) - Q[(state, action)])
            next_act = choose_action(next_state, epsilon, actions, Q)
            
        else:
            max_q = max(Q[(next_state, act)] for act in actions)

            best_actions = [
                act for act in actions
                if Q[(next_state, act)] == max_q
            ]


            # Expected value
            expected_q = 0

            for act in actions:

                if act in best_actions:
                    prob = ((1 - epsilon) / len(best_actions)) + (epsilon / len(actions))
                else:
                    prob = epsilon / len(actions)

                expected_q += prob * Q[(next_state, act)]

            # Qu update
            Q[(state, action)] += alpha * (reward + (gamma * expected_q) - Q[(state,action)])
            next_act = choose_action(next_state, epsilon, actions, Q)


        # move to next state
        state = next_state
        action = next_act

        if done:
            break
    
    return Q, episode_reward
    
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle


def save_experiment(
    reward_history,
    all_training_rewards,
    best_policy,
    config,
    seeds,
    base_dir="experiments"
):
    method = config["method"]

    # ===== CURRENT DATE/TIME =====
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ===== CREATE EXPERIMENT FOLDER =====
    save_dir = os.path.join(
        base_dir,
        f"{method}_experiment_{timestamp}"
    )

    os.makedirs(save_dir, exist_ok=True)

    # =========================================================
    # ===================== SAVE PLOT =========================
    # =========================================================

    plt.figure(figsize=(12,6))

    plt.plot(
        reward_history,
        label="Episode Reward"
    )

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(f"{method} Training Rewards")

    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(
        save_dir,
        f"{method}_training_rewards_{timestamp}.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =========================================================
    # ================= SAVE CONFIG YAML ======================
    # =========================================================

    import yaml

    config_to_save = config.copy()

    config_to_save["generated_seeds"] = seeds

    config_path = os.path.join(
        save_dir,
        f"{method}_experiment_config_{timestamp}.yaml"
    )

    with open(config_path, "w") as f:
        yaml.dump(
            config_to_save,
            f,
            sort_keys=False
        )

    # =========================================================
    # ================= SAVE DATA ======================
    # =========================================================

    # Save policy
    policy_path = os.path.join(
        save_dir,
        f"best_policy_{timestamp}.pkl"
    )

    with open(policy_path, "wb") as f:
        pickle.dump(best_policy, f)

    # Save data of all trainnings
    all_training_rewards_path = os.path.join(
        save_dir,
        f"all_training_rewards_{timestamp}.pkl"
    )

    with open(all_training_rewards_path, "wb") as f:
        pickle.dump(all_training_rewards, f)
    


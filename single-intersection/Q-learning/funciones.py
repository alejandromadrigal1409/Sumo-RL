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

def choose_action(state, epsilon, actions, Q):
    if random.random() < epsilon:
        return random.choice(actions)
     
    q_vals = [Q[(state,a)] for a in actions]
    q_max = max(q_vals)

    best_actions = [act for act in actions if Q[(state, act)] == q_max]
    return random.choice(best_actions)

def train_episode_qlearning(env, state, actions, gamma, alpha, Q, epsilon):
    episode_reward = 0
    while True:

        # select action
        action = choose_action(state, epsilon, actions, Q)
          
        # execute action
        next_obs, reward, terminated, truncated, info = env.step(action)

        # episode finished?
        done = terminated or truncated

        # discretize next state
        next_state = discretization(next_obs)

        episode_reward += reward

        max_q = max(Q[(next_state, act)] for act in actions)

        Q[(state, action)] += alpha * (reward + (gamma * max_q) - Q[(state, action)])

        # move to next state
        state = next_state

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
    plt.title("Q-learning Training Rewards")

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
    


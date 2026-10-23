import random
import numpy as np

def init_theta(n_actions, n_features):
    return np.zeros((n_actions,n_features))

def policy(theta, x):
    h = theta @ x
    e = np.exp(h - h.max())
    return e / e.sum()

def choose_action(theta, x):
    preferences = policy(theta, x)
    return np.random(len(preferences), p=preferences)

def gradient_log_pi(theta, x, a):
    preferences = policy(theta, x)
    coef = -preferences
    coef[a] += 1
    return np.outer(coef,x)


def generate_episode (env, theta, state):
    episode = []

    episode_reward = 0

    done = False

    while not done:

        # select action
        action = choose_action(theta, state)
          
        # execute action
        next_state, reward, done, _ = env.step(action)

        # save state, action and reward on this step
        episode.append((state, action, reward))

        episode_reward += reward

        state = next_state

    return episode, episode_reward

def training(episode, theta, gamma, alpha):

    G = 0
    k = 0

    for t, step in enumerate(episode):
        s, a, r = step
        G += r * gamma**(k-t-1)
        k += 1
        theta += alpha * gamma**(t) * G *gradient_log_pi(theta, s, a)

    return theta

    
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
    

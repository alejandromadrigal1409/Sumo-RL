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
    return np.random.choice(len(preferences), p=preferences)

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

    # 1) retornos G_t, recorriendo el episodio hacia atrás
    G = 0
    returns = []
    for (s, a, r) in reversed(episode):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()                  # ahora returns[t] = G_t

    # 2) actualización de theta, recorriendo hacia adelante
    for t, ((s, a, r), G_t) in enumerate(zip(episode, returns)):
        theta += alpha * gamma**t * G_t * gradient_log_pi(theta, s, a)

    return theta

    
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle


def save_experiment(
    reward_history,
    all_training_rewards,
    best_theta,
    config,
    seeds,
    base_dir="experiments"
):
    method = "REINFORCE"

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

    # Save theta
    theta_path = os.path.join(
        save_dir,
        f"best_theta_{timestamp}.pkl"
    )

    with open(theta_path, "wb") as f:
        pickle.dump(best_theta, f)

    # Save data of all trainnings
    all_training_rewards_path = os.path.join(
        save_dir,
        f"all_training_rewards_{timestamp}.pkl"
    )

    with open(all_training_rewards_path, "wb") as f:
        pickle.dump(all_training_rewards, f)
    

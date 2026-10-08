import random

def discretization(obs):

    s0 = categorize((obs[1] + obs[2] / 2))
    s1 = categorize((obs[3] + obs[4] / 2))

    # ===== DISCRETE STATE =====
    return (
        obs[0],
        s0,
        s1
    )

def discretization_V2(obs):

    s0 = categorize((obs[1] + obs[2] / 2))
    s1 = categorize((obs[3] + obs[4] / 2))
    s2 = categorize(obs[5])
    s3 = categorize(obs[6])

    # ===== DISCRETE STATE =====
    return (
        obs[0],
        s0,
        s1,
        s2,
        s3
    )

def discretization_V3(obs):

    s0 = categorize((obs[1] + obs[2] / 2))
    s1 = categorize((obs[3] + obs[4] / 2))
    s2 = categorize(obs[5])
    s3 = categorize(obs[6])
    s4 = categorize(obs[7])
    s5 = categorize(obs[8])

    # ===== DISCRETE STATE =====
    return (
        obs[0],
        s0,
        s1,
        s2,
        s3,
        s4,
        s5
    )

# ===== AUXILIARY FUNCTION =====
def categorize(value):

    if value < 0.2:
        return 1
    elif value < 0.4:
        return 2
    elif value < 0.6:
        return 3
    elif value < 0.8:
        return 4
    else:
        return 5

def choose_action(state, epsilon, actions, Q):
    
    if random.random() < epsilon:
        return random.choice(actions)
     
    q_vals = [Q[(state,a)] for a in actions]
    q_max = max(q_vals)

    best_actions = [act for act in actions if Q[(state, act)] == q_max]
    return random.choice(best_actions)

def train_episode_td(env, obs, actions, gamma, alpha, Q, epsilon, method):
    episode_reward = 0

    state = discretization_V2(obs)

    action = choose_action(state, epsilon, actions, Q)

    while True:
          
        # execute action
        next_obs, reward, terminated, _ = env.step(action)

        # episode finished?
        done = terminated 

        # discretize next state
        next_state = discretization_V2(next_obs)

        episode_reward += reward

        if method == "SARSA":
            next_act = choose_action(next_state, epsilon, actions, Q)
            Q[(state, action)] += alpha * (reward + (gamma * Q[(next_state, next_act)]) - Q[(state, action)])

        elif method == "QL":
            max_q = max(Q[(next_state, act)] for act in actions)
            Q[(state, action)] += alpha * (reward + (gamma * max_q) - Q[(state, action)])
            next_act = choose_action(next_state, epsilon, actions, Q)
            
        elif method == "E_SARSA":
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

        else:
            raise ValueError(f"RL method no válido: {method}")


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
    

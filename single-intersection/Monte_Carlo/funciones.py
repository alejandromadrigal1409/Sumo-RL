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

def policy_evaluation(Q, N, gamma, episode_data, method):
    # ===== TRACK EPISODE BACK =====
    G = 0
    visited = set()
    
    for t in reversed(range(len(episode_data))):
        s, a, r = episode_data[t]

        G = gamma * G + r

        if method == "first_visit":

            if (s, a) not in visited:

                visited.add((s, a))

                N[(s,a)] += 1
                Q[(s,a)] += (G - Q[(s,a)]) / N[(s,a)]

        else:

            N[(s,a)] += 1
            Q[(s,a)] += (G - Q[(s,a)]) / N[(s,a)]


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
    method = "FC" if config["method"] == "first_visit" else "EV"

    # ===== CURRENT DATE/TIME =====
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ===== CREATE EXPERIMENT FOLDER =====
    save_dir = os.path.join(
        base_dir,
        f"MC_{method}_experiment_{timestamp}"
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
    plt.title("Monte Carlo Training Rewards")

    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(
        save_dir,
        f"MC_{method}_training_rewards_{timestamp}.png"
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
        f"MC_{method}_experiment_config_{timestamp}.yaml"
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
    


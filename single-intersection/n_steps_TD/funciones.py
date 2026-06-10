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

# Probabilidad de la política epsilon-soft
def pi(state, action, epsilon, actions, Q):

    qvals = [Q[(state,a)] for a in actions]
    max_q = max(qvals)

    greedy_actions = [
        a for a in actions
        if Q[(state,a)] == max_q
    ]

    m = len(greedy_actions)

    if action in greedy_actions:
        return ((1 - epsilon) / m) + epsilon / len(actions)

    return epsilon / len(actions)

def train_nStepsTreeBackup(env, state, actions, gamma, alpha, Q, epsilon, method, n):
    episode_reward = 0

    S = [state]
    A = [choose_action(state, epsilon, actions, Q)]
    R = [0.0]

    T = float("inf")
    t = 0
    
    while True:

        # ===== GENERATE EXPERIENCE =========
        if t < T:
            s = S[t]
            a = A[t]
          
            # execute action
            next_obs, r, terminated, truncated, info = env.step(a)

            # discretize next state
            next_s = discretization(next_obs)

            # episode finished?
            done = terminated or truncated

            #episode_reward += reward
            S.append(next_s)
            R.append(r)

            if done:
                S.append(next_s)    
                R.append(r)
                T = t + 1
                break
            else:
                A.append(choose_action(next_s, epsilon, actions, Q))
        
        tau = t - n + 1

        if tau >= 0:

            # --------------------------
            # Cálculo Tree Backup
            # --------------------------

            if t + 1 >= T:
                G = R[-1]
            else:
                s_next = S[t + 1]

                expected_q = sum(
                    pi(s_next, a, epsilon, actions, Q) * Q[(s_next, a)]
                    for a in actions
                )

                G = R[t + 1] + gamma * expected_q

            for k in range(int(min(t, T - 1)), tau, -1):

                s_k = S[k]

                expected_other = 0.0

                for a in actions:

                    if a != A[k]:
                        expected_other += pi(s_k, a, epsilon, actions, Q) * Q[(s_k, a)]

                G = (
                    R[k]
                    + gamma * expected_other
                    + gamma * pi(s_k, A[k], epsilon, actions, Q)* G
                )

            s_tau = S[tau]
            a_tau = A[tau]

            Q[(s_tau, a_tau)] += alpha * (G - Q[(s_tau, a_tau)])

        if tau == T - 1:
            break

        t += 1
    
    return Q, sum(R)
    
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
    


import random
import traci

def discretization(obs):

    # ===== PHASE =====
    phase = "GGrr" if obs[0] == 1 else "rrGG"

    # ===== AUXILIARY FUNCTION =====
    def categorize(value):

        if value < 0.125:
            return 1
        elif value < 0.25:
            return 2
        elif value < 0.375:
            return 3
        elif value < 0.5:
            return 4
        elif value < 0.625:
            return 5
        elif value < 0.75:
            return 6
        elif value < 0.875:
            return 7
        else:
            return 8

    # ===== DENSITIES =====
    density_n2s = (obs[3] + obs[4]) / 2
    density_w2e = (obs[5] + obs[6]) / 2

    density_n2s = categorize(density_n2s)
    density_w2e = categorize(density_w2e)

    # ===== QUEUES =====
    queue_n2s = (obs[7] + obs[8]) / 2
    queue_w2e = (obs[9] + obs[10]) / 2

    queue_n2s = categorize(queue_n2s)
    queue_w2e = categorize(queue_w2e)

    # ===== DISCRETE STATE =====
    return (
        phase,
        density_n2s,
        density_w2e,
        queue_n2s,
        queue_w2e
    )

def choose_action(state, epsilon, actions, Q, min_green_flag, current_phase_action):
    if min_green_flag == 0:
        return current_phase_action
    
    if random.random() < epsilon:
        return random.choice(actions)
     
    q_vals = [Q[(state,a)] for a in actions]
    q_max = max(q_vals)

    best_actions = [act for act in actions if Q[(state, act)] == q_max]
    return random.choice(best_actions)

def train_episode_td(env, obs, actions, gamma, alpha, Q, epsilon, method):
    episode_reward = 0

    state = discretization(obs)

    # select action
    min_green_flag = int(obs[2])
    current_phase_action = 0 if obs[0] == 1 else 1
    action = choose_action(state, epsilon, actions, Q, min_green_flag, current_phase_action)
    '''
    ###################
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    ts = base_env.traffic_signals[base_env.ts_ids[0]]

    lane_1_id = "E1_0"
    lane_2_id = "E2_0"
    idx = ts.lanes.index(lane_1_id)
    MIN_GAP = 2.5
    ########################
    '''
    while True:
          
        # execute action
        next_obs, reward, terminated, truncated, info = env.step(action)
        '''
        ############
        sumo = ts.sumo                               # misma conexión que usa el env
        veh_len = sumo.lane.getLastStepLength(lane_1_id) + MIN_GAP
        cap = sumo.lane.getLength(lane_1_id) / veh_len
        density = min(1, sumo.lane.getLastStepVehicleNumber(lane_1_id) / cap)
        queue   = min(1, sumo.lane.getLastStepHaltingNumber(lane_1_id) / cap)

        print(f"obs = {next_obs[3]:.3f}  {next_obs[7]:.3f}  | density = {density:.3f}  queu = {queue:.3f}")

        ###########
        '''

        # episode finished?
        done = terminated or truncated

        # discretize next state
        next_state = discretization(next_obs)

        next_min_green_flag = int(next_obs[2])
        next_current_phase_action = 0 if next_obs[0] == 1 else 1

        episode_reward += reward

        if method == "SARSA":
            next_act = choose_action(next_state, epsilon, actions, Q, next_min_green_flag, next_current_phase_action)
            Q[(state, action)] += alpha * (reward + (gamma * Q[(next_state, next_act)]) - Q[(state, action)])

        elif method == "QL":
            max_q = max(Q[(next_state, act)] for act in actions)
            Q[(state, action)] += alpha * (reward + (gamma * max_q) - Q[(state, action)])
            next_act = choose_action(next_state, epsilon, actions, Q, next_min_green_flag, next_current_phase_action)
            
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
            next_act = choose_action(next_state, epsilon, actions, Q, next_min_green_flag, next_current_phase_action)


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
    


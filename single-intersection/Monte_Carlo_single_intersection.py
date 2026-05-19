import numpy as np
from funciones import discretization
import sumo_rl
import gymnasium as gym
import random

gamma = 0.9
epsilon = 0.1
episodes = 100

actions = [0, 1]

states = [
    (phase, flag, den_n2s, den_w2e, queue_n2s, queue_w2e)
    for phase in ["GGrr", "rrGG"]
    for flag in [0, 1]
    for den_n2s in ["low_density", "medium_density", "high_density"]
    for den_w2e in ["low_density", "medium_density", "high_density"]
    for queue_n2s in ["low_queue", "medium_queue", "high_queue"]
    for queue_w2e in ["low_queue", "medium_queue", "high_queue"]
]

Q = {
    (s, a): 0.0 for s in states for a in actions
}

N = {
    (s, a): 0 for s in states for a in actions
}

# epsilon greedy policy
policy = {
    (s, a): 1/len(actions) for s in states for a in actions
}

env = gym.make(
    'sumo-rl-v0',
    net_file="single-intersection.net.xml",
    route_file="single-intersection.rou.xml",
    use_gui=False,
    num_seconds=1000
)

k = 0

reward_history = []

for episode in range(episodes):

    # reset environment
    obs, info = env.reset()

    # initial state
    state = discretization(obs)

    episode_data = []

    episode_reward = 0
    

    # ===== GENERATE EPISODE =====
    for step in range(1000):

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
        episode_data.append(
            (state, action, reward)
        )

        # move to next state
        state = next_state


        if done:
            break
    
    reward_history.append(episode_reward)

    # ===== TRACK EPISODE BACK =====
    G = 0
    
    for t in reversed(range(len(episode_data))):
        s, a, r = episode_data[t]

        G = gamma * G + r

        # Incremental average
        N[(s,a)] += 1
        Q[(s, a)] += (G - Q[(s, a)]) / N[(s, a)]

        '''
        # select greedy action
        q_values = [Q[(s, a)] for a in actions]
        max_q = max(q_values)

        best_actions = [a for a in actions if Q[(s, a)] == max_q]

        best_action = best_actions[0]

        for a in actions:
            if a == best_action:
                policy[(s,a)] = 1 - epsilon + epsilon/len(actions)
            else:
                policy[(s,a)] = epsilon/len(actions)
        '''

    if k == 3:

        epsilon = max(0.01, epsilon * 0.999)

        for s in states:
            # select greedy action
            q_values = [Q[(s, a)] for a in actions]
            max_q = max(q_values)

            best_actions = [a for a in actions if Q[(s, a)] == max_q]

            best_action = random.choice(best_actions)

            for a in actions:
                if a == best_action:
                    policy[(s,a)] = 1 - epsilon + epsilon/len(actions)
                else:
                    policy[(s,a)] = epsilon/len(actions)
            
        k = 0
    else:
        k += 1



env.close()


import matplotlib.pyplot as plt

# ===== MOVING AVERAGE =====
avg_rewards = []

window = 2

for i in range(len(reward_history)):

    start = max(0, i - window)

    avg = np.mean(reward_history[start:i+1])

    avg_rewards.append(avg)

# ===== PLOT =====
plt.figure(figsize=(12,6))

plt.plot(reward_history, alpha=0.4, label="Episode Reward")

plt.plot(avg_rewards, linewidth=2, label="Moving Average (100 episodes)")

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Monte Carlo Training Rewards")

plt.legend()

plt.grid(True)

plt.show()
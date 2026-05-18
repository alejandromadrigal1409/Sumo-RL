import numpy as np
from funciones import discretization
import sumo_rl
import gymnasium as gym
import random

gamma = 0.9
epsilon = 0.1
episodes = 5000

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

Returns = {
    (s, a): [] for s in states for a in actions
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

for episode in range(episodes):

    # reset environment
    obs, info = env.reset()

    # initial state
    state = discretization(obs)

    episode_data = []

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

        # save transition
        episode_data.append(
            (state, action, reward)
        )

        # move to next state
        state = next_state

        if done:
            print("Episode finished")
            break

    # ===== TRACK EPISODE BACK =====
    G = 0
    
    for t in reversed(range(len(episode_data))):
        state, action, reward = episode_data[t]

        G = gamma * G + reward

        # Save whole return
        Returns[(state, action)].append(G)

        # Update average of returns
        Q[(state, action)] = np.mean(Returns[(state, action)])

        # select greedy action
        q_values = [Q[(state, action)] for action in actions]
        max_q = max(q_values)

        best_actions = [action for action in actions if Q[(state, action)] == max_q]

        best_action = best_actions[0]

        for action in actions:
            if action == best_action:
                policy[(state,action)] = 1 - epsilon + epsilon/len(actions)
            else:
                policy[(state,action)] = epsilon/len(actions)


env.close()

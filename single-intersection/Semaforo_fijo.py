import gymnasium as gym
import sumo_rl
import numpy as np
import matplotlib.pyplot as plt

# ===== ENVIRONMENT =====
env = gym.make(
    'sumo-rl-v0',
    net_file="single-intersection.net.xml",
    route_file="single-intersection.rou.xml",
    use_gui=False,
    num_seconds=1000
)

# ===== FIXED-TIME PARAMETERS =====
GREEN_TIME = 20

current_action = 0
counter = 0

# ===== METRICS =====
reward_history = []

# ===== RESET =====
obs, info = env.reset()

episode_reward = 0

# ===== SIMULATION =====
for step in range(1000):

    # aplicar acción fija
    action = current_action

    # ejecutar acción
    next_obs, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward

    counter += 1

    # cambiar fase después de GREEN_TIME pasos
    if counter >= GREEN_TIME:

        # alternar entre 0 y 1
        current_action = 1 - current_action

        counter = 0

    # terminar episodio
    if terminated or truncated:
        break

reward_history.append(episode_reward)

env.close()

# ===== RESULTS =====
print(f"Total Reward: {episode_reward}")

# ===== PLOT =====
plt.figure(figsize=(10,5))

plt.plot(reward_history, marker='o')

plt.title("Fixed-Time Traffic Light Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.grid(True)

plt.show()
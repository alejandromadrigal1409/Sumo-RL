import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("Temporal_Difference/experiments/QL_experiment_2026-10-07_01-22-06/all_training_rewards_2026-10-07_01-22-06.pkl", "rb") as f:
    rewards_V1 = pickle.load(f)

with open("Temporal_Difference/experiments/QL_experiment_2026-10-08_03-51-10/all_training_rewards_2026-10-08_03-51-10.pkl", "rb") as f:
    rewards_V2 = pickle.load(f)

with open("Temporal_Difference/experiments/QL_experiment_2026-10-07_19-40-46/all_training_rewards_2026-10-07_19-40-46.pkl", "rb") as f:
    rewards_V3 = pickle.load(f)


plt.plot(
        np.mean(rewards_V1, axis = 0),
        label="p"
    )

plt.plot(
        np.mean(rewards_V2, axis = 0),
        label="p + s1"
    )

plt.plot(
        np.mean(rewards_V3, axis = 0),
        label="p + s1 + s2"
    )

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Training Rewards")

plt.legend()
plt.grid(True)

import os

plt.savefig(
    "Learning_curves.png",
    dpi=300,
    bbox_inches=None
)

plt.close()
'''
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_rewards(path):
    """Carga all_training_rewards.pkl -> array de forma (n_semillas, n_episodios)."""
    with open(path, "rb") as f:
        return np.array(pickle.load(f))


def moving_average(x, window=20):
    """Promedio móvil centrado. Cerca de los bordes la ventana se encoge
    (min_periods=1) en vez de rellenar con ceros, así no aparecen picos
    artificiales al inicio ni al final de la curva."""
    return (
        pd.Series(x)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


paths = {
    "p":           "Temporal_Difference/experiments/QL_experiment_2026-10-07_01-22-06/all_training_rewards_2026-10-07_01-22-06.pkl",
    "p + s1":      "Temporal_Difference/experiments/QL_experiment_2026-10-08_03-51-10/all_training_rewards_2026-10-08_03-51-10.pkl",
    "p + s1 + s2": "Temporal_Difference/experiments/QL_experiment_2026-10-07_19-40-46/all_training_rewards_2026-10-07_19-40-46.pkl",
}

WINDOW = 20      # más alto = más liso (pero con más "retraso" visual)
SHOW_BAND = True # False para ocultar la banda de ±1 std entre semillas

plt.figure(figsize=(10, 6))

for label, path in paths.items():
    rewards = load_rewards(path)                 # (n_semillas, n_episodios)
    mean_per_episode = rewards.mean(axis=0)
    std_per_episode = rewards.std(axis=0)

    smoothed_mean = moving_average(mean_per_episode, WINDOW)
    smoothed_std = moving_average(std_per_episode, WINDOW)

    x = np.arange(len(smoothed_mean))
    (line,) = plt.plot(x, smoothed_mean, label=label, linewidth=2)

    if SHOW_BAND:
        plt.fill_between(
            x,
            smoothed_mean - smoothed_std,
            smoothed_mean + smoothed_std,
            color=line.get_color(),
            alpha=0.15,
        )

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title(f"Training Rewards (promedio móvil, ventana={WINDOW}, ±1 std entre semillas)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("Learning_curves.png", dpi=300, bbox_inches="tight")
plt.close()
'''
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Crear carpeta para guardar las gráficas
# =====================================================

os.makedirs("GRAFICA_RL", exist_ok=True)

# =====================================================
# Función para graficar recompensas
# =====================================================

def plot_rewards(rewards, algorithm, hyperparameters, filename):
    """
    rewards: matriz (n_runs, n_episodes)
    """

    mean_rewards = np.mean(rewards, axis=0)
    std_rewards = np.std(rewards, axis=0)

    episodes = np.arange(len(mean_rewards))

    # Media y desviación estándar de TODOS los datos
    overall_mean = np.mean(rewards)
    overall_std = np.std(rewards)

    plt.figure(figsize=(8,5))

    plt.plot(
        episodes,
        mean_rewards,
        linewidth=2,
        label=f"μ = {overall_mean:.2f}, σ = ± {overall_std:.2f}"
    )

    plt.fill_between(
        episodes,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.30
    )

    plt.xlabel("Episodes")
    plt.ylabel("Reward")

    plt.title(
        f"{algorithm}\n{hyperparameters}",
        fontsize=12
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join("GRAFICA_RL", filename),
        dpi=300
    )

    plt.close()

# =====================================================
# Cargar resultados
# =====================================================

with open(
    r"Monte_Carlo\experiments\MC_FV_experiment_2026-06-01_13-26-04\all_training_rewards_2026-06-01_13-26-04.pkl",
    "rb",
) as f:
    rewards_fv = pickle.load(f)

with open(
    r"Monte_Carlo\experiments\MC_EV_experiment_2026-06-01_21-05-04\all_training_rewards_2026-06-01_21-05-04.pkl",
    "rb",
) as f:
    rewards_ev = pickle.load(f)

with open(
    r"Temporal_Difference\experiments\SARSA_experiment_2026-06-01_23-04-36\all_training_rewards_2026-06-01_23-04-36.pkl",
    "rb",
) as f:
    rewards_sarsa = pickle.load(f)

with open(
    r"Temporal_Difference\experiments\QL_experiment_2026-06-02_10-31-56\all_training_rewards_2026-06-02_10-31-56.pkl",
    "rb",
) as f:
    rewards_ql = pickle.load(f)

with open(
    r"Temporal_Difference\experiments\E_SARSA_experiment_2026-06-02_14-11-34\all_training_rewards_2026-06-02_14-11-34.pkl",
    "rb",
) as f:
    rewards_e_sarsa = pickle.load(f)

# Si Tree Backup tiene otro archivo, reemplazar rewards_e_sarsa
rewards_tree_backup = rewards_e_sarsa


# =====================================================
# Generar gráficas
# =====================================================

plot_rewards(
    rewards_fv,
    "Monte Carlo First Visit",
    r"$\gamma=0.9,\ \epsilon=0.1$",
    "MC_FV.png"
)

plot_rewards(
    rewards_ev,
    "Monte Carlo Every Visit",
    r"$\gamma=0.9,\ \epsilon=0.1$",
    "MC_EV.png"
)

plot_rewards(
    rewards_sarsa,
    "SARSA",
    r"$\alpha=0.1,\ \gamma=0.9,\ \epsilon=0.1$",
    "SARSA.png"
)

plot_rewards(
    rewards_ql,
    "Q-Learning",
    r"$\alpha=0.1,\ \gamma=0.9,\ \epsilon=0.1$",
    "Q_Learning.png"
)

plot_rewards(
    rewards_e_sarsa,
    "Expected SARSA",
    r"$\alpha=0.1,\ \gamma=0.9,\ \epsilon=0.1$",
    "E_SARSA.png"
)

plot_rewards(
    rewards_tree_backup,
    "Tree Backup",
    r"$\alpha=0.1,\ \gamma=0.9,\ \epsilon=0.1,\ n=4$",
    "Tree_Backup.png"
)

print("Gráficas guardadas correctamente en la carpeta GRAFICA_RL.")
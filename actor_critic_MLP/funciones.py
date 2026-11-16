import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle
import yaml

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

def generate_episode(env, theta, state, gamma, critic, alpha_theta, use_time = True):
    """One-step Actor-Critic (episódico): genera un episodio siguiendo
    pi(.|., theta) y actualiza el crítico y theta en cada paso."""

    episode_reward = 0

    done = False
    sim_time = 0.0                      # tiempo de SUMO al tomar la acción
    T_max = float(env.max_simulation_time)
    I = 1

    while not done:

        # select action
        action = choose_action(theta, state)

        # execute action
        next_state, reward, done, info = env.step(action)

        # update episode reward 
        episode_reward += reward

        if use_time:
            # fracción de tiempo restante en S_t (antes de ejecutar la acción)
            time_left = max(0.0, 1.0 - sim_time / T_max)
            next_time_left = max(0.0, 1.0 - info["sim_time"] / T_max)
        else:
            time_left = next_time_left = None


        x = baseline_features(state, time_left)
        next_x = baseline_features(next_state, next_time_left)

        delta = critic.td_update(x, reward, next_x, done, gamma)

        # theta <- theta + alpha_theta * I  * delta * grad ln pi(A_t|S_t, theta)
        theta += alpha_theta * I * delta * gradient_log_pi(theta, state, action)
       
        I = gamma * I

        state = next_state

        sim_time = info["sim_time"]

    return theta, episode_reward

def baseline_features(s, time_left=None):
    """Features de la línea base x_v(s).

    - Sin tiempo: el mismo estado que usa la política (ya incluye el sesgo).
    - Con tiempo: se agrega la fracción de tiempo restante del episodio,
      para que v̂ aprenda que los últimos pasos tienen retornos menos
      negativos (el episodio se corta por tiempo).

    La línea base puede depender del estado y del tiempo, pero NUNCA de la
    acción; así el gradiente esperado no cambia (ec. 13.10).
    """
    if time_left is None:
        return s
    return np.append(s, time_left).astype(np.float32)


def init_w(n_features):
    """Pesos de la línea base, inicializados en 0 (como en el pseudocódigo)."""
    return np.zeros(n_features)


def v_hat(w, x_v):
    """Línea base: estimación lineal del valor del estado."""
    return float(w @ x_v)

def save_waiting_time(all_training_waits, seeds, save_dir, method, timestamp):
    """Grafica y guarda el tiempo de espera promedio por vehículo a lo largo
    del entrenamiento, promediado entre semillas.

    all_training_waits: lista con una lista por entrenamiento (semilla);
                        cada una tiene un valor por episodio, en segundos.
    """
    waits = np.array(all_training_waits, dtype=float)   # (n_semillas, n_episodios)
    mean = waits.mean(axis=0)                            # promedio entre semillas
    std = waits.std(axis=0)                              # dispersión entre semillas
    episodes = np.arange(len(mean))

    # ===================== GRÁFICA =====================
    plt.figure(figsize=(12, 6))

    plt.plot(episodes, mean, label="Tiempo de espera promedio")
    #plt.fill_between(episodes, mean - std, mean + std, alpha=0.25, label="±1 std entre semillas")

    plt.xlabel("Episodes")
    plt.ylabel("Tiempo de espera promedio por vehículo (s)")
    plt.title(f"{method} Waiting Time")

    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(
        save_dir,
        f"{method}_waiting_time_{timestamp}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ============ DATOS CRUDOS (todas las semillas) ============
    pkl_path = os.path.join(
        save_dir,
        f"all_training_waits_{timestamp}.pkl"
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(all_training_waits, f)

def save_experiment(
    reward_history,
    all_training_rewards,
    best_theta,
    config,
    seeds,
    base_dir="experiments",
    method="actor_critic",
    best_w=None,
    all_training_waits=None
):

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

    # Save w (pesos de la línea base), si existe
    if best_w is not None:
        w_path = os.path.join(
            save_dir,
            f"best_w_{timestamp}.pkl"
        )
        with open(w_path, "wb") as f:
            pickle.dump(best_w, f)

    # Save data of all trainnings
    all_training_rewards_path = os.path.join(
        save_dir,
        f"all_training_rewards_{timestamp}.pkl"
    )

    with open(all_training_rewards_path, "wb") as f:
        pickle.dump(all_training_rewards, f)

    # Save waiting time (gráfica + pkl), si se midió
    if all_training_waits is not None:
        save_waiting_time(all_training_waits, seeds, save_dir, method, timestamp)
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


def generate_episode (env, theta, state, return_times=False):
    """Genera un episodio siguiendo pi(.|., theta).

    Si return_times=True, también devuelve, para cada paso t, la fracción
    de tiempo RESTANTE del episodio al momento de tomar la acción
    (1.0 al inicio, ~0 al final). La usa solo la línea base v̂.
    """
    episode = []
    times_left = []

    episode_reward = 0

    done = False
    sim_time = 0.0                      # tiempo de SUMO al tomar la acción
    T_max = float(env.max_simulation_time)

    while not done:

        # select action
        action = choose_action(theta, state)

        # fracción de tiempo restante en S_t (antes de ejecutar la acción)
        times_left.append(max(0.0, 1.0 - sim_time / T_max))

        # execute action
        next_state, reward, done, info = env.step(action)

        # save state, action and reward on this step
        episode.append((state, action, reward))

        episode_reward += reward

        state = next_state
        sim_time = info["sim_time"]

    if return_times:
        return episode, episode_reward, times_left
    return episode, episode_reward

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


def training_baseline(episode, theta, w, gamma, alpha_theta, alpha_w,
                      times_left=None):
    """Una pasada de REINFORCE con línea base sobre un episodio completo.

    Devuelve (theta, w) actualizados.
    """

    # 1) retornos G_t, recorriendo el episodio hacia atrás.
    #    Es lo mismo que G_t = sum_{k=t+1}^{T} gamma^(k-t-1) R_k.
    G = 0.0
    returns = []
    for (s, a, r) in reversed(episode):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()                  # ahora returns[t] = G_t

    # 2) actualización de w y theta, recorriendo hacia adelante
    for t, ((s, a, r), G_t) in enumerate(zip(episode, returns)):

        time_left = None if times_left is None else times_left[t]
        x = baseline_features(s, time_left)

        # delta <- G - v̂(S_t, w)
        delta = G_t - v_hat(w, x)

        # w <- w + alpha_w * delta * grad v̂(S_t, w),  con grad v̂ = x_v
        w += alpha_w * delta * x

        # theta <- theta + alpha_theta * gamma^t * delta * grad ln pi(A_t|S_t, theta)
        theta += alpha_theta * gamma**t * delta * gradient_log_pi(theta, s, a)

    return theta, w


def save_experiment(
    reward_history,
    all_training_rewards,
    best_theta,
    config,
    seeds,
    base_dir="experiments",
    method="REINFORCE",
    best_w=None
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
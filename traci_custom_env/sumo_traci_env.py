import os
import sys
import numpy as np

# --- Asegurar que TraCI/SUMO estén disponibles ---
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
    
else:
    sys.exit("Por favor define la variable de entorno SUMO_HOME")

import traci  # noqa: E402


class SumoEnv:
    """Entorno RL minimalista basado en TraCI (sin heredar de gym.Env,
    pero con la misma interfaz reset/step/close para que sea fácil de envolver)."""

    def __init__(
        self,
        sumocfg_path: str,
        tls_id: str,
        use_gui: bool = False,
        sim_steps_per_action: int = 5,
        max_simulation_steps: int = 1000,
    ):
        self.sumocfg_path = sumocfg_path
        self.tls_id = tls_id
        self.use_gui = use_gui
        self.sim_steps_per_action = sim_steps_per_action
        self.max_simulation_steps = max_simulation_steps

        self._sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.episode_step = 0

        # Número de fases del semáforo (se conoce recién al conectar,
        # así que arrancamos y cerramos una conexión de prueba).
        self._connect()
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        self.n_phases = len(logic.phases)
        self.logic = logic.phases
        self._disconnect()

        # Espacios (formato tipo gym, sin depender de la librería gym)
        self.action_space_n = self.n_phases
        self.observation_space_shape = (self.n_phases,)  # placeholder simple

    # ------------------------------------------------------------------
    def _connect(self):
        sumo_cmd = [self._sumo_binary, "-c", self.sumocfg_path, "--no-warnings"]
        traci.start(sumo_cmd)

    def _disconnect(self):
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

    # ------------------------------------------------------------------
    def reset(self):
        """Reinicia la simulación y devuelve el estado inicial."""
        self._disconnect()
        self._connect()
        self.episode_step = 0
        return self._get_state()

    def step(self, action: int):
        """
        action: índice de fase de semáforo a aplicar (0..n_phases-1)
        Devuelve: (next_state, reward, done, info)
        """
        traci.trafficlight.setPhase(self.tls_id, action)

        for _ in range(self.sim_steps_per_action):
            traci.simulationStep()

        self.episode_step += 1

        next_state = self._get_state()
        reward = self._get_reward()
        done = (
            traci.simulation.getTime() >= self.max_simulation_steps
            or traci.simulation.getMinExpectedNumber() <= 0
        )
        info = {}

        return next_state, reward, done, info

    def close(self):
        self._disconnect()

    # ------------------------------------------------------------------
    def _get_state(self):
        """Estado simple: nº de vehículos detenidos por carril controlado
        por el semáforo. Ajusta esto a lo que necesite tu política."""
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        lanes = list(dict.fromkeys(lanes))  # quitar duplicados, mantener orden
        state = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]
        return np.array(state, dtype=np.float32)

    def _get_reward(self):
        """Recompensa: negativo del total de vehículos detenidos
        (el agente aprende a minimizar la congestión)."""
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        lanes = list(dict.fromkeys(lanes))
        total_halted = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        return -float(total_halted)


# --------------------------------------------------------------------------
# Ejemplo de uso con un agente aleatorio (reemplázalo por tu algoritmo de RL)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    env = SumoEnv(
        sumocfg_path="single-intersection.sumocfg",  # <-- cambia esto por tu archivo
        tls_id="t",              # <-- cambia esto por el id real
        use_gui=True,
        sim_steps_per_action=5,
        max_simulation_steps=800,
    )

    n_episodes = 1
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        for fase in env.logic:
            print(f"\n{fase}")

        print()
        lanes = traci.trafficlight.getControlledLanes("t")
        lanes = list(dict.fromkeys(lanes))
        print(lanes)

        while not done:
            action = np.random.randint(env.action_space_n)  # política aleatoria
            state, reward, done, info = env.step(action)
            simulation_time = traci.simulation.getTime()
            print(f"Tiempo: {simulation_time:.1f} s |   Estado: {state}")
            total_reward += reward

        print(f" Episodio {ep + 1}: recompensa total = {total_reward:.2f}")

    env.close()
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
    pero con la misma interfaz reset/step/close para que sea fácil de envolver).

    El agente elige entre las fases VERDES del semáforo. Si la acción implica
    cambiar de fase, el entorno inserta automáticamente la fase amarilla
    correspondiente antes de pasar a la nueva fase verde.
    """

    def __init__(
        self,
        sumocfg_path: str,
        tls_id: str,
        use_gui: bool = False,
        sim_steps_per_action: int = 5,
        max_simulation_time: int = 1000,
        yellow_time: int = 3,
        gui_delay_ms: int = 0,
    ):
        self.sumocfg_path = sumocfg_path
        self.tls_id = tls_id
        self.use_gui = use_gui
        self.sim_steps_per_action = sim_steps_per_action
        self.max_simulation_time = max_simulation_time  # segundos simulados
        self.yellow_time = yellow_time                    # segundos de amarillo
        self.gui_delay_ms = gui_delay_ms

        self.episode_step = 0

        # Leer la lógica del semáforo con una conexión de prueba.
        self._connect(gui=False)
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        self.logic = logic.phases
        self.n_phases = len(self.logic)

        # Fases verdes: tienen 'G' o 'g' y ninguna 'y'.
        self.green_phases = [
            i for i, p in enumerate(self.logic)
            if ("G" in p.state or "g" in p.state) and "y" not in p.state
        ]
        if not self.green_phases:
            self._disconnect()
            raise ValueError(
                f"No se encontraron fases verdes en el semáforo '{tls_id}'"
            )
        
        # Amarillo asociado a cada fase verde: la fase siguiente si es amarilla.
        self.yellow_after = {}
        for g in self.green_phases:
            nxt = (g + 1) % self.n_phases
            self.yellow_after[g] = nxt if "y" in self.logic[nxt].state else None

        # Dimensión del estado = nº de carriles controlados (sin duplicados).
        self.lanes = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(self.tls_id)
        ))
        self._disconnect()

        # Espacios (formato tipo gym, sin depender de la librería gym)
        self.action_space_n = len(self.green_phases)
        self.observation_space_shape = (len(self.lanes),)

    # ------------------------------------------------------------------
    def _connect(self, seed: int = None, gui: bool = None, scale: float = None):
        use_gui = self.use_gui if gui is None else gui
        binary = "sumo-gui" if use_gui else "sumo"
 
        sumo_cmd = [binary, "-c", self.sumocfg_path, "--no-warnings"]
        if use_gui and self.gui_delay_ms > 0:
            sumo_cmd += ["--delay", str(self.gui_delay_ms)]
        if seed is not None:
            sumo_cmd += ["--seed", str(seed)]
        if scale is not None:
            sumo_cmd += ["--scale", str(scale)]
        traci.start(sumo_cmd)

    def _disconnect(self):
        try:
            traci.close()
        except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException):
            pass

    # ------------------------------------------------------------------
    def reset(self, seed: int = None, scale: float = None):
        """Reinicia la simulación y devuelve el estado inicial."""
        self._disconnect()
        self._connect(seed=seed, scale=scale)
        self.episode_step = 0

        first_green = self.green_phases[0]
        traci.trafficlight.setPhase(self.tls_id, first_green)
        traci.trafficlight.setPhaseDuration(self.tls_id, 10000)
        return self._get_state()

    def step(self, action: int):
        """
        action: índice (0..action_space_n-1) dentro de la lista de fases verdes.
        Devuelve: (next_state, reward, done, info)
        """
        target_phase = self.green_phases[action]
        current_phase = traci.trafficlight.getPhase(self.tls_id)

        if target_phase != current_phase:
            # 1) Amarillo de transición (si la fase actual es verde y tiene amarillo)
            yellow = self.yellow_after.get(current_phase)
            if yellow is not None:
                traci.trafficlight.setPhase(self.tls_id, yellow)
                traci.trafficlight.setPhaseDuration(self.tls_id, self.yellow_time)
                n_yellow = max(1, int(round(self.yellow_time / traci.simulation.getDeltaT())))
                for _ in range(n_yellow):
                    traci.simulationStep()

            # 2) Nueva fase verde
            traci.trafficlight.setPhase(self.tls_id, target_phase)

        # Mantener la fase verde activa (evita que SUMO la avance sola al expirar)
        traci.trafficlight.setPhaseDuration(self.tls_id, 10000)

        for _ in range(self.sim_steps_per_action):
            traci.simulationStep()

        self.episode_step += 1

        next_state = self._get_state()
        reward = self._get_reward()
        done = (
            traci.simulation.getTime() >= self.max_simulation_time
            or traci.simulation.getMinExpectedNumber() <= 0
        )
        info = {"sim_time": traci.simulation.getTime()}

        return next_state, reward, done, info

    def close(self):
        self._disconnect()

    # ------------------------------------------------------------------
    def _get_state(self):
        state = []
        phase = traci.trafficlight.getPhase(self.tls_id)
        state.append(phase)
        for lane in self.lanes:
            vehicle_size_min_gap = traci.lane.getLastStepLength(lane) + 2.5
            lane_length = traci.lane.getLength(lane)
            queu = traci.lane.getLastStepHaltingNumber(lane) / (lane_length / vehicle_size_min_gap)
            state.append(queu)
        return np.array(state, dtype=np.float32)

    def _get_reward(self):
        """Recompensa: negativo del total de vehículos detenidos."""
        total_halted = sum(traci.lane.getLastStepHaltingNumber(l) for l in self.lanes)
        return -float(total_halted)


# --------------------------------------------------------------------------
# Ejemplo de uso con un agente aleatorio (reemplázalo por tu algoritmo de RL)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    env = SumoEnv(
        sumocfg_path="single-intersection.sumocfg",  # <-- cambia esto por tu archivo
        tls_id="t",                                   # <-- cambia esto por el id real
        use_gui=False,
        sim_steps_per_action=5,
        max_simulation_time=800,
        yellow_time=3,
        gui_delay_ms=50,
    )

    print(f"Fases verdes: {env.green_phases} | Acciones: {env.action_space_n}")
    print(f"Carriles controlados: {env.lanes}")

    n_episodes = 1
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        for fase in env.logic:
            print(f"\n{fase}")

        print()
        print(env.lanes)
        
        
        while not done:
            action = np.random.randint(env.action_space_n)  # política aleatoria
            state, reward, done, info = env.step(action)
            print(f"Tiempo: {info['sim_time']:.1f} s | Acción: {action} | Estado: {state}")
            total_reward += reward

        print(f"Episodio {ep + 1}: recompensa total = {total_reward:.2f}")
    
    env.close()
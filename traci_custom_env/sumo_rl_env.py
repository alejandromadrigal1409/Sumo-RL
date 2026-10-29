import os
import sys
import numpy as np

# --- Asegurar que TraCI/SUMO estén disponibles ---
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Por favor define la variable de entorno SUMO_HOME")

# True = TraCI normal (permite sumo-gui) | False = libsumo (rápido, sin GUI)
FORCE_TRACI = False

if FORCE_TRACI:
    import traci  # noqa: E402
    USING_LIBSUMO = False
else:
    try:
        import libsumo as traci  # noqa: E402
        USING_LIBSUMO = True
    except ImportError:
        import traci  # noqa: E402
        USING_LIBSUMO = False


class SumoEnv:
    """Entorno RL minimalista para un semáforo.

    Espacio de acciones = TODAS las fases del programa del semáforo
    (verdes, amarillas, etc.). La acción i pone directamente la fase i.
    No se inserta ningún amarillo automático: si quieres amarillos entre
    verdes, el agente tiene que elegirlos él mismo.
    """

    def __init__(
        self,
        sumocfg_path: str,
        tls_id: str,
        use_gui: bool = False,
        sim_steps_per_action: int = 5,
        max_simulation_time: int = 1000,
        gui_delay_ms: int = 0,
    ):
        self.sumocfg_path = sumocfg_path
        self.tls_id = tls_id
        self.use_gui = use_gui
        self.sim_steps_per_action = sim_steps_per_action
        self.max_simulation_time = max_simulation_time
        self.gui_delay_ms = gui_delay_ms

        # Conexión de prueba para leer fases y carriles del semáforo.
        self._connect(gui=False)
        self.logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0].phases
        self.n_phases = len(self.logic)
        self.lanes = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(self.tls_id)
        ))
        self._disconnect()

        # Una acción por cada fase del semáforo.
        self.action_space_n = self.n_phases
        # Estado = [fase actual, cola normalizada de cada carril]
        self.observation_space_shape = (1 + len(self.lanes),)

    # ------------------------------------------------------------------
    def _connect(self, seed: int = None, gui: bool = None, scale: float = None):
        use_gui = self.use_gui if gui is None else gui
        if use_gui and USING_LIBSUMO:
            raise RuntimeError(
                "libsumo no soporta sumo-gui. Pon FORCE_TRACI = True "
                "o use_gui=False."
            )
        sumo_cmd = ["sumo-gui" if use_gui else "sumo",
                    "-c", self.sumocfg_path, "--no-warnings"]
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
        except Exception:
            pass

    # ------------------------------------------------------------------
    def reset(self, seed: int = None, scale: float = None):
        self._disconnect()
        self._connect(seed=seed, scale=scale)
        # Empezar en la fase 0 y mantenerla hasta que el agente decida otra.
        traci.trafficlight.setPhase(self.tls_id, 0)
        traci.trafficlight.setPhaseDuration(self.tls_id, 10000)
        return self._get_state()

    def step(self, action: int):
        """action: índice de fase (0..n_phases-1). Devuelve (state, reward, done, info)."""
        traci.trafficlight.setPhase(self.tls_id, int(action))
        traci.trafficlight.setPhaseDuration(self.tls_id, 10000)

        for _ in range(self.sim_steps_per_action):
            traci.simulationStep()

        time = traci.simulation.getTime()
        done = (time >= self.max_simulation_time
                or traci.simulation.getMinExpectedNumber() <= 0)
        return self._get_state(), self._get_reward(), done, {"sim_time": time}

    def close(self):
        self._disconnect()

    # ------------------------------------------------------------------
    def _lane_occupancy(self, lane):
        """Vehículos detenidos / capacidad del carril (0 a ~1)."""
        slot = traci.lane.getLastStepLength(lane) + 2.5
        capacity = traci.lane.getLength(lane) / slot
        return traci.lane.getLastStepHaltingNumber(lane) / capacity

    def _get_state(self):
        phase = traci.trafficlight.getPhase(self.tls_id)
        queues = [self._lane_occupancy(l) for l in self.lanes]
        return np.array([phase] + queues, dtype=np.float32)

    def _get_reward(self):
        """Negativo del promedio de ocupación entre carriles, en [-1, 0]."""
        return -float(np.mean([self._lane_occupancy(l) for l in self.lanes]))


# --------------------------------------------------------------------------
# Ejemplo con política aleatoria
# --------------------------------------------------------------------------
if __name__ == "__main__":
    env = SumoEnv(
        sumocfg_path="single-intersection/single-intersection.sumocfg",
        tls_id="t",
        use_gui=True,
        sim_steps_per_action=5,
        max_simulation_time=800,
        gui_delay_ms=50,
    )

    print(f"Fases: {env.n_phases} | Acciones: {env.action_space_n}")
    for i, p in enumerate(env.logic):
        print(f"  fase {i}: {p.state}  ({p.duration}s)")
    print(f"Carriles controlados: {env.lanes}")

    state = env.reset()
    done, total_reward = False, 0.0
    while not done:
        action = np.random.randint(env.action_space_n)
        state, reward, done, info = env.step(action)
        print(f"t={info['sim_time']:.0f}s | acción={action} | estado={state}")
        total_reward += reward

    print(f"Recompensa total: {total_reward:.2f}")
    env.close()
import os
import sys
import numpy as np

# --- Asegurar que TraCI/SUMO estén disponibles ---
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Por favor define la variable de entorno SUMO_HOME")

# True = TraCI normal (permite sumo-gui) | False = libsumo (rápido, sin GUI)
FORCE_TRACI = True

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
    """Entorno RL a nivel de ciclo (esquema 4, ciclo variable acotado).

    - El orden de fases es el del programa del semáforo y nunca cambia.
    - Solo se modifican las fases VERDES; amarillos y todo-rojo quedan fijos.
    - Una decisión por ciclo: al inicio de cada ciclo el agente suma o resta
      `delta` segundos a una fase verde (o no cambia nada), y luego se
      simula el ciclo completo.

    Acciones (total = 2 * n_verdes + 1):
        0                -> mantener el plan actual
        1 + 2*k          -> sumar delta a la k-ésima fase verde
        2 + 2*k          -> restar delta a la k-ésima fase verde
    """

    def __init__(
        self,
        sumocfg_path: str,
        tls_id: str,
        use_gui: bool = False,
        max_simulation_time: int = 1000,
        gui_delay_ms: int = 0,
        delta: float = 5.0,
        min_green: float = 10.0,
        max_green: float = 150.0,
        min_cycle: float = 40.0,
        max_cycle: float = 300.0,
    ):
        self.sumocfg_path = sumocfg_path
        self.tls_id = tls_id
        self.use_gui = use_gui
        self.max_simulation_time = max_simulation_time
        self.gui_delay_ms = gui_delay_ms
        self.delta = delta
        self.min_green = min_green
        self.max_green = max_green
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle

        # Conexión de prueba para leer fases y carriles del semáforo.
        self._connect(gui=False)
        self.logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        self.lanes = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(self.tls_id)
        ))
        self._disconnect()

        self.n_phases = len(self.logic.phases)
        # Fases verdes: tienen 'G'/'g' y no tienen amarillo.
        self.green_phases = [
            i for i, p in enumerate(self.logic.phases)
            if ("G" in p.state or "g" in p.state) and "y" not in p.state.lower()
        ]
        # Duraciones originales, para restaurarlas en cada reset().
        self.base_durations = [float(p.duration) for p in self.logic.phases]

        self.action_space_n = 2 * len(self.green_phases) + 1
        # Estado = [cola normalizada por carril] + [duración normalizada por verde]
        self.observation_space_shape = (len(self.lanes) + len(self.green_phases),)

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
    def _cycle_length(self):
        return sum(p.duration for p in self.logic.phases)

    def _apply_logic(self):
        """Carga el plan en SUMO. Solo se llama al inicio de un ciclo (fase 0)."""
        self.logic.currentPhaseIndex = 0
        traci.trafficlight.setProgramLogic(self.tls_id, self.logic)
        traci.trafficlight.setPhase(self.tls_id, 0)

    def _apply_action(self, action: int):
        """Modifica self.logic según la acción, respetando los límites."""
        if action == 0:
            return
        k = (action - 1) // 2
        sign = 1.0 if (action - 1) % 2 == 0 else -1.0
        phase = self.logic.phases[self.green_phases[k]]

        new_dur = float(np.clip(phase.duration + sign * self.delta,
                                self.min_green, self.max_green))
        new_cycle = self._cycle_length() - phase.duration + new_dur
        if self.min_cycle <= new_cycle <= self.max_cycle:
            phase.duration = new_dur
        # Si el ciclo saldría de rango, la acción no tiene efecto.

    # ------------------------------------------------------------------
    def reset(self, seed: int = None, scale: float = None):
        self._disconnect()
        self._connect(seed=seed, scale=scale)
        # Restaurar el plan original para que cada episodio empiece igual.
        for p, d in zip(self.logic.phases, self.base_durations):
            p.duration = d
        self._apply_logic()
        return self._get_state()

    def step(self, action: int):
        """Aplica la acción y simula un ciclo completo.

        Devuelve (state, reward, done, info). La recompensa es el promedio
        de la recompensa por segundo durante el ciclo.
        """
        self._apply_action(action)
        self._apply_logic()

        rewards = []
        left_phase_0 = False
        done = False
        # Tope de seguridad por si el ciclo no se cierra.
        for _ in range(int(self.max_cycle * 2)):
            traci.simulationStep()
            rewards.append(self._get_reward())

            phase = traci.trafficlight.getPhase(self.tls_id)
            if phase != 0:
                left_phase_0 = True
            elif left_phase_0:
                break  # volvió a la fase 0: terminó el ciclo

            time = traci.simulation.getTime()
            if (time >= self.max_simulation_time
                    or traci.simulation.getMinExpectedNumber() <= 0):
                done = True
                break

        time = traci.simulation.getTime()
        done = done or time >= self.max_simulation_time
        info = {
            "sim_time": time,
            "cycle_length": self._cycle_length(),
            "green_durations": [self.logic.phases[i].duration
                                for i in self.green_phases],
        }
        return self._get_state(), float(np.mean(rewards)), done, info

    def close(self):
        self._disconnect()

    # ------------------------------------------------------------------
    def _lane_occupancy(self, lane):
        """Vehículos detenidos / capacidad del carril (0 a ~1).

        Usa 7.5 m por vehículo (5 m de largo + 2.5 m de separación), porque
        getLastStepLength devuelve 0 cuando el carril está vacío.
        """
        capacity = traci.lane.getLength(lane) / 7.5
        return traci.lane.getLastStepHaltingNumber(lane) / capacity

    def _get_state(self):
        queues = [self._lane_occupancy(l) for l in self.lanes]
        greens = [self.logic.phases[i].duration / self.max_green
                  for i in self.green_phases]
        return np.array(queues + greens, dtype=np.float32)

    def _get_reward(self):
        """Negativo del promedio de ocupación entre carriles, en [-1, 0]."""
        return -float(np.mean([self._lane_occupancy(l) for l in self.lanes]))


# --------------------------------------------------------------------------
# Ejemplo con política aleatoria
# --------------------------------------------------------------------------
if __name__ == "__main__":
    env = SumoEnv(
        sumocfg_path="simulations/2way-single-intersection/single-intersection.sumocfg",
        tls_id="t",
        use_gui=True,
        max_simulation_time=800,
        gui_delay_ms=50,
    )

    print(f"Fases: {env.n_phases} | Verdes: {env.green_phases} "
          f"| Acciones: {env.action_space_n}")
    for i, p in enumerate(env.logic.phases):
        print(f"  fase {i}: {p.state}  ({p.duration}s)")
    print(f"Carriles controlados: {env.lanes}")

    state = env.reset()
    done, total_reward = False, 0.0
   
    while not done:
        action = np.random.randint(env.action_space_n)
        state, reward, done, info = env.step(action)
        print(f"t={info['sim_time']:.0f}s | acción={action} | "
              f"ciclo={info['cycle_length']:.0f}s | verdes={info['green_durations']} "
              f"| r={reward:.3f}")
        total_reward += reward

    print(f"Recompensa total: {total_reward:.2f}")
  
    env.close()
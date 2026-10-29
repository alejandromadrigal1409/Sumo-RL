import os
import sys
import itertools
import numpy as np

# --- Asegurar que TraCI/SUMO estén disponibles ---
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

else:
    sys.exit("Por favor define la variable de entorno SUMO_HOME")

try:
    import libsumo as traci  # noqa: E402
    USING_LIBSUMO = True
except ImportError:
    import traci  # noqa: E402
    USING_LIBSUMO = False


class SumoEnv:
    """Entorno RL multi-semáforo, un solo agente, basado en TraCI/libsumo.

    Controla N semáforos con UNA sola acción conjunta por paso. El espacio
    de acciones es el producto cartesiano de las fases verdes de cada
    semáforo: si cada uno tiene 2 fases verdes, action_space_n = 2**N,
    igual que describes (00, 01, 10, 11 para 2 semáforos).

    El estado sigue siendo un solo vector: se concatenan, en el orden de
    tls_ids, el bloque [fase_actual, cola_carril_1, ..., cola_carril_k]
    de cada semáforo.
    """

    def __init__(
        self,
        sumocfg_path: str,
        tls_ids,  # lista de ids de semáforo, p. ej. ["A", "B"]
        use_gui: bool = False,
        sim_steps_per_action: int = 5,
        max_simulation_time: int = 1000,
        yellow_time: int = 3,
        gui_delay_ms: int = 0,
    ):
        # Por comodidad, si pasan un solo string lo envolvemos en lista.
        if isinstance(tls_ids, str):
            tls_ids = [tls_ids]
        self.tls_ids = list(tls_ids)

        self.sumocfg_path = sumocfg_path
        self.use_gui = use_gui
        self.sim_steps_per_action = sim_steps_per_action
        self.max_simulation_time = max_simulation_time
        self.yellow_time = yellow_time
        self.gui_delay_ms = gui_delay_ms

        self.episode_step = 0

        # ------------------------------------------------------------
        # Sondeo inicial: leer la lógica de CADA semáforo.
        # ------------------------------------------------------------
        self._connect(gui=False)

        self.logic = {}          # tls_id -> lista de fases (objetos SUMO)
        self.green_phases = {}   # tls_id -> [índices de fases verdes]
        self.yellow_after = {}   # tls_id -> {fase_verde: fase_amarilla o None}
        self.lanes = {}          # tls_id -> [carriles controlados, sin duplicados]

        for tls_id in self.tls_ids:
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            phases = logic.phases
            n_phases = len(phases)
            self.logic[tls_id] = phases

            greens = [
                i for i, p in enumerate(phases)
                if ("G" in p.state or "g" in p.state) and "y" not in p.state
            ]
            if not greens:
                self._disconnect()
                raise ValueError(
                    f"No se encontraron fases verdes en el semáforo '{tls_id}'"
                )
            self.green_phases[tls_id] = greens

            yellow_map = {}
            for g in greens:
                nxt = (g + 1) % n_phases
                yellow_map[g] = nxt if "y" in phases[nxt].state else None
            self.yellow_after[tls_id] = yellow_map

            self.lanes[tls_id] = list(dict.fromkeys(
                traci.trafficlight.getControlledLanes(tls_id)
            ))

        self._disconnect()

        # ------------------------------------------------------------
        # Espacio de acciones conjunto: producto cartesiano de las
        # opciones (fases verdes) de cada semáforo.
        # action_combinations[a] = (idx_local_tls0, idx_local_tls1, ...)
        # ------------------------------------------------------------
        per_tls_n_actions = [len(self.green_phases[t]) for t in self.tls_ids]
        self.action_combinations = list(
            itertools.product(*[range(n) for n in per_tls_n_actions])
        )
        self.action_space_n = len(self.action_combinations)

        # Vector de estado = concatenación de [fase, colas...] por semáforo.
        state_len = sum(1 + len(self.lanes[t]) for t in self.tls_ids)
        self.observation_space_shape = (state_len,)

    # ------------------------------------------------------------------
    def _connect(self, seed: int = None, gui: bool = None, scale: float = None):
        use_gui = self.use_gui if gui is None else gui
        if use_gui and USING_LIBSUMO:
            raise RuntimeError(
                "libsumo no soporta sumo-gui (no hay visualización). "
                "Usa use_gui=False, o fuerza TraCI normal comentando el "
                "'import libsumo as traci' si necesitas ver la simulación."
            )
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
        except Exception:
            pass

    # ------------------------------------------------------------------
    def reset(self, seed: int = None, scale: float = None):
        """Reinicia la simulación y devuelve el estado inicial (un solo vector)."""
        self._disconnect()
        self._connect(seed=seed, scale=scale)
        self.episode_step = 0

        for tls_id in self.tls_ids:
            first_green = self.green_phases[tls_id][0]
            traci.trafficlight.setPhase(tls_id, first_green)
            traci.trafficlight.setPhaseDuration(tls_id, 10000)

        return self._get_state()

    def _decode_action(self, action: int):
        """Convierte la acción conjunta (un entero) en la fase verde
        objetivo (índice absoluto) para cada semáforo."""
        local_idxs = self.action_combinations[action]
        return {
            tls_id: self.green_phases[tls_id][local_idxs[i]]
            for i, tls_id in enumerate(self.tls_ids)
        }

    def step(self, action: int):
        """
        action: índice (0..action_space_n-1) dentro de action_combinations.
        Devuelve: (next_state, reward, done, info)
        """
        target_phases = self._decode_action(action)

        # Semáforos cuya fase objetivo difiere de la actual -> necesitan
        # transición (amarillo, si existe).
        changing = {}
        for tls_id in self.tls_ids:
            current = traci.trafficlight.getPhase(tls_id)
            target = target_phases[tls_id]
            if target != current:
                changing[tls_id] = current

        if changing:
            n_yellow = max(1, int(round(self.yellow_time / traci.simulation.getDeltaT())))

            # 1) Poner en amarillo a los que cambian y SÍ tienen amarillo
            #    definido. Los que no tienen amarillo pasan directo a su
            #    fase objetivo (no hay transición que esperar). Los
            #    semáforos que NO cambian simplemente se quedan como están.
            pending_switch = []
            for tls_id, current in changing.items():
                yellow = self.yellow_after[tls_id].get(current)
                if yellow is not None:
                    traci.trafficlight.setPhase(tls_id, yellow)
                    traci.trafficlight.setPhaseDuration(tls_id, self.yellow_time)
                    pending_switch.append(tls_id)
                else:
                    traci.trafficlight.setPhase(tls_id, target_phases[tls_id])

            # 2) Avanzar la ventana de amarillo (común a todos los que
            #    están en transición; los demás semáforos no se tocan y
            #    mantienen su fase verde actual durante estos pasos).
            if pending_switch:
                for _ in range(n_yellow):
                    traci.simulationStep()

                # 3) Pasar a la fase verde objetivo a los que sí tuvieron amarillo.
                for tls_id in pending_switch:
                    traci.trafficlight.setPhase(tls_id, target_phases[tls_id])

        # Mantener todas las fases verdes activas (evita que SUMO las
        # avance solas al expirar su duración).
        for tls_id in self.tls_ids:
            traci.trafficlight.setPhaseDuration(tls_id, 10000)

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
        """Un solo vector: [fase_A, colas_A..., fase_B, colas_B..., ...]."""
        state = []
        for tls_id in self.tls_ids:
            state.append(traci.trafficlight.getPhase(tls_id))
            for lane in self.lanes[tls_id]:
                vehicle_size_min_gap = traci.lane.getLastStepLength(lane) + 2.5
                lane_length = traci.lane.getLength(lane)
                queu = traci.lane.getLastStepHaltingNumber(lane) / (lane_length / vehicle_size_min_gap)
                state.append(queu)
        return np.array(state, dtype=np.float32)

    def _get_reward(self):
        """Recompensa única (escalar) para el agente: negativo del promedio
        de ocupación por congestión, promediado entre TODOS los carriles
        de TODOS los semáforos controlados."""
        total = 0.0
        n_lanes = 0
        for tls_id in self.tls_ids:
            for lane in self.lanes[tls_id]:
                vehicle_size_min_gap = traci.lane.getLastStepLength(lane) + 2.5
                lane_length = traci.lane.getLength(lane)
                capacity = lane_length / vehicle_size_min_gap
                total += traci.lane.getLastStepHaltingNumber(lane) / capacity
                n_lanes += 1
        return -total / n_lanes


# --------------------------------------------------------------------------
# Ejemplo de uso con un agente aleatorio
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Motor de simulación: {'libsumo' if USING_LIBSUMO else 'traci'}")

    env = SumoEnv(
        sumocfg_path="cologne3/cologne3.sumocfg",  # <-- cambia esto por tu archivo
        tls_ids=["TL1", "360086", "360082"],                         # <-- cambia esto por tus ids reales
        use_gui=False,
        sim_steps_per_action=5,
        max_simulation_time=800,
        yellow_time=3,
        gui_delay_ms=50,
    )

    print(f"Combinaciones de acción (action_space_n={env.action_space_n}):")
    for a, combo in enumerate(env.action_combinations):
        print(f"  acción {a}: {dict(zip(env.tls_ids, combo))}")
    print(f"Dimensión del estado: {env.observation_space_shape}")

    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = np.random.randint(env.action_space_n)
        state, reward, done, info = env.step(action)
        print(f"Tiempo: {info['sim_time']:.1f} s | Acción: {action} | Estado: {state}")
        total_reward += reward

    print(f"Recompensa total del episodio: {total_reward:.2f}")
    env.close()
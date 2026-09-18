import gymnasium as gym
import sumo_rl
import traci

env = gym.make(
    'sumo-rl-v0',
    net_file="single-intersection.net.xml",
    route_file="single-intersection.rou.xml",
    use_gui=False,
    num_seconds=1000,
    delta_time=5,
)

obs, info = env.reset()


print("\nStep length (s):", traci.simulation.getDeltaT() / 1000)  # regresa ms, por eso se divide
base_env = env.unwrapped
ts_id = list(base_env.traffic_signals.keys())[0]
ts = base_env.traffic_signals[ts_id]

# Controlled links: índice de link -> lista de (carril_entrada, carril_salida, carril_via)
controlled_links = traci.trafficlight.getControlledLinks(ts_id)

color_map = {
    'G': 'VERDE (prioridad)',
    'g': 'verde (con precaución)',
    'y': 'AMARILLO',
    'r': 'ROJO',
    's': 'stop',
    'u': 'sin prioridad',
}

print(f"\n\nNúmero de acciones (fases green): {ts.num_green_phases}")
print(f"Semáforo: {ts_id}")

for action in range(ts.num_green_phases):
    phase = ts.green_phases[action]
    state = phase.state
    print(f"===== ACCIÓN {action} =====")
    print(f"Fase resultante: '{state}'  (duración base: {phase.duration}s)\n")

    lane_status = {}
    for link_index, char in enumerate(state):
        for (in_lane, out_lane, via) in controlled_links[link_index]:
            status = color_map.get(char, char)
            lane_status.setdefault(in_lane, set()).add(status)

    for lane, statuses in lane_status.items():
        print(f"  Carril '{lane}': {', '.join(statuses)}")
    print()

env.close()
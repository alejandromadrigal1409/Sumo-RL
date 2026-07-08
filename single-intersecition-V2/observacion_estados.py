import os
import gymnasium as gym
import sumo_rl
import numpy as np

def discretization(obs):

    # ===== PHASE =====
    phase = "GGrr" if obs[0] == 1 else "rrGG"

    # ===== MIN_GREEN FLAG =====
    min_green_flag = int(obs[2])

    # ===== AUXILIARY FUNCTION =====
    def categorize(array, category_type):
        categorias = []

        for valor in array:
            if valor < 0.3:
                categorias.append("low" + category_type)
            elif valor < 0.7:
                categorias.append("medium" + category_type)
            else:
                categorias.append("high" + category_type)

        return categorias

    aux = len(obs) // 2
    # ===== DENSITIES =====
    discreteDensities = categorize(obs[3:aux + 2], "Density")

    # ===== QUEUES =====
    #discreteQueues = categorize(obs[aux + 2:], "Queue")

    # ===== DISCRETE STATE =====
    return [phase] + [min_green_flag] + discreteDensities # + discreteQueues
    

# --- Rutas a tus archivos (ajusta según tu proyecto) ---
BASE_DIR = r"C:\Users\aleja\sumo-rl\SimulacionesSUMO\single-intersecition-V2"
NET_FILE = os.path.join(BASE_DIR, "ticoman.net.xml")
ROUTE_FILE = os.path.join(BASE_DIR, "ticoman.rou.xml")

# --- Número de pasos que quieres observar ---
N_STEPS = 10

# creation of environment (single agent, un solo semáforo)
env = gym.make(
    'sumo-rl-v0',
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    # out_csv_name="outputs/montecarlo",  # genera .csv por episodio
    delta_time=5,
    use_gui=False,
    num_seconds=1000,
)

# --- Reiniciar el entorno (obtiene el primer estado/observación) ---
# gymnasium: reset() devuelve (observation, info)
observation, info = env.reset()
print("=== Estado inicial ===")
print("Observación:", observation)
print("Info:", info)

# Después de crear el entorno (antes o después del reset)
ts_id = list(env.unwrapped.traffic_signals.keys())[0]  # tu único semáforo
ts = env.unwrapped.traffic_signals[ts_id]

print("ID del semáforo:", ts_id)
print("Carriles de entrada monitoreados:", ts.lanes)
print("Carriles de salida:", ts.out_lanes)

import traci
lane_id = "820009948#2_1"
MIN_GAP = 2.5

# --- Correr unos pocos pasos y mostrar el estado en cada uno ---
for step in range(N_STEPS):
    # Un solo semáforo -> una sola acción, tomada directamente
    # del action_space del entorno (no hay diccionario de agentes).
    action = env.action_space.sample()

    # gymnasium: step() devuelve 5 valores
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"\n=== Paso {step + 1} ===")
    #print("Acción tomada:", action)
    print("Observación:", observation)
    #print("Recompensa:", reward)
    #print("Info:", info)
    '''
    vehicle_size_min_gap = traci.lane.getLastStepLength(lane_id) + MIN_GAP
    lane_length = traci.lane.getLength(lane_id)

    density = traci.lane.getLastStepVehicleNumber(lane_id) / (lane_length / vehicle_size_min_gap)
    queu = traci.lane.getLastStepHaltingNumber(lane_id) / (lane_length / vehicle_size_min_gap)

    print(f"Carril: {lane_id} , densidad: {density}, cola: {queu}")
    '''
    if terminated or truncated:
        print("La simulación terminó antes de completar todos los pasos.")
        break

prueba = discretization(observation)
print(type(prueba))
print(prueba)

env.close()
print("\nSimulación finalizada.")
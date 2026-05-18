import sumo_rl
import traci
from funciones import discretizacion

env = sumo_rl.SumoEnvironment(
    net_file="single-intersection.net.xml",
    route_file="single-intersection.rou.xml",
    use_gui=True,
    num_seconds=1000
)

obs = env.reset()

ts_id = env.ts_ids[0]

print("ID del semáforo:", ts_id)

# ===== ACCIÓN 0 =====
print("\n=== Acción 0 ===")

actions = {
    ts_id: 0
}

for i in range(20):

    # aplicar acción y obtener resultados RL
    next_obs, reward, done, info = env.step(actions)
    state = discretizacion(next_obs[ts_id])

    # obtener fase actual del semáforo
    fase = traci.trafficlight.getRedYellowGreenState(ts_id)

    print(f"\nPaso {i}")
    print("Fase:", fase)
    print("Estado:", state)
    print("Reward:", reward)

# ===== ACCIÓN 1 =====
print("\n=== Acción 1 ===")

actions = {
    ts_id: 1
}

for i in range(20):

    next_obs, reward, done, info = env.step(actions)
    state = discretizacion(next_obs[ts_id])

    fase = traci.trafficlight.getRedYellowGreenState(ts_id)

    print(f"\nPaso {i}")
    print("Fase:", fase)
    print("Estado:", state)
    print("Reward:", reward)

env.close()
import gymnasium as gym
import sumo_rl

env = gym.make(
    'sumo-rl-v0',
    net_file="single-intersection.net.xml",
    route_file="single-intersection.rou.xml",
    use_gui=False,
    num_seconds=1000,
    delta_time=5,
)

obs, info = env.reset()

print("===== ACTION SPACE =====")
print("Action space:", env.action_space)
print("Tipo:", type(env.action_space))
print("Número de acciones:", env.action_space.n)
for a in range(env.action_space.n):
    print(f"  Acción {a}")

print("\n===== OBSERVATION SPACE =====")
print("Observation space:", env.observation_space)
print("Shape:", env.observation_space.shape)
print("Ejemplo de observación (reset):", obs)

# ===== INFO ADICIONAL DEL SEMÁFORO (via el entorno interno) =====
# env.unwrapped te da acceso al SumoEnvironment real de sumo-rl
base_env = env.unwrapped
ts_id = list(base_env.traffic_signals.keys())[0]
ts = base_env.traffic_signals[ts_id]

print("\n===== DETALLE DEL SEMÁFORO =====")
print("ID del semáforo:", ts_id)
print("Fases green disponibles:", ts.num_green_phases)
print("Fases green (strings de estado):")
for i, phase in enumerate(ts.all_phases):
    print(f"  Fase {i}: {phase.state}  (duración base: {phase.duration}s)")

env.close()
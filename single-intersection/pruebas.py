import traci # Importa la librería traci, que permite comunicar Python con SUMO.
from funciones import get_estado

sumoCmd = [
    "sumo-gui",
    "-c",
    "single-intersection.sumocfg",
    "--random"
]

traci.start(sumoCmd) # Lanza SUMO usando el comando anterior y establece conexión entre Python y SUMO.

current_phase = None
for step in range(1000): # Ejecuta 1000 pasos de simulación. Cada paso suele equivaler a 1 segundo
    
    traci.simulationStep() # Hace avanzar la simulación exactamente un paso.

    s = get_estado()

    print(f"step: {step}, estado: {s}")

traci.close() # Cierra: la conexión TraCI la simulación SUMO
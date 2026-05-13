import traci # Importa la librería traci, que permite comunicar Python con SUMO.

sumoCmd = [
    "sumo-gui",
    "-c",
    "single-intersection.sumocfg",
    "--random"
]

traci.start(sumoCmd) # Lanza SUMO usando el comando anterior y establece conexión entre Python y SUMO.

# Variables para guardar el total acumulado de vehículos detectados.
total_north = 0
total_west = 0

tlsID = "t" # ID del semaforo 

current_phase = None
for step in range(1000): # Ejecuta 1000 pasos de simulación. Cada paso suele equivaler a 1 segundo

    # Muestra el ID de todos los semaforos en el ambiente 
    #print(traci.trafficlight.getIDList())

    cycle = step % 130

    if cycle < 60:
        phase = "GGrr"

    elif cycle < 65:
        phase = "yyrr"

    elif cycle < 125:
        phase = "rrGG"

    else:
        phase = "rryy"

    if phase != current_phase:
        traci.trafficlight.setRedYellowGreenState(
            tlsID,
            phase
        )
        current_phase = phase
    
    traci.simulationStep() # Hace avanzar la simulación exactamente un paso.

    north_queue = (
        traci.lane.getLastStepHaltingNumber("n_t_0")
        +
        traci.lane.getLastStepHaltingNumber("n_t_1")
        )

    west_queue = (
            traci.lane.getLastStepHaltingNumber("w_t_0")
            +
            traci.lane.getLastStepHaltingNumber("w_t_1")
        )

    # lectura de sensores. Aquí se leen dos sensores de tipo induction loop. 
    # Un induction loop es un sensor colocado sobre un carril que detecta vehículos cuando pasan.
    north = (
        # getLastStepVehicleNumber(...)  # “¿cuántos vehículos pasaron por este sensor en el último paso de simulación?”
        # "north0" es el nombre del sensor
        traci.inductionloop.getLastStepVehicleNumber("north0") 
        +
        traci.inductionloop.getLastStepVehicleNumber("north1")
    )

    west = (
        traci.inductionloop.getLastStepVehicleNumber("west0")
        +
        traci.inductionloop.getLastStepVehicleNumber("west1")
    )

    total_north += north
    total_west += west

    print(f"step: {step}, Cola N-S: {north_queue}, Cola W-E: {west_queue}")

print("\nTOTAL NORTH:", total_north)
print("TOTAL WEST:", total_west)

traci.close() # Cierra: la conexión TraCI la simulación SUMO
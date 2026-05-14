import traci

def discretizacion(n):
    if n < 3:
        return 1
    elif n >= 3 and n < 5:
        return 2
    else:
        return 3
    
def get_estado():
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
    
    phase = traci.trafficlight.getPhase("t")

    # Tripleta:

    s = (
        discretizacion(north_queue), 
        discretizacion(west_queue), 
        0 if phase < 2 else 1
    )

    return s
import traci
import numpy as np

def discretization(obs):

    # ===== PHASE =====
    phase = "GGrr" if obs[0] == 1 else "rrGG"

    # ===== MIN_GREEN FLAG =====
    min_green_flag = int(obs[2])

    # ===== AUXILIARY FUNCTION =====
    def categorize(value, category_type):

        if value < 0.3:
            return f"low_{category_type}"

        elif value < 1.5:
            return f"medium_{category_type}"

        else:
            return f"high_{category_type}"

    # ===== DENSITIES =====
    density_n2s = obs[3] + obs[4]
    density_w2e = obs[5] + obs[6]

    density_n2s = categorize(density_n2s, "density")
    density_w2e = categorize(density_w2e, "density")

    # ===== QUEUES =====
    queue_n2s = obs[7] + obs[8]
    queue_w2e = obs[9] + obs[10]

    queue_n2s = categorize(queue_n2s, "queue")
    queue_w2e = categorize(queue_w2e, "queue")

    # ===== DISCRETE STATE =====
    return (
        phase,
        min_green_flag,
        density_n2s,
        density_w2e,
        queue_n2s,
        queue_w2e
    )
    


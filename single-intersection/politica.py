import pickle
from pprint import pformat

with open("Temporal_Difference/experiments/E_SARSA_experiment_2026-09-22_22-48-56/best_policy_2026-09-22_22-48-56.pkl", 
          "rb") as f:
    datos = pickle.load(f)

with open("best_policy_contenido.txt", "w", encoding="utf-8") as f:
    for estado, accion in datos.items():
        f.write("=" * 100 + "\n")
        f.write(f"ESTADO:\n{pformat(estado, width=120)}\n")
        f.write(f"ACCION:\n{pformat(accion, width=120)}\n")
        f.write("\n")
import pickle
from pprint import pformat

experimento = "QL_experiment_2026-10-07_19-40-46/best_policy_2026-10-07_19-40-46.pkl"

with open(f"Temporal_Difference/experiments/{experimento}", 
          "rb") as f:
    datos = pickle.load(f)

with open("best_policy_contenido.txt", "w", encoding="utf-8") as f:
    for estado, accion in datos.items():
        f.write("=" * 100 + "\n")
        f.write(f"ESTADO:\n{pformat(estado, width=120)}\n")
        f.write(f"ACCION:\n{pformat(accion, width=120)}\n")
        f.write("\n")





import csv
import xml.etree.ElementTree as ET
import numpy as np
import sumolib

DIAS  = 25
PASOS = 288                 # intervalos de 5 min en 24 horas
RED   = "del_valle_y_narvarte_vel.net.xml"

# Sensores en el mismo orden que grafo_sensores.csv
with open("sensores.csv", encoding="utf-8") as f:
    sensores = [fila["calle"] for fila in csv.DictReader(f)]
indice = {calle: k for k, calle in enumerate(sensores)}
N = len(sensores)

# Velocidad máxima de cada calle (km/h), para rellenar los intervalos sin vehículos
net = sumolib.net.readNet(RED)
v_max = np.array([net.getEdge(c).getSpeed() * 3.6 for c in sensores])

data = np.zeros((DIAS * PASOS, N, 3), dtype=np.float32)

for d in range(1, DIAS + 1):
    flujo     = np.zeros((PASOS, N))
    ocup_suma = np.zeros((PASOS, N))
    carriles  = np.zeros((PASOS, N))
    vel_pond  = np.zeros((PASOS, N))

    for _, el in ET.iterparse(f"dia{d}_detectores.xml"):
        if el.tag != "interval":
            continue
        t = int(float(el.get("begin")) // 300)          # número de intervalo
        if t < PASOS:                                     # ignora el margen de 24:00 a 24:30
            k = indice[el.get("id")[4:].rsplit("_", 1)[0]]   # "det_123#1_0" -> sensor de "123#1"
            n = int(el.get("nVehContrib"))
            flujo[t, k]     += n
            ocup_suma[t, k] += float(el.get("occupancy"))
            carriles[t, k]  += 1
            if n > 0:
                vel_pond[t, k] += n * float(el.get("speed"))
        el.clear()

    assert (carriles > 0).all(), f"Faltan intervalos en el día {d}"
    ocupacion = ocup_suma / carriles / 100                          # promedio, de % a fracción
    velocidad = np.where(flujo > 0, vel_pond / np.maximum(flujo, 1) * 3.6, v_max)

    inicio = (d - 1) * PASOS
    data[inicio:inicio + PASOS, :, 0] = flujo
    data[inicio:inicio + PASOS, :, 1] = ocupacion
    data[inicio:inicio + PASOS, :, 2] = velocidad
    print(f"Día {d:2d}: {int(flujo.sum()):7d} vehículos detectados, "
          f"{(flujo == 0).mean() * 100:4.1f}% de intervalos sin vehículos")

np.savez("del_valle.npz", data=data)

print("\nForma:", data.shape)
for i, nombre in enumerate(["flow", "occupancy", "speed"]):
    x = data[..., i]
    print(f"{nombre:10s} min={x.min():7.2f}  max={x.max():7.2f}  media={x.mean():7.2f}")

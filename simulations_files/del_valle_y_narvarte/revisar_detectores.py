import xml.etree.ElementTree as ET
from collections import defaultdict
import sumolib

net = sumolib.net.readNet("del_valle_y_narvarte_fijo.net.xml")

# Vehículos por calle (sumando carriles) entre las 8:00 y las 9:00
pico = defaultdict(int)
dia = defaultdict(int)
for _, el in ET.iterparse("dia1_detectores.xml"):
    if el.tag == "interval":
        calle = el.get("id")[4:].rsplit("_", 1)[0]   # "det_123#1_0" -> "123#1"
        n = int(el.get("nVehContrib"))
        dia[calle] += n
        if 28800 <= float(el.get("begin")) < 32400:
            pico[calle] += n
        el.clear()

valores = sorted(pico.values())
print(f"Calles: {len(dia)}")
print(f"Vehículos en la hora pico por calle: mínimo {valores[0]}, mediana {valores[len(valores)//2]}, máximo {valores[-1]}")
print(f"Calles sin ningún vehículo en todo el día: {sum(1 for v in dia.values() if v == 0)}")

print("\nLas 10 calles con MENOS tráfico en la hora pico:")
for calle, v in sorted(pico.items(), key=lambda x: x[1])[:10]:
    e = net.getEdge(calle)
    print(f"  {v:5d} veh/h  {e.getName() or '(sin nombre)':35s} {calle}")

print("\nLas 10 calles con MÁS tráfico en la hora pico:")
for calle, v in sorted(pico.items(), key=lambda x: -x[1])[:10]:
    e = net.getEdge(calle)
    print(f"  {v:5d} veh/h  {e.getName() or '(sin nombre)':35s} {calle}")
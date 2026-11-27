import heapq
import xml.etree.ElementTree as ET
import sumolib

RED   = "del_valle_y_narvarte_vel.net.xml"
MAXD  = 500   # metros: distancia máxima entre dos detectores conectados
PASO  = {"highway.primary", "highway.secondary", "highway.tertiary",
         "highway.primary_link", "highway.secondary_link", "highway.tertiary_link"}

net = sumolib.net.readNet(RED)

# 1. Los sensores: las calles de detectores.add.xml, en el orden en que aparecen
sensores = []
for det in ET.parse("detectores.add.xml").getroot():
    calle = det.get("lane").rsplit("_", 1)[0]       # "123#1_0" -> "123#1"
    if calle not in sensores:
        sensores.append(calle)
indice = {calle: k for k, calle in enumerate(sensores)}

with open("sensores.csv", "w", encoding="utf-8") as f:
    f.write("indice,calle,nombre,carriles,longitud_m\n")
    for k, calle in enumerate(sensores):
        e = net.getEdge(calle)
        f.write(f'{k},{calle},"{e.getName()}",{e.getLaneNumber()},{e.getLength():.1f}\n')

# 2. Las aristas: desde cada sensor, avanzar por la red hasta el siguiente sensor
def es_calle(e):
    return e.getFunction() != "internal" and e.allows("passenger")

aristas = {}
for calle in sensores:
    i = indice[calle]
    e = net.getEdge(calle)
    # distancia desde el detector (a la mitad de la calle) hasta el final de la calle
    pendientes = [(e.getLength() / 2, o.getID()) for o in e.getOutgoing() if es_calle(o)]
    heapq.heapify(pendientes)
    visitadas = set()
    while pendientes:
        d, eid = heapq.heappop(pendientes)          # siempre la más cercana primero
        if eid in visitadas:
            continue
        visitadas.add(eid)
        x = net.getEdge(eid)
        if eid in indice:                           # llegamos a otro sensor
            dist = d + x.getLength() / 2            # hasta su detector, a la mitad
            j = indice[eid]
            if j != i and dist <= MAXD:
                aristas[(i, j)] = min(dist, aristas.get((i, j), float("inf")))
            continue                                # no seguir más allá de un sensor
        if x.getType() not in PASO:                 # no pasar por calles residenciales
            continue
        d_nueva = d + x.getLength()
        if d_nueva > MAXD:
            continue
        for o in x.getOutgoing():
            if es_calle(o) and o.getID() not in visitadas:
                heapq.heappush(pendientes, (d_nueva, o.getID()))

with open("grafo_sensores.csv", "w") as f:
    f.write("from,to,cost\n")
    for (i, j), d in sorted(aristas.items()):
        f.write(f"{i},{j},{d:.1f}\n")

conexiones = {tuple(sorted(k)) for k in aristas}
print(f"Sensores: {len(sensores)}")
print(f"Aristas dirigidas: {len(aristas)}")
print(f"Conexiones (sin contar sentido): {len(conexiones)}")
print(f"Vecinos promedio: {2 * len(conexiones) / len(sensores):.2f}")

import sumolib

RED = "del_valle_y_narvarte_fijo.net.xml"

# Velocidad nueva por tipo de calle, en km/h
NUEVAS = {
    "highway.tertiary":       40,
    "highway.tertiary_link":  40,
    "highway.secondary_link": 40,
    "highway.residential":    30,
}

net = sumolib.net.readNet(RED)
cambios = 0

with open("velocidades.edg.xml", "w", encoding="utf-8") as f:
    f.write("<edges>\n")
    for e in net.getEdges():
        if e.getFunction() == "internal" or e.getType() not in NUEVAS:
            continue
        v = NUEVAS[e.getType()] / 3.6          # SUMO usa m/s
        f.write(f'    <edge id="{e.getID()}" speed="{v:.2f}"/>\n')
        cambios += 1
    f.write("</edges>\n")

print(f"Calles con velocidad nueva: {cambios}")

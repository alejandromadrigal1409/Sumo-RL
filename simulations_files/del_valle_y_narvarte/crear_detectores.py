import sumolib

RED     = "del_valle_y_narvarte_fijo.net.xml"
TIPOS   = {"highway.primary", "highway.secondary"}
LMIN    = 40      # metros: ignora tramos más cortos
PERIODO = 300     # segundos: un resumen cada 5 minutos

net = sumolib.net.readNet(RED)
n_calles, n_detectores = 0, 0

with open("detectores.add.xml", "w", encoding="utf-8") as f:
    f.write("<additional>\n")
    for e in net.getEdges():
        if e.getFunction() == "internal" or e.getType() not in TIPOS or e.getLength() < LMIN:
            continue
        n_calles += 1
        for lane in e.getLanes():
            if not lane.allows("passenger"):
                continue
            pos = lane.getLength() / 2
            f.write(f'    <inductionLoop id="det_{lane.getID()}" lane="{lane.getID()}" '
                    f'pos="{pos:.2f}" period="{PERIODO}" file="detectores.xml"/>\n')
            n_detectores += 1
    f.write("</additional>\n")

print(f"Calles con detector: {n_calles}")
print(f"Detectores (uno por carril): {n_detectores}")
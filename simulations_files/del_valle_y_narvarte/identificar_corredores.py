import sumolib

RED = "del_valle_y_narvarte_vel.net.xml"

# Nombre del corredor -> nombres que tiene en OSM (Eugenia aparece con tres nombres)
AVENIDAS = {
    "Insurgentes":        ["Avenida Insurgentes Sur"],
    "Cuauhtemoc":         ["Avenida Cuauhtémoc"],
    "Xola":               ["Xola"],
    "Eugenia":            ["Eugenia", "Avenida Eugenia", "Avenida Colonia del Valle"],
    "Division del Norte": ["Avenida División del Norte"],
    "Angel Urraza":       ["Ángel Urraza"],
    "Gabriel Mancera":    ["Gabriel Mancera"],
    "Amores":             ["Calle Amores"],
    "Coyoacan":           ["Avenida Coyoacán"],
}

net = sumolib.net.readNet(RED)
calles = [e for e in net.getEdges() if e.getFunction() != "internal" and e.allows("passenger")]

with open("corredores.csv", "w", encoding="utf-8") as csv, open("corredores_sel.txt", "w") as sel:
    csv.write("corredor,desde,hasta,longitud_m\n")
    for nombre, nombres_osm in AVENIDAS.items():
        tramos = {e for e in calles if e.getName() in nombres_osm}
        inicios = [e for e in tramos if not any(i in tramos for i in e.getIncoming())]
        finales = [e for e in tramos if not any(o in tramos for o in e.getOutgoing())]

        for s in inicios:
            mejor = None
            for t in finales:
                ruta, _ = net.getShortestPath(s, t, vClass="passenger")
                if not ruta:
                    continue
                L = sum(e.getLength() for e in ruta)
                L_av = sum(e.getLength() for e in ruta if e in tramos)
                if L > 500 and L_av / L >= 0.85 and (mejor is None or L > mejor[1]):
                    mejor = (t, L, ruta)
            if mejor:
                t, L, ruta = mejor
                csv.write(f"{nombre},{s.getID()},{t.getID()},{L:.0f}\n")
                for e in ruta:
                    sel.write(f"edge:{e.getID()}\n")
                print(f"{nombre:20s} {s.getID():18s} -> {t.getID():18s} {L:6.0f} m")

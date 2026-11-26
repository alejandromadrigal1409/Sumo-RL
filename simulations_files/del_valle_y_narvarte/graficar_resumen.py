import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

t, corriendo, velocidad, detenidos = [], [], [], []
for _, el in ET.iterparse("resumen_dia1.xml"):
    if el.tag == "step":
        t.append(float(el.get("time")) / 3600)            # en horas
        corriendo.append(int(el.get("running")))
        velocidad.append(float(el.get("meanSpeed")) * 3.6)  # en km/h
        detenidos.append(int(el.get("halting")))
        el.clear()

fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
ax[0].plot(t, corriendo);  ax[0].set_ylabel("Vehículos en la red")
ax[1].plot(t, detenidos);  ax[1].set_ylabel("Vehículos detenidos")
ax[2].plot(t, velocidad);  ax[2].set_ylabel("Velocidad media (km/h)")
ax[2].set_xlabel("Hora del día"); ax[2].set_xticks(range(0, 25, 2))
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.show()
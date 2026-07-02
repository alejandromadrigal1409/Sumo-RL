import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Configuración de la gráfica
# ==========================
titulo = "Comparación de Tiempo de Espera"
x_label = "Tiempo de simulación (steps)"
y_label = "Tiempo de espera total (s)"

# Lista de archivos y etiquetas
files = [
    ("RL_raw_data.csv", "E-SARSA"),
    ("FIX_raw_data.csv", "Tiempo Fijo")
]

# ==========================
# Crear figura
# ==========================
plt.figure(figsize=(10, 6))

for filename, label in files:

    # Cargar datos
    df = pd.read_csv(filename)

    xaxis = df.columns[0]
    yaxis = df.columns[1]

    # Calcular promedio y desviación estándar para cada valor de x
    mean = df.groupby(xaxis)[yaxis].mean()
    std = df.groupby(xaxis)[yaxis].std()

    x = mean.index.values

    # Estadísticas globales de todos los datos
    mean_global = df[yaxis].mean()
    std_global = df[yaxis].std()

    # Graficar promedio
    plt.plot(
        x,
        mean,
        linewidth=2,
        label=f"{label} ({mean_global:.2f} ± {std_global:.2f})"
    )

    # Graficar banda de desviación estándar
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.2
    )

# ==========================
# Personalización
# ==========================
plt.title(titulo, fontsize=16)
plt.xlabel(x_label, fontsize=12)
plt.ylabel(y_label, fontsize=12)

# Malla principal y secundaria
plt.grid(True, which="major", linestyle="-", alpha=0.5)
plt.minorticks_on()
plt.grid(True, which="minor", linestyle=":", alpha=0.3)

# Leyenda
plt.legend(
    title="Método (Media ± Desv. Est.)",
    fontsize=10,
    title_fontsize=11
)

plt.tight_layout()

# ==========================
# Guardar figura
# ==========================
plt.savefig(
    "comparacion.png",
    dpi=300,
    bbox_inches="tight"
)

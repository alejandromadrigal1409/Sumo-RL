import argparse, math, os, random, shutil, subprocess, sys

parser = argparse.ArgumentParser()
parser.add_argument("--dia", type=int, default=1)
parser.add_argument("--red", default="del_valle_y_narvarte_vel.net.xml")
parser.add_argument("--resumen", action="store_true", help="guardar diaN_resumen.xml")
parser.add_argument("--conservar-viajes", action="store_true", help="no borrar la carpeta diaN/")
args = parser.parse_args()
dia, red = args.dia, args.red

PERIODOS = [4.2,4.2,4.2,4.2,4.2,2.0,0.83,0.56,0.5,0.59,0.77,0.77,0.77,0.77,0.67,0.67,0.71,0.59,0.5,0.53,0.71,1.0,1.43,2.5]
tasa = [3600 / p for p in PERIODOS]

rng = random.Random(dia)
factor_dia = 0.90 + 0.10 * rng.random()
carpeta = f"dia{dia}"
os.makedirs(carpeta, exist_ok=True)
random_trips = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")
archivos = []

for q in range(96):
    x  = q / 4 + 0.125 - 0.5
    i  = math.floor(x)
    w  = x - i
    i0 = max(0, min(23, i))
    i1 = max(0, min(23, i + 1))
    r  = ((1 - w) * tasa[i0] + w * tasa[i1]) * factor_dia * (0.95 + 0.10 * rng.random())
    p  = round(3600 / r, 3)

    b, e = q * 900, q * 900 + 900
    f = os.path.join(carpeta, f"q{q:02d}.trips.xml")
    rutas = os.path.join(carpeta, f"q{q:02d}.rou.xml")
    subprocess.run([sys.executable, random_trips, "-n", red, "-o", f, "-r", rutas,
                    "-b", str(b), "-e", str(e), "-p", str(p),
                    "--prefix", f"d{dia}q{q:02d}_", "--fringe-factor", "5", "-l",
                    "--validate", "--seed", str(1000 * dia + q)], check=True)
    archivos.append(f)

print(f"Factor del dia {dia}: {factor_dia}")
cmd = ["sumo", "-n", red, "-r", ",".join(archivos), "-a", "detectores.add.xml",
       "-b", "0", "-e", "88200", "--duration-log.statistics",
       "--time-to-teleport", "300", "--no-step-log", "--no-warnings",
       "--output-prefix", f"dia{dia}_"]
if args.resumen:
    cmd += ["--summary-output", "resumen.xml"]
subprocess.run(cmd, check=True)

if not args.conservar_viajes:
    shutil.rmtree(carpeta)

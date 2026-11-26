param([int]$dia = 1)
$periodos = @(4.2,4.2,4.2,4.2,4.2,2.0,0.83,0.56,0.5,0.59,0.77,0.77,0.77,0.77,0.67,0.67,0.71,0.59,0.5,0.53,0.71,1.0,1.43,2.5)
$tasa = $periodos | ForEach-Object { 3600 / $_ }       # vehículos por hora de cada hora
$rand = New-Object System.Random($dia)
$factorDia = 0.90 + 0.10 * $rand.NextDouble()           # entre 0.90 y 1.00
$carpeta = "dia$dia"
New-Item -ItemType Directory -Force -Path $carpeta | Out-Null
$archivos = @()

for ($q = 0; $q -lt 96; $q++) {
    # posición del centro del bloque respecto a los centros de cada hora
    $x  = $q / 4 + 0.125 - 0.5
    $i  = [math]::Floor($x)
    $w  = $x - $i
    $i0 = [math]::Max(0, [math]::Min(23, $i))
    $i1 = [math]::Max(0, [math]::Min(23, $i + 1))
    $r  = ((1 - $w) * $tasa[$i0] + $w * $tasa[$i1]) * $factorDia * (0.95 + 0.10 * $rand.NextDouble())
    $p  = [math]::Round(3600 / $r, 3)

    $b  = $q * 900
    $e  = $b + 900
    $qq = "{0:D2}" -f $q
    $f  = "$carpeta\q$qq.trips.xml"
    $semilla = 1000 * $dia + $q
    python "$env:SUMO_HOME\tools\randomTrips.py" -n del_valle_y_narvarte_fijo.net.xml -o $f -b $b -e $e -p $p --prefix "d${dia}q${qq}_" --fringe-factor 5 -l --validate --seed $semilla 
    $archivos += $f
}

Write-Host "Factor del día $dia : $factorDia"
$lista = $archivos -join ","
sumo -n del_valle_y_narvarte_fijo.net.xml -r $lista -a detectores.add.xml -b 0 -e 88200 --duration-log.statistics --time-to-teleport 300 --no-step-log --no-warnings --summary-output resumen.xml --output-prefix "dia${dia}_"
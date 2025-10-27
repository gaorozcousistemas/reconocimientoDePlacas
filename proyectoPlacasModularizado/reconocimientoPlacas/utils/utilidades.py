import datetime

def formatearFechaHora(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime('%d/%m/%Y %H:%M:%S')

def mostrarEstadisticasFinales(placasDetectadas, estadisticasRangos):
    print(f"\n📊 ESTADÍSTICAS FINALES:")
    print(f"Placas únicas detectadas: {len(placasDetectadas)}")
    print("Rangos más efectivos:")
    for nombre, count in sorted(estadisticasRangos.items(), key=lambda x: x[1], reverse=True):
        print(f"  {nombre}: {count} detecciones")

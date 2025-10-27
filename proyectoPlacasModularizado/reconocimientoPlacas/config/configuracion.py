import numpy as np

# Tiempo de espera para detecciones repetidas (segundos)
TIEMPO_ESPERA_REPETICION = 60

# Rangos HSV para detección de color amarillo en distintas condiciones de luz
RANGOS_AMARILLO = [
    {'bajo': np.array([15, 50, 150]), 'alto': np.array([35, 255, 255]), 'nombre': 'brillante'},
    {'bajo': np.array([15, 80, 80]),  'alto': np.array([35, 255, 200]), 'nombre': 'estandar'},
    {'bajo': np.array([15, 60, 40]),  'alto': np.array([35, 255, 150]), 'nombre': 'oscuro'},
    {'bajo': np.array([20, 40, 180]), 'alto': np.array([30, 150, 255]), 'nombre': 'claro'}
]

import cv2
import numpy as np
import re

# ===== Clase DeteccionPlaca =====
class DeteccionPlaca:
    def __init__(self, placa, bbox, tiempoDeteccion):
        self.placa = placa
        self.bbox = bbox
        self.tiempoPrimerDeteccion = tiempoDeteccion
        self.tiempoUltimaDeteccion = tiempoDeteccion
        self.activa = True
        self.mejorRango = None

    def actualizar(self, bbox, tiempoActual, rango=None):
        self.bbox = bbox
        self.tiempoUltimaDeteccion = tiempoActual
        if rango:
            self.mejorRango = rango

# ===== Funciones de procesamiento =====
def preprocesarPlaca(imagen):
    if imagen.size == 0:
        return None
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    resultados = []
    # ecualización de histograma
    grisEq = cv2.equalizeHist(gris)
    resultados.append(('eq_normal', grisEq))
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    grisClahe = clahe.apply(gris)
    resultados.append(('clahe', grisClahe))
    # Filtro bilateral
    grisBilateral = cv2.bilateralFilter(gris, 5, 50, 50)
    resultados.append(('bilateral', grisBilateral))
    # Combinado
    grisCombinado = cv2.bilateralFilter(grisClahe, 5, 50, 50)
    resultados.append(('combinado', grisCombinado))
    return resultados

def detectarColorPlaca(roi, rangos):
    mejoresResultados = []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    for rango in rangos:
        mask = cv2.inRange(hsv, rango['bajo'], rango['alto'])
        areaTotal = roi.shape[0] * roi.shape[1]
        if areaTotal > 0:
            porcentaje = (cv2.countNonZero(mask) / areaTotal) * 100
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours]
            areaMax = max(areas) if areas else 0
            calidad = porcentaje * (1 + (areaMax / areaTotal))
            mejoresResultados.append({'rango': rango, 'porcentaje': porcentaje, 'calidad': calidad, 'mask': mask})
    mejoresResultados.sort(key=lambda x: x['calidad'], reverse=True)
    return mejoresResultados

def validarPlaca(texto):
    textoLimpio = re.sub(r'[^A-Z0-9]', '', texto.upper())
    if len(textoLimpio) < 5 or len(textoLimpio) > 7:
        return None
    if not re.search(r'[A-Z]', textoLimpio) or not re.search(r'[0-9]', textoLimpio):
        return None
    patrones = [
        r'^[A-Z]{3}[0-9]{3}$',
        r'^[A-Z]{2}[0-9]{4}$',
        r'^[A-Z]{1}[0-9]{5}$',
        r'^[0-9]{3}[A-Z]{3}$',
        r'^[0-9]{2}[A-Z]{4}$',
        r'^[0-9]{1}[A-Z]{5}$'
    ]
    for i in range(len(textoLimpio)-5):
        subcadena = textoLimpio[i:i+6]
        for patron in patrones:
            if re.match(patron, subcadena):
                return subcadena
    return None

def mejorarContrasteTexto(imagen):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    imagenMejorada = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
    kernelCierre = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    imagenMejorada = cv2.morphologyEx(imagenMejorada, cv2.MORPH_CLOSE, kernelCierre)
    return imagenMejorada


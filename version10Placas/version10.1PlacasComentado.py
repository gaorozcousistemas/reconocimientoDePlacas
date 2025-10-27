#!/usr/bin/env python3  # Shebang: permite ejecutar el script con ./script.py en Unix-like.

"""
Refactor de version10Placas.py
Objetivo: mantener la misma lógica funcional pero con mejor organización, claridad y manejo de errores.
"""  # Docstring: resumen del propósito; añadir Usage/Requirements sería útil.

import cv2  # OpenCV: captura y procesamiento de imágenes.
import easyocr  # EasyOCR: lector OCR para extraer texto de imágenes.
import numpy as np  # NumPy: arrays y operaciones numéricas.
import re  # re: expresiones regulares para validar placas.
import time  # time: tiempos y mediciones.
import datetime  # datetime: formateo de fechas.
from ultralytics import YOLO  # YOLO: modelo de detección (ultralytics).
from dataclasses import dataclass  # dataclass: para estructurar la clase de detección.
import logging  # logging: para mensajes estructurados en vez de prints.
import sys  # sys: para exit y manejo de errores críticos.

# ---------------------- CONFIGURACIÓN ----------------------
TIEMPO_ESPERA_REPETICION = 90  # segundos; evita contar la misma placa repetidamente.
FRAME_SKIP = 10  # Procesar detección cada N frames para ahorrar CPU.
CONF_THRESHOLD = 0.4  # Umbral de confianza para las detecciones YOLO.
OCR_CONF_THRESHOLD = 0.3  # Umbral mínimo de confianza del OCR para aceptar texto.
COLOR_PROCESS_THRESHOLD = 15  # % mínimo de píxeles amarillos para intentar OCR.
TRACKING_COLOR_THRESHOLD = 10  # % mínimo para mantener tracking basado en color.
CAMERA_PREFERENCES = [0, 1]  # Indices de cámaras a probar (preferencia).

# Múltiples rangos HSV (idénticos a los definidos originalmente)
RANGOS_AMARILLO = [
    {'bajo': np.array([15, 50, 150]), 'alto': np.array([35, 255, 255]), 'nombre': 'brillante'},
    {'bajo': np.array([15, 80, 80]), 'alto': np.array([35, 255, 200]), 'nombre': 'estandar'},
    {'bajo': np.array([15, 60, 40]), 'alto': np.array([35, 255, 150]), 'nombre': 'oscuro'},
    {'bajo': np.array([20, 40, 180]), 'alto': np.array([30, 150, 255]), 'nombre': 'claro'}
]  # Varios rangos para adaptarse a distintas iluminaciones; considerar calibración externa.

# ---------------------- ESTRUCTURAS ----------------------
@dataclass
class DeteccionPlaca:
    placa: str  # Cadena de la placa válida detectada.
    bbox: tuple  # Bounding box (x1, y1, x2, y2).
    tiempo_primer_deteccion: float  # Timestamp primera detección.
    tiempo_ultima_deteccion: float  # Timestamp última detección.
    activa: bool = True  # Indicador si está activa en el tracking.
    mejor_rango: dict = None  # Rango HSV que mejor detectó el color.

    def actualizar(self, bbox, tiempo_actual, rango=None):
        self.bbox = bbox  # Actualiza el bounding box.
        self.tiempo_ultima_deteccion = tiempo_actual  # Actualiza tiempo de última detección.
        if rango:
            self.mejor_rango = rango  # Opcional: guarda el rango que mejor funcionó.

# ---------------------- UTILIDADES / PREPROCESOS ----------------------
def cargar_modelos(yolo_path='yolov8n.pt', ocr_langs=['en'], gpu=False):
    """Carga modelo YOLO y OCR (easyocr)."""
    logging.info('Cargando modelos...')  # Log inicial.
    try:
        model = YOLO(yolo_path)  # Carga el modelo YOLO desde el path dado.
        logging.info('Modelo YOLO cargado')
    except Exception as e:
        logging.exception('Error cargando YOLO: %s', e)  # Log con traza si falla.
        raise  # Re-lanzar para que el llamador lo maneje.

    try:
        reader = easyocr.Reader(ocr_langs, gpu=gpu)  # Inicializa EasyOCR.
        logging.info('OCR configurado')
    except Exception as e:
        logging.exception('Error configurando OCR: %s', e)
        raise

    return model, reader  # Devuelve ambos objetos listos para usar.

def preprocesar_placa_mejorado(imagen):
    if imagen is None or imagen.size == 0:
        return None  # Si la imagen está vacía, retornar None.

    try:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Convertir a gris.

        resultados = []  # Lista de (nombre_tecnica, imagen_procesada).

        gris_eq = cv2.equalizeHist(gris)  # Ecualización global.
        resultados.append(('eq_normal', gris_eq))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHE local.
        gris_clahe = clahe.apply(gris)
        resultados.append(('clahe', gris_clahe))

        gris_bilateral = cv2.bilateralFilter(gris, 5, 50, 50)  # Filtrado bilateral para suavizar preservando bordes.
        resultados.append(('bilateral', gris_bilateral))

        gris_combinado = cv2.bilateralFilter(gris_clahe, 5, 50, 50)  # Combinación: CLAHE + bilateral.
        resultados.append(('combinado', gris_combinado))

        return resultados  # Retorna varias versiones para probar con OCR.
    except Exception:
        logging.exception('Error en preprocesamiento de placa')
        return None  # En caso de fallo, retornar None.

def detectar_color_placas(roi, rangos=None):
    """Detecta amarillo usando múltiples rangos y devuelve una lista ordenada por 'calidad'."""
    if rangos is None:
        rangos = RANGOS_AMARILLO  # Usa rangos por defecto si no se pasan.

    mejores_resultados = []  # Lista de resultados por rango.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # Convertir ROI a HSV.
    area_total = roi.shape[0] * roi.shape[1]  # Área total de la ROI (px).

    for rango in rangos:
        mask = cv2.inRange(hsv, rango['bajo'], rango['alto'])  # Mascara binaria por rango HSV.
        if area_total <= 0:
            continue  # Evitar división por cero.
        porcentaje = (cv2.countNonZero(mask) / area_total) * 100  # % de píxeles en rango.

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Contornos.
        areas = [cv2.contourArea(cnt) for cnt in contours]  # Áreas de contornos.
        area_maxima = max(areas) if areas else 0  # Area del mayor contorno.
        relacion_area = area_maxima / area_total if area_total > 0 else 0  # Relación con el total.

        calidad = porcentaje * (1 + relacion_area)  # Heurística para ordenar rangos.

        mejores_resultados.append({'rango': rango, 'porcentaje': porcentaje, 'calidad': calidad, 'mask': mask})

    mejores_resultados.sort(key=lambda x: x['calidad'], reverse=True)  # Orden descendente por calidad.
    return mejores_resultados  # Lista con el mejor rango al inicio.

def validar_placa_mejorada(texto):
    try:
        texto_limpio = re.sub(r'[^A-Z0-9]', '', texto.upper())  # Eliminar caracteres no alfanuméricos y pasar a mayúsculas.

        if len(texto_limpio) < 5 or len(texto_limpio) > 7:
            return None  # Longitud fuera de los esperados: descartar.

        if not re.search(r'[A-Z]', texto_limpio) or not re.search(r'[0-9]', texto_limpio):
            return None  # Debe contener letras y números.

        for i in range(len(texto_limpio) - 5):  # Recorre subcadenas de 6 caracteres.
            subcadena = texto_limpio[i:i + 6]
            if (re.match(r'^[A-Z]{3}[0-9]{3}$', subcadena) or
                    re.match(r'^[A-Z]{2}[0-9]{4}$', subcadena) or
                    re.match(r'^[A-Z]{1}[0-9]{5}$', subcadena) or
                    re.match(r'^[0-9]{3}[A-Z]{3}$', subcadena) or
                    re.match(r'^[0-9]{2}[A-Z]{4}$', subcadena) or
                    re.match(r'^[0-9]{1}[A-Z]{5}$', subcadena)):
                return subcadena  # Si alguna subcadena coincide con un patrón válido, retornarla.

        return None  # Si no se encuentra patrón válido, retornar None.
    except Exception:
        return None  # En caso de exception, devolver None (silencioso).

def mejorar_contraste_texto(imagen):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # Kernel pequeño para apertura.
    imagen_mejorada = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)  # Apertura para eliminar pequeños ruidos.
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # Kernel para cierre (muy pequeño).
    imagen_mejorada = cv2.morphologyEx(imagen_mejorada, cv2.MORPH_CLOSE, kernel_cierre)  # Cierre para unir trazos finos.
    return imagen_mejorada  # Retorna imagen lista para OCR.

# ---------------------- CÁMARA ----------------------
def init_camera(preferences=CAMERA_PREFERENCES, width=640, height=480, fps=15):
    cap = None
    for idx in preferences:
        cap = cv2.VideoCapture(idx)  # Intenta abrir el índice de cámara.
        if cap.isOpened():
            logging.info('Cámara abierta: %s', idx)  # Log si se abre correctamente.
            break
        else:
            logging.warning('No se pudo abrir cámara %s', idx)  # Warning si no abre.
    if cap is None or not cap.isOpened():
        raise RuntimeError('No se pudo abrir ninguna cámara')  # Error si no se pudo abrir ninguna.

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # Ajustar ancho.
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # Ajustar alto.
    cap.set(cv2.CAP_PROP_FPS, fps)  # Intentar ajustar FPS.
    return cap  # Retorna objeto VideoCapture.

# ---------------------- BUCLE PRINCIPAL ----------------------
def ejecutar_sistema():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')  # Configura logging básico.

    model, reader = cargar_modelos()  # Carga YOLO y OCR.
    cap = init_camera()  # Inicializa cámara según preferencias.

    placas_activas = {}  # Detecciones actualmente en frame (tracking simple).
    placas_detectadas = {}  # Historial de placas detectadas (para evitar repeticiones).
    estadisticas_rangos = {}  # Contador de qué rango HSV fue más efectivo.

    frame_count = 0  # Contador de frames procesados.
    start_time = time.time()  # Marca de inicio para cálculo de FPS.

    try:
        while True:  # Bucle principal infinito (se detiene con 'q' o KeyboardInterrupt).
            ret, frame = cap.read()  # Lee un frame.
            if not ret:
                time.sleep(0.1)  # Si falla la lectura, esperar y reintentar.
                continue

            frame_count += 1  # Incrementar contador de frames.
            frame_display = frame.copy()  # Copia para dibujar información sin alterar el original.
            tiempo_actual = time.time()  # Timestamp actual.

            # Detección cada FRAME_SKIP frames
            if frame_count % FRAME_SKIP == 0:
                try:
                    results = model(frame, verbose=False, conf=CONF_THRESHOLD)  # Ejecutar YOLO en el frame.

                    for result in results:
                        if result.boxes is None:
                            continue  # Si no hay cajas, saltar.
                        for box in result.boxes:
                            try:
                                conf = float(box.conf[0])  # Intentar leer confianza (tensor).
                            except Exception:
                                conf = float(box.conf)  # Fallback si la estructura difiere.

                            if conf < CONF_THRESHOLD:
                                continue  # Si la confianza es baja, ignorar.

                            coords = box.xyxy[0].cpu().numpy()  # Extraer coordenadas (x1,y1,x2,y2).
                            x1, y1, x2, y2 = map(int, coords)  # Convertir a enteros.

                            height, width = frame.shape[:2]  # Dimensiones del frame.
                            x1, y1 = max(0, x1), max(0, y1)  # Clampear a bordes.
                            x2, y2 = min(width, x2), min(height, y2)

                            if x2 <= x1 or y2 <= y1:
                                continue  # Caja inválida (sin área).

                            roi = frame[y1:y2, x1:x2]  # Recortar la ROI que contiene el objeto.
                            if roi.size == 0:
                                continue  # Si el recorte está vacío, ignorar.

                            resultados_color = detectar_color_placas(roi)  # Detectar amarillo en la ROI.
                            if not resultados_color or resultados_color[0]['porcentaje'] <= COLOR_PROCESS_THRESHOLD:
                                continue  # Si no hay suficiente amarillo, no procesar OCR.

                            mejor_color = resultados_color[0]  # Tomar el mejor rango detectado.

                            tecnicas = preprocesar_placa_mejorado(roi)  # Generar variantes para OCR.
                            if not tecnicas:
                                continue  # Si no hay técnicas, ignorar.

                            # Intentar OCR con cada técnica hasta encontrar placa válida
                            encontrada = False
                            for nombre_tecnica, img_procesada in tecnicas:
                                img_mejorada = mejorar_contraste_texto(img_procesada)  # Limpieza morfológica.
                                try:
                                    ocr_results = reader.readtext(img_mejorada, detail=1, paragraph=False)  # OCR.
                                except Exception:
                                    ocr_results = []  # En caso de fallo OCR, continuar con siguiente técnica.

                                for res in ocr_results:
                                    # easyocr devuelve [bbox, texto, confianza]
                                    try:
                                        texto = res[1]  # Texto detectado.
                                        confianza = res[2]  # Confianza asociada.
                                    except Exception:
                                        continue  # Resultado en formato inesperado.

                                    placa = validar_placa_mejorada(texto)  # Validar si el texto parece placa.
                                    if placa and confianza > OCR_CONF_THRESHOLD:
                                        tiempo_ultima = tiempo_actual  # Timestamp de aceptación.
                                        if placa in placas_activas:
                                            placas_activas[placa].actualizar((x1, y1, x2, y2), tiempo_ultima,
                                                                             mejor_color['rango'])  # Actualizar tracking.
                                        else:
                                            if placa in placas_detectadas:
                                                ultima = placas_detectadas[placa].tiempo_ultima_deteccion
                                                if tiempo_actual - ultima < TIEMPO_ESPERA_REPETICION:
                                                    continue  # Evitar recontar placa recientemente vista.

                                            deteccion = DeteccionPlaca(placa, (x1, y1, x2, y2), tiempo_ultima,
                                                                        tiempo_ultima)  # Nueva detección.
                                            deteccion.mejor_rango = mejor_color['rango']  # Guardar rango usado.
                                            placas_activas[placa] = deteccion  # Añadir a activas.
                                            placas_detectadas[placa] = deteccion  # Añadir a historial.

                                            fecha_hora = datetime.datetime.fromtimestamp(tiempo_ultima).strftime('%d/%m/%Y %H:%M:%S')
                                            logging.info('PLACA DETECTADA: %s | %s | Rango: %s | Técnica: %s | Conf: %.2f',
                                                         placa, fecha_hora, mejor_color['rango']['nombre'], nombre_tecnica, confianza)
                                            # Log informativo con datos de la detección.

                                            estadisticas_rangos[mejor_color['rango']['nombre']] = estadisticas_rangos.get(mejor_color['rango']['nombre'], 0) + 1
                                            encontrada = True  # Marcar que ya se detectó correctamente.
                                            break
                                if encontrada:
                                    break  # Salir del loop de técnicas si ya se encontró una placa válida.

                except Exception:
                    logging.exception('Error durante la detección YOLO/OCR')  # Captura cualquier error en la etapa de detección.

            # Seguimiento y visualización
            placas_a_remover = []  # Lista temporal de placas que ya no deben trackearse.
            for placa, deteccion in list(placas_activas.items()):
                x1, y1, x2, y2 = deteccion.bbox  # Recuperar bbox del tracking.
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    rango_verificacion = deteccion.mejor_rango if deteccion.mejor_rango else RANGOS_AMARILLO[1]  # Rango usado para verificar.
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # Convertir ROI a HSV.
                    mask = cv2.inRange(hsv, rango_verificacion['bajo'], rango_verificacion['alto'])  # Mascara del rango.
                    area = roi.shape[0] * roi.shape[1]
                    if area > 0:
                        pct_amarillo = (cv2.countNonZero(mask) / area) * 100  # % amarillo en ROI.
                        if pct_amarillo > TRACKING_COLOR_THRESHOLD:
                            deteccion.tiempo_ultima_deteccion = time.time()  # Actualizar último tiempo visto.

                            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Dibujar bbox.
                            cv2.putText(frame_display, f"PLACA: {placa}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 255, 0), 2)  # Escribir placa.
                            fecha_deteccion = datetime.datetime.fromtimestamp(deteccion.tiempo_primer_deteccion).strftime('%d/%m/%Y %H:%M:%S')
                            cv2.putText(frame_display, f"Entrada: {fecha_deteccion}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (255, 255, 255), 1)  # Mostrar fecha de entrada.
                            tiempo_en_cuadro = int(time.time() - deteccion.tiempo_primer_deteccion)
                            cv2.putText(frame_display, f"Tiempo: {tiempo_en_cuadro}s", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        (255, 255, 255), 1)  # Mostrar tiempo en cuadro.
                            continue  # Mantener tracking para esta placa si aún hay color.

                placas_a_remover.append(placa)  # Si no se cumple la condición, marcar para remover.

            for placa in placas_a_remover:
                placas_activas.pop(placa, None)  # Remover placas no vistas.

            elapsed = time.time() - start_time  # Tiempo desde inicio.
            fps = frame_count / elapsed if elapsed > 0 else 0  # Calcular FPS aproximado.

            info_rangos = 'Rangos: '  # String con top rangos usados.
            for nombre, count in list(estadisticas_rangos.items())[:3]:
                info_rangos += f"{nombre}({count}) "  # Agregar top 3 de rangos.

            cv2.putText(frame_display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Mostrar FPS.
            cv2.putText(frame_display, f"Activas: {len(placas_activas)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)  # Mostrar cantidad de placas activas.
            cv2.putText(frame_display, info_rangos, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)  # Mostrar rangos.

            cv2.imshow('Sistema de Placas - Adaptativo a Iluminación', frame_display)  # Ventana de visualización.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Salir si se presiona 'q'.

    except KeyboardInterrupt:
        logging.info('Programa interrumpido por usuario')  # Manejo de interrupción manual.
    finally:
        cap.release()  # Liberar cámara.
        cv2.destroyAllWindows()  # Cerrar ventanas OpenCV.

        logging.info('ESTADÍSTICAS FINALES:')
        logging.info('Placas únicas detectadas: %d', len(placas_detectadas))  # Log cantidad de placas únicas.
        for nombre, count in sorted(estadisticas_rangos.items(), key=lambda x: x[1], reverse=True):
            logging.info('  %s: %d', nombre, count)  # Log top rangos por uso.

if __name__ == '__main__':
    try:
        ejecutar_sistema()  # Ejecuta la aplicación principal.
    except Exception as e:
        logging.exception('Error crítico en ejecución: %s', e)  # Log de error crítico.
        sys.exit(1)  # Salida con código de error.

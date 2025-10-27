import cv2
import easyocr
import numpy as np
import re
import time
import datetime
from ultralytics import YOLO

# --- imports para BD y worker ---
import sqlite3
import os
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event

# ===== Parámetros globales mejorados =====
TIEMPO_ESPERA_REPETICION = 60  # 1 minuto en segundos

# MÚLTIPLES RANGOS HSV PARA DIFERENTES CONDICIONES DE ILUMINACIÓN
rangos_amarillo = [
    {'bajo': np.array([15, 50, 150]), 'alto': np.array([35, 255, 255]), 'nombre': 'brillante'},
    {'bajo': np.array([15, 80, 80]),  'alto': np.array([35, 255, 200]), 'nombre': 'estandar'},
    {'bajo': np.array([15, 60, 40]),  'alto': np.array([35, 255, 150]), 'nombre': 'oscuro'},
    {'bajo': np.array([20, 40, 180]), 'alto': np.array([30, 150, 255]), 'nombre': 'claro'}
]

# ===== Estructura para almacenar información de placas =====
class DeteccionPlaca:
    def __init__(self, placa, bbox, tiempo_deteccion):
        self.placa = placa
        self.bbox = bbox
        self.tiempo_primer_deteccion = tiempo_deteccion
        self.tiempo_ultima_deteccion = tiempo_deteccion
        self.activa = True
        self.mejor_rango = None  # Guardar el rango que mejor funcionó

    def actualizar(self, bbox, tiempo_actual, rango=None):
        self.bbox = bbox
        self.tiempo_ultima_deteccion = tiempo_actual
        if rango:
            self.mejor_rango = rango

# ===== CARGA MODELOS =====
print("Cargando modelos...")

# Cargar YOLO
try:
    model = YOLO('yolov8n.pt')
    print("✅ Modelo YOLO cargado correctamente")
except Exception as e:
    print(f"❌ Error cargando YOLO: {e}")
    exit()

# Configurar OCR
try:
    reader = easyocr.Reader(['en'], gpu=False)
    print("✅ OCR configurado correctamente")
except Exception as e:
    print(f"❌ Error configurando OCR: {e}")
    exit()

# ====== CONFIGURACIÓN SQLITE + WORKER (INTEGRADO) ======
DB_FOLDER = r"C:\placas\reconocimientoDePlacas\version10Placas\database"
DB_PATH = Path(DB_FOLDER) / "placas.sqlite"

# Crear carpeta si no existe
os.makedirs(DB_FOLDER, exist_ok=True)

# Mostrar rutas (útil para depurar)
print("DB_FOLDER:", DB_FOLDER)
print("DB_PATH (relativo):", DB_PATH)
print("DB_PATH (absoluto):", DB_PATH.resolve())

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detecciones_placas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detection_ts TEXT NOT NULL,   -- 'YYYY-MM-DD HH:MM:SS'
    plate TEXT NOT NULL,
    rango TEXT,
    tecnica TEXT
);
"""
PRAGMAS_AND_INDEXES = [
    "PRAGMA journal_mode = WAL;",
    "PRAGMA synchronous = NORMAL;",
    "PRAGMA foreign_keys = ON;",
    "CREATE INDEX IF NOT EXISTS idx_plate ON detecciones_placas(plate);",
    "CREATE INDEX IF NOT EXISTS idx_detection_ts ON detecciones_placas(detection_ts);"
]

write_queue = Queue(maxsize=2000)
_stop_event = Event()
_worker_thread = None
BATCH_SIZE = 20
FLUSH_INTERVAL = 1.0  # segundos

def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()
    try:
        cur.execute(CREATE_TABLE_SQL)
        for stmt in PRAGMAS_AND_INDEXES:
            try:
                cur.execute(stmt)
            except Exception:
                pass
        conn.commit()
    finally:
        conn.close()
    print("✅ Base de datos inicializada:", DB_PATH)

def _worker_loop():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=True)
    try:
        conn.execute("PRAGMA busy_timeout = 5000;")
    except Exception:
        pass
    cur = conn.cursor()
    buffer = []
    last_flush = time.time()

    while not _stop_event.is_set() or not write_queue.empty() or buffer:
        try:
            item = write_queue.get(timeout=0.25)
            buffer.append(item)
            write_queue.task_done()
        except Empty:
            item = None

        now = time.time()
        if (len(buffer) >= BATCH_SIZE) or (buffer and (now - last_flush) >= FLUSH_INTERVAL) or (_stop_event.is_set() and buffer):
            try:
                cur.executemany(
                    "INSERT INTO detecciones_placas (detection_ts, plate, rango, tecnica) VALUES (?, ?, ?, ?);",
                    buffer
                )
                conn.commit()
            except sqlite3.OperationalError as oe:
                print(f"[DB WORKER] OperationalError: {oe} - reintentando...")
                retry_count = 0
                while retry_count < 5:
                    try:
                        time.sleep(0.2 * (retry_count + 1))
                        cur.executemany(
                            "INSERT INTO detecciones_placas (detection_ts, plate, rango, tecnica) VALUES (?, ?, ?, ?);",
                            buffer
                        )
                        conn.commit()
                        break
                    except Exception:
                        retry_count += 1
                else:
                    print("[DB WORKER] Falló insertar batch tras reintentos. Buffer descartado.")
            except Exception as e:
                print(f"[DB WORKER] Error insertando batch: {e}")
            finally:
                buffer.clear()
                last_flush = time.time()
    try:
        conn.close()
    except Exception:
        pass

def start_worker():
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _stop_event.clear()
        _worker_thread = Thread(target=_worker_loop, daemon=True)
        _worker_thread.start()
        print("✅ Worker de DB iniciado.")

def stop_worker(wait_seconds=2.0):
    _stop_event.set()
    if _worker_thread is not None:
        _worker_thread.join(timeout=wait_seconds)
        print("✅ Worker de DB detenido.")

# ===== Funciones de procesamiento MEJORADAS =====
def preprocesar_placa_mejorado(imagen):
    if imagen.size == 0:
        return None
    try:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        resultados = []
        gris_eq = cv2.equalizeHist(gris)
        resultados.append(('eq_normal', gris_eq))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gris_clahe = clahe.apply(gris)
        resultados.append(('clahe', gris_clahe))
        gris_bilateral = cv2.bilateralFilter(gris, 5, 50, 50)
        resultados.append(('bilateral', gris_bilateral))
        gris_combinado = cv2.bilateralFilter(gris_clahe, 5, 50, 50)
        resultados.append(('combinado', gris_combinado))
        return resultados
    except Exception:
        return None

def detectar_color_placas(roi):
    mejores_resultados = []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    for rango in rangos_amarillo:
        mask = cv2.inRange(hsv, rango['bajo'], rango['alto'])
        area_total = roi.shape[0] * roi.shape[1]
        if area_total > 0:
            porcentaje = (cv2.countNonZero(mask) / area_total) * 100
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(cnt) for cnt in contours]
            area_maxima = max(areas) if areas else 0
            relacion_area = area_maxima / area_total if area_total > 0 else 0
            calidad = porcentaje * (1 + relacion_area)
            mejores_resultados.append({
                'rango': rango,
                'porcentaje': porcentaje,
                'calidad': calidad,
                'mask': mask
            })
    mejores_resultados.sort(key=lambda x: x['calidad'], reverse=True)
    return mejores_resultados

def validar_placa_mejorada(texto):
    try:
        texto_limpio = re.sub(r'[^A-Z0-9]', '', texto.upper())
        if len(texto_limpio) < 5 or len(texto_limpio) > 7:
            return None
        if not re.search(r'[A-Z]', texto_limpio) or not re.search(r'[0-9]', texto_limpio):
            return None
        for i in range(len(texto_limpio) - 5):
            subcadena = texto_limpio[i:i+6]
            if (re.match(r'^[A-Z]{3}[0-9]{3}$', subcadena) or
                re.match(r'^[A-Z]{2}[0-9]{4}$', subcadena) or
                re.match(r'^[A-Z]{1}[0-9]{5}$', subcadena) or
                re.match(r'^[0-9]{3}[A-Z]{3}$', subcadena) or
                re.match(r'^[0-9]{2}[A-Z]{4}$', subcadena) or
                re.match(r'^[0-9]{1}[A-Z]{5}$', subcadena)):
                return subcadena
        return None
    except Exception:
        return None

def mejorar_contraste_texto(imagen):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    imagen_mejorada = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    imagen_mejorada = cv2.morphologyEx(imagen_mejorada, cv2.MORPH_CLOSE, kernel_cierre)
    return imagen_mejorada

# ===== Captura de video =====
print("Inicializando cámara...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara 0. Intentando con cámara 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ No se pudo abrir ninguna cámara")
        exit()

# Configurar cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

print("🔍 Sistema de detección mejorado - Adaptativo a condiciones de luz")
print("💡 Usando múltiples rangos de color y técnicas de preprocesamiento")

# Inicializar DB y worker
init_db()
start_worker()

# Variables de estado
placas_activas = {}
placas_detectadas = {}
frame_count = 0
start_time = time.time()
estadisticas_rangos = {}  # Para aprender qué rangos funcionan mejor

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1
        frame_display = frame.copy()
        tiempo_actual = time.time()

        # ==== DETECCIÓN MEJORADA (cada 10 frames) ====
        if frame_count % 10 == 0:
            try:
                results = model(frame, verbose=False, conf=0.4)
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            if conf < 0.4:
                                continue

                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            height, width = frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width, x2), min(height, y2)
                            if x2 <= x1 or y2 <= y1:
                                continue

                            roi = frame[y1:y2, x1:x2]
                            if roi.size == 0:
                                continue

                            resultados_color = detectar_color_placas(roi)
                            if resultados_color and resultados_color[0]['porcentaje'] > 15:
                                mejor_color = resultados_color[0]
                                tecnicas_preprocesamiento = preprocesar_placa_mejorado(roi)
                                if tecnicas_preprocesamiento:
                                    for nombre_tecnica, img_procesada in tecnicas_preprocesamiento:
                                        img_mejorada = mejorar_contraste_texto(img_procesada)
                                        try:
                                            ocr_results = reader.readtext(img_mejorada, detail=1, paragraph=False)
                                            if ocr_results:
                                                for texto, confianza in [(res[1], res[2]) for res in ocr_results]:
                                                    placa = validar_placa_mejorada(texto)
                                                    if placa and confianza > 0.3:
                                                        # Verificar si es nueva detección
                                                        if placa in placas_activas:
                                                            placas_activas[placa].actualizar(
                                                                (x1, y1, x2, y2), tiempo_actual, mejor_color['rango']
                                                            )
                                                        else:
                                                            if placa in placas_detectadas:
                                                                ultima_deteccion = placas_detectadas[placa].tiempo_ultima_deteccion
                                                                if tiempo_actual - ultima_deteccion < TIEMPO_ESPERA_REPETICION:
                                                                    continue

                                                            # NUEVA DETECCIÓN EXITOSA
                                                            deteccion = DeteccionPlaca(placa, (x1, y1, x2, y2), tiempo_actual)
                                                            deteccion.mejor_rango = mejor_color['rango']
                                                            placas_activas[placa] = deteccion
                                                            placas_detectadas[placa] = deteccion

                                                            fecha_hora = datetime.datetime.fromtimestamp(tiempo_actual).strftime('%d/%m/%Y %H:%M:%S')
                                                            print(f"✅ PLACA DETECTADA: {placa}")
                                                            print(f"   📅 {fecha_hora}")
                                                            print(f"   🎨 Rango: {mejor_color['rango']['nombre']}")
                                                            print(f"   🔧 Técnica: {nombre_tecnica}")
                                                            print(f"   📊 Confianza: {confianza:.2f}")

                                                            # Registrar estadísticas del rango exitoso
                                                            if mejor_color['rango']['nombre'] in estadisticas_rangos:
                                                                estadisticas_rangos[mejor_color['rango']['nombre']] += 1
                                                            else:
                                                                estadisticas_rangos[mejor_color['rango']['nombre']] = 1

                                                            # ----- ENCOLAR PARA GUARDAR EN BD -----
                                                            try:
                                                                timestamp_iso = datetime.datetime.fromtimestamp(tiempo_actual).strftime('%Y-%m-%d %H:%M:%S')
                                                                rango_nombre = mejor_color['rango']['nombre'] if mejor_color and 'rango' in mejor_color else None
                                                                tecnica_nombre = nombre_tecnica
                                                                write_queue.put_nowait((timestamp_iso, placa, rango_nombre, tecnica_nombre))
                                                            except Exception:
                                                                print("[WARNING] Cola de escritura llena - detección no guardada")

                                                            break  # Salir después de primera detección exitosa
                                        except Exception:
                                            continue
            except Exception:
                pass

        # ==== SEGUIMIENTO Y VISUALIZACIÓN ====
        placas_a_remover = []

        for placa, deteccion in placas_activas.items():
            x1, y1, x2, y2 = deteccion.bbox

            # Verificar si la placa sigue visible
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                rango_verificacion = deteccion.mejor_rango if deteccion.mejor_rango else rangos_amarillo[1]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, rango_verificacion['bajo'], rango_verificacion['alto'])

                area = roi.shape[0] * roi.shape[1]
                if area > 0:
                    pct_amarillo = (cv2.countNonZero(mask) / area) * 100

                    if pct_amarillo > 10:
                        deteccion.tiempo_ultima_deteccion = tiempo_actual

                        cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame_display, f"PLACA: {placa}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        fecha_deteccion = datetime.datetime.fromtimestamp(
                            deteccion.tiempo_primer_deteccion
                        ).strftime('%d/%m/%Y %H:%M:%S')

                        cv2.putText(frame_display, f"Entrada: {fecha_deteccion}",
                                   (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                        tiempo_en_cuadro = int(tiempo_actual - deteccion.tiempo_primer_deteccion)
                        cv2.putText(frame_display, f"Tiempo: {tiempo_en_cuadro}s",
                                   (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        continue

            placas_a_remover.append(placa)

        # Remover placas que ya no están en cuadro
        for placa in placas_a_remover:
            if placa in placas_activas:
                del placas_activas[placa]

        # ==== INFORMACIÓN EN PANTALLA MEJORADA ====
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        info_rangos = "Rangos: "
        for nombre, count in list(estadisticas_rangos.items())[:3]:
            info_rangos += f"{nombre}({count}) "

        cv2.putText(frame_display, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Activas: {len(placas_activas)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_display, info_rangos, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.imshow("Sistema de Placas - Adaptativo a Iluminación", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⏹️ Programa interrumpido")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()

    # detener worker y esperar que vacíe la cola
    stop_worker()

    print(f"\n📊 ESTADÍSTICAS FINALES:")
    print(f"Placas únicas detectadas: {len(placas_detectadas)}")
    print("Rangos más efectivos:")
    for nombre, count in sorted(estadisticas_rangos.items(), key=lambda x: x[1], reverse=True):
        print(f"  {nombre}: {count} detecciones")

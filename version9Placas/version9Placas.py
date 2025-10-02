import cv2
import easyocr
import numpy as np
import re
import time
import datetime
from ultralytics import YOLO

# ===== Parámetros globales =====
UMBRAL_AMARILLO = 25
MAX_FRAMES_SIN_PLACA = 30
TIEMPO_ESPERA_REPETICION = 60  # 1 minuto en segundos

# Rangos HSV para amarillo
amarillo_bajo = np.array([15, 50, 50])
amarillo_alto = np.array([35, 255, 255])

# ===== Estructura para almacenar información de placas =====
class DeteccionPlaca:
    def __init__(self, placa, bbox, tiempo_deteccion):
        self.placa = placa
        self.bbox = bbox
        self.tiempo_primer_deteccion = tiempo_deteccion
        self.tiempo_ultima_deteccion = tiempo_deteccion
        self.activa = True
    
    def actualizar(self, bbox, tiempo_actual):
        self.bbox = bbox
        self.tiempo_ultima_deteccion = tiempo_actual
    
    def puede_detectar_nuevamente(self, tiempo_actual):
        """Verifica si ha pasado el tiempo mínimo para nueva detección"""
        tiempo_transcurrido = tiempo_actual - self.tiempo_ultima_deteccion
        return tiempo_transcurrido > TIEMPO_ESPERA_REPETICION

# ===== Cargar modelos =====
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

# ===== Funciones de procesamiento =====
def preprocesar_placa(imagen):
    """Preprocesa la imagen para mejorar OCR"""
    if imagen.size == 0:
        return None
        
    try:
        # Convertir a grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Mejorar contraste
        gris = cv2.equalizeHist(gris)
        
        # Reducir ruido
        gris = cv2.medianBlur(gris, 3)
        
        # Umbral adaptativo
        binaria = cv2.adaptiveThreshold(
            gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binaria
    except Exception as e:
        return None

def validar_placa(texto):
    """Valida si el texto es una placa colombiana válida"""
    try:
        # Limpiar texto
        texto_limpio = re.sub(r'[^A-Z0-9]', '', texto.upper())
        
        # Debe tener exactamente 6 caracteres
        if len(texto_limpio) != 6:
            return None
        
        # Debe contener al menos una letra y al menos un número
        if not re.search(r'[A-Z]', texto_limpio) or not re.search(r'[0-9]', texto_limpio):
            return None
        
        return texto_limpio
    except Exception as e:
        return None

def filtrar_texto_ocr(resultados_ocr):
    """Filtra resultados OCR para encontrar la placa más probable"""
    try:
        for texto, confianza in resultados_ocr:
            placa = validar_placa(texto)
            if placa:
                return placa
                
            # Buscar subcadenas de 6 caracteres
            texto_limpio = re.sub(r'[^A-Z0-9]', '', texto.upper())
            for i in range(len(texto_limpio) - 5):
                subcadena = texto_limpio[i:i+6]
                placa = validar_placa(subcadena)
                if placa:
                    return placa
                    
        return None
    except Exception as e:
        return None

def formatear_fecha_hora(tiempo_unix):
    """Formatea fecha y hora para mostrar"""
    return datetime.datetime.fromtimestamp(tiempo_unix).strftime('%d/%m/%Y %H:%M:%S')

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
cap.set(cv2.CAP_PROP_FPS, 30)

print("🔍 Buscando placa... (presiona 'q' para salir)")
print("💡 Las placas detectadas no se repetirán por 1 minuto")

# Variables de estado
placas_activas = {}  # Diccionario de placas actualmente en cuadro
placas_detectadas = {}  # Historial completo de detecciones
frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al leer frame de la cámara")
            time.sleep(0.1)
            continue
            
        frame_count += 1
        frame_display = frame.copy()
        tiempo_actual = time.time()
        
        # ==== DETECCIÓN DE NUEVAS PLACAS (cada 15 frames) ====
        if frame_count % 15 == 0:
            try:
                results = model(frame, verbose=False, conf=0.5)
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            if conf < 0.5:
                                continue
                                
                            # Obtener coordenadas
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            
                            # Validar coordenadas
                            height, width = frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width, x2), min(height, y2)
                            
                            if x2 <= x1 or y2 <= y1:
                                continue
                                
                            roi = frame[y1:y2, x1:x2]
                            if roi.size == 0:
                                continue
                            
                            # Verificar color amarillo
                            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                            mask = cv2.inRange(hsv, amarillo_bajo, amarillo_alto)
                            area = roi.shape[0] * roi.shape[1]
                            if area == 0:
                                continue
                                
                            pct_amarillo = (cv2.countNonZero(mask) / area) * 100
                            
                            if pct_amarillo < UMBRAL_AMARILLO:
                                continue
                            
                            # Procesar OCR
                            roi_procesado = preprocesar_placa(roi)
                            if roi_procesado is None:
                                continue
                                
                            try:
                                ocr_results = reader.readtext(roi_procesado, detail=1, paragraph=False)
                                if ocr_results:
                                    textos_confianzas = [(res[1], res[2]) for res in ocr_results]
                                    placa = filtrar_texto_ocr(textos_confianzas)
                                    
                                    if placa:
                                        # Verificar si es una nueva detección o una placa existente
                                        if placa in placas_activas:
                                            # Actualizar placa existente
                                            placas_activas[placa].actualizar((x1, y1, x2, y2), tiempo_actual)
                                        else:
                                            # Verificar si ya fue detectada recientemente
                                            if placa in placas_detectadas:
                                                ultima_deteccion = placas_detectadas[placa].tiempo_ultima_deteccion
                                                if tiempo_actual - ultima_deteccion < TIEMPO_ESPERA_REPETICION:
                                                    print(f"⏰ Placa {placa} ignorada (detección reciente)")
                                                    continue
                                            
                                            # Nueva detección
                                            deteccion = DeteccionPlaca(placa, (x1, y1, x2, y2), tiempo_actual)
                                            placas_activas[placa] = deteccion
                                            placas_detectadas[placa] = deteccion
                                            
                                            fecha_hora = formatear_fecha_hora(tiempo_actual)
                                            print(f"✅ NUEVA PLACA: {placa} - {fecha_hora}")
                            except Exception as e:
                                continue
            except Exception as e:
                print(f"⚠️ Error en detección: {e}")
        
        # ==== SEGUIMIENTO Y DIBUJADO DE PLACAS ACTIVAS ====
        placas_a_remover = []
        
        for placa, deteccion in placas_activas.items():
            x1, y1, x2, y2 = deteccion.bbox
            
            # Verificar si la placa sigue en cuadro
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, amarillo_bajo, amarillo_alto)
                area = roi.shape[0] * roi.shape[1]
                if area > 0:
                    pct_amarillo = (cv2.countNonZero(mask) / area) * 100
                    
                    if pct_amarillo > 15:
                        # Placa todavía visible - actualizar tiempo
                        deteccion.tiempo_ultima_deteccion = tiempo_actual
                        
                        # Dibujar bounding box e información
                        cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        
                        # Texto de la placa
                        cv2.putText(frame_display, f"PLACA: {placa}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Fecha y hora de primera detección
                        fecha_deteccion = formatear_fecha_hora(deteccion.tiempo_primer_deteccion)
                        cv2.putText(frame_display, f"Entrada: {fecha_deteccion}", 
                                   (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # Tiempo en cuadro
                        tiempo_en_cuadro = int(tiempo_actual - deteccion.tiempo_primer_deteccion)
                        cv2.putText(frame_display, f"Tiempo: {tiempo_en_cuadro}s", 
                                   (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        continue
            
            # Si llegamos aquí, la placa ya no está visible
            placas_a_remover.append(placa)
        
        # Remover placas que ya no están en cuadro
        for placa in placas_a_remover:
            if placa in placas_activas:
                tiempo_en_cuadro = tiempo_actual - placas_activas[placa].tiempo_primer_deteccion
                print(f"🚗 Placa {placa} salió del cuadro después de {int(tiempo_en_cuadro)} segundos")
                del placas_activas[placa]
        
        # ==== MOSTRAR INFORMACIÓN EN PANTALLA ====
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Información general
        cv2.putText(frame_display, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Placas activas: {len(placas_activas)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Total detectadas: {len(placas_detectadas)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Tiempo de espera configurado
        cv2.putText(frame_display, f"Espera: {TIEMPO_ESPERA_REPETICION}s", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow("Detección de Placas Colombianas - Sistema de Parqueadero", frame_display)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⏹️ Programa interrumpido por el usuario")
except Exception as e:
    print(f"❌ Error crítico: {e}")
finally:
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

    # Mostrar resumen final
    print("\n📊 RESUMEN FINAL DE DETECCIONES:")
    if placas_detectadas:
        for placa, deteccion in sorted(placas_detectadas.items()):
            fecha = formatear_fecha_hora(deteccion.tiempo_primer_deteccion)
            print(f"  {placa} - Primera detección: {fecha}")
    else:
        print("  No se detectaron placas")

    print(f"\n✅ Programa terminado. Total de placas únicas: {len(placas_detectadas)}")
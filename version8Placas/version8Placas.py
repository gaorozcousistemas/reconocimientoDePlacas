import cv2
import easyocr
import numpy as np
import re
import time
from ultralytics import YOLO

# ===== Parámetros globales =====
UMBRAL_AMARILLO = 25
MAX_FRAMES_SIN_PLACA = 30

# Rangos HSV para amarillo
amarillo_bajo = np.array([15, 50, 50])
amarillo_alto = np.array([35, 255, 255])

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
        print(f"⚠️ Error en preprocesamiento: {e}")
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
        print(f"⚠️ Error en validación de placa: {e}")
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
        print(f"⚠️ Error en filtrado OCR: {e}")
        return None

# ===== Captura de video =====
print("Inicializando cámara...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara 0. Intentando con cámara 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ No se pudo abrir ninguna cámara")
        print("💡 Verifica que la cámara esté conectada y disponible")
        exit()

# Configurar cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

print("🔍 Buscando placa... (presiona 'q' para salir)")
print("💡 Asegúrate de que la placa esté bien iluminada y visible")

# Variables de estado
placa_actual = None
bbox_actual = None
frames_sin_placa = 0
frame_count = 0
start_time = time.time()
historial_placas = {}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al leer frame de la cámara")
            time.sleep(0.1)
            continue
            
        frame_count += 1
        frame_display = frame.copy()
        
        # Procesar detección cada 15 frames para mejor rendimiento
        if frame_count % 15 == 0 or placa_actual is None:
            try:
                # Detección con YOLO
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
                                        # Actualizar historial
                                        historial_placas[placa] = historial_placas.get(placa, 0) + 1
                                        
                                        if historial_placas[placa] >= 1:
                                            placa_actual = placa
                                            bbox_actual = (x1, y1, x2, y2)
                                            frames_sin_placa = 0
                                            print(f"✅ Placa detectada: {placa}")
                            except Exception as e:
                                print(f"⚠️ Error en OCR: {e}")
                                continue
            except Exception as e:
                print(f"⚠️ Error en detección: {e}")
        
        # Dibujar bounding box si hay placa detectada
        if placa_actual and bbox_actual:
            x1, y1, x2, y2 = bbox_actual
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame_display, placa_actual, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Tracking simple
            frames_sin_placa += 1
            if frames_sin_placa > MAX_FRAMES_SIN_PLACA:
                print(f"ℹ️ Placa {placa_actual} perdida")
                placa_actual = None
                bbox_actual = None
        
        # Mostrar información
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        cv2.putText(frame_display, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if placa_actual:
            cv2.putText(frame_display, f"Placa: {placa_actual}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame_display, "Buscando placa...", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Detección de Placas Colombianas", frame_display)
        
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

    # Mostrar resumen
    print("\n📊 Resumen de detecciones:")
    if historial_placas:
        for placa, count in sorted(historial_placas.items(), key=lambda x: x[1], reverse=True):
            print(f"  {placa}: {count} detecciones")
    else:
        print("  No se detectaron placas")

    print("✅ Programa terminado")
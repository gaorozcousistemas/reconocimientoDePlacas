import cv2
import easyocr
import numpy as np
import re
import time
import datetime
from ultralytics import YOLO

# ===== Parámetros para fuente FE de alta seguridad =====
TIEMPO_ESPERA_REPETICION = 120

# Rangos HSV optimizados para placas colombianas oficiales
rangos_amarillo = [
    {'bajo': np.array([20, 100, 100]), 'alto': np.array([30, 255, 255])},  # Amarillo oficial
]

# Dimensiones conocidas de la fuente FE (proporciones)
PROPORCION_FE_AUTOS = (70/32)  # 70mm x 32mm = ~2.18 relación aspecto
PROPORCION_FE_MOTOS = (47/21.5)  # ~2.18 (similar)

# ===== Estructura para placas =====
class DeteccionPlaca:
    def __init__(self, placa, bbox, tiempo_deteccion, tipo="auto"):
        self.placa = placa
        self.bbox = bbox
        self.tiempo_primer_deteccion = tiempo_deteccion
        self.tiempo_ultima_deteccion = tiempo_deteccion
        self.tipo = tipo  # "auto" o "moto"

# ===== Cargar modelos =====
print("Cargando modelos para fuente FE...")

try:
    model = YOLO('yolov8n.pt')
    print("✅ Modelo YOLO cargado")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

try:
    reader = easyocr.Reader(['en'], gpu=False)
    print("✅ OCR configurado")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ===== Funciones ESPECIALIZADAS para fuente FE =====
def preprocesar_fuente_fe(imagen):
    """Preprocesamiento específico para fuente FE con alto relieve"""
    if imagen.size == 0:
        return None
    
    try:
        # Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # ENFATIZAR EL ALTO RELIEVE (sombras y luces)
        # 1. Realce de bordes para capturar el relieve
        edges = cv2.Canny(gris, 50, 150)
        
        # 2. Operación morfológica para unir caracteres del relieve
        kernel = np.ones((2,2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # 3. Combinar con imagen original para mantener textura
        gris_enfatizado = cv2.addWeighted(gris, 0.7, edges_dilated.astype(np.uint8) * 255, 0.3, 0)
        
        # 4. Alto contraste para el relieve
        gris_contrast = cv2.convertScaleAbs(gris_enfatizado, alpha=2.0, beta=0)
        
        # 5. Umbral adaptativo para manejar sombras del relieve
        thresh = cv2.adaptiveThreshold(
            gris_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 8
        )
        
        return thresh
        
    except Exception as e:
        return None

def verificar_proporciones_fe(roi):
    """Verifica si las dimensiones coinciden con fuente FE"""
    try:
        height, width = roi.shape[:2]
        if height == 0 or width == 0:
            return False, "auto"
        
        relacion_aspecto = width / height
        
        # Las placas con fuente FE tienen relación aspecto específica
        if 1.8 <= relacion_aspecto <= 2.5:  # Rango para fuente FE
            # Determinar si es auto o moto por tamaño absoluto
            if width > 200:  # Placa de auto más ancha
                return True, "auto"
            else:
                return True, "moto"
        return False, "auto"
    except:
        return False, "auto"

def mejorar_contraste_relieve(imagen):
    """Técnica específica para alto relieve"""
    try:
        # Enfoque en las sombras creadas por el relieve
        lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Realce del canal L (luminancia) donde se ve el relieve
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l)
        
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    except:
        return imagen

def validar_formato_placa_colombiana(texto):
    """Validación estricta de formato colombiano con fuente FE"""
    try:
        texto_limpio = re.sub(r'[^A-Z0-9]', '', texto.upper())
        
        # Debe tener exactamente 6 caracteres
        if len(texto_limpio) != 6:
            return None
        
        # Formatos específicos de Colombia con fuente FE
        formatos_validos = [
            r'^[A-Z]{3}[0-9]{3}$',  # ABC123 (vehículos particulares)
            r'^[A-Z]{2}[0-9]{4}$',  # AB1234 (motos)
            r'^[A-Z]{1}[0-9]{5}$',  # A12345 (formatos especiales)
            r'^[0-9]{3}[A-Z]{3}$',  # 123ABC (público)
        ]
        
        for formato in formatos_validos:
            if re.match(formato, texto_limpio):
                return texto_limpio
        
        return None
    except:
        return None

# ===== Captura de video =====
print("Inicializando cámara...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ No se pudo abrir cámara")
        exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🔍 Sistema especializado para fuente FE de alta seguridad")
print("💡 Optimizado para alto relieve de placas colombianas")

# Variables de estado
placas_activas = {}
placas_detectadas = {}
frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame_count += 1
        frame_display = frame.copy()
        tiempo_actual = time.time()
        
        # ==== DETECCIÓN ESPECIALIZADA (cada 10 frames) ====
        if frame_count % 10 == 0:
            try:
                results = model(frame, verbose=False, conf=0.6)
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            if float(box.conf[0]) < 0.6:
                                continue
                                
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                            
                            if x2 <= x1 or y2 <= y1:
                                continue
                                
                            roi = frame[y1:y2, x1:x2]
                            if roi.size == 0:
                                continue
                            
                            # VERIFICACIÓN DE PROPORCIONES FUENTE FE
                            es_fe, tipo_vehiculo = verificar_proporciones_fe(roi)
                            if not es_fe:
                                continue
                            
                            # VERIFICACIÓN DE COLOR AMARILLO OFICIAL
                            try:
                                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                                mask = cv2.inRange(hsv, rangos_amarillo[0]['bajo'], rangos_amarillo[0]['alto'])
                                area = roi.shape[0] * roi.shape[1]
                                if area > 0 and (cv2.countNonZero(mask) / area) * 100 < 25:
                                    continue
                            except:
                                continue
                            
                            # PROCESAMIENTO ESPECIALIZADO PARA FUENTE FE
                            try:
                                # Mejorar contraste para el relieve
                                roi_mejorado = mejorar_contraste_relieve(roi)
                                
                                # Preprocesamiento específico para fuente FE
                                roi_procesado = preprocesar_fuente_fe(roi_mejorado)
                                
                                if roi_procesado is not None:
                                    # OCR con configuración optimizada para relieve
                                    ocr_results = reader.readtext(
                                        roi_procesado, 
                                        detail=1, 
                                        paragraph=False,
                                        text_threshold=0.4,  # Más bajo para relieve
                                        low_text=0.3
                                    )
                                    
                                    if ocr_results:
                                        for texto, confianza in [(res[1], res[2]) for res in ocr_results]:
                                            # Validación ESTRICTA del formato colombiano
                                            placa = validar_formato_placa_colombiana(texto)
                                            
                                            if placa and confianza > 0.5:
                                                if placa in placas_activas:
                                                    placas_activas[placa].tiempo_ultima_deteccion = tiempo_actual
                                                    placas_activas[placa].bbox = (x1, y1, x2, y2)
                                                else:
                                                    if placa in placas_detectadas:
                                                        if tiempo_actual - placas_detectadas[placa].tiempo_ultima_deteccion < TIEMPO_ESPERA_REPETICION:
                                                            continue
                                                    
                                                    # NUEVA DETECCIÓN CONFIRMADA
                                                    deteccion = DeteccionPlaca(placa, (x1, y1, x2, y2), tiempo_actual, tipo_vehiculo)
                                                    placas_activas[placa] = deteccion
                                                    placas_detectadas[placa] = deteccion
                                                    
                                                    fecha_hora = datetime.datetime.now().strftime('%H:%M:%S')
                                                    print(f"🚗 PLACA DETECTADA: {placa}")
                                                    print(f"   📋 Tipo: {tipo_vehiculo.upper()}")
                                                    print(f"   ⏰ Hora: {fecha_hora}")
                                                    print(f"   🔍 Confianza: {confianza:.2f}")
                                                    
                            except Exception as e:
                                continue
                                
            except Exception as e:
                pass
        
        # ==== VISUALIZACIÓN MEJORADA ====
        placas_a_remover = []
        
        for placa, deteccion in list(placas_activas.items()):
            x1, y1, x2, y2 = deteccion.bbox
            
            if tiempo_actual - deteccion.tiempo_ultima_deteccion > 5:
                placas_a_remover.append(placa)
            else:
                # Color según tipo de vehículo
                color = (0, 255, 0) if deteccion.tipo == "auto" else (255, 255, 0)  # Verde para autos, Amarillo para motos
                
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_display, f"{placa} ({deteccion.tipo})", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                tiempo_en_cuadro = int(tiempo_actual - deteccion.tiempo_primer_deteccion)
                cv2.putText(frame_display, f"{tiempo_en_cuadro}s", 
                           (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for placa in placas_a_remover:
            if placa in placas_activas:
                print(f"📤 {placa} salió del área")
                del placas_activas[placa]
        
        # ==== UI INFORMATIVA ====
        fps = frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        
        cv2.putText(frame_display, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Placas activas: {len(placas_activas)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_display, "Fuente FE - Alto Relieve", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow("Sistema Fuente FE - Placas Colombianas", frame_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⏹️ Sistema detenido")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n📊 Resumen: {len(placas_detectadas)} placas detectadas")
import cv2
import easyocr
import numpy as np
import re
import time
import datetime
from ultralytics import YOLO
+
# ===== Parámetros globales mejorados =====
TIEMPO_ESPERA_REPETICION = 60  # 1 minuto en segundos

# MÚLTIPLES RANGOS HSV PARA DIFERENTES CONDICIONES DE ILUMINACIÓN
rangos_amarillo = [
    # Amarillo brillante (sol directo)
    {
        'bajo': np.array([15, 50, 150]),
        'alto': np.array([35, 255, 255]),
        'nombre': 'brillante'
    },
    # Amarillo estándar (día nublado)
    {
        'bajo': np.array([15, 80, 80]),
        'alto': np.array([35, 255, 200]),
        'nombre': 'estandar'
    },
    # Amarillo oscuro (sombra/tarde)
    {
        'bajo': np.array([15, 60, 40]),
        'alto': np.array([35, 255, 150]),
        'nombre': 'oscuro'
    },
    # Amarillo muy claro (placa desgastada)
    {
        'bajo': np.array([20, 40, 180]),
        'alto': np.array([30, 150, 255]),
        'nombre': 'claro'
    }
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

# ===== Funciones de procesamiento MEJORADAS =====
def preprocesar_placa_mejorado(imagen):
    """Preprocesamiento avanzado para mejorar OCR en diferentes condiciones"""
    if imagen.size == 0:
        return None
        
    try:
        # Convertir a grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # MÚLTIPLES TÉCNICAS DE PREPROCESAMIENTO
        resultados = []
        
        # Técnica 1: Ecualización de histograma normal
        gris_eq = cv2.equalizeHist(gris)
        resultados.append(('eq_normal', gris_eq))
        
        # Técnica 2: CLAHE (mejor para variaciones locales de iluminación)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gris_clahe = clahe.apply(gris)
        resultados.append(('clahe', gris_clahe))
        
        # Técnica 3: Filtrado bilateral (preserva bordes)
        gris_bilateral = cv2.bilateralFilter(gris, 5, 50, 50)
        resultados.append(('bilateral', gris_bilateral))
        
        # Técnica 4: Combinación CLAHE + bilateral
        gris_combinado = cv2.bilateralFilter(gris_clahe, 5, 50, 50)
        resultados.append(('combinado', gris_combinado))
        
        return resultados
        
    except Exception as e:
        return None

def detectar_color_placas(roi):
    """Detecta amarillo usando múltiples rangos y selecciona el mejor"""
    mejores_resultados = []
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    for rango in rangos_amarillo:
        mask = cv2.inRange(hsv, rango['bajo'], rango['alto'])
        area_total = roi.shape[0] * roi.shape[1]
        
        if area_total > 0:
            porcentaje = (cv2.countNonZero(mask) / area_total) * 100
            
            # Calcular calidad de la máscara (menos ruido = mejor)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(cnt) for cnt in contours]
            area_maxima = max(areas) if areas else 0
            relacion_area = area_maxima / area_total if area_total > 0 else 0
            
            calidad = porcentaje * (1 + relacion_area)  # Penalizar máscaras ruidosas
            
            mejores_resultados.append({
                'rango': rango,
                'porcentaje': porcentaje,
                'calidad': calidad,
                'mask': mask
            })
    
    # Ordenar por calidad y devolver los mejores
    mejores_resultados.sort(key=lambda x: x['calidad'], reverse=True)
    return mejores_resultados

def validar_placa_mejorada(texto):
    """Validación más flexible de placas"""
    try:
        # Limpiar texto
        texto_limpio = re.sub(r'[^A-Z0-9]', '', texto.upper())
        
        # Permitir 5-7 caracteres por posibles errores OCR
        if len(texto_limpio) < 5 or len(texto_limpio) > 7:
            return None
        
        # Debe contener al menos una letra y al menos un número
        if not re.search(r'[A-Z]', texto_limpio) or not re.search(r'[0-9]', texto_limpio):
            return None
        
        # Buscar subcadena de 6 caracteres que cumpla los requisitos
        for i in range(len(texto_limpio) - 5):
            subcadena = texto_limpio[i:i+6]
            
            # Verificar formato de placa colombiana
            if (re.match(r'^[A-Z]{3}[0-9]{3}$', subcadena) or  # ABC123
                re.match(r'^[A-Z]{2}[0-9]{4}$', subcadena) or  # AB1234
                re.match(r'^[A-Z]{1}[0-9]{5}$', subcadena) or  # A12345
                re.match(r'^[0-9]{3}[A-Z]{3}$', subcadena) or  # 123ABC
                re.match(r'^[0-9]{2}[A-Z]{4}$', subcadena) or  # 12ABCD
                re.match(r'^[0-9]{1}[A-Z]{5}$', subcadena)):   # 1ABCDE
                return subcadena
        
        return None
    except Exception as e:
        return None

def mejorar_contraste_texto(imagen):
    """Enfocar el área de texto mediante operaciones morfológicas"""
    # Operación de apertura para enfocar texto
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    imagen_mejorada = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
    
    # Operación de cierre para unir caracteres rotos
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
                results = model(frame, verbose=False, conf=0.4)  # Menor confianza inicial
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            if conf < 0.4:  # Umbral más bajo para capturar más candidatos
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
                            
                            # DETECCIÓN DE COLOR MEJORADA
                            resultados_color = detectar_color_placas(roi)
                            
                            # Procesar solo si hay suficiente color amarillo
                            if resultados_color and resultados_color[0]['porcentaje'] > 15:
                                mejor_color = resultados_color[0]
                                
                                # PROCESAMIENTO OCR MEJORADO
                                tecnicas_preprocesamiento = preprocesar_placa_mejorado(roi)
                                
                                if tecnicas_preprocesamiento:
                                    for nombre_tecnica, img_procesada in tecnicas_preprocesamiento:
                                        # Aplicar mejora de contraste de texto
                                        img_mejorada = mejorar_contraste_texto(img_procesada)
                                        
                                        try:
                                            ocr_results = reader.readtext(img_mejorada, detail=1, paragraph=False)
                                            if ocr_results:
                                                for texto, confianza in [(res[1], res[2]) for res in ocr_results]:
                                                    placa = validar_placa_mejorada(texto)
                                                    
                                                    if placa and confianza > 0.3:  # Confianza mínima baja
                                                        # Verificar si es nueva detección
                                                        if placa in placas_activas:
                                                            placas_activas[placa].actualizar(
                                                                (x1, y1, x2, y2), tiempo_actual, mejor_color['rango']
                                                            )
                                                        else:
                                                            # Verificar tiempo de espera
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
                                                            
                                                            break  # Salir después de primera detección exitosa
                                        except Exception as e:
                                            continue
                                            
            except Exception as e:
                pass
        
        # ==== SEGUIMIENTO Y VISUALIZACIÓN ====
        placas_a_remover = []
        
        for placa, deteccion in placas_activas.items():
            x1, y1, x2, y2 = deteccion.bbox
            
            # Verificar si la placa sigue visible
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                # Usar el rango que funcionó mejor para esta placa
                rango_verificacion = deteccion.mejor_rango if deteccion.mejor_rango else rangos_amarillo[1]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, rango_verificacion['bajo'], rango_verificacion['alto'])
                
                area = roi.shape[0] * roi.shape[1]
                if area > 0:
                    pct_amarillo = (cv2.countNonZero(mask) / area) * 100
                    
                    if pct_amarillo > 10:  # Umbral más bajo para tracking
                        deteccion.tiempo_ultima_deteccion = tiempo_actual
                        
                        # DIBUJADO MEJORADO
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
        
        # Mostrar estadísticas de rangos que funcionan
        info_rangos = "Rangos: "
        for nombre, count in list(estadisticas_rangos.items())[:3]:  # Mostrar top 3
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
    
    print(f"\n📊 ESTADÍSTICAS FINALES:")
    print(f"Placas únicas detectadas: {len(placas_detectadas)}")
    print("Rangos más efectivos:")
    for nombre, count in sorted(estadisticas_rangos.items(), key=lambda x: x[1], reverse=True):
        print(f"  {nombre}: {count} detecciones")
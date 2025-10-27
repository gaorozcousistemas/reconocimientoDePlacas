import cv2
import time
import datetime

from config.configuracion import RANGOS_AMARILLO, TIEMPO_ESPERA_REPETICION
from models.modelos import cargarYolo, cargarOcr
from processing.procesamiento import preprocesarPlaca, detectarColorPlaca, validarPlaca, mejorarContrasteTexto
from database.baseDatos import iniciarBaseDatos, iniciarWorker, detenerWorker, colaEscritura
from utils.utilidades import formatearFechaHora, mostrarEstadisticasFinales

# ===== Inicializar modelos =====
print("Cargando modelos...")
modeloYolo = cargarYolo('yolov8n.pt')
lectorOcr = cargarOcr(['en'], gpu=False)

# ===== Inicializar Base de Datos y Worker =====
iniciarBaseDatos()
iniciarWorker()

# ===== Variables de estado =====
placasActivas = {}
placasDetectadas = {}
contadorFrames = 0
inicioTiempo = time.time()
estadisticasRangos = {}

# ===== Inicializar cámara =====
print("Inicializando cámara...")
camara = cv2.VideoCapture(0)
if not camara.isOpened():
    print("❌ No se pudo abrir la cámara 0. Intentando con cámara 1...")
    camara = cv2.VideoCapture(1)
    if not camara.isOpened():
        print("❌ No se pudo abrir ninguna cámara")
        exit()

camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camara.set(cv2.CAP_PROP_FPS, 15)

print("🔍 Sistema de detección de placas - Adaptativo a condiciones de luz")

try:
    while True:
        ret, frame = camara.read()
        if not ret:
            time.sleep(0.1)
            continue

        contadorFrames += 1
        frameDisplay = frame.copy()
        tiempoActual = time.time()

        # ===== DETECCIÓN CADA 10 FRAMES =====
        if contadorFrames % 10 == 0:
            try:
                resultadosYolo = modeloYolo(frame, verbose=False, conf=0.4)
                for resultado in resultadosYolo:
                    if resultado.boxes is not None:
                        for caja in resultado.boxes:
                            confianza = float(caja.conf[0])
                            if confianza < 0.4:
                                continue

                            x1, y1, x2, y2 = map(int, caja.xyxy[0].cpu().numpy())
                            alto, ancho = frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(ancho, x2), min(alto, y2)
                            if x2 <= x1 or y2 <= y1:
                                continue

                            roi = frame[y1:y2, x1:x2]
                            if roi.size == 0:
                                continue

                            resultadosColor = detectarColorPlaca(roi, RANGOS_AMARILLO)
                            if resultadosColor and resultadosColor[0]['porcentaje'] > 15:
                                mejorColor = resultadosColor[0]
                                tecnicasPreprocesamiento = preprocesarPlaca(roi)
                                if tecnicasPreprocesamiento:
                                    for nombreTecnica, imgProcesada in tecnicasPreprocesamiento:
                                        imgMejorada = mejorarContrasteTexto(imgProcesada)
                                        try:
                                            ocrResults = lectorOcr.readtext(imgMejorada, detail=1, paragraph=False)
                                            if ocrResults:
                                                for texto, conf in [(res[1], res[2]) for res in ocrResults]:
                                                    placa = validarPlaca(texto)
                                                    if placa and conf > 0.3:
                                                        # ===== Verificar si es nueva detección =====
                                                        if placa in placasActivas:
                                                            placasActivas[placa].actualizar(
                                                                (x1, y1, x2, y2), tiempoActual, mejorColor['rango']
                                                            )
                                                        else:
                                                            if placa in placasDetectadas:
                                                                ultimaDeteccion = placasDetectadas[placa].tiempoUltimaDeteccion
                                                                if tiempoActual - ultimaDeteccion < TIEMPO_ESPERA_REPETICION:
                                                                    continue

                                                            # ===== Nueva detección exitosa =====
                                                            from processing.procesamiento import DeteccionPlaca
                                                            deteccion = DeteccionPlaca(placa, (x1, y1, x2, y2), tiempoActual)
                                                            deteccion.mejorRango = mejorColor['rango']
                                                            placasActivas[placa] = deteccion
                                                            placasDetectadas[placa] = deteccion

                                                            fechaHora = formatearFechaHora(tiempoActual)
                                                            print(f"✅ PLACA DETECTADA: {placa}")
                                                            print(f"   📅 {fechaHora}")
                                                            print(f"   🎨 Rango: {mejorColor['rango']['nombre']}")
                                                            print(f"   🔧 Técnica: {nombreTecnica}")
                                                            print(f"   📊 Confianza: {conf:.2f}")

                                                            # ===== Estadísticas de rangos =====
                                                            nombreRango = mejorColor['rango']['nombre']
                                                            if nombreRango in estadisticasRangos:
                                                                estadisticasRangos[nombreRango] += 1
                                                            else:
                                                                estadisticasRangos[nombreRango] = 1

                                                            # ===== Encolar para guardar en BD =====
                                                            try:
                                                                timestampIso = datetime.datetime.fromtimestamp(tiempoActual).strftime('%Y-%m-%d %H:%M:%S')
                                                                colaEscritura.put_nowait((timestampIso, placa, nombreRango, nombreTecnica))
                                                            except Exception:
                                                                print("[WARNING] Cola de escritura llena - detección no guardada")
                                                            break
                                        except Exception:
                                            continue
            except Exception:
                pass

        # ===== SEGUIMIENTO Y VISUALIZACIÓN =====
        placasAEliminar = []

        for placa, deteccion in placasActivas.items():
            x1, y1, x2, y2 = deteccion.bbox
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                rangoVerificacion = deteccion.mejorRango if deteccion.mejorRango else RANGOS_AMARILLO[1]
                mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), rangoVerificacion['bajo'], rangoVerificacion['alto'])
                area = roi.shape[0] * roi.shape[1]
                if area > 0:
                    pctAmarillo = (cv2.countNonZero(mask) / area) * 100
                    if pctAmarillo > 10:
                        deteccion.tiempoUltimaDeteccion = tiempoActual
                        cv2.rectangle(frameDisplay, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frameDisplay, f"PLACA: {placa}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        fechaEntrada = formatearFechaHora(deteccion.tiempoPrimerDeteccion)
                        cv2.putText(frameDisplay, f"Entrada: {fechaEntrada}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        tiempoEnCuadro = int(tiempoActual - deteccion.tiempoPrimerDeteccion)
                        cv2.putText(frameDisplay, f"Tiempo: {tiempoEnCuadro}s", (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        continue
            placasAEliminar.append(placa)

        for placa in placasAEliminar:
            if placa in placasActivas:
                del placasActivas[placa]

        # ===== Información en pantalla =====
        elapsed = time.time() - inicioTiempo
        fps = contadorFrames / elapsed if elapsed > 0 else 0
        infoRangos = "Rangos: "
        for nombre, count in list(estadisticasRangos.items())[:3]:
            infoRangos += f"{nombre}({count}) "

        cv2.putText(frameDisplay, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frameDisplay, f"Activas: {len(placasActivas)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frameDisplay, infoRangos, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        cv2.imshow("Sistema de Placas - Adaptativo a Iluminación", frameDisplay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⏹️ Programa interrumpido por usuario")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    camara.release()
    cv2.destroyAllWindows()
    detenerWorker()
    mostrarEstadisticasFinales(placasDetectadas, estadisticasRangos)


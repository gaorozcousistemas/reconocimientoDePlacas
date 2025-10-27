from ultralytics import YOLO
import easyocr

def cargarYolo(pathModelo):
    try:
        modelo = YOLO(pathModelo)
        print("✅ Modelo YOLO cargado correctamente")
        return modelo
    except Exception as e:
        print(f"❌ Error cargando YOLO: {e}")
        exit()

def cargarOcr(idiomas, gpu=False):
    try:
        lector = easyocr.Reader(idiomas, gpu=gpu)
        print("✅ OCR configurado correctamente")
        return lector
    except Exception as e:
        print(f"❌ Error configurando OCR: {e}")
        exit()

# Importar librerías necesarias
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import imutils
import re

#Carga Imagen de Entreda
image = cv2.imread('./imagenes/imagen3.jpg')

#Iniciar modelos
model = YOLO('best.pt')  # Modelo YOLOv8 para detección de placas
ocr = PaddleOCR(use_angle_cls=True, lang='es')  # Modelo PaddleOCR para reconocimiento de texto

# Ejecutar YOLO sobre la imagen
results = model(image)
#print(results[0].boxes)

for result in results:
    # Filtrar solo las detecciones de clase "placa" (cls == 0)
    index_plates = (result.boxes.cls == 0).nonzero(as_tuple=True)[0]
    #print(index_plates)

    for idx in index_plates:
        # Obtener confianza de la caja
        conf = result.boxes.conf[idx].item()
        if conf > 0.7:
            # Obtener las coordenadas de la caja
            xyxy = result.boxes.xyxy[idx].squeeze().tolist()
            x1, y1 = int(xyxy[0]), int(xyxy[1])
            x2, y2 = int(xyxy[2]), int(xyxy[3])
            
            # Recortar imagen de la placa con padding
            plate_image = image[y1:y2, x1:x2]

            # Ejecutar OCR con PaddleOCR
            result_ocr = ocr.predict(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
            #print(result_ocr)

            # Ordenar los textos detectados de izquierda a derecha
            boxes = result_ocr[0]['rec_boxes']
            texts = result_ocr[0]['rec_texts']
            left_to_right = sorted(zip(boxes, texts), key=lambda x: min(x[0][::2]))
            print(f"left_to_right:", left_to_right)
            
            # Filtrar por whitelist (solo letras mayúsculas y números)
            whitelist_pattern = re.compile(r'^[A-Z0-9]+$')
            left_to_right = ''.join([t for _, t in left_to_right])
            output_text = ''.join([t for t in left_to_right if whitelist_pattern.fullmatch(t)])
            print(f"output_text: {output_text}")
            
            # Visualización
            cv2.imshow("plate_image", plate_image)
            # Dibujar resultados sobre la imagen
            cv2.rectangle(image, (x1 - 10, y1 - 35), (x2 + 10, y2-(y2 -y1)), (0, 255, 0), -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, output_text, (x1-7, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
# Mostrar imagen final
cv2.imshow("Image", imutils.resize(image, width=720))
cv2.waitKey(0)
cv2.destroyAllWindows()
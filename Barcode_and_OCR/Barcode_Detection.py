import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pyzbar import pyzbar
from ultralytics import YOLO
from paddleocr import PaddleOCR
import json
import tempfile

ocr_model = PaddleOCR(lang='en', use_angle_cls=True)

def extraer_imagenes(directorio):
    imagenes = []
    for archivo in os.listdir(directorio):
        if archivo.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            ruta_completa = os.path.join(directorio, archivo)
            imagen = Image.open(ruta_completa)
            imagenes.append((archivo, imagen))
    return imagenes

def extraer_subimagenes(imagen, box_coords):
    subimagenes = []
    for box in box_coords:
        x1, y1, x2, y2 = box
        subimagen = imagen[y1:y2, x1:x2]
        subimagenes.append(subimagen)
    return subimagenes

def filtro1(imagen_procesada):
    imagen_procesada1 = cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    image = clahe.apply(imagen_procesada1)
    roi1 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    roi1 = cv2.GaussianBlur(roi1, (3, 3), 2)
    imagen_procesada1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    image = clahe.apply(imagen_procesada1)
    roi1 = cv2.GaussianBlur(image, (5, 5), 2)
    
    return roi1

def filtro2(imagen):
    norm_img = np.zeros((imagen.shape[0], imagen.shape[1]), dtype=np.uint8)
    imagen = cv2.normalize(imagen, norm_img, 0, 255, cv2.NORM_MINMAX)

    def set_image_dpi(im):
        length_x, width_y = im.size
        factor = min(1, float(1024.0 / length_x))
        size = int(factor * length_x), int(factor * width_y)
        im_resized = im.resize(size, Image.LANCZOS)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_filename = temp_file.name
        im_resized.save(temp_filename, dpi=(300, 300))
        return temp_filename

    imagen_pil = Image.fromarray(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    temp_filename = set_image_dpi(imagen_pil)
    imagen = cv2.imread(temp_filename)

    def remove_noise(image):
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)

    imagen = remove_noise(imagen)

    kernel = np.ones((5, 5), np.uint8)
    imagen = cv2.erode(imagen, kernel, iterations=0)

    def get_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    imagen = get_grayscale(imagen)

    def thresholding(image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    imagen = thresholding(imagen)

    return imagen

def guardar_en_json(data, nombre_archivo):
    with open(nombre_archivo, 'w') as f:
        json.dump(data, f, indent=4)

def mostrar_subimagenes(imagenes, grosor=2, umbral=0.65, output_json='resultados.json'):
    resultados = []
    
    for nombre, imagen in imagenes:
        imagen_cv2 = cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
        results = model.predict(source=imagen_cv2, stream=True, verbose=False)
        
        box_coords = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                if conf > umbral:
                    x1, y1, x2, y2 = map(int, box)
                    box_coords.append((x1, y1, x2, y2))

        subimagenes = extraer_subimagenes(imagen_cv2, box_coords)
        
        for i, subimagen in enumerate(subimagenes):
            imagen_filtrada = filtro1(subimagen)

            codigos_barras = pyzbar.decode(imagen_filtrada)
            info_imagen = {"id": nombre, "nombre_imagen": nombre, "codigos_barras": [], "ocr": []}

            for codigo in codigos_barras:
                x, y, w, h = codigo.rect
                cv2.rectangle(subimagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
                texto_codigo = codigo.data.decode('utf-8')
                tipo_codigo = codigo.type
                cv2.putText(subimagen, f'{tipo_codigo}: {texto_codigo}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"Encontrado código de barras ({tipo_codigo}): {texto_codigo}")
                info_imagen["codigos_barras"].append({"tipo": tipo_codigo, "texto": texto_codigo, "coordenadas": [x, y, w, h]})

            if not codigos_barras:
                imagen_filtrada = filtro2(subimagen)
                codigos_barras = pyzbar.decode(imagen_filtrada)
                info_imagen = {"id": nombre, "nombre_imagen": nombre, "codigos_barras": [], "ocr": []}

                for codigo in codigos_barras:
                    x, y, w, h = codigo.rect
                    cv2.rectangle(subimagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    texto_codigo = codigo.data.decode('utf-8')
                    tipo_codigo = codigo.type
                    cv2.putText(subimagen, f'{tipo_codigo}: {texto_codigo}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print(f"Encontrado código de barras ({tipo_codigo}): {texto_codigo}")
                    info_imagen["codigos_barras"].append({"tipo": tipo_codigo, "texto": texto_codigo, "coordenadas": [x, y, w, h]})

            if not codigos_barras:
                imagen_filtrada = filtro1(subimagen)
                print("Sin código de barras legible")
                print("Comenzando con el reconocimiento OCR")
                
                result1 = ocr_model.ocr(imagen_filtrada, cls=True)
                texts = [res1[1][0] for res1 in result1[0]]
                boxes1 = [res1[0] for res1 in result1[0]]
                ocr_info = []

                if len(texts) > 1:
                    texts = [' '.join(texts)]
                for text, box in zip(texts, boxes1):
                    print(f"Código OCR encontrado: {text}")
                    ocr_info.append({"texto": text, "coordenadas": box})
                info_imagen["ocr"].extend(ocr_info)
                if not texts:
                    print("Vuelva a sacar la foto")

            cv2.imshow(f'{nombre} - Box {i+1}', imagen_filtrada)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            resultados.append(info_imagen)
    
    guardar_en_json(resultados, output_json)

model = YOLO("barras.pt")

path = r'G:\INNOVAI\Test'
imagenes = extraer_imagenes(path)

output_json_path = os.path.join(path, 'resultados.json')
mostrar_subimagenes(imagenes, grosor=9, umbral=0.10, output_json=output_json_path)

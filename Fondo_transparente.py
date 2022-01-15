import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import datetime


# Iniciamos la función de selfie segmentation de mediapipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation


prueba1 = "yered.jpg"
prueba2 = "micro.jpg"


# Leemos la imagen
img = cv2.imread(prueba2)
# Redimensionamos la imagen
alto, ancho, _ = img.shape
#img = cv2.resize(img, (int((ancho/7)),int((alto/7))))
print(img.shape)


tiempo = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
img_name = f"Imagen Procesada {tiempo}.png"
pre = "foto.png"


with mp_selfie_segmentation.SelfieSegmentation(model_selection = 0) as selfie_segmentation:
    # Convertimos la imagen de BGR a RGB antes del procesamiento
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Teniendo la imagen en RGB podemos aplicarle el método de selfie segmentation, lo que hace es
    # buscar el contorno de la persona.
    result = selfie_segmentation.process(img_rgb)

    # Binarizamos el resultado anterior para que solo haya 0 y 1 (o sea en blanco y negro).
    _, thresh = cv2.threshold(result.segmentation_mask, 0.75, 255, cv2.THRESH_BINARY)
    # Imprimimos 
    #print(thresh.dtype)
    # Cambiamos de float32 a uint8
    thresh = thresh.astype(np.uint8)
    # Suavizamos la imagen
    #thresh = cv2.medianBlur(thresh, 13)

    # Invertimos el negro a blanco y el blanco a negro
    thresh_not = cv2.bitwise_not(thresh)

    # -- Proceso para encontrar el contorno de la persona -- #
    img_contours = cv2.findContours(thresh_not, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = sorted(img_contours, key=cv2.contourArea)

    for i in img_contours:
        if cv2.contourArea(i) > 100:
            break
    # -- Fin de proceso para encontrar el contorno de la persona -- #
    
    
    bg = cv2.imread("canal_alpha.png")
    bg = cv2.resize(bg, (300, 300))

    fondo = cv2.bitwise_and(bg, bg, mask=thresh_not)

    fg = cv2.bitwise_and(img, img, mask=thresh)

    

    salida = cv2.add(fondo, fg)
    
    cv2.imwrite(pre, salida)
    
def procesar(imagen):
    img = Image.open(imagen)
    img = img.convert("RGBA")

    imgnp = np.array(img)

    white = np.sum(imgnp[:,:,:3], axis=2)
    white_mask = np.where(white == 255*3, 1, 0)

    alpha = np.where(white_mask, 0, imgnp[:,:,-1])

    imgnp[:,:,-1] = alpha 

    img = Image.fromarray(np.uint8(imgnp))
    img.save(img_name, "PNG")
    print("Completado!")



procesar(pre)

#cv2.imshow("img", img)
#cv2.imshow("result_mask", result.segmentation_mask)
#cv2.imshow("threshold", thresh)
#cv2.imshow("threshold_not", thresh_not)
#cv2.imshow("new_img", fondo)
#cv2.imshow("fg", fg)
#cv2.imshow("Salida", salida)

import cv2
import numpy as np
import glob
import argparse

#paso de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i","--images",required = False,
                help = "ruta a la carpeta de la secuencia de imagenes")
ap.add_argument("-o","--out",required = False,
                help = "ruta al archivo de video resultante")

args = vars(ap.parse_args())
path_in = args['images']
path_out = args['out']



#cargamos el clasificador hog
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


#lista de archivos ordenados alfabeticamente
list = sorted(glob.glob((path_in+'/*png')))

#video de salida
fourcc = cv2.VideoWriter_fourcc('X','V', 'I','D')
out = cv2.VideoWriter(path_out,fourcc, 33.0, (640,480))


for i in range(0,len(list)):


    print('Procesando: ',(i/len(list))*100,"%")


    img = cv2.imread(list[i])
    rects, weights = hog.detectMultiScale(img,
                                          winStride=(8, 8),
                                          padding=(32, 32),
                                          scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (125, 255, 0), 2)

    out.write(img)
    cv2.imshow('frame', img)



print('FIN')
out.release()
cv2.destroyAllWindows()
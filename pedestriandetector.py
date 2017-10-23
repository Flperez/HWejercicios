import cv2
import numpy as np
import glob,os
import argparse

#paso de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i","--images",required =True,
                help = "ruta a la carpeta de la secuencia de imagenes")
ap.add_argument("-o","--out",required = True,
                help = "ruta al archivo de video resultante")

args = vars(ap.parse_args())
path_in = args['images']
path_out = args['out']




#cargamos el clasificador hog
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


#lista de archivos ordenados alfabeticamente

#video de salida
fourcc = cv2.VideoWriter_fourcc('X','V', 'I','D')
out = cv2.VideoWriter(path_out,fourcc, 33.0, (640,480))

for infile in sorted(glob.glob((path_in + '/*png'))):
    file, ext = os.path.splitext(infile)


    print('Procesando: ',infile)
    img = cv2.imread(infile)
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
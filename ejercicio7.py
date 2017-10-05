import numpy as np
import argparse
import glob, os
import cv2
from collections import deque
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#funciones
def nueva_ruta(ruta_img,path_out):
    k = ruta_img.rfind("/")
    name = ruta_img[k+1:]
    return path_out+"/"+name






#paso de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i","--images",required = False,
                help = "ruta a la carpeta de la secuencia de imagenes")
ap.add_argument("-o","--out",required = False,
                help = "ruta a la carpeta con la imagen resultante")


args = vars(ap.parse_args())
path_in = args['images']
path_out = args['out']



if __name__ == "__main__":

    bajos = np.array([29, 43, 126], dtype=np.uint8)
    altos = np.array([88, 255, 255], dtype=np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    num = 10
    colax = deque(maxlen=num)
    colay = deque(maxlen=num)
    cont = 0
    for infile in sorted(glob.glob((path_in+'/*jpg')),key=numericalSort):
        file, ext = os.path.splitext(infile)
        print("Procesando: ",infile)
        im = cv2.imread(infile)
        hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,bajos,altos)

        #Erosion
        cv2.erode(src=mask,dst=mask, kernel=kernel, iterations=2)

        #dilatacion
        cv2.dilate(src=mask,dst=mask, kernel=kernel, iterations=2)

        #Calculo del area maxima
        _,contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]



        j = np.argmax(areas) #indice del contorno con mayor area
        cv2.drawContours(mask,contours,j,(255,0,0))


        (x,y),radius = cv2.minEnclosingCircle(contours[j])
        radius = int(radius)
        center =(int(x),int(y))
        cv2.circle(im, center, radius, (0, 0, 255), 2)
        cv2.circle(mask, center, radius, (0, 0, 255), 2)


        # pintar trayectoria
        ## cola circular
        #actualizar cola
        colax.append(center[0])
        colay.append(center[1])
        cont=cont+1

        #Se han rellenado los 10 valores
        if cont>colax.maxlen:
            for k in range(0,colax.maxlen-1):
                cv2.line(im,(colax[k],colay[k]),(colax[k+1],colay[k+1]),(255,0,0),k)




        #for para pintar
        r1 = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
        r2 = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow("mask", r2)
        cv2.imshow("image", r1)
        cv2.waitKey(100)

    cv2.destroyAllWindows()








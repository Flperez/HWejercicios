import numpy as np
import argparse
import glob, os
import cv2
from collections import deque

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

    for infile in sorted(glob.glob((path_in+'/*jpg'))):
        file, ext = os.path.splitext(infile)
        print("Procesando: ",infile)
        im = cv2.imread(infile)
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,bajos,altos)

        #Erosion
        cv2.erode(src=mask,dst=mask, kernel=kernel, iterations=2)

        #dilatacion
        cv2.dilate(src=mask,dst=mask, kernel=kernel, iterations=2)

        edges = cv2.Canny(mask, 1, 2)
        _,contours, hier = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]

        max = np.amax(areas)
        j= np.argmax(areas)

        #cv2.drawContours(im,[contours[j]],0,(0,0,255),2)
        #cv2.drawContours(mask, [contours[j]], 0, (0, 0, 255), 2)

        mu = cv2.moments(contours[j])
        x = int(mu['m10']/mu['m00'])
        y = int(mu['m01']/mu['m00'])

        _,radius = cv2.minEnclosingCircle(contours[j])
        radius = int(radius)

        cv2.circle(im,(x,y),radius,(0,0,255),2)

        resized = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("mask",mask)
        cv2.imshow("image",resized)
        cv2.waitKey(100)

        #pintar trayectoria
        #cola circular
        #actualizar cola
        #for para pintar


    cv2.destroyAllWindows()








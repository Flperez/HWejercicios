import numpy as np
import argparse
from PIL import Image
import glob, os
import cv2
from PIL import ImageFilter

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


    for infile in sorted(glob.glob((path_in+'/*jpg'))):
        file, ext = os.path.splitext(infile)
        print("Procesando: ",infile)
        im = cv2.imread(infile)
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)


        mask = cv2.inRange(hsv,bajos,altos)

        ruta = nueva_ruta(infile,path_out)
        cv2.imwrite(ruta,mask)



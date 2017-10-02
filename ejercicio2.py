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

def output_mask(altos,bajos,image):
    size = image.shape
    mask = np.zeros(size, np.uint8)

    for x in range(size[0]):
        for y in range(size[1]):
            in_range_h = image[x, y, 0] >= bajos[0] and image[x, y, 0] < altos[0]
            in_range_s = image[x, y, 1] >= bajos[1] and image[x, y, 1] < altos[1]
            in_range_v = image[x, y, 2] >= bajos[2] and image[x, y, 2] < altos[2]
            mask[x, y] = 255*(in_range_h and in_range_s and in_range_v)
    return mask




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

    for infile in sorted(glob.glob((path_in+'/*jpg'))):
        file, ext = os.path.splitext(infile)
        print("Procesando: ",infile)
        im = cv2.imread(infile)
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)

        bajos = (29,43,126)
        altos = (88,255,255)
        mask = output_mask(altos,bajos,hsv)
        print("Calculado la mascara")

        cv2.imshow("mask",mask)
        cv2.imshow("image original",im)

        ruta = nueva_ruta(infile,path_out)
        cv2.waitKey(2)
        cv2.destroyAllWindows()
        cv2.imwrite(ruta,mask)



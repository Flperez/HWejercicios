import numpy as np
import argparse
from PIL import Image
import glob, os
from PIL import ImageFilter

#funciones
def nueva_ruta(ruta_img,path_out,j):
    k = ruta_img.rfind("/")
    name = ruta_img[k+1:]
    return path_out+"/"+name+"_%d"%j



#paso de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i","--images",required = False,
                help = "ruta a la carpeta de la secuencia de imagenes")
ap.add_argument("-o","--out",required = False,
                help = "ruta a la carpeta con el aumentado de datos")
ap.add_argument("-f","--factor",required = False, type=int,
                help = "factor de aumento de datos (5,10,20)")

args = vars(ap.parse_args())
path_in = args['images']
path_out = args['out']
factor = args['factor']

#lista de archivos ordenados alfabeticamente
list = sorted(glob.glob((path_in+'/*png')))


if __name__ == "__main__":

    for infile in sorted(glob.glob((path_in+'/*png'))):
        file, ext = os.path.splitext(infile)
        im = Image.open(infile)
        print("Procesando: ",infile)
        for j in range (1,factor+1):



            #Aumentado de datos
            result = im.filter(ImageFilter.GaussianBlur(2+8*(j/(factor+1))))
            size = int(im.width*(0.25+2.25*(j/factor+1))), int(im.height*(0.25+2.25*(j/factor+1)))
            result = result.resize(size,0)
            result = result.rotate(360*(j/factor+1))
        
            #Gurdando datos
            ruta_save = nueva_ruta(file,path_out,j)
            print("\tGuardando: ",ruta_save)
            result.save(ruta_save, "PNG")

            








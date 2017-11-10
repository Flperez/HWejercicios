import argparse
from PIL import Image, ImageFilter
import glob, os
from random import randint


# funciones
def nueva_ruta(ruta_in, ruta_out, indice):
    k = ruta_in.rfind("/")
    name = ruta_in[k + 1:]
    return ruta_out + "/" + name + "_%d" % indice

def nueva_img(img):

    new = img

    #Gaussian Blur
    bool_filter = randint(0, 1)
    if bool_filter == 1:
        porcentaje = randint(0, 100)
        new = new.filter(ImageFilter.GaussianBlur(2 + (porcentaje/100)*8))
        print("\tAplicado un filtro Gaussiano con radio: ",2+(porcentaje/100)*8)



    #Resize
    bool_resize = randint(0, 1)
    if bool_resize == 1:
        porcentaje = randint(0, 100)
        width, height = new.size
        width = width * (0.25 + (2.25 * (porcentaje/100)))
        height = height *  (0.25 + (2.25 * (porcentaje/100)))

        size = int(width),int(height)
        new = new.resize(size, 0)
        print("\tImagen redimensionada a una escala de ",(0.25 + 2.25 * (porcentaje/100)))

    # Flip left
    bool_flip_left_right = randint(0, 1)
    if bool_flip_left_right == 1:
        new = new.transpose(Image.FLIP_LEFT_RIGHT)
        print("\tAplicada un flip de izquierda a derecha")

    # Flip top
    bool_flip_top_bottom = randint(0, 1)
    if bool_flip_top_bottom == 1:
        new = new.transpose(Image.FLIP_TOP_BOTTOM)
        print("\tAplicada un flip de arriba hacia abajo")

    # Rotate
    bool_rotate = randint(0, 1)
    if bool_rotate == 1:
        j= randint(1,3)
        new = new.rotate(j*90)
        print("\tImagen rotada con un angulo de ",90*j)

    return new



if __name__ == "__main__":

    # paso de argumentos
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dataset", required=True,
                    help="ruta a la carpeta de la secuencia de imagenes")
    ap.add_argument("-f","--factor", required=False,
                    help="factor de aumento de datos ")
    ap.add_argument("-o", "--output_dataset", required=True,
                    help="ruta a la carpeta con el aumentado de datos")

    args = vars(ap.parse_args())
    print(args)
    path_in = args['input_dataset']
    path_out = args['output_dataset']
    factor = int(args['factor'])


    #Bucle para aumentar los datos
    for infile in sorted(glob.glob((path_in + '/*png'))):
        file, ext = os.path.splitext(infile)
        img = Image.open(infile)
        print("Procesando: ", infile)

        for j in range(0, factor ):

            # Aumentado de datos
            result = nueva_img(img=img)

            # Gurdando datos
            ruta_save = nueva_ruta(ruta_in=file, ruta_out = path_out, indice=j)
            print("\tGuardando: ", ruta_save)
            result.save(ruta_save,PNG)

    print("Los datos han sido aumentados.")









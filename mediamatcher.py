import numpy as np
import cv2
import argparse
import glob
import matplotlib.pyplot as plt

N = 10 #umbral

def  numMatches(rutaImgScene,des1,detector):
    '''
    codigo copiado del ejemplo:
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher

    Funcion que recibe la ruta a la imagen con la que queremos comparar junto con el
    descriptor de la imagen plantilla y el detector que vamos a usar

    :param rutaImgScene: ruta a la imagen del dataset
    :param des1: descriptor de la primera imagen
    :param detector: algoritmo que usaremos para extraer los puntos de interes
    :return: numero de matches
    '''

    img2 = cv2.imread(rutaImgScene, 0)  # queryImage


    # find the keypoints and descriptors with SIFT
    kp2, des2 = detector.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    num_matches = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            num_matches = num_matches+1

    print("Imagen: ",rutaImgScene,
          " con ",num_matches," matches")
    return num_matches

#paso de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-q","--query",required = False,
                help = "ruta a la imagen de consulta")
ap.add_argument("-c","--covers",required = False,
                help = "ruta a la carpeta con mis caratulas")

args = vars(ap.parse_args())
rutaImgObject = args['query']
rutaImgCovers = args['covers']

if __name__ == "__main__":

    img = cv2.imread(rutaImgObject)  # queryImage
    img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    _, des1 = sift.detectAndCompute(img1, None)

    infile = sorted(glob.glob((rutaImgCovers+'/*jpg')))
    list = [numMatches(rutaImgScene=rut,des1=des1,detector=sift) for rut in infile]
    arg_max = int(np.argmax(list))

    if list[arg_max]>N:
        # mostramos los resultados
        print("\nLa imagen que mas se asemeja a la introducida es: \n ",
              infile[arg_max]," con ",list[arg_max]," matches")


        img2 = cv2.imread(infile[arg_max])

        plt.subplot(121)
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title("imagen de consulta")
        plt.subplot(122)
        img2 = cv2.cvtColor(src=img2, code=cv2.COLOR_BGR2RGB)
        plt.imshow(img2)
        plt.title("imagen correspondiente")
        plt.show()



    else:
        print("No hay ninguna imagen en la lista que supere el umbral: ",N)








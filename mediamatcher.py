import numpy as np
import cv2
import argparse
import glob

N = 10

def  numMatches(rutaImgScene,des1,descriptor):
    '''

    :param rutaImgScene: ruta a la imagen del dataset
    :param des1: descriptor de la primera imagen
    :param descriptor: algoritmo que usaremos para extraer los puntos de interes
    :return: numero de matches
    '''
    img2 = cv2.imread(rutaImgScene, 0)  # queryImage


    # find the keypoints and descriptors with SIFT
    kp2, des2 = descriptor.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    cont = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            cont = cont + 1

    return cont

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

    img1 = cv2.imread(rutaImgObject, 0)  # queryImage


    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    _, des1 = sift.detectAndCompute(img1, None)

    infile = sorted(glob.glob((rutaImgCovers+'/*png')))
    list = [numMatches(rutaImgScene=rut,des1=des1,descriptor=sift) for rut in infile]
    arg_max = int(np.argmax(list))

    if list[arg_max]>N:
        print(infile[arg_max])
    else:
        print("No hay ninguna imagen en la lista que supere el umbral: ",N)








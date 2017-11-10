import cv2
import numpy as np
import argparse
import pytesseract
from PIL import Image
from non_maxima_supression import non_max_suppression_fast


def unique_rows(a):
    '''
    Eliminar las filas duplicadas
    https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array
    '''
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))



def text_detection(img):
    '''
    Codigo copiado del modulo de text de opencv_contrib-3.3.0: "textdetection.py"

    '''
    path = "/home/f/libs/OP-3.3/opencv_contrib-3.3.0/modules/text/samples"

    # Extract channels to be processed individually
    channels = cv2.text.computeNMChannels(img)
    # Append negative channels to detect ER- (bright regions over dark background)
    cn = len(channels) - 1
    for c in range(0, cn):
        channels.append((255 - channels[c]))


    # Apply the default cascade classifier to each independent channel (could be done in parallel)
    print("Extracting Class Specific Extremal Regions from " + str(len(channels)) + " channels ...")
    print("    (...) this may take a while (...)")
    ROI=np.array([])
    for channel in channels:

        erc1 = cv2.text.loadClassifierNM1('trained_classifierNM1.xml')

        er1 = cv2.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

        erc2 = cv2.text.loadClassifierNM2('trained_classifierNM2.xml')
        er2 = cv2.text.createERFilterNM2(erc2, 0.5)

        regions = cv2.text.detectRegions(channel, er1, er2)

        rects=cv2.text.erGrouping(img, channel, [r.tolist() for r in regions])
        if len(rects)==1:
            ROI=np.append(ROI,rects)

    ROI=np.reshape(ROI,(-1,4))
    ROI=ROI.astype(int)
    return ROI

def contained(A,B):
    if (A[0]>=B[0] and A[1]>=B[1]
        and A[2]<=B[2] and A[3]<=B[3]):
        return True
    else:
        return False

def delete_rects_contained(rects):
    '''
    Eliminamos las BB contenidas dentro de otras BB
    :param rects:
    :return:
    '''
    #Vector que contiene los indices que contienen las filas que deben eliminarse
    deleted = np.array([])

    # Eliminamos los rectangulos duplicados
    rects = unique_rows(rects)

    #Calculamos las areas ya que una BB contenida en otra tendra una area menor
    areas = [(rects[k,0]-rects[k,2])*(rects[k,1]-rects[k,3]) for k in range(0,len(rects))]


    #Eliminamos los rectangulos contenidos dentro de otros rectangulos
    for i in range(0,len(rects)):
        for j in range(0,len(rects)):
            if areas[i]<=areas[j] and i !=j:
                if contained(rects[i],rects[j])==True:
                        deleted=np.append(deleted,i)

    deleted = deleted.astype(int)
    rects = np.delete(rects,(deleted),axis=0)
    return rects

def text_recognition(rects,image):

    offset = 10
    for r in range(0, np.shape(rects)[0]):
        rect = rects[r]
        image_to_recognise = image[rect[1]-offset:rect[1] + rect[3]+offset,
                             rect[0]-offset:rect[0] + rect[2]+offset]
        text = Image.fromarray(image_to_recognise)
        text = pytesseract.image_to_string(text)

        #Si ha reconocido algun texto en el rectangulo
        if text:
            print("Se ha reconocido en el rectangulo ",rect,":",text)
            cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)
            cv2.putText(image,text, (rect[0], rect[1]-offset),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

    return image








if __name__=="__main__":

    # Paso de argumentos
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="ruta a la imagen de entrada")
    ap.add_argument("-v", "--video", required=False, help="ruta al video de entrada")
    args = vars(ap.parse_args())
    path_image = args['image']
    path_video = args['video']

    if path_image and path_video:
        print("Debe introducir o un video o una imagen pero no ambos")



    ########## Video ##########
    if path_video and not path_image: #Se ha introducido un video y no una imagen
        # cargamos video de entrada
        cap = cv2.VideoCapture(path_video)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:

                rects_sin_filtrar = text_detection(img=frame)
                rects_non_maxima = non_max_suppression_fast(rects_sin_filtrar, 0.8)
                rects_filtrados = delete_rects_contained(rects_non_maxima)
                vis = text_recognition(rects=rects_filtrados, image=frame)

                cv2.imshow("Text detection result", vis)
                cv2.waitKey(33)
            else:
                cv2.destroyAllWindows()

    ########## Imagen ##########
    if path_image and not path_video: #Se ha introducido una imagen y no un video
        img = cv2.imread(path_image)

        rects_sin_filtrar = text_detection(img=img)
        rects_non_maxima = non_max_suppression_fast(rects_sin_filtrar,0.8)
        rects_filtrados = delete_rects_contained(rects_non_maxima)
        vis = text_recognition(rects=rects_filtrados,image=img)

        # Visualization
        cv2.imshow("Text detection result", vis)
        cv2.waitKey()
        cv2.destroyAllWindows()




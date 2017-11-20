import cv2
import numpy as np
import argparse
import pytesseract
from PIL import Image
from filtro import filtro




def text_detection(img):
    '''
    Codigo copiado del modulo de text de opencv_contrib-3.3.0: "textdetection.py"

    '''

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
            cv2.putText(image,text, (rect[0], rect[1]-5),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

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
                mifiltro = filtro(rects_sin_filtrar)
                rects_filtrados = mifiltro.filtrar()
                vis = text_recognition(rects=rects_filtrados, image=frame)

                cv2.imshow("Text detection result", vis)
                cv2.waitKey(33)
            else:
                cv2.destroyAllWindows()

    ########## Imagen ##########
    if path_image and not path_video: #Se ha introducido una imagen y no un video
        img = cv2.imread(path_image)

        rects_sin_filtrar = text_detection(img=img)
        mifiltro = filtro(rects_sin_filtrar)
        rects_filtrados=mifiltro.filtrar()
        vis = text_recognition(rects=rects_filtrados,image=img)

        # Visualization
        cv2.imshow("Text detection result", vis)
        cv2.waitKey()
        cv2.destroyAllWindows()




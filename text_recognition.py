import cv2
import numpy as np
import argparse

def textrecognition(img,pathname):
    '''
    Codigo copiado del modulo de text de opencv_contrib-3.3.0: "textdetection.py"

    '''

    vis = img.copy()
    # Extract channels to be processed individually
    channels = cv2.text.computeNMChannels(img)
    # Append negative channels to detect ER- (bright regions over dark background)
    cn = len(channels) - 1
    for c in range(0, cn):
        channels.append((255 - channels[c]))

    # Apply the default cascade classifier to each independent channel (could be done in parallel)
    print("Extracting Class Specific Extremal Regions from " + str(len(channels)) + " channels ...")
    print("    (...) this may take a while (...)")
    for channel in channels:

        erc1 = cv2.text.loadClassifierNM1(pathname + '/trained_classifierNM1.xml')
        er1 = cv2.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

        erc2 = cv2.text.loadClassifierNM2(pathname + '/trained_classifierNM2.xml')
        er2 = cv2.text.createERFilterNM2(erc2, 0.5)

        regions = cv2.text.detectRegions(channel, er1, er2)

        rects = cv2.text.erGrouping(img, channel, [r.tolist() for r in regions])
        # rects = cv2.text.erGrouping(img,channel,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)

        # Visualization
        for r in range(0, np.shape(rects)[0]):
            rect = rects[r]
            cv2.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0), 2)
            cv2.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 255), 1)

    return vis


#Paso de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = False, help = "ruta a la imagen de entrada")
ap.add_argument("-v","--video",required = False, help = "ruta al video de entrada")
args = vars(ap.parse_args())
path_image = args['image']
path_video= args['video']


pathname = "/home/f/opencv_contrib-3.3.0/modules/text/samples"


if __name__=="__main__":

    if path_video and not path_image: #No se ha introducido una imagen
        # cargamos video de entrada
        cap = cv2.VideoCapture(path_video)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.flip(frame, 0)
                vis = textrecognition(img=frame,pathname=pathname)
                cv2.imshow("Text detection result", vis)
                cv2.imshow("Original image", frame)
                cv2.waitKey(33)



    if path_image and not path_video:
        img = cv2.imread(path_image)
        vis = img.copy()
        vis = textrecognition(img=img,pathname=pathname)
        # Visualization
        cv2.imshow("Text detection result", vis)
        cv2.imshow("Original image",img)
        cv2.waitKey()

    if path_image and path_video:
        print("Debe introducir o un video o una imagen pero no ambos")

    cv2.destroyAllWindows()




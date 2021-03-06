import numpy as np
import cv2
import argparse



if __name__=="__main__":

    # Paso de argumentos
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="ruta al video de entrada")
    args = vars(ap.parse_args())
    path_video = args['image']

    '''
    Codigo copiado del modulo de tracking de opencv_contrib-3.3.0: "tracker.py"
    '''

    cv2.namedWindow("tracking")
    camera = cv2.VideoCapture(path_video)

    ok, image=camera.read()
    if not ok:
        print('Failed to read video')
        exit()
    bbox = cv2.selectROI("tracking", image)
    tracker = cv2.TrackerMIL_create()
    init_once = False

    while camera.isOpened():
        ok, image=camera.read()
        if not ok:
            print ('no image to read')
            break

        if not init_once:
            ok = tracker.init(image, bbox)
            init_once = True

        ok, newbox = tracker.update(image)
        print (ok, newbox)

        if ok:
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(image, p1, p2, (200,0,0))

        cv2.imshow("tracking", image)
        k = cv2.waitKey(1) & 0xff
        if k == 'q' : break # q pressed

    cv2.destroyAllWindows()
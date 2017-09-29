import cv2
#import numpy as np
import argparse



ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",required = False, help = "ruta a donde guarda el archivo de video de entrada")
ap.add_argument("-o","--out",required = False, help = "ruta a donde guarda el archivo de video resultante")


args = vars(ap.parse_args())

path_in = args['video']
path_out = args['out']
path_cascade_face = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalcatface.xml'
path_cascade_eyes = '/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml'





faceCascade = cv2.CascadeClassifier(path_cascade_face)
eyesCascade = cv2.CascadeClassifier(path_cascade_eyes)


#cargamos video de entrada
cap = cv2.VideoCapture(path_in)
#video de salida
fourcc = cv2.VideoWriter_fourcc('X’, ‘V’, ‘I’,’D')
out = cv2.VideoWriter(path_out,fourcc, 33.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame, 0)
        frame_original = frame.copy()
        faces = faceCascade.detectMultiScale(frame,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (125, 255, 0), 2)
            eyes = eyesCascade.detectMultiScale(frame,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in eyes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (125, 255, 0), 2)

        out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
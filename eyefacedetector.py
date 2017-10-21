import cv2
import argparse



ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",required = True, help = "ruta a donde guarda el archivo de video de entrada")
ap.add_argument("-o","--out",required = True, help = "ruta a donde guarda el archivo de video resultante")


args = vars(ap.parse_args())

path_in = args['video']
path_out = args['out']



faceCascade = cv2.CascadeClassifier('/home/f/PycharmProjects/HWejercicios/haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('/home/f/PycharmProjects/HWejercicios/haarcascade_eye.xml')


#cargamos video de entrada
cap = cv2.VideoCapture(path_in)
#video de salida
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I','D')
out = cv2.VideoWriter(path_out,fourcc, 33.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame_gray = cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(frame_gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            eyes = eyesCascade.detectMultiScale(frame_gray,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in eyes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(33)

        if key == 113: #letra q
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
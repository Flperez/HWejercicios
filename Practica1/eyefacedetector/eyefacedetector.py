import cv2
import argparse
import numpy as np


if __name__=="__main__":
    #Argumentos
    ap = argparse.ArgumentParser()
    ap.add_argument("-v","--video",required = True, help = "ruta a donde guarda el archivo de video de entrada")
    ap.add_argument("-o","--out",required = True, help = "ruta a donde guarda el archivo de video resultante")
    args = vars(ap.parse_args())
    path_in = args['video']
    path_out = args['out']


    #Clasificadores
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')


    #cargamos video de entrada
    cap = cv2.VideoCapture(path_in)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #video de salida
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I','D')
    out = cv2.VideoWriter(path_out,fourcc, 33.0, (640,480))


    print('Si desea salir del video presione la letra <<q>>')
    print("Si desea no visualizar la deteccion presione 'n'")
    mode = "visualization"

    #contador para la cantidad de frames procesados
    cont = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:

            frame_gray = cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)
            frame_eyes = np.zeros(frame.shape, dtype='uint8')

            #Deteccion caras
            faces = faceCascade.detectMultiScale(frame_gray,
                                                 scaleFactor=1.1,
                                                 minNeighbors=5,
                                                 flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                #Deteccion ojos
                frame_eyes[y:y + h,x:x + w] = frame_gray[ y:y + h,x:x + w]
                eyes = eyesCascade.detectMultiScale(frame_eyes,
                                                    scaleFactor=1.1,
                                                    minNeighbors=5,
                                                    minSize=(30, 30),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
                for (x, y, w, h) in eyes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

            #guardamos el video resultante
            out.write(frame)

            if mode == "visualization":
                cv2.imshow('frame', frame)
                key = cv2.waitKey(33)

            if key == ord('q'):
                break
            if key == ord('n'):
                mode="no_visualization"
                cv2.destroyAllWindows()

            #mostrar proceso
            print("Frame actual: ",cont," Process: ",(cont/length)*100,"%")
            cont+=1

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
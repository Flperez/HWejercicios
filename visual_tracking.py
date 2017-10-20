import numpy as np
import argparse
import cv2
from collections import deque


num=5







#paso de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",required = True,
                help = "ruta al video de entrada")
ap.add_argument("-m","--min_values",nargs='+', type=int,required = True,
                help = "Valores minimos para la mascara en escala HSV")
ap.add_argument("-M","--max_values",nargs='+', type=int,required = True,
                help = "Valores maximos para la mascara en escala HSV")
ap.add_argument("-o","--output",required = False,
                help = "ruta al video resultante")


args = vars(ap.parse_args())
path_in = args['video']
min_values = args['min_values']
max_values = args['max_values']
path_out = args['output']

if path_out:
    # video de salida
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter(path_out, fourcc, 33.0, (1280, 1024))


if __name__ == "__main__":

    bajos = np.array(min_values, dtype=np.uint8)
    altos = np.array(max_values, dtype=np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    colax = deque(maxlen=num)
    colay = deque(maxlen=num)
    for i in range(0,10):
        colax.append(None)
        colay.append(None)
    cont = 0

    # cargamos video de entrada
    cap = cv2.VideoCapture(path_in)



    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.GaussianBlur(frame,(5,5),0)
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,bajos,altos)

            #Erosion
            cv2.erode(src=mask,dst=mask, kernel=kernel, iterations=2)

            #dilatacion
            cv2.dilate(src=mask,dst=mask, kernel=kernel, iterations=2)

            #Calculo del area maxima
            _,contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours]



            j = np.argmax(areas) #indice del contorno con mayor area


            (x,y),radius = cv2.minEnclosingCircle(contours[j])
            radius = int(radius)
            center =(int(x),int(y))
            cv2.circle(frame, center, radius, (0, 0, 255), 2)
            cv2.circle(frame, center, 2, (0, 255, 0), 15)


            # pintar trayectoria
            ## cola circular
            #actualizar cola
            colax.appendleft(center[0])
            colay.appendleft(center[1])
            cont=cont+1


            num_lineas = colax.maxlen-colax.count(None)
            #Se han rellenado los 10 valores
            if num_lineas>=2:
                for k in range(0,num_lineas-1):
                    cv2.line(frame,(colax[k],colay[k]),(colax[k+1],colay[k+1]),(255,0,0),num_lineas-k)

            #for para pintar
            r1 = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            r2 = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)

            cv2.imshow("mask", r2)
            cv2.imshow("image", r1)

            cv2.waitKey(33)



            if path_out:
                out.write(frame)


    print("fin")
    cap.release()
    if path_out:
        out.release()
    cv2.destroyAllWindows()










import cv2
import numpy as np
import glob
import argparse
import re

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts



#paso de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i","--images",required = False,
                help = "ruta a la carpeta de salida")
ap.add_argument("-o","--output",required=  True,
                help = "Ruta al video de salida")

args = vars(ap.parse_args())
path_in = args['images']
path_out = args['output']

# video de salida
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

if __name__=="__main__":



    if path_in: #queremos crear un video con una secuencia de imagenes
        out = cv2.VideoWriter(path_out, fourcc, 33.0, (1280, 1024))

        for infile in sorted(glob.glob(path_in+'/*jpg'), key=numericalSort):

            frame = cv2.imread(infile)

            out.write(frame)
            cv2.imshow('frame', frame)
            cv2.waitKey(33)
        print("\nGuardado video en: ",path_out)
        out.release()
        cv2.destroyAllWindows()


    else: #queremos grabar con la cam
        out = cv2.VideoWriter(path_out, fourcc, 33.0, (640,480))

        mode = 'visualization'
        cap = cv2.VideoCapture(0)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.flip(frame, 0)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    mode = 'save'

                if mode == 'save':
                    out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()


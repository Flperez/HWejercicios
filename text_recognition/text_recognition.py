import cv2
import numpy as np
import argparse
import pytesseract
from PIL import Image

def non_max_suppression_fast(boxes, overlapThresh):
    '''
    Código copiado de:
    https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python
    Se utilizara para refinar los rectangulos

    :param boxes:
    :param overlapThresh:
    :return:
    '''
    # if there are no boxes, return an empty list

    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")





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

        erc1 = cv2.text.loadClassifierNM1(path+'/trained_classifierNM1.xml')
        er1 = cv2.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

        erc2 = cv2.text.loadClassifierNM2(path+'/trained_classifierNM2.xml')
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

# TODO: mejorar el algoritmo de borrado
def delete_rects_contained(rects):
    '''
    Eliminamos las BB contenidas dentro de otras BB
    :param rects:
    :return:
    '''
    deleted = np.array([])
    areas = [(rects[k,0]-rects[k,2])*(rects[k,1]-rects[k,3]) for k in range(0,len(rects))]
    fin = 0
    for i in range(0,len(rects)):
        for j in range(0,len(rects)):
            if areas[i]<=areas[j] and i !=j:
                if contained(rects[i],rects[j])==True:
                    print("[",i,"]","El rectangulo: ",rects[i]," esta contenido en: ",rects[j])
                    deleted=np.append(deleted,i)
    deleted =deleted.astype(int)
    rects=np.delete(rects,(deleted),axis=0)
    return rects

def text_recognition(rects,image):
    print("Reconocimiento de texto")
    offset = 10
    # Visualization
    for r in range(0, np.shape(rects)[0]):
        rect = rects[r]
        image_to_recognise = image[rect[1]-offset:rect[1] + rect[3]+offset,
                             rect[0]-offset:rect[0] + rect[2]+offset]
        text = Image.fromarray(image_to_recognise)
        text = pytesseract.image_to_string(text)
        if text:


            print(rect)
            cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)
            cv2.putText(image,text, (rect[0], rect[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

    return image





#Paso de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = False, help = "ruta a la imagen de entrada")
ap.add_argument("-v","--video",required = False, help = "ruta al video de entrada")
args = vars(ap.parse_args())
path_image = args['image']
path_video= args['video']



if __name__=="__main__":



    if path_video and not path_image: #Se ha introducido un video y no una imagen
        # cargamos video de entrada
        cap = cv2.VideoCapture(path_video)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                rects = text_detection(img=frame)
                vis = text_recognition(rects=rects, image=frame)

                cv2.imshow("Text detection result", vis)
                cv2.waitKey(33)



    if path_image and not path_video: #Se ha introducido una imagen y no un video
        img = cv2.imread(path_image)


        rects = text_detection(img=img)
        print("rects: ",rects)
        rects = non_max_suppression_fast(rects,0.8)
        print("Non_maxima: ",rects)
        rects = delete_rects_contained(rects)
        print("Eliminados los contenidos: \n",rects)

        vis = text_recognition(rects=rects,image=img)

        # Visualization
        cv2.imshow("Text detection result", vis)
        cv2.waitKey()

    if path_image and path_video:
        print("Debe introducir o un video o una imagen pero no ambos")

    cv2.destroyAllWindows()




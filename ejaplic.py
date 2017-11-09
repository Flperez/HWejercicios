import cv2
import matplotlib.pyplot as plt
import numpy as np

path = "/home/f/MEGAsync/Aplicaciones_industriales/Tema4-Ejercicio/cheque.png"
# Lectura
img = cv2.imread(path)
gray_img = cv2.cvtColor(img,code=cv2.COLOR_RGB2GRAY)
plt.set_cmap('gray')
plt.imshow(gray_img)

# Umbralización elemental
binary_img_1 = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

plt.imshow(binary_img_1)
plt.show()

# Umbralización avanzada
# OpenCV incluye una serie de funciones más potentes para realizar esta tarea.
# http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
# Reescalado

w,h = binary_img_1.shape
ratio = w/h

wp=int(800*ratio)
size = 800,wp

binary_800x600_img = np.zeros((600,800))

binary_800x600 = cv2.resize(binary_img_1, size , interpolation =
cv2.INTER_LINEAR)
binary_800x600_img[0:wp,0:800]=binary_800x600


plt.imshow(binary_800x600_img)
plt.show()

# Corrección inclinacion

lines = cv2.HoughLines(binary_img_1,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow("img",img)
cv2.waitKey()

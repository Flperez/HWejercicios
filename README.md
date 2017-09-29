# HWejercicios

# Lista de ejercicios

## Ejercicio 1:
### Desarrollar un módulo de aumentado de datos para redes de convolución en
python

1. Descargar subconjunto de imágenes de ImageNet
    tiny-imagenet200.zip
2. Utilizar el módulo PIL / pillow para cargar/salvar/modificar imágenes
3. Aumentado de datos soportado: blurring, resize, rotate, transpose
- Blur radius: [2, 10]
- Resize: [0.25, 2.5]
- rotate/transpose: PIL.Image.FLIP_LEFT_RIGHT, PIL.Image.FLIP_TOP_BOTTOM, PIL.Image.ROTATE_90,
PIL.Image.ROTATE_180, PIL.Image.ROTATE_270 or PIL.Image.TRANSPOSE

4. El programa recibe un factor de aumentado (5x, 10x, 20x), generando (5, 10, 20) imágenes por
cada imagen de entrada. El nuevo dataset resultante del aumentado se guardará en un directorio
de salida.


## Ejercicio 2
### Embrión para un sistema de seguimiento visual por color en python

Objetivo: encontrar regiones candidatas a objeto

Pasos a seguir:

1. Leer una imagen de una secuencia (pillow)
2. Convertir de RGB a formato HSV (pillow)
3. Dado el rango de color de nuestro objeto, umbralizar el fotograma (numpy arrays)
-  29<H>88, 43<S<255, 126<V<255
4. Presentar regiones candidatas a objeto en la imagen (pillow)
-  blend imagen original y umbralización
5. Salvar blend en un directorio de salida (os)
- conservar nombre original de imagen


## Ejercicio 3
### Uso de matplotlib


- Objetivo: comparar resultados obtenidos por un algoritmo de reconocimiento 
de objetos 3D
 - por cada objeto reconocido se conoce: area2D, area3D, complejidad (número entero)
    - se desea analizar cada característica por separado 
        - generar una gráfica por cada una de ellas
    
- Dos ficheros:
    - detection.csv, groundtruth.csv
- Pasos a seguir:
1. Leer ficheros csv y cargar datos en 2 numpy arrrays (
numpy.loadtxt)
2. Generar una gráfica de barras para el area2D
- cada barra muestra % de objetos cuya area2D difere del original en un rango establecido
        - matplotlib.pyplot.bar
- de forma adicional, la gráfica debe mostrar el % de objetos para el que no se calculo area

    - fallo en el cálculo se marca con carácter - en el fichero
3. Generar gráfica equivalente para area3D
4. Generar una tercera gráfica donde se muestren las diferencias a nivel de complejidad
5. Salvar las gráficas en disco en formato png

- Uso de funciones recomendadas
    - np.count_nonzero, np.isnan, np.su

   
    

#Ejercicio 4
## Uso del notebook de Jupyter
- Convertir el Codelab de Matplotlib en un notebook


## Ejercicio 5
### Detector de caras y ojos
- Detectar caras y ojos en los actores de una película
- Utilizar clasificadores cascada (2001) previamente entrenados
    - el módulo de OpenCV de detección de objetos permite trabajar con estos clasificadores


## Ejercicio 6
### Detector de peatones:
1. Debemos cargar un video de entrada (video_in)
2. Obtener los frames
3. Aplicar el detector de peatones de HOG
4. Pintar con un rectángulo el peatón y guardar el video resultante (video_out)

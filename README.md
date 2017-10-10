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
-  29<H<88, 43<S<255, 126<V<255
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

   
    

## Ejercicio 4
### Uso del notebook de Jupyter
- Convertir el Codelab de Matplotlib en un notebook


## Ejercicio 5
### Detector de caras y ojos
- Detectar caras y ojos en los actores de una película
- Utilizar clasificadores cascada (2001) previamente entrenados
    - el módulo de OpenCV de detección de objetos permite trabajar con estos clasificadores


## Ejercicio 6
### Detector de peatones:
1. Debemos cargar un video de enProcesando:  59.8 %
trada (video_in)
2. Obtener los frames
3. Aplicar el detector de peatones de HOG
4. Pintar con un rectángulo el peatón y guardar el video resultante (video_out)


## Ejercicio 7
### Continuación del ejercicio 2

- Tras la segmentación
1. Eliminar pequeños focos de ruido
    -  Aplicar dos operaciones de erosión consecutivas
        - elemento estructurante cuadrado 3x3
    - Aplicar dos operaciones de dilatación consecutivas
        - elemento estructurante cuadrado 3x3
2.  Detectar distintos contornos que aparecen en la imagen (cv2.findContours )
- Quedarse con el de mayor área
3. Presentar resultado del seguimiento en pantalla

## Ejercicio 7 = mediamatcher.py
### Descriptores 2d

Estamos interesados en identificar imágenes. Para ello vamos a trabajar en un enfoque basado en la detección y matching de puntos característicos. Por suerte, OpenCV presenta un módulo que implementa diferentes descriptores (SURF, SIFT, ORB, etc...) que simplifican la identificación de puntos característicos en imagen. Para comenzar a familiarizarnos con estos puntos característicos lo primero que haremos será seguir el tutorial que se nos proporciona desde la documentación de OpenCV.En él se detallan los pasos a seguir para identificar puntos homólogos en dos imágenes distintas.
Una vez tengamos el código del tutorial funcionando lo vamos a utilizar para identificar imágenes similares. Para ello nos crearemos un conjunto de fotos (podemos hacerlas con el móvil o descargarlas de internet) de carátulas de libros, cds, dvds...(lo que el alumno considere).A partir de aquí el alumno desarrollará una función en Python que reciba dos argumentos:

- Ruta donde se encuentra la imagen a emparejar
- Colección de imágenes
    - mediamatcher.py –query=./cover_The_Hobbit –covers=./my_media_database/

La función deberá mostrar en una ventana la imagen con la queda emparejada la imagen de consulta. De todas las imágenes se seleccionará aquella que obtenga mayor número de ​matches (o puntos emparejados). Fijaremos un umbral de un mínimo de N puntos emparejados. Si varias imágenes superan dicho umbral, se mostrará aquella que presente mayor número de matches.
​
candidata
## Ejercicio 8
### Reconocimiento de texto

Aplicar el módulo de detección y reconocimiento de texto (text) de OpenCV al problema de identificación de matrículas
- El programa puede recibir como entrada una imagen o un vídeo, mostrando el resultado de la detección/reconocimiento en una ventana
- En su versión más simple, el programa debería responder a la siguiente interfaz:

    - python text_recognition.py --image=./media/car.png
    - python text_recognition.py --video=./media/car.avi
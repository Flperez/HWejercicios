# HWejercicios

# Lista de ejercicios

## Ejercicio 1:
Desarrollar un módulo de aumentado de datos para redes de convolución en
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
Embrión para un sistema de seguimiento visual por color en python

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

## Ejercicio 4

## Ejercicio 5
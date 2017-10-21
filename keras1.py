import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from random import randint
import numpy as np
import matplotlib.pyplot as plt

# Reading and setting the input dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_dim = 784  # 28*28
x_train = x_train.reshape(x_train.shape[0], input_dim)
x_test = x_test.reshape(x_test.shape[0], input_dim)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# Building the linear classifier

model = Sequential()
model.add(Dense(units=num_classes, input_dim=input_dim, activation='softmax'))

model.compile( loss='categorical_crossentropy',
optimizer= 'sgd',
metrics=['accuracy'])






# Training the linear classifier
batch_size = 128
nb_epoch = 20

model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size)


# Model evaluation

score = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score[0])
print('Test accuracy:', score[1])

# Inference (predict): testing the trained model over some images (test set)

#TODO: place your code here

#visualizamos una imagen al azar
j = randint(0,10000)
img = x_test[j,:].reshape((28,28))
img = img*255
img = np.array(img,dtype='uint8')

plt.imshow(img,cmap='gray')
plt.title("Imagen al azar")
plt.show()
predictions = model.predict(x_test[[j],:],batch_size=batch_size)
print("El numero de la imagen es: ",np.argmax(predictions))


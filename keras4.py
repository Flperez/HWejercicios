import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from random import randint
import numpy as np
import matplotlib.pyplot as plt


# Reading and setting the input dataset
# Reading and setting the input dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Building the Lenet-5 CNN

model = Sequential()

model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Training the Lenet-5 CNN
batch_size = 128
nb_epoch = 20

model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size)



# Lenet-5 CNN evaluation

score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])



# Inference (predict): testing the trained Lenet-5 over some images (test set)

#TODO: place your code here


j = randint(0,10000)
img = x_test[j,:].reshape((32,32))
img = img*255
img = np.array(img,dtype='uint8')

plt.imshow(img,cmap='gray')
plt.title("Numero elegido")
plt.show()
predictions = model.predict(x_test[[j],:],batch_size=batch_size)
print("La clase de la imagen es: ",np.argmax(predictions))

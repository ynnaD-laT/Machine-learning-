
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from matplotlib import pyplot as plt
import seaborn as sn
import time

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train.shape

x_train = x_train / 255
x_test = x_test / 255

index = 5

plt.imshow(x_train[index], cmap = plt.cm.binary)
print(y_train[index])

x_train_flat = x_train.reshape(len(x_train), 28*28)
x_test_flat = x_test.reshape(len(x_test), 28*28)
x_train_flat.shape

"""# (A.1) Three-Layer Perceptron"""

model_relu = keras.Sequential([
    keras.layers.Dense(256, input_shape = (784,), activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')

])
model_relu.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
start_time = time.time()
model_relu.fit(x_train_flat, y_train, epochs = 5, batch_size=64, validation_split = 0.2)
training_time = time.time() - start_time
print("Training time: ", training_time)

"""# (A.2) Activation Sigmond"""

model_sigmoid = keras.Sequential([
    keras.layers.Dense(256, input_shape = (784,), activation = 'sigmoid'),
    keras.layers.Dense(128, activation = 'sigmoid'),
    keras.layers.Dense(10, activation = 'softmax')

])
model_sigmoid.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

start_time = time.time()
model_sigmoid.fit(x_train_flat, y_train, epochs = 5, batch_size = 64, validation_split = 0.2)
training_time = time.time() - start_time
print("Training time: ", training_time)

"""# (A.3) Optimizer to SGD

"""

model_relu = keras.Sequential([
    keras.layers.Dense(256, input_shape = (784,), activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')

])
model_relu.compile(
    optimizer = "SGD",
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
start_time = time.time()
model_relu.fit(x_train_flat, y_train, epochs = 5, batch_size=64, validation_split = 0.2)
training_time = time.time() - start_time
print("Training time: ", training_time)

"""# (A.4) 5-layer perceptron


"""

model_relu = keras.Sequential([
    keras.layers.Dense(256, input_shape = (784,), activation = 'relu'),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')

])
model_relu.compile(
    optimizer = "adam",
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
start_time = time.time()
model_relu.fit(x_train_flat, y_train, epochs = 5, batch_size=64, validation_split = 0.2)
training_time = time.time() - start_time
print("Training time: ", training_time)

"""# (A.4) Activation Sigmoid 5-layer"""

model_relu = keras.Sequential([
    keras.layers.Dense(256, input_shape = (784,), activation = 'sigmoid'),
    keras.layers.Dense(256, activation = 'sigmoid'),
    keras.layers.Dense(128, activation = 'sigmoid'),
    keras.layers.Dense(128, activation = 'sigmoid'),
    keras.layers.Dense(10, activation = 'softmax')

])
model_relu.compile(
    optimizer = "adam",
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
start_time = time.time()
model_relu.fit(x_train_flat, y_train, epochs = 5, batch_size=64, validation_split = 0.2)
training_time = time.time() - start_time
print("Training time: ", training_time)

"""# (A.4) Change optimizer to SGD"""

model_relu = keras.Sequential([
    keras.layers.Dense(256, input_shape = (784,), activation = 'relu'),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')

])
model_relu.compile(
    optimizer = "SGD",
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
start_time = time.time()
model_relu.fit(x_train_flat, y_train, epochs = 5, batch_size=64, validation_split = 0.2)
training_time = time.time() - start_time
print("Training time: ", training_time)

"""# Task B: CNN
## (B.1): 4-layer CNN
"""

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

from keras import layers
from keras import models

model_CNN = models.Sequential()
model_CNN.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_CNN.add(layers.MaxPooling2D((2, 2)))
model_CNN.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_CNN.add(layers.MaxPooling2D((2, 2)))
model_CNN.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_CNN.add(layers.Flatten())
model_CNN.add(layers.Dense(10, activation='softmax'))

model_CNN.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

start_time = time.time()
model_CNN.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split = 0.2)
training_time = time.time() - start_time

print("Training time: ", training_time)

"""# VGG-16 CNN"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input
import time

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

def preprocess_mnist(images):
    images = tf.expand_dims(images, -1)
    images = tf.image.resize(images, [32, 32])
    images = tf.repeat(images, 3, axis=-1)
    return images.numpy() / 255.0

train_images = preprocess_mnist(train_images)
test_images = preprocess_mnist(test_images)


train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)


model_VGG = models.Sequential([
    Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='tanh', padding='same'),
    layers.Conv2D(32, (3, 3), activation='tanh', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='tanh', padding='same'),
    layers.Conv2D(64, (3, 3), activation='tanh', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='tanh'),
    layers.Dense(10, activation='softmax')
])

model_VGG.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


start_time = time.time()
model_VGG.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split = 0.2)
training_time = time.time() - start_time
print("Training time: ", training_time)

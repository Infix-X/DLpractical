import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Conv2D,Dense,MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()



print(X_train.shape)



X_train[0].min(), X_train[0].max()


X_train = (X_train - 0.0) / (255.0 - 0.0)
X_test = (X_test - 0.0) / (255.0 - 0.0)
X_train[0].min(), X_train[0].max()


def plot_digit(image, digit, plt, i):
  plt.subplot(4, 5, i + 1)
  plt.imshow(image, cmap=plt.get_cmap('gray'))
  plt.title(f"Digit: {digit}")
  plt.xticks([])
  plt.yticks([])
  plt.figure(figsize=(16, 10))


for i in range(20):
  plot_digit(X_train[i], y_train[i], plt, i)
  plt.show()

X_train = X_train.reshape((X_train.shape + (1,)))
X_test = X_test.reshape((X_test.shape + (1,)))

y_train = np.array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9], dtype=np.uint8)
print(y_train[0:20])

model = Sequential([
Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
MaxPooling2D((2, 2)),
Flatten(),
Dense(100, activation="relu"),
Dense(10, activation="softmax")
])


from tensorflow.keras.optimizers import SGD
optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer,
loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.summary()


(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(len(x_train), len(y_train))


model.fit(X_train, y_train, epochs=10, batch_size=32)


plt.figure(figsize=(16, 10))
for i in range(20):
 image = random.choice(X_test).squeeze()
 digit = np.argmax(model.predict(image.reshape((1, 28, 28, 1)))[0],
 axis=-1)
 plot_digit(image, digit, plt, i)
plt.show()


plt.figure(figsize=(16, 10))
for i in range(20):
    image = random.choice(X_test).squeeze()
    digit = np.argmax(model.predict(image.reshape((1, 28, 28, 1)), axis=-1)
    # Make sure you have the correct model for predictions
    # Also, make sure you have defined the plot_digit function
    plot_digit(image, digit, plt, i)

plt.show()

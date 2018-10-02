from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# loading data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(len(train_labels))
print(train_images)
print(train_labels)

digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

print(test_images.shape)
print(len(test_images))
print(test_images)
print(test_labels)

# preprocessing data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# building the network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=5, batch_size=128)

# evalauting metrics
train_loss, train_acc = network.evaluate(train_images, train_labels)
print('train_acc:', train_acc)
print('train_loss:', train_loss)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print('test_loss:', test_loss)

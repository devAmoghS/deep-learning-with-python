from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

model = models.Sequential()
# a convnet take an input tensor of shape (image_height, image_width, image_channels)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# printing the model architecture
model.summary()

# flattening 3D tensor to 1D tensor
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
# performs a 10-way classification
model.add(layers.Dense(10, activation='softmax'))

# printing the model architecture
model.summary()

# Loading the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocessing images into floating-point tensors
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

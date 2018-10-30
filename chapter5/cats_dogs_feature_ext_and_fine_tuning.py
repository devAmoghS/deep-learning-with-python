import os

from keras import layers
from keras import models, optimizers
from keras.applications import VGG16
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

base_dir = '/home/amogh/Downloads/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)


def get_model():
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    print('This is the number of trainable weights before unfreezing some of the conv base: ',
          len(model.trainable_weights))

    conv_base.trainable = True

    # fine tuning starts here
    set_trainable = False  # type: bool
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        layer.trainable = set_trainable

    model.summary()

    print('This is the number of trainable weights after unfreezing some of the conv base: ',
          len(model.trainable_weights))

    model.compile(
        optimizer=optimizers.RMSprop(lr=0.00001),
        loss='binary_crossentropy',
        metrics=['acc']
    )
    return model


def train_model(model):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50
    )
    return history


def plot_learning_curves(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def plot_smooth_learning_curves(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, smooth_curve(acc), 'bo', label='Training acc')
    plt.plot(epochs, smooth_curve(val_acc), 'b', label='Validation acc')
    plt.title('Smoothed Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, smooth_curve(loss), 'bo', label='Training loss')
    plt.plot(epochs, smooth_curve(val_loss), 'b', label='Validation loss')
    plt.title('Smoothed Training and validation loss')
    plt.legend()
    plt.show()


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def evaluate_model(model):
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
    test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
    print('test_acc:', test_acc)
    print('test_loss: ', test_loss)


model = get_model()
history = train_model(model)
plot_learning_curves(history)
plot_smooth_learning_curves(history)
evaluate_model(model)

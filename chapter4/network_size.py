from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Take only the top 10000 most frequently occurring words.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def explore():
    print(train_data.shape)
    print(train_labels.shape)

    print(test_data.shape)
    print(test_labels.shape)

    print(train_data[0])
    print(train_labels[0])

    print(max(max(sequence) for sequence in train_data))

    # word_index is a dictionary mapping: word -> int indices
    word_index = imdb.get_word_index()
    # reversing it, mapping becomes int indices -> word
    reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join(reversed_word_index.get(i-3, '?') for i in train_data[0])
    print(decoded_review)


explore()

"""Encoding the integer sequence into a binary matrix"""


def vectorise_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    # print(results)
    # print(sequences)
    # print(enumerate(sequences))
    for i, sequence in enumerate(sequences):
        # print(i, sequence)
        results[i, sequence] = 1.
    return results


# Vectorise features
x_train = vectorise_sequences(train_data)
x_test = vectorise_sequences(test_data)
print(x_train[0])
print(x_train[0].shape)


# Vectorise labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(y_train[0])
print(y_train[0].shape)

""" The input data are vectors and the labels are scalars (1s and 0s). The network
    we choose is a simple stack of Dense layers with `relu` activation
    ====================
    NETWORK ARCHITECTURE
    ====================
    - Two intermediate layers with 16 hidden units each (activation = 'relu')
      To zero out the negative values
    - A third layer that will be the output the scalar prediction(sentiment of the current review)
      (activation = 'sigmoid') To output the score between 0 and 1 (how likely is the review positive)
"""


def get_original_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def get_smaller_model():
    model = models.Sequential()
    model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def get_larger_model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


# Model and Layers Definition


x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


def get_losses(model):
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    return history.history['loss'], history.history['val_loss']


original_loss, original_val_loss = get_losses(get_original_model())
smaller_loss, smaller_val_loss = get_losses(get_smaller_model())
larger_loss, larger_val_loss = get_losses(get_original_model())


epochs = range(1, len(original_loss) + 1)

plt.plot(epochs, original_loss, 'bo', label='Original model loss')
plt.plot(epochs, smaller_loss, 'b', label='Smaller model loss')
plt.plot(epochs, larger_loss, 'ro', label='Larger model loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.show()


plt.plot(epochs, original_val_loss, 'bo', label='Original val model loss')
plt.plot(epochs, smaller_val_loss, 'b', label='Smaller val model loss')
plt.plot(epochs, larger_val_loss, 'ro', label='Larger val model loss')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()

import os

import numpy as np
import matplotlib.pyplot as plt
from keras import preprocessing
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras_preprocessing.text import Tokenizer

"""Preprocessing labels of raw IMDB data"""
imdb_dir = '/home/amogh/Downloads/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)

    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()

            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

"""Tokenising the text of raw IMDB data"""

maxlen = 20
training_samples = 500
validation_samples = 10000
max_words = 10000

tokeniser = Tokenizer(num_words=max_words)
tokeniser.fit_on_texts(texts)
sequences = tokeniser.texts_to_sequences(texts)
word_index = tokeniser.word_index
print('Found %s unique tokens.' % len(word_index))

data = preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# splits the data into training and validation
# shuffles the data because its is ordered
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

embedding_dim = 100

"""Model Definition"""
model = Sequential()
# generates the 3D tensors of shape(max_words, maxlen, embedding_dim)
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# flattens the tensor to shape(max_words, maxlen * embedding_dim)
model.add(Flatten())

# adding the classifier on top
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

"""Training and Evaluation"""
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
model.save_weights('pre_trained_glove_model.h5')

"""Plotting the results"""
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

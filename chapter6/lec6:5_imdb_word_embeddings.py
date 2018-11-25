from keras import preprocessing
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

# no. of words to consider as features
max_features = 10000
# cuts off the text after these many words
maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# turns list of integers into 2D integer tensors of shape(samples, maxlen)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
# generates the 3D tensors of shape(samples, maxlen, 8)
model.add(Embedding(10000, 8, input_length=maxlen))
# flattens the tensor to shape(samples, maxlen * 8)
model.add(Flatten())

# adding the classifier on top
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

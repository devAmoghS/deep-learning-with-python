from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

"""Data Exploration"""
# print(train_data.shape)
# print(test_data.shape)
# print(train_targets)

"""Normalization of data"""

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
# test data will be normalized with mean and std of train
test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    # train_data.shape[1] gives columns (# of features)
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(64, activation='relu'))
    # activation is 'linear' in the last layer
    model.add(layers.Dense(1))
    # 'mse' is a widely used loss function for scalar regression
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    return model


"""K-folds validation"""
k = 4
num_val_samples = len(train_data) // k

num_epochs = 500
all_mae_history = []

for i in range(k):
    print("processing fold # ", i)
    # prepare validation data from partition #k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # prepare the training data from other partitions
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    # building the keras model
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=1, verbose=0)

    # evaluate the model on the validation data
    # print(history.history.keys())
    mae_history = history.history['val_mean_absolute_error']
    # val_mse, val_mae = model.evaluate(val_data, val_targets)
    all_mae_history.append(mae_history)

# print(all_mae_history)
avg_mae_history = [
    np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]

"""Plotting validation scores"""
plt.plot(range(1, len(avg_mae_history) + 1), avg_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smoothed_mae_history = smooth_curve(avg_mae_history[10:])
plt.plot(range(1, len(smoothed_mae_history) + 1), smoothed_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# Training the final model

model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)

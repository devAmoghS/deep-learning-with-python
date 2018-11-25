import os

import matplotlib.pyplot as plt
import numpy as np

data_dir = '/home/amogh/Downloads/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values


def my_plot():
    temp = float_data[:, 1]
    # Temperature over the full temporal range of the dataset
    plt.plot(range(len(temp)), temp)
    plt.figure()
    # Temperature over the first 10 days of the dataset
    plt.plot(range(1440), temp[:1440])
    plt.show()
    return


my_plot()

# Normalizing the data
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

print(float_data.shape)


# Generator yielding timeseries samples and their targets
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples[:, ::-1, :], targets


# Preparing the training, validation, and test generators
lookback = 1440  # 10 days
step = 6  # 1 hour
delay = 144  # 1 day
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay,
                      min_index=0, max_index=200000, shuffle=True,
                      step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay,
                      min_index=200001, max_index=300000,
                      step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay,
                      min_index=300001, max_index=None,
                      step=step, batch_size=batch_size)

# // batch_size is missing in the book!
val_steps = (300000 - 200001 - lookback) // batch_size
print(val_steps)
test_steps = (len(float_data) - 300001 - lookback) // batch_size
print(test_steps)


def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        # print(mae)
        batch_maes.append(mae)
    return np.mean(batch_maes)


ans = evaluate_naive_method()
print(ans * std[1])

import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            # assign a unique index to each unique word. nothing is 0 indexed
            token_index[word] = len(token_index) + 1

# stores the words as vectors of size 1000
dimensionality = 1000
# vectorise the samples. Consider only the first `max-length` words in each sample
max_length = 10

results = np.zeros(shape=(len(samples), max_length, dimensionality))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1

print(results)

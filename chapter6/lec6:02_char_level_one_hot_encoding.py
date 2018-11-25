import numpy as np
import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable
# length of characters is 100 (all printable ASCII characters)
token_index = dict(zip(range(1, len(characters) + 1), characters))

# vectorise the samples. Consider only the first `max-length` characters in each sample
max_length = 50

results = np.zeros(shape=(len(samples), max_length, max(token_index.keys()) + 1))

for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1

print(results)

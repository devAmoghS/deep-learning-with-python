import numpy as np
import os

from keras import models
from keras.models import load_model
from keras_preprocessing import image
from matplotlib import pyplot as plt

model = load_model('cats_and_dogs_small_2.h5')
model.summary()

base_dir = '/home/amogh/Downloads/cats_and_dogs_small'
test_dir = os.path.join(base_dir, 'test')
img_path = os.path.join(test_dir, 'cats', 'cat.1700.jpg')
img = image.load_img(img_path, target_size=(150, 150))

img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.      # scaling the input as before

print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

# Extracts the output of top eight layers
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)
# visualising the fourth channel
plt.matshow(first_layer_activation[0, :, :, 6], cmap='viridis')
plt.show()

# visualising every channel in every intermediate activation
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    # number of features in feature maps
    n_features = layer_activation.shape[-1]

    # shape of feature map is (l, size, size, n_features)
    size = layer_activation.shape[1]

    # tiles the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]

            channel_image -= channel_image.mean()
            channel_image /= channel_image.std() + 1e-5
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size: (col + 1) * size,
                         row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

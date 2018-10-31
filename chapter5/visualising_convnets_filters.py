import numpy as np
from matplotlib import pyplot as plt
from keras.applications import VGG16
from keras import backend as K

model = VGG16(weights='imagenet', include_top=False)

# layer_name = 'block3_conv1'
# filter_index = 0


# function to convert a tensor into a valid image
def deprocess_image(x):
    # normalizing the tensor
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clips to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # returns a list of tensor of size, hence we pick the first element
    grads = K.gradients(loss, model.input)[0]

    # smoothing of the gradient: divide by its L2 norm
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # fetching numpy output values, given input values
    iterate = K.function([model.input], [loss, grads])

    # now we define a loop to perform stochastic gradient descent
    input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

    # magnitude for each gradient update
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        # adjust the input image in the direction that maximize the loss
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)


# Generating a grid of all filter response patterns in a layer
for layer_name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']:
    print('Running for layer: ', layer_name)
    # TODO: check for `size=64`
    size = 150
    margin = 5

    # empty black images to store results
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    for i in range(8):
        for j in range(8):
            filter_img = generate_pattern(layer_name=layer_name,
                                          filter_index=i + (j * 8),
                                          size=size)

            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size

            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            print(i, j)
            print(filter_img.shape)
            print(results[horizontal_start: horizontal_end, vertical_start: vertical_end, :].shape)
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img[:, :, :]

    plt.figure(figsize=(20, 20))
    plt.title(layer_name)
    plt.imshow(results)
    plt.show()

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
from keras import backend as K
import cv2
import matplotlib.pyplot as plt

model = VGG16(weights='imagenet')  # we include the classifier here
img_path = '/home/amogh/Downloads/creative_commons_elephant.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)  # converts to float32 numpy array of shape
x = np.expand_dims(x, axis=0)  # adds a dimension to give shape: (1, 224, 224, 3)
x = preprocess_input(x)  # performs channle wise color normalisation

preds = model.predict(x)
print('Predicted: ', decode_predictions(preds, top=3)[0])

# Setting up the Grad-CAM algorithm

# 386 is the index of "African elephant" class, which was maximally activated
african_elephant_output = model.output[:386]
last_conv_layer = model.get_layer('block5_conv3')

# gradient of the "African elephant" class with regard to output feature map of 'block5_conv5'
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
# each entry is mean intensity of gradient over specific channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

# Multiplies each channel in the feature map array by "how important the channel is" wrt "elephant" class
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# channel wise mean of the feature map is the heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)

# heat map post-processing
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

# superimposing heatmap with the original image
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # resize heatmap to same size as original image
heatmap = np.uint8(255 * heatmap)                            # converts heatmap to RGB
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img                       # 0.4 is the intensity factor
cv2.imwrite('/home/amogh/Downloads/elephant_cam.jpg', superimposed_img)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

tf.experimental.numpy.experimental_enable_numpy_behavior()
tf.config.run_functions_eagerly(True)

train = [2, 6]
test = [3, 7]
batch_size = 2 
target_size = (256, 256)
initial_j = 1.0  # Initial value for the learnable parameter j

# Define the image path format
image_path_format = r'C:\Users\jenna\AppData\Local\Programs\Python\Python312\Lib\site-packages\cv2\data\{}_image.tiff'
mask_path_format = r'C:\Users\jenna\AppData\Local\Programs\Python\Python312\Lib\site-packages\cv2\data\{}_mask.tiff'

# Read train images and masks
train_images = []
train_masks = []
for i in train:
    img = cv2.imread(image_path_format.format(i), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path_format.format(i), cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, target_size)
    resized_mask = cv2.resize(mask, target_size)
    train_images.append(resized_img)  # Append resized images
    train_masks.append(resized_mask)  # Append resized masks

# Read test images and masks
test_images = []
test_masks = []
for i in test:
    img = cv2.imread(image_path_format.format(i), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path_format.format(i), cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, target_size)
    resized_mask = cv2.resize(mask, target_size)
    test_images.append(resized_img)  # Append resized images
    test_masks.append(resized_mask)  # Append resized masks

# Convert lists to numpy arrays and add channel dimension
train_images = np.expand_dims(np.array(train_images), axis=-1)
train_masks = np.expand_dims(np.array(train_masks), axis=-1)
test_images = np.expand_dims(np.array(test_images), axis=-1)
test_masks = np.expand_dims(np.array(test_masks), axis=-1)

# Normalize the images
train_images = train_images / 255.0
train_masks = train_masks / 255.0
test_images = test_images / 255.0
test_masks = test_masks / 255.0

# Define custom layers
class TopHatLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TopHatLayer, self).__init__()

    def build(self, input_shape):
        self.kernel = tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=tf.float32)
        self.kernel = self.kernel[:, :, tf.newaxis]

    def call(self, inputs):
        erosion = tf.nn.erosion2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', dilations=[1, 1, 1, 1])
        dilation = tf.nn.dilation2d(erosion, self.kernel, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', dilations=[1, 1, 1, 1])
        top_hat = inputs - dilation
        return top_hat

class BlackHatLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(BlackHatLayer, self).__init__()

    def build(self, input_shape):
        self.kernel = tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=tf.float32)
        self.kernel = self.kernel[:, :, tf.newaxis]

    def call(self, inputs):
        dilation = tf.nn.dilation2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', dilations=[1, 1, 1, 1])
        erosion = tf.nn.erosion2d(dilation, self.kernel, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC', dilations=[1, 1, 1, 1])
        black_hat = erosion - inputs
        return black_hat

class MaxTransformLayer(tf.keras.layers.Layer):
    def __init__(self, initial_j):
        super(MaxTransformLayer, self).__init__()
        self.j = tf.Variable(initial_j, dtype=tf.float32, trainable=True)

    def call(self, inputs):
        top_hat, black_hat = tf.split(inputs, num_or_size_splits=2, axis=-1)
        D_j_n = tf.maximum(top_hat, black_hat)
        w_j = 1 / (2 * self.j + 1)
        weighted_D_j_n = w_j * D_j_n
        return weighted_D_j_n

# Define the model using the functional API
inputs = tf.keras.Input(shape=(256, 256, 1))
top_hat = TopHatLayer()(inputs)
black_hat = BlackHatLayer()(inputs)
concatenated = tf.keras.layers.Concatenate(axis=-1)([top_hat, black_hat])
max_transform = MaxTransformLayer(initial_j)(concatenated)
outputs = max_transform  # This is where you can add more layers as needed
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_masks, batch_size=batch_size, epochs=10, validation_data=(test_images, test_masks))

# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_masks)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp 
import cv2
import scipy
import torch
from skimage.morphology import binary_opening, binary_closing, thin, disk, remove_small_objects, remove_small_holes
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage.util import invert

tf.experimental.numpy.experimental_enable_numpy_behavior()
tf.config.run_functions_eagerly(True)

train = [2, 6]
test = [3, 7]
batch_size = 2 
target_size = (256, 256)

# Define the image path format
image_path_format = r'C:\Users\jenna\AppData\Local\Programs\Python\Python312\Lib\site-packages\cv2\data\{}_image.tiff'
mask_path_format = r'C:\Users\jenna\AppData\Local\Programs\Python\Python312\Lib\site-packages\cv2\data\{}_mask.tiff'
# Read train images and masks
train_images = []
train_masks = []
test_images = []
test_masks = []
original_train_sizes = []
original_test_sizes = []

for i in train:
    img = cv2.imread(image_path_format.format(i), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path_format.format(i), cv2.IMREAD_GRAYSCALE)
    img = np.array(img)
    mask = np.array(mask)
    original_train_sizes.append(img.shape[:2])  # Store the original size
    resized_img = cv2.resize(img, target_size)
    resized_mask = cv2.resize(mask, target_size)
    train_images.append(resized_img)  # Append resized images
    train_masks.append(resized_mask)  # Append resized masks

for i in test:
    img = cv2.imread(image_path_format.format(i), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path_format.format(i), cv2.IMREAD_GRAYSCALE)
    img = np.array(img)
    mask = np.array(mask)
    original_test_sizes.append(img.shape[:2])  # Store the original size
    resized_img = cv2.resize(img, target_size)
    resized_mask = cv2.resize(mask, target_size)
    test_images.append(resized_img)  # Append resized images
    test_masks.append(resized_mask)  # Append resized masks
    
# Convert lists to numpy arrays
train_images = np.array(train_images)
train_masks = np.array(train_masks)
test_images = np.array(test_images)
test_masks = np.array(test_masks)

# Add channel dimension to input images
train_images = np.expand_dims(train_images, axis=-1)
train_masks = np.expand_dims(train_masks, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)
test_masks = np.expand_dims(test_masks, axis=-1)

# Convert numpy arrays to tensors
train_images = tf.convert_to_tensor(train_images)
train_masks = tf.convert_to_tensor(train_masks)
test_images = tf.convert_to_tensor(test_images)
test_masks = tf.convert_to_tensor(test_masks)

# Print shapes for debugging
print(train_images.shape)
print(train_masks.shape)
print(test_images.shape)
print(test_masks.shape)

train_images = tf.cast(train_images, tf.float32) / 255.0
train_masks = tf.cast(train_masks, tf.float32) / 255.0
test_images = tf.cast(test_images, tf.float32) / 255.0
test_masks = tf.cast(test_masks, tf.float32) / 255.0

# Custom loss function for image segmentation
def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - (numerator + 1) / (denominator + 1)  # Add smoothing to avoid division by zero

# Define custom layers
class TopHatLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TopHatLayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(5, 5, 1, 1), initializer='ones', trainable=False, name='tophat_kernel')

    def call(self, inputs):
        # Perform the top hat operation (morphological opening - original image)
        opening = tf.nn.conv2d(inputs, self.kernel, [1, 1, 1, 1], 'SAME')
        top_hat = inputs - opening
        return top_hat

class BlackHatLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(BlackHatLayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(5, 5, 1, 1), initializer='ones', trainable=False, name='blackhat_kernel')

    def call(self, inputs):
        # Perform the black hat operation (morphological closing - original image)
        dilation = tf.nn.conv2d(inputs, self.kernel, [1, 1, 1, 1], 'SAME')
        black_hat = dilation - inputs
        return black_hat

class MultiScaleFocusLayer(tf.keras.layers.Layer):
    def __init__(self, scales):
        super(MultiScaleFocusLayer, self).__init__()
        self.scales = scales

    def build(self, input_shape):
        self.conv_layers = []
        for scale in self.scales:
            conv_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation=None, padding='same')
            self.conv_layers.append(conv_layer)

    def call(self, inputs):
        # Perform convolutions at multiple scales and combine using weighted summation
        multiscale_output = 0
        for scale, conv_layer in zip(self.scales, self.conv_layers):
            conv_output = conv_layer(inputs)
            multiscale_output += conv_output * self._compute_weight(scale)
        return multiscale_output
    def _compute_weight(self, scale):
        return 1 / (2 * scale + 1)

class FocusRegionLayer(tf.keras.layers.Layer):
    def __init__(self, window_size, iterations, threshold, batch_size):
        super(FocusRegionLayer, self).__init__()
        self.window_size = window_size
        self.iterations = iterations
        self.threshold = threshold
        self.batch_size = batch_size

    def call(self, inputs):
        # Extract the focus measurement values
        R_n_M = inputs
        # R_n_M = tf.expand_dims(inputs, axis=0)

        # Median filtering
        median_filtered = self.median_filter(R_n_M)

        # Morphological skeleton extraction
        skeleton = self.morphological_skeleton(median_filtered, self.threshold)

        # Median filtering again
        R_n_MS = self.median_filter(skeleton)
        R_n_MS_bool = R_n_MS > self.threshold
        # Determine focus regions based on thresholding
        P_n_d = tf.cast(tf.reduce_max(inputs) - tf.reduce_max(inputs), dtype=tf.float32)
        P_n_d_bool = inputs > self.threshold
        # Combine skeleton and thresholded focus regions
        R_n_d_bool = tf.math.logical_or(R_n_MS_bool, P_n_d_bool)
        # Create trimaps
        R_n_d = tf.cast(R_n_d_bool, dtype=tf.float32)
        max_focus = tf.reduce_max(R_n_d, axis=-1)
        max_focus = tf.expand_dims(max_focus, axis=-1)
        max_focus = tf.tile(max_focus, multiples=[1, 1, 1, 1])
        max_focus_bool = max_focus == 0
        # T_n_0 = tf.where(tf.logical_xor(R_n_d_bool, max_focus_bool)) 
        # max_focus = tf.logical_and(R_n_d_bool, max_focus_bool)
        # T_n_0 = tf.logical_xor(R_n_d_bool, max_focus_bool)
        # T_n = tf.where(R_n_d, tf.where(max_focus, 1.0, 0.5), 0.0)

        return median_filtered

    def median_filter(self, input_data):
        # Median filtering using a sliding window approach
        # Prepare the kernel for median filtering
        kernel_shape = [self.window_size[0], self.window_size[1], 1, 1]
        kernel = tf.ones(kernel_shape) / (self.window_size[0] * self.window_size[1])

        # Apply depthwise convolution for median filtering
        median_filtered = tf.nn.depthwise_conv2d(input_data, kernel, strides=[1, 1, 1, 1], padding='SAME')
        return median_filtered

    def morphological_skeleton(self, input,threshold):
        # Thresholding using TensorFlow operations
        mask = input < threshold  # Adjust the threshold as needed

        # Perform morphological operations using TensorFlow functions
        mask = tf.cast(input < threshold, tf.float32)
        mask = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', activation='relu')(mask)
        mask = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(mask)
        mask = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', activation='relu')(mask)
        mask = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(mask)
        return mask


model = tf.keras.Sequential([
    # Input layer specifying the input shape including batch size
    tf.keras.layers.InputLayer(input_shape=(256, 256, 1)),
    # Add layers as needed
    TopHatLayer(),  # Top Hat layer
    BlackHatLayer(),  # Black Hat layer
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),  # Batch normalization
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),  # Batch normalization
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),  # Batch normalization
    tf.keras.layers.Conv2D(filters=3, kernel_size=1, activation='relu', padding='same'),  # Expecting 3 channels here for MultiScaleFocusLayer
    MultiScaleFocusLayer(scales=[1, 2, 3]),  # Multi-scale focus layer
    tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid', padding='same'),
    FocusRegionLayer(window_size=(3, 3), iterations=3, threshold=0.5, batch_size=batch_size)  # Median filtering layer
])

# Compile the model with appropriate loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=dice_loss, metrics=['accuracy'])

history = model.fit(train_images, train_masks, epochs=10, validation_data=(test_images, test_masks), batch_size=2) # Explicitly set batch size

# Retrieve loss and accuracy from training history
loss = history.history['loss'][-1]
accuracy = history.history['accuracy'][-1]

print("Test Dice Loss:", loss)
print("Test Accuracy:", accuracy)

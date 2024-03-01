import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp 
import cv2

train = [2, 6]
test = [3, 7]

target_size = (256, 256)

# Define the image path format
image_path_format = r'C:\Users\jenna\AppData\Local\Programs\Python\Python312\Lib\site-packages\cv2\data\{}_image.tiff'

# Read train images and masks
train_images = []
train_masks = []
original_train_sizes = []
original_test_sizes = []

for i in train:
    img = cv2.imread(image_path_format.format(i), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(image_path_format.format(i), cv2.IMREAD_GRAYSCALE)
    img = np.array(img)
    mask = np.array(mask)
    original_train_sizes.append(img.shape[:2])  # Store the original size
    resized_img = cv2.resize(img, target_size)
    resized_mask = cv2.resize(mask, target_size)
    train_images.append(resized_img)  # Append resized images
    train_masks.append(resized_mask)  # Append resized masks

# Read test images and masks
test_images = []
test_masks = []

for i in test:
    img = cv2.imread(image_path_format.format(i), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(image_path_format.format(i), cv2.IMREAD_GRAYSCALE)
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

# Print shapes for debugging
print(train_images.shape)
print(train_masks.shape)
print(test_images.shape)
print(test_masks.shape)

train_images = train_images.astype('float32') / 255.0
train_masks = train_masks.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
test_masks = test_masks.astype('float32') / 255.0

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
            multiscale_output += conv_output
        return multiscale_output

class MultiScaleFocusMeasureLayer(tf.keras.layers.Layer):
    def __init__(self, scales):
        super(MultiScaleFocusMeasureLayer, self).__init__()
        self.scales = scales

    def build(self, input_shape):
        self.conv_layers = []
        for scale in self.scales:
            conv_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=scale, activation=None, padding='same')
            self.conv_layers.append(conv_layer)

    def call(self, inputs):
        # Perform convolutions at multiple scales and combine using weighted summation
        multiscale_output = 0
        for scale, conv_layer in zip(self.scales, self.conv_layers):
            conv_output = conv_layer(inputs)
            multiscale_output += conv_output
        return multiscale_output

class FocusRegionLayer(tf.keras.layers.Layer):
    def __init__(self, threshold):
        super(FocusRegionLayer, self).__init__()
        self.threshold = threshold

    def call(self, inputs):
        max_value = tf.reduce_max(inputs, axis=-1, keepdims=True)
        focus_regions = tf.where(inputs > max_value - self.threshold, 1, 0)
        return focus_regions

class MedianFilterLayer(tf.keras.layers.Layer):
    def __init__(self, window_size):
        super(MedianFilterLayer, self).__init__()
        self.window_size = window_size

    def call(self, inputs):
        # Prepare the kernel for median filtering
        kernel_shape = [self.window_size[0], self.window_size[1], 1, 1]
        kernel = tf.ones(kernel_shape) / (self.window_size[0] * self.window_size[1])
        
        # Apply depthwise convolution for median filtering
        median_filtered = tf.nn.depthwise_conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='SAME')
        return median_filtered


class SkeletonExtractionLayer(tf.keras.layers.Layer):
    def __init__(self, iterations):
        super(SkeletonExtractionLayer, self).__init__()
        self.iterations = iterations

    def call(self, inputs):
        skeleton = tf.keras.morphology.skeletonize(inputs, iterations=self.iterations)
        return skeleton

class TrimapsLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        uncertain_regions = tf.where(tf.logical_and(inputs == 0, tf.reduce_max(inputs, axis=-1) == 1), 0.5, 0)
        trimaps = tf.where(inputs == 1, 1, uncertain_regions)
        return trimaps

# Define your neural network model
model = tf.keras.Sequential([
    # Input layer specifying the input shape
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
    tf.keras.layers.Conv2D(filters=3, kernel_size=1, activation='relu', padding='same'),  # Number of channels should match the number of scales
    MultiScaleFocusLayer(scales=[1, 2, 3]),  # Multi-scale focus layer
    tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid', padding='same'),
    MedianFilterLayer(window_size=(3, 3))  # Median filtering layer
])

# Compile the model with appropriate loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=dice_loss, metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_masks, epochs=50, validation_data=(test_images, test_masks), batch_size=2) # Explicitly set batch size

# Retrieve loss and accuracy from training history
loss = history.history['loss'][-1]
accuracy = history.history['accuracy'][-1]

print("Test Dice Loss:", loss)
print("Test Accuracy:", accuracy)
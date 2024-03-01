import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

train = [2, 3, 6]
test = [7]

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
        self.num_scales = len(self.scales)
        # Initialize weights for each scale
        self.weights_list = [1 / (2 * j + 1) for j in self.scales]
    
    def call(self, inputs):
        # Ensure the input shape is compatible
        if inputs.shape[-1] != self.num_scales:
            raise ValueError(f"Number of input channels ({inputs.shape[-1]}) does not match the number of scales ({self.num_scales}).")
        
        # Compute focus measurement value D^n_j(x,y) for each scale and take the maximum
        D_n = None
        for i, j in enumerate(self.scales):
            # Compute D^n_j(x,y) for the current scale
            D_j_n = inputs[..., i] * self.weights_list[i]
            # Update D_n if it's None or take the maximum
            D_n = D_j_n if D_n is None else tf.maximum(D_n, D_j_n)
        
        return D_n

# Define your neural network model
model = tf.keras.Sequential([
    # Add layers as needed
    TopHatLayer(),  # Top Hat layer
    BlackHatLayer(),  # Black Hat layer
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    # Add more layers...
    tf.keras.layers.Conv2D(filters=3, kernel_size=1, activation='relu', padding='same'),  # Number of channels should match the number of scales
    MultiScaleFocusLayer(scales=[1, 2, 3]),  # Multi-scale focus layer
    # Add more layers...
    tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid', padding='same')
])

# Compile the model with appropriate loss function
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

# Train the model
model.fit(train_images, train_masks, epochs=10, validation_data=(test_images, test_masks), batch_size=2) # Explicitly set batch size

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_images, test_masks)

print("Test Dice Loss:", loss)
print("Test Accuracy:", accuracy)
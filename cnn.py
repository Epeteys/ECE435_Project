def conv2d(x, W, b, s=1):
    conv = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='VALID')
    conv = tf.nn.bias_add(conv, b)
    return tf.nn.relu(conv)

def maxpooling2D(x, k=2):
    return tf.nn.max_pool(x,
                         ksize=[1, k, k, 1],
                         strides=[1, k, k, 1],
                         padding="SAME")
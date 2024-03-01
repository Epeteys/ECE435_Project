def get_weights_biases(mu, sigma):
    weights = {
        'wc1' : tf.Variable(tf.truncated_normal([5, 5, 1, 6], mu, sigma)),
        'wc2' : tf.Variable(tf.truncated_normal([5, 5, 6, 16], mu, sigma)),
        'wd1' : tf.Variable(tf.truncated_normal([400, 120], mu, sigma)),
        'out' : tf.Variable(tf.truncated_normal([120, n_classes], mu, sigma))
    }

    biases = {
        'bc1' : tf.Variable(tf.zeros([6])),
        'bc2' : tf.Variable(tf.zeros([16])),
        'bd1' : tf.Variable(tf.zeros([120])),
        'out' : tf.Variable(tf.zeros([n_classes]))
    }

    return weights, biases
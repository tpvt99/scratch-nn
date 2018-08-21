import tensorflow as tf
import numpy as np
from utils import batch_shuffle
from mnist_load import load_data

weight_size = 3
weight_channels = 125
no_cnn_layers = 10

fc_dims = [100, 20, 10]

epoches = 10
learning_rate = 0.01
batch_size = 100

def placeholder_initializer(w, h, c, out_c):
    X = tf.placeholder(tf.float32, shape = [None, h, w, c], name="X")
    Y = tf.placeholder(tf.float32, shape = [None, out_c], name="Y")

    return X, Y

def wb_initializer(no_layer, f, prev_filters, channels):
    W = tf.get_variable("K"+str(no_layer), shape = [f, f, prev_filters, channels], initializer = tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b"+str(no_layer), shape = [channels], initializer = tf.initializers.zeros())

    return W, b

# simple convolution + RELU
def CNN_layer(no_layer, X, weight_size, weight_channel):
    prev_channels = X.shape[3]
    W, b = wb_initializer(no_layer, weight_size, prev_channels, weight_channel)

    Z = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = "SAME")
    A = tf.nn.relu(Z)

    return A, W

def feed_forward(X, no_cnn_layers, weight_size, weight_channel, fc1, fc2):

    # CNN layers
    for i_layer in range(no_cnn_layers):
        A, W = CNN_layer(i_layer, X, weight_size, weight_channel)
        X = A

    # Fully-connected layers
    F = tf.contrib.layers.flatten(X)

    W1 = tf.get_variable("W1", shape = [F.shape[1], fc1], initializer = tf.contrib.layers.xavier_initializer())
    F1 = tf.matmul(F, W1)
    A1 = tf.nn.relu(F1)

    W2 = tf.get_variable("W2", shape = [fc1, fc2], initializer = tf.contrib.layers.xavier_initializer())
    F2 = tf.matmul(A1, W2)
    
    A2 = tf.nn.softmax(F2)
        
    return A2

def compute_cost(A, Y):
    cost = -tf.reduce_mean(tf.multiply(Y, tf.log(A + 1e-4)) + tf.multiply((1-Y), tf.log(1-A+1e-4)))

    return cost


def model(X_train, Y_train, X_test, Y_test, epoches = 10, learning_rate = 1e-4, batch_size = 100, weight_size = 3, weight_channels = 128, no_cnn_layers = 10):
    m, w, h, c = X_train.shape

    X, Y = placeholder_initializer(w, h, c, 10)
    
    A = feed_forward(X, no_cnn_layers, weight_size, weight_channels, 100, 10)

    cost = compute_cost(A, Y)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

    y_pred_class = tf.argmax(A, axis = 1)
    y_true_class = tf.argmax(Y, axis = 1)
    y_pred = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(y_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        seed = 0
        for i in range(epoches):
            seed += 1
            mini_batches = batch_shuffle(X_train, Y_train, batch_size, seed = seed)
            total_batch = int(m/batch_size)
            batch_cost = 0

            for mini_batch in mini_batches:
                X_batch, Y_batch = mini_batch

                costx, _ = sess.run([cost, optimizer], feed_dict = {X: X_batch, Y : Y_batch})

                batch_cost += costx / total_batch

            acc = sess.run(accuracy, feed_dict = {X : X_train[0:1000,:], Y: Y_train[0:1000,:]})

            print("Cost after %i: %f, Accuracy: %f" %(i, batch_cost, acc))
        acc = sess.run(accuracy, feed_dict = {X : X_test, Y: Y_test})
        print(acc)


if __name__ == "__main__":
    train, test = load_data()
    X_train, Y_train = train
    X_test, Y_test = test

    model(X_train, Y_train, X_test, Y_test, epoches = epoches, learning_rate = learning_rate, batch_size = batch_size, weight_size = weight_size, weight_channels = weight_channels, no_cnn_layers = no_cnn_layers)

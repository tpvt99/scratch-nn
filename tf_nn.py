import tensorflow as tf
import numpy as np
from utils import batch_shuffle_nn
from mnist_load import load_data

layers_dims = [784, 500, 200, 100, 50, 20, 10] # include the input layers and output layers

learning_rate = 0.001
epoches = 100
batch_size = 1000

# X has [m, dimension], m is the total number of examples
def placeholder_initializer(input_dimension, output_dimension):
    X = tf.placeholder(tf.float32, shape = [None, input_dimension], name = "X")
    Y = tf.placeholder(tf.float32, shape = [None, output_dimension], name = "Y")

    return X, Y

def parameters_initializer(layers_dimension):
    parameters = {}
    for i in range(len(layers_dimension)-1):
        parameters["W" + str(i+1)] = tf.get_variable("W" + str(i+1), initializer = tf.contrib.layers.xavier_initializer(), shape = [layers_dimension[i+1], layers_dimension[i]])
        parameters["b" + str(i+1)] = tf.get_variable("b" + str(i+1), initializer = tf.initializers.zeros(), shape = [layers_dimension[i+1]])
    return parameters

def forward_propagation(X, parameters):
    number_layers = len(parameters)//2

    for i in range(number_layers-1):
        Z = tf.matmul(X, tf.transpose(parameters["W" + str(i+1)]))
        Z = tf.nn.bias_add(Z, parameters["b" + str(i+1)])
        A = tf.nn.relu(Z)
        X = A

    Z = tf.matmul(X, tf.transpose(parameters["W" + str(number_layers)]))
    Z = tf.nn.bias_add(Z, parameters["b" + str(number_layers)])
    A = tf.nn.softmax(Z)

    return A

# could customize the cost function
def compute_cost(A, Y):
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = A))

    cost = -tf.reduce_mean(tf.multiply(Y, tf.log(A + 1e-4)) + tf.multiply((1-Y), tf.log(1-A+1e-4)))
    return cost


def model(X_train, Y_train, X_test, Y_test, layer_dimensions, learning_rate = 1e-3, epoches = 100, batch_size = 10):
    m = X_train.shape[0]
    input_dimension = X_train.shape[1]
    output_dimension = Y_train.shape[1]
    X, Y = placeholder_initializer(input_dimension, output_dimension)
    parameters = parameters_initializer(layer_dimensions)

    A = forward_propagation(X, parameters)
    cost = compute_cost(A, Y)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

    y_pred_class = tf.argmax(A, axis = 1)
    y_true_class = tf.argmax(Y, axis = 1)
    y_acc = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(y_acc, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        seed = 0
        for i in range(epoches):
            seed += 1
            mini_batches = batch_shuffle_nn(X_train, Y_train, batch_size, seed = seed)
            total_batch = int(m/batch_size)
            batch_cost = 0

            for mini_batch in mini_batches:
                X_batch, Y_batch = mini_batch

                costx,_ = sess.run([cost, optimizer], feed_dict = {X: X_batch, Y : Y_batch})

                batch_cost += costx / total_batch

            acc = sess.run(accuracy, feed_dict = {X : X_train[0:1000,:], Y: Y_train[0:1000,:]})

            print("Cost after %i: %f, Accuracy: %f" %(i, batch_cost, acc))
        acc = sess.run(accuracy, feed_dict = {X : X_test, Y: Y_test})
        print(acc)


if __name__ == "__main__":
    train, test = load_data()
    X_train, Y_train = train
    X_test, Y_test = test
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    model(X_train, Y_train, X_test, Y_test, layer_dimensions = layers_dims, learning_rate = learning_rate, epoches = epoches, batch_size = batch_size)

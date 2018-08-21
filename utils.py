import numpy as np
import matplotlib.pyplot as plt
import math

def batch_shuffle(X, Y, batch_size, seed):

    [m, n_H, n_W, c] = X.shape

    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    number_of_batches = int(m / batch_size)

    for i in range(number_of_batches):
        batch_X = shuffled_X[i*batch_size : i*batch_size + batch_size,:,:,:]
        batch_Y = shuffled_Y[i*batch_size : i*batch_size + batch_size,:]
        mini_batches.append((batch_X, batch_Y))

    if m % batch_size != 0:
        batch_X_final = shuffled_X[number_of_batches*batch_size : m,:,:,:]
        batch_Y_final = shuffled_Y[number_of_batches*batch_size : m,:]
        mini_batches.append((batch_X_final, batch_Y_final))


    return mini_batches

def batch_shuffle_nn(X, Y, batch_size, seed):

    [m, c] = X.shape

    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    number_of_batches = int(m / batch_size)

    for i in range(number_of_batches):
        batch_X = shuffled_X[i*batch_size : i*batch_size + batch_size,:]
        batch_Y = shuffled_Y[i*batch_size : i*batch_size + batch_size,:]
        mini_batches.append((batch_X, batch_Y))

    if m % batch_size != 0:
        batch_X_final = shuffled_X[number_of_batches*batch_size : m,:]
        batch_Y_final = shuffled_Y[number_of_batches*batch_size : m,:]
        mini_batches.append((batch_X_final, batch_Y_final))

    return mini_batches

def plot_images(images, true_class, pred_class):
    
    fig, axes = plt.subplots(1,1)
    fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
    img_shape = [28,28]

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap = 'binary')

        if pred_class is None:
            xlabel = "True: {0}".format(true_class[i])
        else:
            xlabel = "True: {0}, Pred: {0}".format(true_class[i], pred_class[i])

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

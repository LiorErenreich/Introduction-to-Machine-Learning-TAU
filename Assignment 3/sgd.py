#################################
# Your name: Lior Erenreich
#################################


import numpy as np
import numpy.random
import scipy
from matplotlib import pyplot as plt
from scipy.special import expit
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.


    :param data: a numpy array of shape (n, d), representing the input data points
    :param labels: a numpy array of shape (n,), representing the labels of the input data points
    :param C: a float, representing the regularization strength
    :param eta_0: a float, representing the learning rate
    :param T: an integer, representing the number of iterations to run the algorithm

    :return:
        w - a numpy array of shape (d,), representing the learned weight vector
        accs - a float, representing the accuracy of the learned model

    """
    n, d = data.shape

    # Initialize the weight vector w and bias b to zero
    w = np.zeros(d)

    # Run the stochastic gradient descent algorithm for T iterations
    for t in range(1, T + 1):

        # Select a random index i in the range [0, n)
        i = np.random.randint(n)

        # Compute the learning rate for the current iteration
        eta_t = eta_0 / t

        # If the current sample is misclassified, update the weight vector
        if labels[i] * np.dot(w, data[i]) < 1:
            w = (1 - eta_t) * w + eta_t * C * labels[i] * data[i]

        # Otherwise, update the weight vector with the regularization term
        else:
            w = (1 - eta_t) * w

    # Return the learned weight vector and the accuracy
    return w, accuracy_score(validation_data, validation_labels, w)


def SGD_log(data, labels, eta_0, T):
    """
    Performs stochastic gradient descent to minimize the log-loss objective function
    with L2 regularization.
    :param data: ndarray of shape (n,d)
    :param labels: ndarray of shape (n,)
    :param eta_0: initial learning rate
    :param T: number of iterations

    :return:
        - ndarray of shape (d,) - the learned weight vector
        - train_losses: list of length T - contains the loss at each iteration
    """
    w = np.zeros(data.shape[1])
    for t in range(1, T+1):
        eta_t = eta_0 / t

        # Sample a random point
        i = np.random.randint(data.shape[0])
        y_i = labels[i]
        x_i = data[i]

        # Compute the gradient of the objective function at w
        grad = -y_i*expit(-y_i*np.dot(w, x_i))*x_i

        # Update w with the learning rate
        w = (1 - eta_t)*w + eta_t*grad
    return w, accuracy_score(validation_data, validation_labels, w)


#################################

# Place for additional code

#################################

def accuracy_score(data, labels, w):
    y_predict = np.sign(np.dot(data, w))
    accuracy = np.mean(y_predict == labels)
    return accuracy

def plot_accuracy_as_function_of_eta(eta_0_range, num_iter, T, C, is_hinge):
    """
    Find the best eta_0 to use for SGD on the loss objective function with L2
    regularization, using cross-validation on the validation set.
    :return:
        best_eta_0: float - the best learning rate to use
    """
    accuracy_per_eta = []

    for eta_0 in eta_0_range:
        sum_accuracy = 0
        for i in range(num_iter):
            if (is_hinge):
                w, accuracy = SGD_hinge(train_data, train_labels, C, eta_0, T)
            else:
                w, accuracy = SGD_log(train_data, train_labels, eta_0, T)
            sum_accuracy += accuracy
        accuracy_per_eta.append(sum_accuracy / num_iter)

    if (is_hinge):
        plt.title("1(a) Accuracy of SGD for Hinge loss as a Function of \u03B70")
    else:
        plt.title("2(a) Accuracy of SGD for Log Loss as a Function of \u03B70")
    plt.xlabel('\u03B70')
    plt.ylabel('Average Accuracy')
    plt.xscale('log')
    plt.plot(eta_0_range, accuracy_per_eta, marker='o')
    plt.show()
    return eta_0_range[np.argmax(accuracy_per_eta)]


def q_1_a(T, eta_0_range, num_iter, C):
    return plot_accuracy_as_function_of_eta(eta_0_range, num_iter, T, C, True)


def q_1_b(T, eta_0, num_iter, C_range):
    accuracy_to_Cs = []
    for C in C_range:
        sum_accuracy = 0
        for i in range(num_iter):
            w, accuracy = SGD_hinge(train_data, train_labels, C, eta_0, T)
            sum_accuracy += accuracy
        accuracy_to_Cs.append(sum_accuracy / num_iter)

    plt.title("1(b) Accuracy of SGD for Hinge Loss as a Function of C")
    plt.xlabel('C')
    plt.ylabel('Average Validation Accuracy')
    plt.semilogx(C_range, accuracy_to_Cs, 'o-')
    plt.show()
    print(f'1(b) Best C: {C_range[np.argmax(accuracy_to_Cs)]}')
    return C_range[np.argmax(accuracy_to_Cs)]


def q_1_c(eta_0, C, T):
    w, _ = SGD_hinge(train_data, train_labels, C, eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')  # According to the guidance in the assignment.
    plt.title('1(c) Resulting w as an image for the best C and \u03B70, T = 20,000')
    plt.colorbar()
    plt.show()
    return w

def q_1_d(w):
    return accuracy_score(test_data, test_labels, w)

def q_2_a(T, num_iter, eta_0_range):
    return plot_accuracy_as_function_of_eta(eta_0_range, num_iter, T, 1, False)

def q_2_b(T, eta_0):

    # Train the classifier using the best eta_0 found in part (a)
    w, accuracy = SGD_log(train_data, train_labels, eta_0, T)

    # Show the resulting w as an image
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')  # According to the guidance in the assignment.
    plt.colorbar()
    plt.title("2(b) Learned Weights")
    plt.show()

    print("2(b) Accuracy on the test set: {:.4f}%".format(accuracy * 100))


def q_2_c(T, eta_0):
    """
    Train the logistic regression classifier for T iterations and plot the norm of w as a function of iteration.

    Args:

    - T: int, the number of iterations for which to run SGD
    - eta_0: float, the learning rate parameter

    Returns:

    - norm_history: list of floats, the norm of w after each iteration
    """
    # initialize w
    d = train_data.shape[1]
    w = np.zeros(d)

    # SGD loop
    norm_history = []
    for t in range(1, T+1):
        eta_t = eta_0 / t
        i = np.random.choice(train_data.shape[0])
        x_i, y_i = train_data[i], train_labels[i]
        y_hat_i = scipy.special.expit(w.dot(x_i))
        gradient = (1 - y_hat_i) * y_i * x_i - y_hat_i * x_i
        w = w + eta_t * gradient
        norm_history.append(np.linalg.norm(w))

    # plot norm history
    plt.plot(range(1, T+1), norm_history)
    plt.xlabel('Iteration')
    plt.ylabel('Norm of w')
    plt.title('2(c) Norm of w as a function of iteration')
    plt.show()

    return norm_history

if __name__ == '__main__':
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    # define range of C values to try
    val_range = [10 ** i for i in range(-5, 6)]  # η0 = 10^−5, 10^−4, . . . , 10^4, 10^5
    num_iter = 10  # 10 runs according to the assignment.
    T = 1000  # For 1 (a) and 2 (a) the value of T is 1,000 according to the assignment.
    C = 1  # For 1 (a) the value of C is 1 according to the assignment.
    best_eta_0 = q_1_a(T, val_range, num_iter, C)
    best_c = q_1_b(T, best_eta_0, num_iter, val_range)
    Tc = 20000  # For 1 (c) and 2 (b - c) the value of T is 20,000 according to the assignment.
    w = q_1_c(best_eta_0, best_c, Tc)  # "Using the best C, η0 you found, train the classifier, but for T = 20000".
    print(f'1(d) The accuracy of the best classifier on the test is {100*q_1_d(w)}%.')
    best_eta_0 = q_2_a(T, num_iter, val_range)
    q_2_b(Tc, best_eta_0)
    q_2_c(Tc, best_eta_0)

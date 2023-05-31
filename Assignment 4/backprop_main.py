import numpy as np
from matplotlib import pyplot as plt

import backprop_data

import backprop_network



training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)

net = backprop_network.Network([784, 40, 10])

net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)


def question_a():
    print("-----(a)------")
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

def question_b():
    print("-----(b)------")
    rates = [0.001, 0.01, 0.1, 1, 10, 100]
    num_rates = len(rates)
    train_accuracy = np.empty(num_rates, dtype=object)
    train_loss = np.empty(num_rates, dtype=object)
    test_accuracy = np.empty(num_rates, dtype=object)

    for rate_index, learning_rate in enumerate(rates):
        print("(b) Rate " + str(learning_rate))
        net = backprop_network.Network([784, 40, 10])
        train_accuracy[rate_index], train_loss[rate_index], test_accuracy[rate_index] = net.SGD(training_data, epochs=30, mini_batch_size=10,
                                                                     learning_rate=learning_rate, test_data=test_data)
        plt.plot(np.arange(30), train_accuracy[rate_index], label=r"rate = {}".format(learning_rate))


    plt.xlabel(r"Epochs", fontsize=13)
    plt.ylabel(r"Accuracy", fontsize=13)
    plt.title(r"(b) Training Accuracy", fontsize=19)
    plt.legend()
    plt.show()

    for rate_index, learning_rate in enumerate(rates):
        plt.plot(np.arange(30), train_loss[rate_index], label="rate = {}".format(learning_rate))

    plt.xlabel(r"Epochs", fontsize=13)
    plt.ylabel(r"$\ell (\mathcal{W})$", fontsize=13)
    plt.title(r"(b) Training loss $\ell (\mathcal{W})$", fontsize=19)
    plt.legend()
    plt.show()

    for rate_index, learning_rate in enumerate(rates):
        plt.plot(np.arange(30), test_accuracy[rate_index], label="rate = {}".format(learning_rate))

    plt.xlabel(r"Epochs", fontsize=13)
    plt.ylabel(r"Accuracy", fontsize=13)
    plt.title(r"(b) Test Accuracy", fontsize=19)
    plt.legend()
    plt.show()

def question_c():
    print("-----(c)------")
    training_data, test_data = backprop_data.load(train_size=50000,test_size=10000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1,test_data=test_data)


question_a()
question_b()
question_c()
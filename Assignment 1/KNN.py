import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

def knn(train_set, labels, query_image, num_k):
    """ The function implements the k-NN algorithm to return a prediction of the query image, given the train images
    and labels. The function uses the k nearest neighbors, using the Euclidean L2 metric. In case of a tie between
    the k labels of neighbors, it chooses an arbitrary option.

    :param train_set: a set of train images
    :param labels: a vector of labels, corresponding to the images
    :param query_image: a query image
    :param num_k: a number k (nearest neighbors)
    :return: a prediction of the query image
    """

    dist_train_to_image = calc_dist(train_set, query_image)
    idx_k = np.argsort(dist_train_to_image)[:num_k]

    k_labels = np.take(labels, idx_k).astype(int)
    return np.bincount(k_labels).argmax()


def calc_dist(set_train, image):
    return np.linalg.norm(set_train - image, axis=1)


# Question 1 - Visualizing the Hoeffding bound
# Part (a)
N = 200000
n = 20
samples = np.random.binomial(1, 0.5, size=(N, n))
X_bar = np.mean(samples, axis=1)

# Part (b)
epsilons = np.linspace(0, 1, 50) #Takes 50 values of ϵ ∈ [0, 1] (as shown in the assignment)
p_empirical = []
for eps in epsilons:
    p_empirical.append(np.mean(np.abs(X_bar - 0.5) > eps))
plt.plot(epsilons, p_empirical, label="Empirical Probability")

# Part (c)
hoeffding_bound = 2 * np.exp(-2 * n * epsilons**2)
plt.plot(epsilons, hoeffding_bound, label="Hoeffding Bound")

plt.title("c. Probability vs. $\epsilon$")
plt.xlabel("$\epsilon$")
plt.ylabel("Probability")
plt.legend()
plt.show()

# Question 2 - Nearest Neighbor
# Load MNIST dataset with sklearn (as shown in the assignment)
mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']
# Define the training and test sets of images (as shown in the assignment)
db_size = 70000
T = 10000
idx = np.random.RandomState(0).choice(db_size, 11000)
train = data[idx[:T], :].astype(int)
train_labels = labels[idx[:T]]
test = data[idx[T:], :].astype(int)
test_labels = labels[idx[T:]]

test_results = np.zeros(1000)
n = 1000
k = np.arange(1, 101)
k_results = np.zeros(100)
for i in range(100):
    for j in range(len(test_results)):
        test_results[j] = knn(train[:n], train_labels[:n], test[j], k[i])
    k_results[i] = np.mean(test_results == test_labels.astype(int))

print("b. The accuracy for k=10, n=1000 is ", k_results[10]*100, "%.")
plt.plot(k, k_results)
plt.title("c. Accuracy vs. k")
plt.xlabel("Number of Nearest Neighbors (k)")
plt.ylabel("Accuracy")
plt.show()

n = np.arange(100, 5001, 100)
n_results = np.zeros(50)
for i in range(50):
    for j in range(len(test_results)):
        test_results[j] = knn(train[:n[i]], train_labels[:n[i]], test[j], 1)
    n_results[i] = np.linalg.norm(test_results == test_labels.astype(int), 1)/1000

plt.plot(n, n_results)
plt.title("d. Accuracy vs. n")
plt.xlabel('n')
plt.ylabel('Accuracy')
plt.show()




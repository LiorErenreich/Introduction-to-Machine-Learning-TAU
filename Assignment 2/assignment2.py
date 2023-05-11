#################################
# Your name: Lior Erenreich
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """
    def sample_from_D(self, m: int):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two-dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x = np.sort(np.random.uniform(0, 1, m))
        y = np.array([self.random_choose(self.define_x_intersection(x)[i]) for i in range(m)]).reshape(m, )
        # Return a 2D array where the first column is x and the second column is y
        return np.column_stack((x, y))

    def experiment_m_range_erm(self, first: int, last: int, step: int, k: int, T: int):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two-dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """

        # Generate the range of m values
        ns = np.arange(first, last + 1, step)
        # Run the experiment T times
        errs = np.array([self.empirical_true_err(m, k) for m in ns for _ in range(T)])
        errs = errs.reshape(len(ns), T, 2)
        # Calculate the average empirical and true error
        np_array_avg_errs = np.asarray(np.mean(errs, axis=1))

        # Plot the empirical and true errors as a function of n
        plt.title("Average Empirical and True Errors as a Function of Sample Size")
        plt.xlabel("n")
        plt.ylabel("Error")
        plt.plot(ns, np_array_avg_errs[:, 0], label="Empirical Error")
        plt.plot(ns, np_array_avg_errs[:, 1], label="True Error")
        plt.legend()
        plt.show()

        return np_array_avg_errs

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """

        # Create an array of k values to try
        k_choices = np.arange(k_first, k_last + 1, step)

        # Initialize arrays to store the empirical and true errors for each k
        empirical_err = np.zeros(k_choices.shape[0])
        true_err = np.zeros(k_choices.shape[0])

        # Sample data from the distribution
        sample = self.sample_from_D(m)

        # Loop over the k values and calculate the ERM intervals and errors
        for k in range(k_choices.shape[0]):
            ERM_intervals, ERM_empirical_err = intervals.find_best_interval(sample[:, 0], sample[:, 1], k_choices[k])
            empirical_err[k] = ERM_empirical_err / m
            true_err[k] = self.true_err(ERM_intervals)

        # Plot the empirical and true errors as a function of k
        plt.scatter(k_choices, true_err)
        plt.scatter(k_choices, empirical_err)
        plt.plot(k_choices, empirical_err, label='Empirical Error')
        plt.plot(k_choices, true_err, label='True Error')
        plt.legend()
        plt.title('Empirical and True Errors as a Function of k')
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.show()

        return np.argmin(empirical_err)

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # Generate a sample of size m from the distribution D
        sample = self.sample_from_D(m)
        # Shuffle the sample
        np.random.shuffle(sample)
        # Split the sample into training and validation sets (80-20 split)
        s1, s2 = sample[:int(m * 0.8)], sample[int(m * 0.8):]
        # Sort the training set by x-values
        s1 = s1[s1[:, 0].argsort()]
        # Initialize variables to keep track of best k and its empirical error
        # For each k value (1 to 10), find the best interval hypothesis using ERM algorithm and calculate empirical
        # Return the best k value
        return np.argmin([self.empirical_err(s2, intervals.find_best_interval(s1[:, 0], s1[:, 1], k)[0]) for k in np.arange(1, 11)]) + 1

    #################################
    # Additional methods

    def define_x_intersection(self, x):
        """
        Generate an array of the intersection of intervals 1 and 2
        """

        return ~((x > 0.2) & (x < 0.4)) & ~((x > 0.6) & (x < 0.8))

    def random_choose(self, param):
        """
        Randomly choose a float number from the given list based on the parameter.

        Args:
            param: a boolean parameter to decide which list to use for random choice

        Returns:
            A float number from [0.0, 1.0] list based on the given parameter with probabilities of
            [0.2, 0.8] or [0.9, 0.1]

        """
        return np.random.choice([0.0, 1.0], size=1, p=(lambda: [0.2, 0.8] if param else [0.9, 0.1])())[0]

    def empirical_true_err(self, n: int, k: int):
        """
        Runs the ERM algorithm and calculates the empirical error and true error
        for a given sample size and maximum number of intervals.

        Args:
            n: An integer, the size of the data sample.
            k: An integer, the maximum number of intervals.

        Returns:
            A tuple of two floats representing the empirical error and the true
            error, respectively.
        """
        sample = self.sample_from_D(n)
        best_intervals, err_cnt = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
        return err_cnt / n, self.true_err(best_intervals)

    def true_err(self, intervals):
        """
        Calculate the true error of the hypothesis corresponding to the given intervals.

        The true error is defined as the sum of the probabilities of the two sets of intervals not covered
        by the given intervals. The first set of intervals has label 1 and a high probability, and the second
        set of intervals has label 1 and a low probability.

        Args:
            intervals (list): A list of tuples, where each tuple represents an interval.

        Returns:
            float: The true error of the hypothesis corresponding to the given intervals.
        """
        # Calculate the intersection between the current intervals and the intervals with high and low probability
        len_cur_high_p = self.intersection_len(intervals, [(0, 0.2), (0.4, 0.6), (0.8, 1)])
        len_cur_low_p = self.intersection_len(intervals, [(0.2, 0.4), (0.6, 0.8)])

        # Calculate and return the true error using the lengths of the intervals
        return 0.8 * (0.6 - len_cur_high_p) + 0.1 * (0.4 - len_cur_low_p) + 0.2 * len_cur_high_p + 0.9 * len_cur_low_p

    def intersection_len(self, l1, l2) -> int:
        """
        Calculate the length of the intersection between the intervals in list1 and list2.

        Args:
            l1 (List[Tuple]): A list of tuples representing intervals.
            l2 (List[Tuple]): A list of tuples representing intervals.

        Returns:
            int: The length of the intersection between the intervals in list1 and list2.
        """
        start = np.maximum(np.array(l1)[:, 0].reshape(-1, 1), np.array(l2)[:, 0])
        end = np.minimum(np.array(l1)[:, 1].reshape(-1, 1), np.array(l2)[:, 1])
        mask = start < end
        return np.sum((end - start) * mask)

    def empirical_err(self, sample, intervals):
        """
        Calculate the empirical error of a given hypothesis, represented by a list of intervals, on a given sample.

        Args:
            sample (np.ndarray): A two-dimensional array containing pairs drawn from the distribution P.
            intervals (List[Tuple]): A list of tuples, where each tuple represents an interval in the hypothesis.

        Returns:
            float: The empirical error of the hypothesis on the given sample.
        """
        return sum([self.zero_one_loss(intervals, x, y) for x, y in sample]) / len(sample)

    def zero_one_loss(self, intervals, x, y) -> int:
        """
        Calculates the zero-one loss of a hypothesis h(x) with respect to a label y and an input value x.

        Args:
            intervals (List[Tuple[float, float]]): A list of intervals defining the hypothesis.
            x (float): The input value to evaluate the hypothesis.
            y (int): The true label corresponding to the input value (0 or 1).

        Returns:
            int: 0 if h(x) = y, else 1.
        """
        x_in_interval = self.is_in_interval(np.array(intervals), x)
        if (x_in_interval and y == 1) or (not x_in_interval and y == 0):
            return 0
        return 1

    def is_in_interval(self, intervals, x) -> bool:
        """
        Check if x is in the intervals in list_intervals.

        Args:
            intervals: An array of intervals represented by tuples.
            x: A float number in [0,1].

        Returns:
            True if x is in any of the intervals, False otherwise.
        """
        return np.any((intervals[:, 0] <= x) & (x <= intervals[:, 1]))

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.cross_validation(1500)


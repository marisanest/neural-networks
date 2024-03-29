from tqdm import tqdm
import numpy as np

class KNearestNeighbor(object):
    """ a kNN classifier with Euclidean distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_vectorized(X)
        elif num_loops == 1:
            dists = self.compute_distances_with_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_with_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #####################################################################
        # TODO (0):                                                        #
        # Loop over num_test (outer loop) and num_train (inner loop) and    #
        # compute the Euclidean distance between the ith test point and the #
        # jth training point, and store the result in dists[i, j]. You      #
        # should not use a loop over dimension.                             #
        #####################################################################
        for i in tqdm(range(num_test), ascii=False, desc="euclidean distance calculation"):
            for j in range(num_train):
                dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return dists

    def compute_distances_vectorized(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO (3):                                                            #
        # Compute the Euclidean distance between all test points and all        #
        # training points without using any explicit loops, and store the       #
        # result in dists.                                                      #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # Hint: Try to formulate the Euclidean distance using matrix            #
        #       multiplication and two broadcast sums.                          #
        #########################################################################
        dists = np.sqrt(np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(self.X_train**2, axis=1) - 2 * np.dot(X, self.X_train.T))
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to the ith test point.
            closest_y = []

            #########################################################################
            # TODO (0):                                                             #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            closest_y = self.y_train[np.argsort(dists[i])][:k]
            #########################################################################
            # TODO (0):                                                             #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            y_pred[i] = np.argmax(np.bincount(closest_y))
            #########################################################################
            #                           END OF YOUR CODE                            #
            #########################################################################
        return y_pred

    @classmethod
    def cross_validate(cls, num_folds, X_train_folds, y_train_folds):

        accuracies = []

        for i in tqdm(range(num_folds)):
            X_train = cls.flat_list(X_train_folds[:i] + X_train_folds[i+1:])
            y_train = cls.flat_list(y_train_folds[:i] + y_train_folds[i+1:])
            X_test = X_train_folds[i]
            y_test = y_train_folds[i]

            accuracy = cls.validate(X_train, y_train, X_test, y_test)
            accuracies.append(accuracy)

        return accuracies

    @classmethod
    def validate(cls, X_train, y_train, X_test, y_test):
        classifier = cls()  # KNearestNeighbor()
        classifier.train(X_train, y_train)
        dists = classifier.compute_distances_vectorized(X_test)
        y_test_pred = classifier.predict_labels(dists)
        num_correct = np.sum(y_test_pred == y_test)
        accuracy = float(num_correct) / len(X_test)
        return accuracy

    @staticmethod
    def flat_list(multi_list):
        return np.array([instance for sublist in multi_list for instance in sublist])

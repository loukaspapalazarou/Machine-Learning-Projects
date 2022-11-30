import random
import math
from tools import file_to_list

# Eucledian distance between 2 vectors
def euclidean(a, b):
    d = 0
    for i in range(len(a)):
        d += math.pow(a[i] - b[i], 2)
    return d

# Gaussian function for 2 vectors
def gaussian(a, b, sigma):
    t1 = -1 * euclidean(a, b)
    t2 = 2 * math.pow(sigma, 2)
    return math.exp(t1 / t2)

# This class represents a hidden node
class Node:
    def __init__(self, center_vector, sigma):
        self.center_vector = center_vector
        self.sigma = sigma
        self.phi = 0
    
    # the gaussian function for a given vector
    def calculate_phi(self, input_vector):
        phi = gaussian(self.center_vector, input_vector, self.sigma)
        self.phi = phi
        return phi
    
    # update center based on equations
    def update_center(self, input_vector, step_error, weight, learning_rate):
        for i, _ in enumerate(self.center_vector):
            s = learning_rate * step_error * weight * self.phi
            s *= (input_vector[i] - self.center_vector[i]) / math.pow(self.sigma, 2)
            self.center_vector[i] += s
    
    # update sigma based on equations
    def update_sigma(self, input_vector, step_error, weight, learning_rate):
        s = learning_rate * step_error * weight * self.phi
        e = euclidean(input_vector, self.center_vector)
        s = s / math.pow(self.sigma, 3)
        self.sigma += s


# Class for the Radial basis function net
class RBF:
    def __init__(self, hidden_nodes, outputs, centers, sigma, centers_lr, sigma_lr, weights_lr):
        self.sigma = sigma
        self.centers_lr = centers_lr
        self.weights_lr = weights_lr
        self.sigma_lr = sigma_lr

        self.hidden_nodes = []
        for center_vector in centers:
            self.hidden_nodes.append(Node(center_vector, sigma))
        
        self.weights = []
        for _ in range(hidden_nodes):
            self.weights.append(random.uniform(-1.0, 1.0))
    
    # updates based on equations
    def update_weights(self, input_vector, error):
        for i, h in enumerate(self.hidden_nodes):
            self.weights[i] += self.weights_lr * error * gaussian(h.center_vector, input_vector, h.sigma)

    def update_centers(self, input_vector, step_error):
        for h, w in zip(self.hidden_nodes, self.weights):
            h.update_center(input_vector, step_error, w, self.weights_lr)

    def update_sigmas(self, input_vector, step_error):
        for i, h in enumerate(self.hidden_nodes):
            self.hidden_nodes[i].update_sigma(input_vector, step_error, self.weights[i], self.sigma_lr)
    
    def calculate_output(self, input_vector):
        output = 0
        for h, w in zip(self.hidden_nodes, self.weights):
            output += w * h.calculate_phi(input_vector)
        return output

    # one train step is going thruogh all the training data
    # calculating the output, error and updating based on the error
    def train_step(self, train_file):
        error_sum = 0
        data = file_to_list(train_file)
        for i, input_vector in enumerate(data):
            # expected output is the first value in the input vector
            expected_output = input_vector.pop(0)
            actual_output = self.calculate_output(input_vector)
            step_error = expected_output - actual_output
            self.update_weights(input_vector, step_error)
            self.update_centers(input_vector, step_error)
            self.update_sigmas(input_vector, step_error)
            error_sum += math.pow(step_error, 2)
        return error_sum/2

    # one test step is going thruogh all the training data
    # calculating the output and error
    def test_step(self, test_file):
        error_sum = 0
        data = file_to_list(test_file)
        for i, input_vector in enumerate(data):
            expected_output = input_vector.pop(0)
            actual_output = self.calculate_output(input_vector)
            step_error = expected_output - actual_output
            error_sum += math.pow(step_error, 2)
        return error_sum/2

    # perform all of training and testing while keeping track of all errors
    def train_and_test(self, epochs, train_file, test_file, verbose=False):
        train_error = []
        test_error = []
        for epoch in range(epochs):
            if verbose:
                print("Epoch:", epoch, end="\t")
            train_e = self.train_step(train_file)
            test_e = self.test_step(test_file)
            if verbose:
                print("Train error:", format(train_e, '.16f'), "\tTest error:", format(test_e, '.16f'))
            train_error.append(train_e)
            test_error.append(test_e)
        return train_error, test_error
import math
import random
from os.path import exists
import matplotlib.pyplot as plt
import preprocessing


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


# Convert character to list of bits
def char_to_bits(c):
    bits = [0] * 26
    bits[ord(c) - 65] = 1
    return bits


# Convert list of bits to character
def bits_to_char(bits):
    for i in range(len(bits)):
        if bits[i] == 1:
            return chr(i + 65)


# Each neuron has its value, delta error, inbound and outbound connections
# It also includes all its operations to calculate its value, error and update its weights
class Neuron:
    def __init__(self, isBias=False):
        self.inbound_connections = None
        self.inbound_connections = []
        self.outbound_connections = []
        self.delta = 0
        self.isBias = isBias
        if self.isBias:
            self.value = 1
        else:
            self.value = round(random.uniform(-1.0, 1.0), 2)

    def calculate_value(self):
        if self.isBias:
            self.value = 1
            return
        total_sum = 0
        for conn in self.inbound_connections:
            # Check which side of the connection is not me
            if conn.n1 == self:
                total_sum += conn.n2.value * conn.weight
            else:
                total_sum += conn.n1.value * conn.weight
        self.value = sigmoid(total_sum)

    def calculate_delta(self):
        if self.isBias:
            return
        # Calculate delta values
        total_sum = 0
        for out_conn in self.outbound_connections:
            if out_conn.n1 == self:
                total_sum += out_conn.n2.delta * out_conn.weight
            else:
                total_sum += out_conn.n1.delta * out_conn.weight
        self.delta = sigmoid_deriv(self.value) * total_sum

    def update_weights(self, learning_rate):
        # Update inbound weights
        for in_conn in self.inbound_connections:
            if in_conn.n1 == self:
                in_conn.weight -= learning_rate * self.delta * in_conn.n2.value
            else:
                in_conn.weight -= learning_rate * self.delta * in_conn.n1.value


# Define connection class, 2 neurons and their weight
class Connection:
    def __init__(self, n1, n2):
        self.weight = round(random.uniform(-1.0, 1.0), 2)
        self.n1 = n1
        self.n2 = n2


# Network has a list for each layer
class Network:
    def __init__(self, input_layer_neurons, hidden_layer_1_neurons, hidden_layer_2_neurons, output_layer_neurons):
        self.input_layer = [Neuron(isBias=True)]
        self.hidden_layer_1 = [Neuron(isBias=True)]
        if hidden_layer_2_neurons > 0:
            self.hidden_layer_2 = [Neuron(isBias=True)]
        else:
            self.hidden_layer_2 = []
        self.output_layer = []

        # Create layers
        for _ in range(input_layer_neurons):
            self.input_layer.append(Neuron())

        for _ in range(hidden_layer_1_neurons):
            self.hidden_layer_1.append(Neuron())

        for _ in range(hidden_layer_2_neurons):
            self.hidden_layer_2.append(Neuron())

        for _ in range(output_layer_neurons):
            self.output_layer.append(Neuron())

        # Connect layers
        # in -> l1
        for in_neuron in self.input_layer:
            for l1_neuron in self.hidden_layer_1:
                c = Connection(in_neuron, l1_neuron)
                if not l1_neuron.isBias:
                    l1_neuron.inbound_connections.append(c)
                    in_neuron.outbound_connections.append(c)

        # If a second hidden layer exists: l1 -> l2 -> out
        if hidden_layer_2_neurons > 0:
            for l1_neuron in self.hidden_layer_1:
                for l2_neuron in self.hidden_layer_2:
                    c = Connection(l1_neuron, l2_neuron)
                    if not l2_neuron.isBias:
                        l2_neuron.inbound_connections.append(c)
                        l1_neuron.outbound_connections.append(c)

            for l2_neuron in self.hidden_layer_2:
                for out_neuron in self.output_layer:
                    c = Connection(l2_neuron, out_neuron)
                    if not out_neuron.isBias:
                        out_neuron.inbound_connections.append(c)
                        l2_neuron.outbound_connections.append(c)

        else:
            # l1 -> out
            for l1_neuron in self.hidden_layer_1:
                for out_neuron in self.output_layer:
                    c = Connection(l1_neuron, out_neuron)
                    if not out_neuron.isBias:
                        out_neuron.inbound_connections.append(c)
                        l1_neuron.outbound_connections.append(c)

    # Make a prediction given an input
    def predict(self, inputs):
        # Place inputs in input layer
        i = 0
        for in_neuron in self.input_layer:
            if not in_neuron.isBias:
                in_neuron.value = inputs[i]
                i += 1

        for l1_neuron in self.hidden_layer_1:
            l1_neuron.calculate_value()

        for l2_neuron in self.hidden_layer_2:
            l2_neuron.calculate_value()

        for out_neuron in self.output_layer:
            out_neuron.calculate_value()

        # returns a list of the values of the output neurons
        output = []
        for out_neuron in self.output_layer:
            output.append(out_neuron.value)
        return output

    # Average error given an expected output
    def calculate_error(self, expected_output, actual_output):
        error_sum = 0
        for i in range(len(self.output_layer)):
            self.output_layer[i].delta = sigmoid_deriv(actual_output[i]) * (actual_output[i] - expected_output[i])
            error_sum += abs(actual_output[i] - expected_output[i])
        # calculate average error of all bits
        return error_sum / len(self.output_layer)

    # Predict, then learn from mistake
    # expected output is a capital letter
    def learn(self, inputs, expected_output, learning_rate):
        actual_output = self.predict(inputs)

        error = self.calculate_error(expected_output, actual_output)

        for l2_neuron in self.hidden_layer_2:
            l2_neuron.calculate_delta()

        for l1_neuron in self.hidden_layer_1:
            l1_neuron.calculate_delta()

        for in_neuron in self.input_layer:
            in_neuron.calculate_delta()

        # update weights
        for out_neuron in self.output_layer:
            out_neuron.update_weights(learning_rate)

        for l2_neuron in self.hidden_layer_2:
            l2_neuron.update_weights(learning_rate)

        for l1_neuron in self.hidden_layer_1:
            l1_neuron.update_weights(learning_rate)

        for in_neuron in self.input_layer:
            in_neuron.update_weights(learning_rate)

        return error

    # Train, test and create files
    def evaluate(self, epochs, learning_rate, training_file, testing_file, verbose=False, scenario=""):
        errors_file = open("results/errors" + str(scenario) + ".txt", 'w')
        successrate_file = open("results/successrate" + str(scenario) + ".txt", 'w')

        training_errors_log = []
        testing_errors_log = []

        training_successrate_log = []
        testing_successrate_log = []

        for i in range(epochs):
            print("Epoch", i + 1)

            # Train
            correct = 0
            error_sum = 0
            f = open(training_file)
            lines = f.readlines()
            random.shuffle(lines)
            training_size = len(lines)
            f.close()
            for line in lines:
                line = line.split()
                expected_output = char_to_bits(line[0])
                expected_letter = line.pop(0)
                inputs = [float(num) for num in line]
                error_sum += self.learn(inputs, expected_output, learning_rate)
                if self.output_to_char() == expected_letter:
                    correct += 1
            training_successrate = correct / training_size * 100
            training_error = error_sum / training_size
            training_errors_log.append(training_error)
            training_successrate_log.append(training_successrate)
            print(f'Training success rate: {training_successrate}%')
            print(f'Training error: {training_error}')

            # Test
            correct = 0
            error_sum = 0
            f = open(testing_file)
            lines = f.readlines()
            random.shuffle(lines)
            testing_size = len(lines)
            f.close()
            for line in lines:
                line = line.split()
                expected_letter = line.pop(0)
                inputs = [float(num) for num in line]
                expected_output = char_to_bits(expected_letter)
                actual = self.predict(inputs)
                error_sum += self.calculate_error(expected_output, actual)
                actual = self.output_to_char()
                if verbose:
                    print(expected_letter, end=" ")
                if actual == expected_letter:
                    correct += 1
                    if verbose:
                        print(actual, "Correct!")
                elif verbose:
                    print(actual)
            testing_successrate = correct / testing_size * 100
            testing_error = error_sum / testing_size
            testing_successrate_log.append(testing_successrate)
            testing_errors_log.append(testing_error)
            print(f'Testing success rate: {testing_successrate}%')
            print(f'Testing error: {testing_error}')
            print()

            error_line = str(i) + "\t" + str(training_error) + "\t" + str(testing_error) + "\n"
            successrate_line = str(i) + "\t" + str(training_successrate) + "\t" + str(testing_successrate) + "\n"
            errors_file.write(error_line)
            successrate_file.write(successrate_line)

        errors_file.close()
        successrate_file.close()

        return training_errors_log, training_successrate_log, testing_errors_log, testing_successrate_log

    # Returns the max valued bit, the one the network "chose" as its answer
    def output_value(self):
        actual = []
        for out_neuron in self.output_layer:
            actual.append(out_neuron.value)
        max_index = actual.index(max(actual))
        return self.output_layer[max_index].value

    # Returns the letter representation of the list of bits
    def output_to_char(self):
        actual = []
        for out_neuron in self.output_layer:
            actual.append(out_neuron.value)
        max_index = actual.index(max(actual))
        temp = [0] * 26
        temp[max_index] = 1
        return bits_to_char(temp)


if __name__ == "__main__":

    # Import parameters
    parameters = preprocessing.get_params("parameters.txt")
    HIDDEN_LAYER_1 = parameters[0]
    HIDDEN_LAYER_2 = parameters[1]
    INPUT_LAYER = parameters[2]
    OUTPUT_LAYER = parameters[3]
    LEARNING_RATE = parameters[4]
    MOMENTUM = parameters[5]
    EPOCHS = parameters[6]
    TRAINING_FILE = parameters[7]
    TESTING_FILE = parameters[8]

    # Create training and testing files
    if not exists(TRAINING_FILE) or not exists(TESTING_FILE):
        preprocessing.process_files()

    # Initialize network
    network = Network(INPUT_LAYER, HIDDEN_LAYER_1, HIDDEN_LAYER_2, OUTPUT_LAYER)

    print("Starting training with:")
    print("Epochs:", EPOCHS)
    print("Neurons in hidden layer 1:", HIDDEN_LAYER_1)
    print("Neurons in hidden layer 2:", HIDDEN_LAYER_2)
    print("Learning rate:", LEARNING_RATE)
    print()

    scenario = ""
    train_err, train_succ, test_err, test_succ = network.evaluate(EPOCHS, LEARNING_RATE, TRAINING_FILE,
                                                                  TESTING_FILE, verbose=False, scenario=scenario)

    print()
    print("Epochs:", EPOCHS)
    print("Neurons in hidden layer 1:", HIDDEN_LAYER_1)
    print("Neurons in hidden layer 2:", HIDDEN_LAYER_2)
    print("Learning rate:", LEARNING_RATE)

    plt.ylim(0, 100)
    plt.plot(train_succ, color="blue")
    plt.plot(test_succ, color="red")
    plt.legend(["Train Data", "Test Data"], loc='best')
    plt.savefig("results/successrate" + str(scenario) + ".png")

    plt.clf()

    plt.plot(train_err, color="blue")
    plt.plot(test_err, color="red")
    plt.legend(["Train Data", "Test Data"], loc='best')
    plt.savefig("results/errors" + str(scenario) + ".png")

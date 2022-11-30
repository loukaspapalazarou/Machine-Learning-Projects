from tools import *
from RBF import *
import os

if __name__ == "__main__":
    os.system("cls")
    RAW_DATA_FILE = "selwood.txt"
    parameters = get_parameters("parameters.txt")

    HIDDEN_NEURONS = parameters[0]
    INPUT_NEURONS = parameters[1]
    OUTPUT_NEURONS = parameters[2]
    CENTERS_LEARNING_RATE = parameters[3][0]
    SIGMA_LEARNING_RATE = parameters[3][1]
    WEIGHT_LEARNING_RATE = parameters[3][2]
    SIGMA = parameters[4]
    EPOCHS = parameters[5]
    CENTERS_FILE = parameters[6]
    TRAIN_FILE = parameters[7]
    TEST_FILE = parameters[8]
    
    # get dimension of centers
    dimension = get_data_dimension(TRAIN_FILE)
    process_data(RAW_DATA_FILE, TRAIN_FILE, TEST_FILE)
    # create a file with random centers as initials
    create_centers_file(CENTERS_FILE, dimension, HIDDEN_NEURONS)
    CENTERS = file_to_list(CENTERS_FILE)

    rbf = RBF(
            hidden_nodes=HIDDEN_NEURONS,
            outputs=OUTPUT_NEURONS,
            centers=CENTERS,
            sigma=SIGMA,
            centers_lr=CENTERS_LEARNING_RATE,
            sigma_lr=SIGMA_LEARNING_RATE,
            weights_lr=WEIGHT_LEARNING_RATE
        )

    # train and test
    train_errors, test_errors = rbf.train_and_test(EPOCHS, TRAIN_FILE, TEST_FILE, verbose=True)

    # save
    save_plot(train_errors, test_errors, "results.png")
    save_errors(train_errors, test_errors, "results.txt")
    save_weights(rbf.weights, "weights.txt")


package main

import (
	"fmt"
	"kohonensom/som"
	"kohonensom/tools"
	"os"
)

func main() {
	// Get parameters from file
	EXPERIMENT_NAME, DIMENSION, INITIAL_LEARNING_RATE, INPUTS, EPOCHS := tools.GetParameters("parameters.txt")
	fmt.Printf("Experiment name: %s\n------------------------\n\n", EXPERIMENT_NAME)

	// Initialize SOM variables
	grid, weights, input_layer, n0, sigma0 := som.Initalize(DIMENSION, INPUTS, INITIAL_LEARNING_RATE)
	fmt.Println()

	training_errors := []float64{}
	testing_errors := []float64{}

	training_data := tools.GetData("training.txt")
	testing_data := tools.GetData("testing.txt")

	// Start training
	for epoch := 0; epoch < EPOCHS; epoch++ {
		fmt.Println("Epoch ", epoch)
		fmt.Println("---------")
		training_error := som.TrainStep(DIMENSION, epoch, EPOCHS, training_data, input_layer, weights, n0, sigma0)
		testing_error := som.TestStep(DIMENSION, epoch, testing_data, input_layer, weights)
		// append errors to error list
		training_errors = append(training_errors, training_error)
		testing_errors = append(testing_errors, testing_error)
		// print current error
		fmt.Println("Training error: ", training_error)
		fmt.Println("Testing error: ", testing_error)
		fmt.Println()
	}

	som.LabelData(DIMENSION, grid, weights, testing_data)
	som.ShowMap(DIMENSION, grid)

	// create output folder
	path := "results/" + EXPERIMENT_NAME
	os.RemoveAll(path)
	os.MkdirAll(path, os.ModePerm)

	// Write results
	tools.WriteErrorsToFile(path, training_errors, testing_errors)
	tools.WriteMapToFile(path, DIMENSION, grid)
	tools.PlotErrors(path, training_errors, testing_errors)

}

// SOM tools

package som

import (
	"fmt"
	"kohonensom/tools"
	"math"
)

// I, J are positions, K is input count
type Key struct {
	I, J, K int
}

// Initialize
func Initalize(dimension int, inputs int, initial_learning_rate float64) ([][]string, map[Key]float64, []float64, float64, float64) {
	fmt.Print("Initializing...")

	// n0 is the initial learning rate and sigma0 is set to half of the dimension
	n0 := initial_learning_rate
	sigma0 := float64(dimension) / 2.0

	grid := make([][]string, dimension)

	for i := range grid {
		grid[i] = make([]string, dimension)
	}

	// The weights are a triplet of i,j positions and k, the corresponding input neuron
	weights := make(map[Key]float64)
	input_layer := make([]float64, inputs)
	for i := range weights {
		weights[i] = tools.RandomFloat()
	}

	// Initialize weights
	for i := 0; i < dimension; i++ {
		for j := 0; j < dimension; j++ {
			for k := 0; k < inputs; k++ {
				weights[Key{I: i, J: j, K: k}] = tools.RandomFloat()
			}
		}
	}
	fmt.Println("done")
	return grid, weights, input_layer, n0, sigma0
}

// Winner function
func findWinner(dimension int, input_layer []float64, weights map[Key]float64) (int, int) {
	min_distance := math.MaxFloat64
	winner_i := 0
	winner_j := 0
	// Iterate grid
	for i := 0; i < dimension; i++ {
		for j := 0; j < dimension; j++ {
			// find min distance
			temp := distance(input_layer, weights, i, j)
			if temp < min_distance {
				winner_i = i
				winner_j = j
				min_distance = temp
			}
		}
	}
	return winner_i, winner_j
}

// Distance of input to a given node
func distance(input_layer []float64, weights map[Key]float64, node_i, node_j int) float64 {
	d := 0.0
	for k := 0; k < len(input_layer); k++ {
		d += math.Pow(input_layer[k]-weights[Key{I: node_i, J: node_j, K: k}], 2)
	}
	return math.Sqrt(d)

}

// Eucledian distance used for neighborhood function
func eucledianDistance(winner_i, winner_j, other_i, other_j float64) float64 {
	i_distance := winner_i - other_i
	j_distance := winner_j - other_j
	return math.Pow(i_distance, 2) + math.Pow(j_distance, 2)
}

// Neighborhood function
func neighborhoodFunction(winner_i, winner_j, other_i, other_j, sigma float64) float64 {
	return math.Exp(-(eucledianDistance(winner_i, winner_j, other_i, other_j) / (2 * math.Pow(sigma, 2))))
}

// Update weights
func updateWeights(dimension int, weights map[Key]float64, input_layer []float64, winner_i, winner_j, n, sigma float64) {
	for i := 0; i < dimension; i++ {
		for j := 0; j < dimension; j++ {
			// Calculcate h, the coefficient to repsresent the neighborhood
			h := neighborhoodFunction(winner_i, winner_j, float64(i), float64(j), sigma)
			for k := 0; k < len(input_layer); k++ {
				weights[Key{I: i, J: j, K: k}] += n * h * (input_layer[k] - weights[Key{I: i, J: j, K: k}])
			}
		}
	}
}

// Calculate new learning rate given current epoch
func calculateN(sigma0 float64, current_epoch, total_epochs int) float64 {
	if current_epoch == 0 {
		return sigma0
	}
	T := total_epochs
	J := float64(T) / math.Log10(sigma0)
	return sigma0 * math.Exp(-float64(current_epoch)/J)
}

// Calculate new sigma rate given current epoch
func calculateSigma(n0 float64, current_epoch, total_epochs int) float64 {
	if current_epoch == 0 {
		return n0
	}
	T := total_epochs
	return n0 * math.Exp(-float64(current_epoch)/float64(T))
}

// One training step, tests and updates
func TrainStep(dimension, epoch, total_epochs int, training_data [][]string, input_layer []float64, weights map[Key]float64, n0, sigma0 float64) float64 {
	fmt.Printf("Training (%d records) | ", len(training_data))
	n := calculateN(n0, epoch, total_epochs)
	sigma := calculateSigma(sigma0, epoch, total_epochs)

	training_error := 0.0
	// tools.ShuffleRecords(training_data)

	// for every record in training set
	for record_cnt := 0; record_cnt < len(training_data); record_cnt++ {
		record_str := training_data[record_cnt][1:]
		input_layer = tools.RecordToFloatList(record_str)

		winner_i, winner_j := findWinner(dimension, input_layer, weights)
		training_error += math.Pow(distance(input_layer, weights, winner_i, winner_j), 2)

		updateWeights(dimension, weights, input_layer, float64(winner_i), float64(winner_j), n, sigma)

		// Keep track of progress in %
		progress := float32(record_cnt+1) * 100 / float32(len(training_data))
		if progress == float32(int(progress)) && int(progress)%10 == 0 {
			fmt.Print(progress)
			if int(progress) == 100 {
				fmt.Println("%")
			} else {
				fmt.Print("%...")
			}
		}
	}

	training_error = training_error / float64(len(training_data))
	return training_error
}

// One testing step
func TestStep(dimension, epoch int, testing_data [][]string, input_layer []float64, weights map[Key]float64) float64 {
	fmt.Printf("Testing (%d records) | ", len(testing_data))
	testing_error := 0.0

	// for every record in training set
	for record_cnt := 0; record_cnt < len(testing_data); record_cnt++ {
		record_str := testing_data[record_cnt][1:]
		input_layer = tools.RecordToFloatList(record_str)

		winner_i, winner_j := findWinner(dimension, input_layer, weights)
		testing_error += math.Pow(distance(input_layer, weights, winner_i, winner_j), 2)

		progress := float32(record_cnt+1) * 100 / float32(len(testing_data))
		if progress == float32(int(progress)) && int(progress)%10 == 0 {
			fmt.Print(progress)
			if int(progress) == 100 {
				fmt.Println("%")
			} else {
				fmt.Print("%...")
			}
		}
	}

	testing_error = testing_error / float64(len(testing_data))
	return testing_error
}

// Label the data
func LabelData(dimension int, grid [][]string, weights map[Key]float64, test_data [][]string) {
	fmt.Print("Labeling...")
	for i := 0; i < dimension; i++ {
		for j := 0; j < dimension; j++ {
			min_distance := math.MaxFloat64
			label := "_"
			for k := 0; k < len(test_data); k++ {
				// Calculate distance of current record to current node
				record := tools.RecordToFloatList(test_data[k][1:])
				l := test_data[k][0]
				d := distance(record, weights, i, j)
				if d < min_distance {
					min_distance = d
					label = l
				}
			}
			grid[i][j] = label
		}
	}
	fmt.Println("done")
}

// Print map
func ShowMap(dimension int, grid [][]string) {
	for i := 0; i < dimension; i++ {
		for j := 0; j < dimension; j++ {
			fmt.Print(grid[i][j], " ")
		}
		fmt.Println()
	}
}

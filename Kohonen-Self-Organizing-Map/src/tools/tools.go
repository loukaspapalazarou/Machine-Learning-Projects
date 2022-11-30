// General tools

package tools

import (
	"bufio"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

// Generate random floats
func RandomFloat() float64 {
	rand.Seed(time.Now().UnixNano())
	r := rand.Float64()
	return r
}

// Read input data from given file
func GetData(filename string) [][]string {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var records [][]string
	for scanner.Scan() {
		r := strings.Split(scanner.Text(), " ")
		records = append(records, r)
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	return records
}

// Get parameters from file
func GetParameters(filename string) (string, int, float64, int, int) {
	params := []string{}
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		r := strings.Split(scanner.Text(), " ")
		params = append(params, r[1])
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	NAME := params[0]
	DIMENSION, _ := strconv.Atoi(params[1])
	INITIAL_LEARNING_RATE, _ := strconv.ParseFloat(params[2], 64)
	INPUTS, _ := strconv.Atoi(params[3])
	EPOCHS, _ := strconv.Atoi(params[4])

	return NAME, DIMENSION, INITIAL_LEARNING_RATE, INPUTS, EPOCHS
}

// Shuffle the records
func ShuffleRecords(r [][]string) {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(r), func(i, j int) { r[i], r[j] = r[j], r[i] })
}

// Convert a record from strings (as read) to float
func RecordToFloatList(r []string) []float64 {
	f := []float64{}
	for i := 0; i < len(r); i++ {
		val, _ := strconv.ParseFloat(r[i], 64)
		f = append(f, val)
	}
	return f
}

// Write the labeled map in a file
func WriteMapToFile(filename string, dimension int, grid [][]string) {
	filename += "/clustering.txt"
	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.Truncate(filename, 0); err != nil {
		log.Fatal(err)
	}
	for i := 0; i < dimension; i++ {
		for j := 0; j < dimension; j++ {

			if _, err := f.Write([]byte(grid[i][j] + " ")); err != nil {
				log.Fatal(err)
			}

		}
		if _, err := f.Write([]byte("\n")); err != nil {
			log.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
}

// Write errors in a file
func WriteErrorsToFile(filename string, training_errors, testing_errors []float64) {
	filename += "/results.txt"
	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.Truncate(filename, 0); err != nil {
		log.Fatal(err)
	}
	for i := 0; i < len(training_errors); i++ {
		s := strconv.Itoa(i) + " "
		s += strconv.FormatFloat(training_errors[i], 'g', -1, 64) + " "
		s += strconv.FormatFloat(testing_errors[i], 'g', -1, 64) + "\n"
		if _, err := f.Write([]byte(s)); err != nil {
			log.Fatal(err)
		}
	}
}

// Convert list of points to point object for plotting
func listToPoints(list []float64) plotter.XYs {
	pts := make(plotter.XYs, len(list))
	for i := range pts {
		pts[i].X = float64(i)
		pts[i].Y = list[i]
	}
	return pts
}

// Create visualization of errors
func PlotErrors(filename string, training_errors, testing_errors []float64) {
	filename += "/errors.png"

	p := plot.New()

	p.Title.Text = "Error plot"
	p.X.Label.Text = "Epoch"
	p.Y.Label.Text = "Error"

	err := plotutil.AddLinePoints(p,
		"Training", listToPoints(training_errors),
		"Testing", listToPoints(testing_errors))
	if err != nil {
		panic(err)
	}

	if err := p.Save(6*vg.Inch, 6*vg.Inch, filename); err != nil {
		panic(err)
	}
}

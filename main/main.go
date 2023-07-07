package main

import (
	act "74_Basic_Neural_Network/activation"
	ld "74_Basic_Neural_Network/layerDense"
	"74_Basic_Neural_Network/loss"
	sp "74_Basic_Neural_Network/spiral_data"
	"74_Basic_Neural_Network/utils"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	sliceX, rowX, colX := sp.ReadFromFile("./spiral_data/spiral_data.txt")
	sliceY, _, _ := sp.ReadFromFile("./spiral_data/spiral_dataY.txt")
	inputsX := mat.NewDense(rowX, colX, sliceX)
	inputsY := mat.NewDense(len(sliceY), 1, sliceY)

	dense1 := ld.NewLayer_Dense(2, 3)
	var activation1 act.ReLU
	dense2 := ld.NewLayer_Dense(3, 3)
	var activation2 act.Softmax
	var err *loss.CategoricalCrossEntropy = loss.NewCrossEntropy()

	dense1.Forward(inputsX)
	activation1.Forward(dense1.Output)

	dense2.Forward(activation1.Output)
	activation2.Forward(dense2.Output)
	fa := mat.Formatted(activation2.Output, mat.Squeeze(), mat.Excerpt(5))
	fmt.Println(fa)
	los := loss.Calculate(err, activation2.Output, inputsY)
	fmt.Println("loss:", los)

	predictions := utils.ArgmaxDense(activation2.Output)
	accuracy := utils.Accuracy(predictions, inputsY)
	fmt.Printf("acc: %.2f\n", accuracy)
}

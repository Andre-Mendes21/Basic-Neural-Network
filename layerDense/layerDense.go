package LayerDense

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type Layer_Dense struct {
	Inputs   *mat.Dense
	Dinputs  *mat.Dense
	Weights  *mat.Dense
	Dweights *mat.Dense
	Biases   *mat.VecDense
	Dbiases  *mat.VecDense
	Output   *mat.Dense
}

func NewLayer_Dense(n_inputs, n_neurons int) *Layer_Dense {
	dist := distuv.Normal{Mu: 0, Sigma: 1}
	data := make([]float64, n_inputs*n_neurons)
	for i := range data {
		data[i] = 0.01 * dist.Rand()
	}
	var inputs mat.Dense
	var dinputs mat.Dense
	weights := mat.NewDense(n_inputs, n_neurons, data)
	var dweights mat.Dense
	biases := mat.NewVecDense(n_neurons, nil)
	var dbiases mat.VecDense
	var output mat.Dense
	return &Layer_Dense{
		Inputs:   &inputs,
		Dinputs:  &dinputs,
		Weights:  weights,
		Dweights: &dweights,
		Biases:   biases,
		Dbiases:  &dbiases,
		Output:   &output,
	}
}

func (l *Layer_Dense) Forward(inputs *mat.Dense) {
	l.Inputs = mat.DenseCopyOf(inputs)
	l.Output.Mul(inputs, l.Weights)
	rows, _ := l.Output.Dims()
	for i := 0; i < rows; i++ {
		var aux mat.VecDense
		aux.AddVec(l.Output.RowView(i), l.Biases)
		l.Output.SetRow(i, aux.RawVector().Data)
	}
}

func (l *Layer_Dense) Backward(dvalues *mat.Dense) {
	fmt.Println(l.Inputs.T().Dims())
	l.Dweights.Mul(l.Inputs.T(), dvalues)

	dvalRows, dvalCols := dvalues.Dims()
	aux := mat.NewVecDense(dvalCols, nil)
	for i := 0; i < dvalCols; i++ {
		var val float64
		for j := 0; j < dvalRows; j++ {
			val += dvalues.At(j, i)
		}
		aux.SetVec(i, val)
	}
	l.Dbiases = mat.VecDenseCopyOf(aux)
	l.Dinputs.Mul(dvalues, l.Weights.T())
}

func (l *Layer_Dense) FormatMatrix(m mat.Matrix, cutoff int) fmt.Formatter {
	return mat.Formatted(m, mat.Squeeze(), mat.Excerpt(cutoff))
}

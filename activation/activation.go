package activationFuncs

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type Activation interface {
	Forward(inputs *mat.Dense)
}

type ReLU struct {
	Inputs  *mat.Dense
	Dinputs *mat.Dense
	Output  *mat.Dense
}

func (r *ReLU) Forward(inputs *mat.Dense) {
	rows, cols := inputs.Dims()
	aux := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			max := math.Max(0, inputs.At(i, j))
			aux.Set(i, j, max)
		}
	}
	r.Inputs = mat.DenseCopyOf(inputs)
	r.Output = mat.DenseCopyOf(aux)
}

func (r *ReLU) Backward(dvalues *mat.Dense) {
	r.Dinputs = mat.DenseCopyOf(dvalues)
	rows, cols := r.Dinputs.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if r.Inputs.At(i, j) <= 0 {
				r.Inputs.Set(i, j, 0)
			}
		}
	}
}

type Softmax struct {
	Output *mat.Dense
}

func (s *Softmax) Forward(inputs *mat.Dense) {
	rows, cols := inputs.Dims()
	s.Output = mat.NewDense(rows, cols, nil)

	exp_values := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			row_max := mat.Max(inputs.RowView(i))
			val := math.Exp(inputs.At(i, j) - row_max)
			exp_values.Set(i, j, val)
		}
	}

	// row-wise summation
	exp_sum := mat.NewVecDense(rows, nil)
	for i := 0; i < rows; i++ {
		var val float64
		for j := 0; j < cols; j++ {
			val += exp_values.At(i, j)
		}
		exp_sum.SetVec(i, val)
	}

	probs := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := exp_values.At(i, j) / exp_sum.AtVec(i)
			probs.Set(i, j, val)
		}
	}
	s.Output = mat.DenseCopyOf(probs)
}

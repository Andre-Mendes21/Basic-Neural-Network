package utils

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

func DenseIdentity(size int) (identity *mat.Dense) {
	identity = mat.NewDense(size, size, nil)
	for i := 0; i < size*size; i++ {
		identity.Set(i, i, 1)
	}

	return identity
}

func Accuracy(v mat.Vector, m mat.Matrix) (accuracy float64) {
	predictions := mat.VecDenseCopyOf(v)
	class := mat.DenseCopyOf(m)
	rows, cols := class.Dims()
	var selected []float64
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if predictions.AtVec(i) == class.At(i, j) {
				selected = append(selected, 1.0)
			} else {
				selected = append(selected, 0.0)
			}
		}
	}
	return stat.Mean(selected, nil)
}

func ArgmaxVector(a mat.Vector) int {
	aux := mat.VecDenseCopyOf(a)
	max := math.SmallestNonzeroFloat64
	var maxIndex int = 0
	for i := 0; i < a.Len(); i++ {
		if aux.AtVec(i) > max {
			max = aux.AtVec(i)
			maxIndex = i
		}
	}
	return maxIndex
}

func ArgmaxDense(a mat.Matrix) *mat.VecDense {
	aux := mat.DenseCopyOf(a)
	rows, _ := aux.Dims()

	argMax := mat.NewVecDense(rows, nil)
	for i := 0; i < rows; i++ {
		val := ArgmaxVector(aux.RowView(i))
		argMax.SetVec(i, float64(val))
	}
	return argMax
}

func clamp(v, lo, hi float64) float64 {
	return math.Min(math.Max(v, lo), hi)
}

func ClampVector(a mat.Vector, lo, hi float64) *mat.VecDense {
	aux := mat.VecDenseCopyOf(a)
	for i := 0; i < a.Len(); i++ {
		val := a.AtVec(i)
		aux.SetVec(i, clamp(val, lo, hi))
	}
	return aux
}

func ClampDense(m mat.Matrix, lo, hi float64) *mat.Dense {
	aux := mat.DenseCopyOf(m)
	rows, _ := aux.Dims()
	for i := 0; i < rows; i++ {
		clampledVec := ClampVector(aux.RowView(i), lo, hi)
		aux.SetRow(i, clampledVec.RawVector().Data)
	}
	return aux
}

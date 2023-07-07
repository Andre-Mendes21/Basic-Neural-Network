package loss

import (
	"74_Basic_Neural_Network/utils"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

const (
	LOBOUND = 1e-7
	HIBOUND = 1 - 1e-7
)

type Loss interface {
	Forward(*mat.Dense, *mat.Dense) *mat.VecDense
}

func Calculate(l Loss, output, y *mat.Dense) float64 {
	sampleLosses := l.Forward(output, y)

	dataLoss := stat.Mean(sampleLosses.RawVector().Data, nil)

	return dataLoss
}

type CategoricalCrossEntropy struct {
	Dinputs *mat.Dense
	Output  *mat.VecDense
}

func NewCrossEntropy() *CategoricalCrossEntropy {
	var dinputs mat.Dense
	var output mat.VecDense
	return &CategoricalCrossEntropy{
		Dinputs: &dinputs,
		Output:  &output,
	}
}

func (c *CategoricalCrossEntropy) Forward(yPred, yTrue *mat.Dense) *mat.VecDense {
	rowsPred, _ := yPred.Dims()
	yPredClipped := mat.DenseCopyOf(yPred)

	yPredClipped = utils.ClampDense(yPredClipped, LOBOUND, HIBOUND)

	_, colsTrue := yTrue.Dims()
	correctConfidences := mat.NewVecDense(rowsPred, nil)
	if colsTrue == 1 {
		for i := 0; i < rowsPred; i++ {
			val := yPredClipped.At(i, int(yTrue.At(i, 0)))
			correctConfidences.SetVec(i, val)
		}
	} else {
		var aux *mat.Dense
		aux.Mul(yPredClipped, yTrue)
		row, col := aux.Dims()
		for i := 0; i < row; i++ {
			var val float64
			for j := 0; j < col; j++ {
				val += aux.At(i, j)
			}
			correctConfidences.SetVec(i, val)
		}
	}

	negativeLogLikelihoods := mat.NewVecDense(correctConfidences.Len(), nil)
	for i := 0; i < correctConfidences.Len(); i++ {
		val := -math.Log(correctConfidences.AtVec(i))
		negativeLogLikelihoods.SetVec(i, val)
	}
	c.Output = mat.VecDenseCopyOf(negativeLogLikelihoods)
	return negativeLogLikelihoods
}

func (c *CategoricalCrossEntropy) Backward(dvalues, yTrue *mat.Dense) {
	_, labels := dvalues.Dims()
	_, yCols := yTrue.Dims()

	if yCols == 1 {
		aux := utils.DenseIdentity(labels).ColView(int(yTrue.At(0, 0)))
		eyeVec := mat.VecDenseCopyOf(aux)
		eyeVec.ScaleVec(-1, aux)
		temp := eyeVec.SliceVec(0, eyeVec.Len())
		c.Dinputs.Mul(temp, dvalues)

	}
}

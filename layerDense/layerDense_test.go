package LayerDense

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestForward(t *testing.T) {
	inputs := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})

	biases := mat.NewVecDense(3, []float64{1, 2, 3})
	weights := mat.NewDense(2, 3, []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5})
	test := NewLayer_Dense(2, 3)
	test.Biases.CopyVec(biases)
	test.Weights.Copy(weights)

	test.Forward(inputs)
	got := test.Output
	// want := mat.NewDense(3, 3, []float64{1.5, 1.5, 1.5, 3.5, 3.5, 3.5, 5.5, 5.5, 5.5})
	want := mat.NewDense(3, 3, []float64{2.5, 3.5, 4.5, 4.5, 5.5, 6.5, 6.5, 7.5, 8.5})

	if mat.Equal(got, want) == false {
		fg := mat.Formatted(got, mat.Squeeze())
		fw := mat.Formatted(want, mat.Squeeze())
		t.Errorf("Forward() = false\nwant:\n%v\n\ngot:\n%v\n", fw, fg)
	}

}

func TestBackward(t *testing.T) {
	inputs := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	dvalues := mat.NewDense(2, 3, []float64{10, 11, 12, 13, 14, 15})
	weights := mat.NewDense(2, 3, []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5})

	test := NewLayer_Dense(2, 3)
	test.Inputs = mat.DenseCopyOf(inputs)
	test.Weights = mat.DenseCopyOf(weights)

	test.Backward(dvalues)
	gotDweight := test.Dweights
	gotDbiases := test.Dbiases
	gotDinputs := test.Dinputs

	wantDweight := mat.NewDense(3, 3, []float64{62, 67, 72, 85, 92, 99, 108, 117, 126})
	wantDbiases := mat.NewVecDense(3, []float64{23, 25, 27})
	wantDinputs := mat.NewDense(2, 2, []float64{16.5, 16.5, 21.0, 21.0})

	if mat.Equal(gotDweight, wantDweight) == false {
		fg := mat.Formatted(gotDweight, mat.Squeeze())
		fw := mat.Formatted(wantDweight, mat.Squeeze())
		t.Errorf("Forward() = false\nwant:\n%v\n\ngot:\n%v\n", fw, fg)
	}
	if mat.Equal(gotDbiases, wantDbiases) == false {
		fg := mat.Formatted(gotDbiases, mat.Squeeze())
		fw := mat.Formatted(wantDbiases, mat.Squeeze())
		t.Errorf("Forward() = false\nwant:\n%v\n\ngot:\n%v\n", fw, fg)
	}
	if mat.Equal(gotDinputs, wantDinputs) == false {
		fg := mat.Formatted(gotDinputs, mat.Squeeze())
		fw := mat.Formatted(wantDinputs, mat.Squeeze())
		t.Errorf("Forward() = false\nwant:\n%v\n\ngot:\n%v\n", fw, fg)
	}
}

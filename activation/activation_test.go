package activationFuncs

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestReLUForward(t *testing.T) {
	test1 := mat.NewDense(2, 3, []float64{1, 2, 3, -1, -2, -3})
	test2 := mat.NewDense(1, 1, []float64{-0.5})
	test3 := mat.NewDense(2, 2, []float64{-1, -1, -5, -0.000001})
	test4 := mat.NewDense(2, 2, []float64{1, 1, 5, 0.000001})
	var activation1 ReLU
	var activation2 ReLU
	var activation3 ReLU
	var activation4 ReLU
	activation1.Forward(test1)
	activation2.Forward(test2)
	activation3.Forward(test3)
	activation4.Forward(test4)
	got1 := activation1.Output
	got2 := activation2.Output
	got3 := activation3.Output
	got4 := activation4.Output
	var inputs = []*mat.Dense{
		got1,
		got2,
		got3,
		got4,
	}

	want1 := mat.NewDense(2, 3, []float64{1, 2, 3, 0, 0, 0})
	want2 := mat.NewDense(1, 1, []float64{0})
	want3 := mat.NewDense(2, 2, []float64{0, 0, 0, 0})
	want4 := mat.NewDense(2, 2, []float64{1, 1, 5, 0.000001})
	var wants = []*mat.Dense{
		want1, want2, want3, want4,
	}

	for i, val := range inputs {
		if !mat.Equal(val, wants[i]) {
			fg := mat.Formatted(val, mat.Squeeze())
			fw := mat.Formatted(wants[i], mat.Squeeze())
			t.Errorf("ReLU.Forward() = false\nwant:\n%v\n\ngot:\n%v\n", fw, fg)
		}
	}
}

func TestSoftmaxForward(t *testing.T) {
	test1 := mat.NewDense(2, 3, []float64{1, 2, 3, -1, -2, -3})
	test2 := mat.NewDense(1, 1, []float64{-0.5})
	test3 := mat.NewDense(2, 2, []float64{-1, -1, -5, -0.000001})
	test4 := mat.NewDense(2, 2, []float64{1, 1, 5, 0.000001})
	var activation1 Softmax
	var activation2 Softmax
	var activation3 Softmax
	var activation4 Softmax
	activation1.Forward(test1)
	activation2.Forward(test2)
	activation3.Forward(test3)
	activation4.Forward(test4)
	got1 := activation1.Output
	got2 := activation2.Output
	got3 := activation3.Output
	got4 := activation4.Output
	var inputs = []*mat.Dense{
		got1,
		got2,
		got3,
		got4,
	}

	want1 := mat.NewDense(2, 3, []float64{
		0.09003057, 0.24472847, 0.66524096,
		0.66524096, 0.24472847, 0.09003057,
	})

	want2 := mat.NewDense(1, 1, []float64{1.0})
	want3 := mat.NewDense(2, 2, []float64{
		0.5, 0.5,
		0.00669286, 0.99330714,
	})
	want4 := mat.NewDense(2, 2, []float64{
		0.5, 0.5,
		0.99330714, 0.00669286,
	})
	var wants = []*mat.Dense{
		want1, want2, want3, want4,
	}

	for i, val := range inputs {
		if !mat.Equal(val, wants[i]) {
			fg := mat.Formatted(val, mat.Squeeze())
			fw := mat.Formatted(wants[i], mat.Squeeze())
			t.Errorf("Softmax.Forward() = false\nwant:\n%v\n\ngot:\n%v\n", fw, fg)
		}
	}
}

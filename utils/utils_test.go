package utils

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

const (
	LOBOUND = 1e-7
	HIBOUND = 1 - 1e-7
)

func TestClampVector(t *testing.T) {
	test1 := mat.NewVecDense(5, []float64{0, 0, 1 - 1e-6, 1, 3})
	test2 := mat.NewVecDense(7, []float64{5, 6, 7, 8, 0, 0, 0})
	test3 := mat.NewVecDense(7, []float64{0.33, 0.8, 0.9, 0.77, 0.1, 0.5, 0.5})
	var tests = []*mat.VecDense{test1, test2, test3}

	var gots []*mat.VecDense
	for _, elem := range tests {
		gots = append(gots, ClampVector(elem, LOBOUND, HIBOUND))
	}

	want1 := mat.NewVecDense(5, []float64{1.000000e-07, 1.000000e-07, 9.999990e-01, 9.999999e-01, 9.999999e-01})
	want2 := mat.NewVecDense(7, []float64{9.999999e-01, 9.999999e-01, 9.999999e-01, 9.999999e-01, 1.000000e-07, 1.000000e-07, 1.000000e-07})
	want3 := mat.NewVecDense(7, []float64{0.33, 0.8, 0.9, 0.77, 0.1, 0.5, 0.5})
	var wants = []*mat.VecDense{want1, want2, want3}

	for i, val := range gots {
		if !mat.EqualApprox(val, wants[i], 0.5e-7) {
			fg := mat.Formatted(gots[i], mat.Squeeze())
			fw := mat.Formatted(wants[i], mat.Squeeze())
			t.Errorf("ClampVector() = false\nwant:\n%v\n\ngot:\n%v\n", fw, fg)
		}
	}
}

func TestClampDense(t *testing.T) {
	test1 := mat.NewDense(3, 2, []float64{0, 0, 1 - 1e-6, 1, 3, 4})
	test2 := mat.NewDense(4, 2, []float64{5, 6, 7, 8, 0, 0, 0, 4})
	test3 := mat.NewDense(4, 2, []float64{0.33, 0.8, 0.9, 0.77, 0.1, 0.5, 0.5, 0.23})
	var tests = []*mat.Dense{test1, test2, test3}

	var gots []*mat.Dense
	for _, elem := range tests {
		gots = append(gots, ClampDense(elem, LOBOUND, HIBOUND))
	}

	want1 := mat.NewDense(3, 2, []float64{1.000000e-07, 1.000000e-07, 9.999990e-01, 9.999999e-01, 9.999999e-01, 9.999999e-01})
	want2 := mat.NewDense(4, 2, []float64{9.999999e-01, 9.999999e-01, 9.999999e-01, 9.999999e-01, 1.000000e-07, 1.000000e-07, 1.000000e-07, 9.999999e-01})
	want3 := mat.NewDense(4, 2, []float64{0.33, 0.8, 0.9, 0.77, 0.1, 0.5, 0.5, 0.23})
	var wants = []*mat.Dense{want1, want2, want3}

	for i, val := range gots {
		if !mat.EqualApprox(val, wants[i], 0.5e-7) {
			fg := mat.Formatted(gots[i], mat.Squeeze())
			fw := mat.Formatted(wants[i], mat.Squeeze())
			t.Errorf("ClampVector() = false\nwant:\n%v\n\ngot:\n%v\n", fw, fg)
		}
	}
}

func TestArgmaxVector(t *testing.T) {
	test1 := mat.NewVecDense(5, []float64{0, 0, 1 - 1e-6, 1, 3})
	test2 := mat.NewVecDense(7, []float64{5, 6, 7, 8, 0, 0, 0})
	test3 := mat.NewVecDense(7, []float64{0.33, 0.8, 0.9, 0.77, 0.1, 0.5, 0.5})
	var tests = []*mat.VecDense{test1, test2, test3}

	var gots []int
	for _, elem := range tests {
		gots = append(gots, ArgmaxVector(elem))
	}

	want1 := 4
	want2 := 3
	want3 := 2
	var wants = []int{want1, want2, want3}

	for i, got := range gots {
		if got != wants[i] {
			t.Errorf("ArgmaxVector() = false\nwant:\n%v\n\ngot:\n%v\n", wants[1], got)
		}
	}
}

func TestArgmaxDense(t *testing.T) {
	test1 := mat.NewDense(3, 2, []float64{0, 0, 1 - 1e-6, 1, 3, 4})
	test2 := mat.NewDense(4, 2, []float64{5, 6, 7, 8, 0, 0, 0, 4})
	test3 := mat.NewDense(4, 2, []float64{0.33, 0.8, 0.9, 0.77, 0.1, 0.5, 0.5, 0.23})
	var tests = []*mat.Dense{test1, test2, test3}

	var gots []*mat.VecDense
	for _, elem := range tests {
		gots = append(gots, ArgmaxDense(elem))
	}

	want1 := mat.NewVecDense(3, []float64{0, 1, 1})
	want2 := mat.NewVecDense(4, []float64{1, 1, 0, 1})
	want3 := mat.NewVecDense(4, []float64{1, 0, 1, 0})
	var wants = []*mat.VecDense{want1, want2, want3}

	for i, val := range gots {
		if !mat.Equal(val, wants[i]) {
			fg := mat.Formatted(gots[i], mat.Squeeze())
			fw := mat.Formatted(wants[i], mat.Squeeze())
			t.Errorf("ClampVector() = false\nwant:\n%v\n\ngot:\n%v\n", fw, fg)
		}
	}
}

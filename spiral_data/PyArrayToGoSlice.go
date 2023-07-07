package spiralData

import (
	"os"
	"strconv"
)

func pyArrayToFloatSlice(pyArray []byte) (goFloat64Slice []float64, row, col int) {
	var goByteSlice []byte
	for _, val := range pyArray {
		switch string(val) {
		case "[":
		case "]":
		case "\r":
		case "\n":
			row++
		default:
			goByteSlice = append(goByteSlice, val)
		}
	}
	for i := 0; i < len(goByteSlice); i++ {
		var aux []byte
		var j int
		for j = i; j < len(goByteSlice) && string(goByteSlice[j]) != " "; j++ {
			aux = append(aux, goByteSlice[j])
		}
		if len(aux) > 0 {
			parsedFloat, _ := strconv.ParseFloat(string(aux), 64)
			goFloat64Slice = append(goFloat64Slice, parsedFloat)
		}
		i = j
	}
	return goFloat64Slice, row + 1, 2
}

func ReadFromFile(name string) ([]float64, int, int) {
	pyArray, _ := os.ReadFile(name)
	return pyArrayToFloatSlice(pyArray)
}

package steady

import "math"

// sigmoid returns 1 / (1 + exp(-x)).
func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// dot computes the dot product of two equal-length float32 slices.
func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// PredictLogits writes OVA logistic probabilities for each label into out.
// hidden is the embedding vector (length dim). weights is numLabels × dim
// row-major, bias is numLabels floats. out must have length >= numLabels.
func PredictLogits(hidden, weights, bias, out []float32) {
	for i := range out {
		row := weights[i*len(hidden) : (i+1)*len(hidden)]
		out[i] = sigmoid(dot(hidden, row) + bias[i])
	}
}

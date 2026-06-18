// Package steady provides zero-allocation text classification in pure Go.
//
// Models are trained offline and loaded at runtime via memory-mapped files.
// Classification runs the byteSteady encoder, OVA logistic regression,
// Platt scaling, and conformal prediction in a single sub-millisecond pass
// with zero heap allocations.
//
// Quick start:
//
//	m, _ := steady.Load("model.bin")
//	defer m.Close()
//	m.SetLabelNames([]string{"identity", "constraint", "decision", "fact", "preference", "episode"})
//	result := m.Classify("I am a Go developer")
//	if !result.IsEmpty() {
//		fmt.Println(result.Kinds[0], result.Confidences[0])
//	}
//
// Training:
//
//	go run ./cmd/steady -input data.txt -output model.bin -epochs 20 -bucket 2000000 -dim 64
//
// Input format: __label__classname Text here
package steady

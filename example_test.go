package steady_test

import (
	"fmt"
	"os"
	"strings"

	"github.com/xDarkicex/steady"
)

// Example_classify demonstrates loading a model and classifying text.
func Example_classify() {
	// Generate a small synthetic model for the example.
	lines := []string{
		"__label__identity I am a software engineer",
		"__label__constraint secrets must not be shared",
		"__label__decision let us use Redis",
		"__label__fact the server runs on port 443",
		"__label__preference I prefer Go over Python",
		"__label__episode yesterday I fixed a bug",
	}
	// Repeat for minimum training data
	input := strings.Repeat(strings.Join(lines, "\n")+"\n", 30)
	tmpDir, _ := os.MkdirTemp("", "steady_example_*")
	defer os.RemoveAll(tmpDir)
	inputPath := tmpDir + "/train.txt"
	outputPath := tmpDir + "/model.bin"
	os.WriteFile(inputPath, []byte(input), 0644)

	cfg := steady.DefaultTrainConfig()
	cfg.Input = inputPath
	cfg.Output = outputPath
	cfg.Bucket = 500
	cfg.Dim = 8
	cfg.Epochs = 10
	cfg.LR = 0.2
	cfg.Seed = 42
	if err := steady.Train(cfg); err != nil {
		panic(err)
	}

	m, err := steady.Load(outputPath)
	if err != nil {
		panic(err)
	}
	defer m.Close()
	m.SetLabelNames(cfg.LabelNames)

	result := m.Classify("I am a software engineer")
	if result.IsEmpty() {
		fmt.Println("noise")
	} else {
		fmt.Println(result.Kinds[0])
	}
	// Output:
	// identity
}

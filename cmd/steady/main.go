// Command steady trains and evaluates text classification models using
// byteSteady embeddings, OVA logistic regression, Platt scaling, and
// conformal prediction.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/xDarkicex/steady"
)

func main() {
	cfg := steady.DefaultTrainConfig()
	var lrFlag float64

	flag.StringVar(&cfg.Input, "input", "", "training data path (__label__kind text format)")
	flag.StringVar(&cfg.Output, "output", "model.bin", "output model artifact path")
	flag.IntVar(&cfg.Bucket, "bucket", cfg.Bucket, "embedding table rows")
	flag.IntVar(&cfg.Dim, "dim", cfg.Dim, "embedding dimension")
	flag.IntVar(&cfg.Epochs, "epochs", cfg.Epochs, "SGD epochs")
	flag.Float64Var(&lrFlag, "lr", float64(cfg.LR), "learning rate")
	flag.Parse()

	cfg.LR = float32(lrFlag)

	if cfg.Input == "" {
		fmt.Fprintln(os.Stderr, "usage: steady train -input <data.txt> [-output model.bin]")
		os.Exit(1)
	}

	if err := steady.Train(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "steady: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Training complete:", cfg.Output)
}

# steady Quick Start

## Preferred Settings

After extensive hyperparameter tuning on a 60,000-example multilingual dataset
across 6 classes, the following settings produce Q ≈ 0.08 and >95% calibrated
confidence on correct classifications:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `-bucket` | 2,000,000 | Large enough to avoid byte n-gram hash collisions. 20K is too small — variable n-grams (3–7 bytes) from real text collide heavily. |
| `-dim` | 64 | Good balance of capacity vs. model size. 32 works for simple problems, 128 if accuracy needed. |
| `-epochs` | 20 | 10–25 range works. More epochs after Q stabilizes adds no benefit. |
| `-lr` | 0.1 | Linear decay to 0 over epochs. Standard SGD default. |

## Training

```bash
# Train a model from labeled data
go run ./cmd/steady \
  -input training_data.txt \
  -output model.bin \
  -bucket 2000000 \
  -dim 64 \
  -epochs 20
```

## Input Format

One example per line. 6 labels by default:

```
__label__identity I am a Go developer working in Berlin
__label__constraint API keys must never be committed to version control
__label__decision I decided to use PostgreSQL instead of MySQL
__label__fact The latency p99 is around 200ms
__label__preference I prefer Neovim over VS Code any day
__label__episode Yesterday I deployed to production and everything went smoothly
```

## Inference

```go
m, _ := steady.Load("model.bin")
defer m.Close()
m.SetLabelNames([]string{"identity", "constraint", "decision", "fact", "preference", "episode"})

result := m.Classify("oh btw I prefer Neovim over VS Code")
// result.Kinds = ["preference"]
// result.Confidences = [0.97]

result = m.Classify("hmm interesting point")
// result.IsEmpty() = true  (noise rejected by conformal prediction)
```

## Model Size

| Bucket | Dim | Labels | File Size |
|--------|-----|--------|-----------|
| 20,000 | 64 | 6 | 5 MB |
| 200,000 | 64 | 6 | 51 MB |
| 2,000,000 | 64 | 6 | 488 MB |

The embedding table is mmap'd at runtime — only accessed pages are resident.
OS page cache handles memory pressure automatically.

## How It Works

1. **byteSteady encoder** — raw text is encoded by sliding multi-scale byte
   n-grams (4, 8, 12, 16 bytes) through FNV-1a hash → embedding row lookup.
   Rows are averaged into a 64-dim hidden vector.
2. **OVA logistic head** — one sigmoid per label over the hidden vector.
   Independent per-class probabilities (not softmax — a text can be both
   identity AND constraint simultaneously).
3. **Platt scaling** — raw logits are calibrated via logistic regression
   (A×logit + B) fit on a held-out set.
4. **Conformal prediction** — labels with non-conformity ≤ Q are included
   in the prediction set. Empty set = noise/out-of-distribution.

## Tuning Guide

**Q too high (>0.3):** Model isn't confident. Increase epochs, bucket, or dim.
Use `ClassifyDebug` to inspect raw logits — check if correct label consistently
scores higher than others.

**Q too low (<0.01):** Overconfident, possible overfitting. Reduce epochs or dim,
check that calibration set is representative.

**All predictions empty:** Platt parameters haven't converged. The calibration
optimizer needs ~2000 iterations at lr=1.0. If Platt A values are near zero,
re-train with more calibration data.

**Specific classes missing:** That class may be underrepresented. Check training
data distribution. Minimum 5,000 examples per class recommended.

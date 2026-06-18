package steady

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"unsafe"

	"github.com/xDarkicex/memory"
)

const (
	// modelMagic is the 4-byte identifier for steady model files ("BYTE").
	modelMagic   uint32 = 0x42595445
	modelVersion uint32 = 1
	headerSize   int64  = 32
)

// Model holds a loaded classification model. All fields are read-only after Load.
type Model struct {
	table       []float32 // mmap'd embedding table, bucket × dim
	weights     []float32 // OVA weights, numLabels × dim
	bias        []float32 // OVA bias, numLabels
	plattA      []float32 // Platt slope, numLabels
	plattB      []float32 // Platt intercept, numLabels
	q           float32   // conformal quantile
	bucket      int
	dim         int
	numLabels   int
	labelNames  []string
	modelPool   *memory.Pool
	scratchPool *memory.Pool
}

// Load opens a model file and returns the loaded Model. The caller must call
// Close to release resources.
func Load(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("steady: open model: %w", err)
	}
	defer f.Close()

	var hdr [headerSize]byte
	if _, err := f.Read(hdr[:]); err != nil {
		return nil, fmt.Errorf("steady: read header: %w", err)
	}
	magic := binary.LittleEndian.Uint32(hdr[0:4])
	if magic != modelMagic {
		return nil, errors.New("steady: not a steady model file")
	}
	version := binary.LittleEndian.Uint32(hdr[4:8])
	if version != modelVersion {
		return nil, fmt.Errorf("steady: unsupported model version %d", version)
	}

	bucket := int(binary.LittleEndian.Uint32(hdr[8:12]))
	dim := int(binary.LittleEndian.Uint32(hdr[12:16]))
	minN := binary.LittleEndian.Uint32(hdr[16:20])
	maxN := binary.LittleEndian.Uint32(hdr[20:24])
	numLabels := int(binary.LittleEndian.Uint32(hdr[24:28]))
	_ = minN
	_ = maxN

	if bucket <= 0 || dim <= 0 || numLabels <= 0 {
		return nil, errors.New("steady: invalid header dimensions")
	}

	// Map the embedding table.
	tableBytes := bucket * dim * 4
	tableRaw, err := memory.MmapFileReadOnly(int(f.Fd()), headerSize, tableBytes)
	if err != nil {
		return nil, fmt.Errorf("steady: mmap table: %w", err)
	}
	table := unsafe.Slice((*float32)(unsafe.Pointer(&tableRaw[0])), bucket*dim)

	modelPool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  16 * 1024 * 1024,
		SlabSize:  1024 * 1024,
		SlabCount: 8,
	})
	if err != nil {
		memory.Munmap(tableRaw)
		return nil, fmt.Errorf("steady: create model pool: %w", err)
	}

	scratchPool, err := memory.NewPool(memory.AllocatorConfig{
		PoolSize:  2 * 1024 * 1024,
		SlabSize:  256 * 1024,
		SlabCount: 4,
	})
	if err != nil {
		modelPool.Free()
		memory.Munmap(tableRaw)
		return nil, fmt.Errorf("steady: create scratch pool: %w", err)
	}

	// Seek past the embedding table (already mmap'd).
	if _, err := f.Seek(headerSize+int64(tableBytes), 0); err != nil {
		memory.Munmap(tableRaw)
		modelPool.Free()
		scratchPool.Free()
		return nil, fmt.Errorf("steady: seek past table: %w", err)
	}

	// Read the small arrays from the file (after the table).
	weights := memory.MustPoolSlice[float32](modelPool, numLabels*dim)
	weights = weights[:numLabels*dim]
	readFloats(f, weights)

	bias := memory.MustPoolSlice[float32](modelPool, numLabels)
	bias = bias[:numLabels]
	readFloats(f, bias)

	plattA := memory.MustPoolSlice[float32](modelPool, numLabels)
	plattA = plattA[:numLabels]
	readFloats(f, plattA)

	plattB := memory.MustPoolSlice[float32](modelPool, numLabels)
	plattB = plattB[:numLabels]
	readFloats(f, plattB)

	var qRaw [4]byte
	if _, err := f.Read(qRaw[:]); err != nil {
		memory.Munmap(tableRaw)
		modelPool.Free()
		scratchPool.Free()
		return nil, fmt.Errorf("steady: read quantile: %w", err)
	}
	q := float32fromle(qRaw)

	m := &Model{
		table:       table,
		weights:     weights,
		bias:        bias,
		plattA:      plattA,
		plattB:      plattB,
		q:           q,
		bucket:      bucket,
		dim:         dim,
		numLabels:   numLabels,
		modelPool:   modelPool,
		scratchPool: scratchPool,
	}
	m.setDefaultLabelNames()
	return m, nil
}

// Close releases resources held by the model.
func (m *Model) Close() error {
	if m.table != nil {
		tableRaw := unsafe.Slice((*byte)(unsafe.Pointer(&m.table[0])), len(m.table)*4)
		if err := memory.Munmap(tableRaw); err != nil {
			return err
		}
		m.table = nil
	}
	if m.modelPool != nil {
		m.modelPool.Free()
		m.modelPool = nil
	}
	if m.scratchPool != nil {
		m.scratchPool.Free()
		m.scratchPool = nil
	}
	return nil
}

func readFloats(f *os.File, dst []float32) {
	binary.Read(f, binary.LittleEndian, dst)
}

func float32fromle(b [4]byte) float32 {
	return float32frombits(binary.LittleEndian.Uint32(b[:]))
}

func float32frombits(u uint32) float32 {
	return *(*float32)(unsafe.Pointer(&u))
}

// setDefaultLabelNames populates labelNames with generic "label_0".."label_N-1"
// defaults. Callers should use SetLabelNames to override with meaningful names.
func (m *Model) setDefaultLabelNames() {
	names := make([]string, m.numLabels)
	for i := range names {
		names[i] = fmt.Sprintf("label_%d", i)
	}
	m.labelNames = names
}

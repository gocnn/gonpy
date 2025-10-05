// Package npy provides support for reading and writing tensors in NPY and NPZ formats.
// The NPY format is used for single tensors, while NPZ is a zipped archive for multiple named tensors.
// This implementation follows the NPY format specification and supports a subset of data types
// commonly used in machine learning frameworks like Candle.
//
// Note: This package assumes the existence of a Tensor type, DType enum, Shape struct, and Device.
// These are placeholders and should be replaced with actual types from your ML framework.
// For demonstration, minimal definitions are provided.
//
// Supported DTypes: BF16, F16, F32, F64, I64, U32, U8.
// Fortran order is not supported for reading/writing.

package gonpy

import (
	"archive/zip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
)

const (
	npyMagicString = "\x93NUMPY"
	npySuffix      = ".npy"
)

// DType represents the data type of the tensor.
// This is a placeholder; in a real framework, this would be an enum from the core package.
type DType string

const (
	DTypeBF16   DType = "bf16"
	DTypeF16    DType = "f16"
	DTypeF32    DType = "f32"
	DTypeF64    DType = "f64"
	DTypeI64    DType = "i64"
	DTypeU32    DType = "u32"
	DTypeU8     DType = "u8"
	DTypeF8E4M3 DType = "f8e4m3"
)

// Shape represents the shape of the tensor.
// This is a placeholder; typically a struct with methods like ElemCount().
type Shape []int

func (s Shape) ElemCount() int {
	count := 1
	for _, dim := range s {
		count *= dim
	}
	return count
}

// Tensor represents a multi-dimensional array.
// This is a placeholder; in a real framework, this would have more methods.
type Tensor struct {
	Data   interface{} // e.g., []float32, []uint16 for f16, etc.
	Shape  Shape
	DType  DType
	Device string // e.g., "cpu"
}

// String returns a string representation of the tensor.
func (t *Tensor) String() string {
	var dataStr string
	switch d := t.Data.(type) {
	case []float32:
		dataStr = fmt.Sprintf("%v", d)
	case []float64:
		dataStr = fmt.Sprintf("%v", d)
	case []int64:
		dataStr = fmt.Sprintf("%v", d)
	case []uint32:
		dataStr = fmt.Sprintf("%v", d)
	case []uint16:
		dataStr = fmt.Sprintf("%v", d)
	case []byte:
		dataStr = fmt.Sprintf("%v", d)
	case []int8:
		dataStr = fmt.Sprintf("%v", d)
	default:
		dataStr = fmt.Sprintf("%v", d)
	}
	return fmt.Sprintf("&{%s %v %s %s}", dataStr, t.Shape, t.DType, t.Device)
}

// ErrorNpy is a custom error type for NPY-related errors.
type ErrorNpy struct {
	Msg string
}

func (e ErrorNpy) Error() string {
	return fmt.Sprintf("npy error: %s", e.Msg)
}

// readHeader reads the NPY header from the reader.
func readHeader(r io.Reader) (string, error) {
	magic := make([]byte, len(npyMagicString))
	if _, err := io.ReadFull(r, magic); err != nil {
		return "", err
	}
	if string(magic) != npyMagicString {
		return "", ErrorNpy{Msg: "magic string mismatch"}
	}

	var version [2]byte
	if _, err := io.ReadFull(r, version[:]); err != nil {
		return "", err
	}

	var headerLenLen int
	switch version[0] {
	case 1:
		headerLenLen = 2
	case 2:
		headerLenLen = 4
	default:
		return "", ErrorNpy{Msg: fmt.Sprintf("unsupported version %d", version[0])}
	}

	headerLenBytes := make([]byte, headerLenLen)
	if _, err := io.ReadFull(r, headerLenBytes); err != nil {
		return "", err
	}

	headerLen := int(binary.LittleEndian.Uint32(append(headerLenBytes, 0, 0)[:4])) // Pad to 4 bytes if needed

	header := make([]byte, headerLen)
	if _, err := io.ReadFull(r, header); err != nil {
		return "", err
	}

	return string(header), nil
}

// Header represents the parsed NPY header.
type Header struct {
	Descr        DType
	FortranOrder bool
	Shape        Shape
}

// String formats the header as a string for writing.
func (h *Header) String() (string, error) {
	fortranOrder := "False"
	if h.FortranOrder {
		fortranOrder = "True"
	}

	var shapeStr string
	if len(h.Shape) == 0 {
		shapeStr = "()"
	} else {
		parts := make([]string, len(h.Shape))
		for i, dim := range h.Shape {
			parts[i] = strconv.Itoa(dim)
		}
		shapeStr = "(" + strings.Join(parts, ",") + ",)"
	}

	var descr string
	switch h.Descr {
	case DTypeBF16:
		return "", ErrorNpy{Msg: "bf16 is not supported for writing"}
	case DTypeF16:
		descr = "f2"
	case DTypeF32:
		descr = "f4"
	case DTypeF64:
		descr = "f8"
	case DTypeI64:
		descr = "i8"
	case DTypeU32:
		descr = "u4"
	case DTypeU8:
		descr = "u1"
	case DTypeF8E4M3:
		return "", ErrorNpy{Msg: "f8e4m3 is not supported for writing"}
	default:
		return "", ErrorNpy{Msg: fmt.Sprintf("unsupported dtype %s", h.Descr)}
	}

	return fmt.Sprintf("{'descr': '<%s', 'fortran_order': %s, 'shape': %s, }", descr, fortranOrder, shapeStr), nil
}

// parseHeader parses the header string into a Header struct.
func parseHeader(headerStr string) (*Header, error) {
	// Trim outer braces and whitespace
	headerStr = strings.Trim(headerStr, "{} \t\n\r,")

	// Simple parser: split by top-level commas
	re := regexp.MustCompile(`(?s)'([^']*)':\s*([^,]*?)(?:,\s*|$)`)

	matches := re.FindAllStringSubmatch(headerStr, -1)
	if len(matches) == 0 {
		return nil, ErrorNpy{Msg: "unable to parse header"}
	}

	partMap := make(map[string]string)
	for _, match := range matches {
		if len(match) != 3 {
			continue
		}
		key := strings.Trim(match[1], "' ")
		value := strings.Trim(match[2], "' ")
		partMap[key] = value
	}

	fortranOrder := false
	if fo, ok := partMap["fortran_order"]; ok {
		switch fo {
		case "False":
			fortranOrder = false
		case "True":
			fortranOrder = true
		default:
			return nil, ErrorNpy{Msg: fmt.Sprintf("unknown fortran_order %s", fo)}
		}
	}

	descrStr, ok := partMap["descr"]
	if !ok || descrStr == "" {
		return nil, ErrorNpy{Msg: "no descr in header"}
	}
	if strings.HasPrefix(descrStr, ">") {
		return nil, ErrorNpy{Msg: fmt.Sprintf("big-endian descr %s not supported", descrStr)}
	}
	descrStr = strings.Trim(descrStr, "=<>|")
	var descr DType
	switch descrStr {
	case "e", "f2":
		descr = DTypeF16
	case "f", "f4":
		descr = DTypeF32
	case "d", "f8":
		descr = DTypeF64
	case "q", "i8":
		descr = DTypeI64
	case "B", "u1":
		descr = DTypeU8
	case "I", "u4":
		descr = DTypeU32
	case "?", "b1":
		descr = DTypeU8 // Bool as U8
	default:
		return nil, ErrorNpy{Msg: fmt.Sprintf("unrecognized descr %s", descrStr)}
	}

	shapeStr, ok := partMap["shape"]
	if !ok {
		return nil, ErrorNpy{Msg: "no shape in header"}
	}
	shapeStr = strings.Trim(shapeStr, "() ,")
	var shape Shape
	if shapeStr != "" {
		parts := strings.Split(shapeStr, ",")
		shape = make(Shape, len(parts))
		for i, p := range parts {
			dim, err := strconv.Atoi(strings.TrimSpace(p))
			if err != nil {
				return nil, err
			}
			shape[i] = dim
		}
	}

	return &Header{
		Descr:        descr,
		FortranOrder: fortranOrder,
		Shape:        shape,
	}, nil
}

// readData reads the tensor data from the reader based on shape and dtype.
// Returns the data as interface{} (typed slice).
func readData(shape Shape, dtype DType, r io.Reader) (interface{}, error) {
	elemCount := shape.ElemCount()

	switch dtype {
	case DTypeBF16:
		data := make([]uint16, elemCount) // Assume bf16 as uint16 bits
		if err := binary.Read(r, binary.LittleEndian, data); err != nil {
			return nil, err
		}
		return data, nil
	case DTypeF16:
		data := make([]uint16, elemCount) // Assume f16 as uint16 bits
		if err := binary.Read(r, binary.LittleEndian, data); err != nil {
			return nil, err
		}
		return data, nil
	case DTypeF32:
		data := make([]float32, elemCount)
		if err := binary.Read(r, binary.LittleEndian, data); err != nil {
			return nil, err
		}
		return data, nil
	case DTypeF64:
		data := make([]float64, elemCount)
		if err := binary.Read(r, binary.LittleEndian, data); err != nil {
			return nil, err
		}
		return data, nil
	case DTypeI64:
		data := make([]int64, elemCount)
		if err := binary.Read(r, binary.LittleEndian, data); err != nil {
			return nil, err
		}
		return data, nil
	case DTypeU32:
		data := make([]uint32, elemCount)
		if err := binary.Read(r, binary.LittleEndian, data); err != nil {
			return nil, err
		}
		return data, nil
	case DTypeU8:
		data := make([]byte, elemCount)
		if _, err := io.ReadFull(r, data); err != nil {
			return nil, err
		}
		return data, nil
	case DTypeF8E4M3:
		data := make([]int8, elemCount) // Assume f8e4m3 as int8 bits
		if err := binary.Read(r, binary.LittleEndian, data); err != nil {
			return nil, err
		}
		return data, nil
	default:
		return nil, ErrorNpy{Msg: fmt.Sprintf("unsupported dtype %s", dtype)}
	}
}

// ReadNPY reads a single tensor from an NPY file.
func ReadNPY(path string) (*Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	headerStr, err := readHeader(f)
	if err != nil {
		return nil, err
	}

	header, err := parseHeader(headerStr)
	if err != nil {
		return nil, err
	}
	if header.FortranOrder {
		return nil, ErrorNpy{Msg: "fortran order not supported"}
	}

	data, err := readData(header.Shape, header.Descr, f)
	if err != nil {
		return nil, err
	}

	return &Tensor{
		Data:   data,
		Shape:  header.Shape,
		DType:  header.Descr,
		Device: "cpu", // Assume CPU
	}, nil
}

// ReadNPZ reads all named tensors from an NPZ file.
func ReadNPZ(path string) ([]struct {
	Name   string
	Tensor *Tensor
}, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	var result []struct {
		Name   string
		Tensor *Tensor
	}
	for _, file := range r.File {
		rc, err := file.Open()
		if err != nil {
			return nil, err
		}
		defer rc.Close()

		name := strings.TrimSuffix(file.Name, npySuffix)

		headerStr, err := readHeader(rc)
		if err != nil {
			return nil, err
		}

		header, err := parseHeader(headerStr)
		if err != nil {
			return nil, err
		}
		if header.FortranOrder {
			return nil, ErrorNpy{Msg: "fortran order not supported"}
		}

		data, err := readData(header.Shape, header.Descr, rc)
		if err != nil {
			return nil, err
		}

		result = append(result, struct {
			Name   string
			Tensor *Tensor
		}{
			Name: name,
			Tensor: &Tensor{
				Data:   data,
				Shape:  header.Shape,
				DType:  header.Descr,
				Device: "cpu",
			},
		})
	}
	return result, nil
}

// ReadNPZByName reads specific named tensors from an NPZ file.
func ReadNPZByName(path string, names []string) ([]*Tensor, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	var result []*Tensor
	for _, name := range names {
		fileName := name + npySuffix
		file, err := r.Open(fileName)
		if err != nil {
			if errors.Is(err, zip.ErrFormat) || strings.Contains(err.Error(), "not found") {
				return nil, ErrorNpy{Msg: fmt.Sprintf("no array for %s in %s", name, path)}
			}
			return nil, err
		}
		defer file.Close()

		headerStr, err := readHeader(file)
		if err != nil {
			return nil, err
		}

		header, err := parseHeader(headerStr)
		if err != nil {
			return nil, err
		}
		if header.FortranOrder {
			return nil, ErrorNpy{Msg: "fortran order not supported"}
		}

		data, err := readData(header.Shape, header.Descr, file)
		if err != nil {
			return nil, err
		}

		result = append(result, &Tensor{
			Data:   data,
			Shape:  header.Shape,
			DType:  header.Descr,
			Device: "cpu",
		})
	}
	return result, nil
}

// writeData writes the tensor data to the writer.
func writeData(w io.Writer, data interface{}) error {
	switch d := data.(type) {
	case []uint16: // BF16 or F16
		return binary.Write(w, binary.LittleEndian, d)
	case []float32:
		return binary.Write(w, binary.LittleEndian, d)
	case []float64:
		return binary.Write(w, binary.LittleEndian, d)
	case []int64:
		return binary.Write(w, binary.LittleEndian, d)
	case []uint32:
		return binary.Write(w, binary.LittleEndian, d)
	case []byte:
		_, err := w.Write(d)
		return err
	case []int8: // F8E4M3
		return binary.Write(w, binary.LittleEndian, d)
	default:
		return ErrorNpy{Msg: "unsupported data type for writing"}
	}
}

// Write writes the tensor to the writer in NPY format.
func (t *Tensor) Write(w io.Writer) error {
	if _, err := w.Write([]byte(npyMagicString)); err != nil {
		return err
	}
	if _, err := w.Write([]byte{1, 0}); err != nil { // Version 1.0
		return err
	}

	header := &Header{
		Descr:        t.DType,
		FortranOrder: false,
		Shape:        t.Shape,
	}
	headerStr, err := header.String()
	if err != nil {
		return err
	}

	// Pad to 16-byte alignment
	totalPrefixLen := len(npyMagicString) + 2 + 2 + len(headerStr) // Magic + version + len + header
	pad := (16 - (totalPrefixLen % 16)) % 16
	headerStr += strings.Repeat(" ", pad) + "\n"

	headerLen := uint16(len(headerStr))
	lenBytes := make([]byte, 2)
	binary.LittleEndian.PutUint16(lenBytes, headerLen)
	if _, err := w.Write(lenBytes); err != nil {
		return err
	}

	if _, err := w.Write([]byte(headerStr)); err != nil {
		return err
	}

	return writeData(w, t.Data)
}

// WriteNPY writes the tensor to an NPY file.
func (t *Tensor) WriteNPY(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return t.Write(f)
}

// WriteNPZ writes multiple named tensors to an NPZ file.
func WriteNPZ(path string, tensors map[string]*Tensor) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	zw := zip.NewWriter(f)
	defer zw.Close()

	for name, tensor := range tensors {
		w, err := zw.Create(name + npySuffix)
		if err != nil {
			return err
		}
		if err := tensor.Write(w); err != nil {
			return err
		}
	}
	return nil
}

// NpzTensors provides lazy loading of tensors from an NPZ file.
type NpzTensors struct {
	indexPerName map[string]int
	path         string
}

// NewNpzTensors creates a new lazy loader for an NPZ file.
func NewNpzTensors(path string) (*NpzTensors, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	indexPerName := make(map[string]int)
	for i, file := range r.File {
		name := strings.TrimSuffix(file.Name, npySuffix)
		indexPerName[name] = i
	}

	return &NpzTensors{
		indexPerName: indexPerName,
		path:         path,
	}, nil
}

// Names returns the list of tensor names in the NPZ file.
func (n *NpzTensors) Names() []string {
	names := make([]string, 0, len(n.indexPerName))
	for name := range n.indexPerName {
		names = append(names, name)
	}
	return names
}

// GetShapeAndDType returns the shape and dtype for a named tensor without loading data.
func (n *NpzTensors) GetShapeAndDType(name string) (Shape, DType, error) {
	index, ok := n.indexPerName[name]
	if !ok {
		return nil, "", fmt.Errorf("cannot find tensor %s", name)
	}

	r, err := zip.OpenReader(n.path)
	if err != nil {
		return nil, "", err
	}
	defer r.Close()

	rc, err := r.File[index].Open()
	if err != nil {
		return nil, "", err
	}
	defer rc.Close()

	headerStr, err := readHeader(rc)
	if err != nil {
		return nil, "", err
	}

	header, err := parseHeader(headerStr)
	if err != nil {
		return nil, "", err
	}

	return header.Shape, header.Descr, nil
}

// Get loads a named tensor from the NPZ file.
func (n *NpzTensors) Get(name string) (*Tensor, error) {
	index, ok := n.indexPerName[name]
	if !ok {
		return nil, fmt.Errorf("cannot find tensor %s", name)
	}

	r, err := zip.OpenReader(n.path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	rc, err := r.File[index].Open()
	if err != nil {
		return nil, err
	}
	defer rc.Close()

	headerStr, err := readHeader(rc)
	if err != nil {
		return nil, err
	}

	header, err := parseHeader(headerStr)
	if err != nil {
		return nil, err
	}
	if header.FortranOrder {
		return nil, ErrorNpy{Msg: "fortran order not supported"}
	}

	data, err := readData(header.Shape, header.Descr, rc)
	if err != nil {
		return nil, err
	}

	return &Tensor{
		Data:   data,
		Shape:  header.Shape,
		DType:  header.Descr,
		Device: "cpu",
	}, nil
}

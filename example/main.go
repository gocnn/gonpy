package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gocnn/gonpy"
)

func main() {
	// Example 1: Writing a single tensor to an NPY file
	fmt.Println("Example 1: Writing a tensor to an NPY file")
	data := []float32{1.0, 2.0, 3.0, 4.0}
	tensor := &gonpy.Tensor{
		Data:   data,
		Shape:  gonpy.Shape{2, 2},
		DType:  gonpy.DTypeF32,
		Device: "cpu",
	}

	npyPath := "test.npy"
	if err := tensor.WriteNPY(npyPath); err != nil {
		log.Fatalf("Failed to write NPY file: %v", err)
	}
	fmt.Printf("Wrote tensor to %s\n", npyPath)

	// Example 2: Reading a tensor from an NPY file
	fmt.Println("\nExample 2: Reading a tensor from an NPY file")
	readTensor, err := gonpy.ReadNPY(npyPath)
	if err != nil {
		log.Fatalf("Failed to read NPY file: %v", err)
	}
	fmt.Printf("Read tensor: %s\n", readTensor)

	// Example 3: Writing multiple tensors to an NPZ file
	fmt.Println("\nExample 3: Writing multiple tensors to an NPZ file")
	tensors := map[string]*gonpy.Tensor{
		"tensor1": {
			Data:   []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
			Shape:  gonpy.Shape{2, 3},
			DType:  gonpy.DTypeF32,
			Device: "cpu",
		},
		"tensor2": {
			Data:   []int64{10, 20, 30},
			Shape:  gonpy.Shape{3},
			DType:  gonpy.DTypeI64,
			Device: "cpu",
		},
	}

	npzPath := "test.npz"
	if err := gonpy.WriteNPZ(npzPath, tensors); err != nil {
		log.Fatalf("Failed to write NPZ file: %v", err)
	}
	fmt.Printf("Wrote tensors to %s\n", npzPath)

	// Example 4: Reading all tensors from an NPZ file
	fmt.Println("\nExample 4: Reading all tensors from an NPZ file")
	loadedTensors, err := gonpy.ReadNPZ(npzPath)
	if err != nil {
		log.Fatalf("Failed to read NPZ file: %v", err)
	}
	for _, nt := range loadedTensors {
		fmt.Printf("Tensor '%s': %s\n", nt.Name, nt.Tensor)
	}

	// Example 5: Reading specific tensors by name from an NPZ file
	fmt.Println("\nExample 5: Reading specific tensors by name")
	names := []string{"tensor1"}
	selectedTensors, err := gonpy.ReadNPZByName(npzPath, names)
	if err != nil {
		log.Fatalf("Failed to read tensors by name: %v", err)
	}
	for i, t := range selectedTensors {
		fmt.Printf("Selected tensor '%s': %s\n", names[i], t)
	}

	// Example 6: Using NpzTensors for lazy loading
	fmt.Println("\nExample 6: Lazy loading with NpzTensors")
	npzTensors, err := gonpy.NewNpzTensors(npzPath)
	if err != nil {
		log.Fatalf("Failed to create NpzTensors: %v", err)
	}

	// List available tensor names
	names = npzTensors.Names()
	fmt.Println("Available tensor names:", names)

	// Get shape and dtype without loading data
	for _, name := range names {
		shape, dtype, err := npzTensors.GetShapeAndDType(name)
		if err != nil {
			log.Fatalf("Failed to get shape and dtype for %s: %v", name, err)
		}
		fmt.Printf("Tensor '%s' - Shape: %v, DType: %s\n", name, shape, dtype)
	}

	// Load a specific tensor
	tensor1, err := npzTensors.Get("tensor1")
	if err != nil {
		log.Fatalf("Failed to load tensor 'tensor1': %v", err)
	}
	fmt.Printf("Loaded tensor 'tensor1': %s\n", tensor1)

	// Clean up
	if err := os.Remove(npyPath); err != nil {
		log.Printf("Failed to remove %s: %v", npyPath, err)
	}
	if err := os.Remove(npzPath); err != nil {
		log.Printf("Failed to remove %s: %v", npzPath, err)
	}
}

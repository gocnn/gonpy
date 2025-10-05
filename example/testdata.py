import numpy as np
import pathlib

# Define output directory
output_dir = pathlib.Path("data")
output_dir.mkdir(exist_ok=True)

# Modern dtype objects (fully supported in NumPy 2.0+)
dtype_list = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]

for dtype in dtype_list:
    ii = 1
    dtype_str = np.dtype(dtype).str[1:]  # e.g., 'i1', 'u1', 'f4', etc.
    for dims in (1, 2):
        for order in ("C", "F"):
            # Create base array
            data = np.arange(8, dtype=dtype)

            if dims == 2:
                # Elegant reshape to consistent (4, 2) shape with specified order
                data = data.reshape(4, 2, order=order)

            # Construct filename matching original format
            fname = output_dir / f"{dtype_str}_{ii}.npy"
            np.save(fname, data)
            ii += 1

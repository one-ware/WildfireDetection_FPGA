import sys
import os

try:
    import onnx
except Exception as e:
    print("Missing dependency 'onnx'. Install with: pip install onnx")
    raise

try:
    import numpy as np
except Exception:
    print("Missing dependency 'numpy'. Install with: pip install numpy")
    raise


def count_params(onnx_path):
    model = onnx.load(onnx_path)
    total = 0
    for init in model.graph.initializer:
        dims = list(init.dims)
        if len(dims) == 0:
            size = 1
        else:
            size = int(np.prod(dims))
        total += size
    return total


if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        "models/deeplab_binary.onnx",
        "models/unet_seg.onnx",
    ]

    for f in files:
        if not os.path.exists(f):
            print(f"{f}: NOT FOUND")
            continue
        try:
            cnt = count_params(f)
            print(f"{f}: {cnt:,} parameters")
        except Exception as e:
            print(f"{f}: ERROR - {e}")

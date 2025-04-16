# TensorLens

TensorLens is a minimalistic Python library to trace and visualise tensors. It allows you to trace tensors, normalize them, and store them for later use. It also provides an interactive viewer for inspecting these tensors. Currently supporting only 1D,2D and 3D tensors.

## Installation

Install TensorLens from PyPI:

```bash
pip install tensorlens
```

For a quick demo

```bash
uv run --with tensorlens tensorlens
```

## Usage

It can be used to visualise and manipulate tensors using UI. for example this code visualises GPT2 state dict tensors

```bash
import torch
import numpy as np
from transformers import GPT2Model, GPT2Config
from tensorlens.tensorlens import trace, viewer

[trace(key, tensor.detach().cpu().numpy()) for key, tensor in GPT2Model.from_pretrained('gpt2-large').state_dict().items()]

viewer(height='100%')
```


### Trace Operator

The core operation of TensorLens is the `trace` function. 

```python
import numpy as np
from tensorlens.tensorlens import trace

# Example: Tracing a 2D tensor
tensor = np.random.randint(-100, 100, size=(10, 10))
trace("my_tensor", tensor)
```

#### Parameters:

- `key`: A unique string identifier for the tensor.
- `tensor`: The tensor to trace (must be a NumPy array).
- `normalize_range`: Tuple specifying the range for clipping (default: `(-1.0, 1.0)`).
- `normalization`: Defines the normalization strategy:
  - `"clip"`: Clips values within the specified range.
  - `"minmax"`: Scales values between [-128, 127] based on tensor's min and max.
  - `"zscore"`: Normalizes based on z-score and clips outliers.
  - `"none"`: No normalization (assumes the tensor is already in the desired range).

### Viewer

The library includes a web-based viewer for exploring your tensors. You can run a local server or view the tensors in Jupyter/Colab notebooks.

```python
from tensorlens.tensorlens import viewer

# Start the viewer server
viewer(host="127.0.0.1", port=8000, notebook=True)
```

### Command-Line Interface

#### Options:
- `--debug`: Run in debug server mode.
- `--notebook`: Run in notebook/Colab mode.
- `--workers`: Number of server workers.
- `--host`: Host to bind.
- `--port`: Port to bind.
- `--tensordata_path`: Path to store tensor data.
- `--downsample_threshold`: Points threshold after which downsampling occurs.

## Example

```python
from tensorlens import trace, viewer
import numpy as np

# Trace a few tensors
trace("demo_1d", np.random.randint(-100, 100, size=30))
trace("demo_2d", np.random.randint(-100, 100, size=(20, 20)))
trace("demo_3d", np.random.randint(-100, 100, size=(20, 20, 20)))

# Start the viewer server
viewer(port=8080, debug=True)
```

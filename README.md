# TensorLens

demo in seconds with `uvx tensorlens` if you have uv installed.


![image](https://github.com/user-attachments/assets/d9161c40-4edb-4657-a550-2b9c2f56c83f)

TensorLens is a minimalistic Python library to trace and visualise tensors. It also provides an interactive viewer for inspecting these tensors. Currently supporting only 1D,2D and 3D tensors.

## Installation

try in colab in 3 lines:

![image](https://github.com/user-attachments/assets/31605c78-957b-48e4-a501-25776d3fd63a)

<a href="https://colab.research.google.com/github/attentionmech/tensorlens/blob/main/tensorlens/notebooks/tensorlens_setup_demo.ipynb" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<br><br>

Source of data is backend, hence you can change params like normalisation strategy and they will reflect immediately:

![image](https://github.com/user-attachments/assets/73b19e86-b188-4b1e-af0e-c70c0e0ce76b)

<br><br>

Install TensorLens from PyPI:

```bash
pip install tensorlens
```

For a quick demo

```bash
uv run --with tensorlens tensorlens
```

### Demos with LLMs

Have done some demos with models and they can be run via my other script repo smolbox (or you can just use the code directly).

- [activations](https://github.com/attentionmech/smolbox/blob/main/smolbox/tools/inspect/tensorlens_activations.py)
- [weights](https://github.com/attentionmech/smolbox/blob/main/smolbox/tools/inspect/tensorlens_weights.py)
- [attention](https://github.com/attentionmech/smolbox/blob/main/smolbox/tools/inspect/tensorlens_attention.py)


## Usage

It can be used to visualise and manipulate tensors using UI. for example this code visualises GPT2 state dict tensors

```python
import torch
import numpy as np
from transformers import GPT2Model, GPT2Config
from tensorlens.tensorlens import trace, viewer

[trace(key, tensor.detach().cpu().numpy()) for key, tensor in GPT2Model.from_pretrained('gpt2-large').state_dict().items()]

viewer(height='100%')
```

you can run the above code via `uv` like this `uv run --with torch,transformers,tensorlens demo.py` where `demo.py` is file where you pasted this example.

If you are superlazy, you can run this recipe via my other project `smolbox`

`uv run --with git+https://github.com/attentionmech/smolbox smolbox inspect/tensorlens_weights`


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

IMPORTANT NOTE: in viewer there is option to execute some numpy operations too. Although I have done some basic checks, but running code received from UI in python is not considered safe, so ensure you are running this in secure trusted network. Will introduce a flag to disable that feature in next release.

### Command-Line Interface

#### Options:
- `--debug`: Run in debug server mode.
- `--notebook`: Run in notebook/Colab mode.
- `--workers`: Number of server workers.
- `--host`: Host to bind.
- `--port`: Port to bind.
- `--downsample_threshold`: Points threshold after which downsampling occurs (default 1M).

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

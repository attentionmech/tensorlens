import argparse
import os
import socket
import threading
import time
from typing import Literal, Tuple

import IPython.display
import numpy as np
from flask import Flask
from gunicorn.app.base import BaseApplication

from tensorlens.core import global_store
from tensorlens.web.server import app

NormalizationStrategy = Literal["clip", "minmax", "zscore", "none"]


def _start_dev_server(app: Flask, host: str, port: int):
    app.run(host=host, port=port, debug=True)


def _start_prod_server(app: Flask, host: str, port: int, workers: int):
    class GunicornApp(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.app = app
            super().__init__()

        def load(self):
            return self.app

        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key, value)

    options = {
        "bind": f"{host}:{port}",
        "workers": workers,
    }
    GunicornApp(app, options).run()


def _in_colab():
    try:
        import google.colab

        IN_COLAB = True
    except:
        IN_COLAB = False
    return IN_COLAB


def _is_port_open(host: str, port: int) -> bool:
    """Check if the port is already open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex((host, port))
        return result == 0  # 0 means the port is open


def _start_colab_server(
    app: Flask, host: str = "127.0.0.1", port: int = 8000, width=400, height=400
):
    if not _is_port_open(host, port):

        def run():
            app.run(host=host, port=port, debug=False, use_reloader=False)

        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()

        time.sleep(3)

    if _in_colab():
        js = f"""
        (async () => {{
            const url = new URL(await google.colab.kernel.proxyPort({port}, {{'cache': false}}));
            const iframe = document.createElement('iframe');
            iframe.src = url;
            iframe.setAttribute('width', '{width}');
            iframe.setAttribute('height', '{height}');
            iframe.setAttribute('frameborder', 0);
            document.body.appendChild(iframe);
        }})();
        """
        return IPython.display.display(IPython.display.Javascript(js))
    else:
        iframe_html = f"""
        <iframe src="http://127.0.0.1:{port}" width="{width}" height="{height}" frameborder="0"></iframe>
        """
        return IPython.display.HTML(iframe_html)


def trace(
    key: str,
    tensor: np.ndarray,
    normalize_range: Tuple[float, float] = (-1.0, 1.0),
    normalization: NormalizationStrategy = "clip",
) -> None:
    """
    Traces a tensor by normalizing it and converting it to int8.

    Parameters:
        key (str): Identifier for storing the tensor.
        tensor (np.ndarray): Input NumPy array.
        normalize_range (Tuple[float, float], optional): Used in 'clip' strategy. Default is (-1.0, 1.0).
        normalization (str): Strategy for normalization. Options: 'clip', 'minmax', 'zscore', 'none'.

    Raises:
        TypeError, ValueError
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError("Only NumPy arrays are supported.")
    if not isinstance(key, str) or not key:
        raise ValueError("Key must be a non-empty string.")

    norm_tensor = normalize_to_int8(tensor, normalize_range, normalization)
    global_store.INMEMORY_TENSORS[key] = norm_tensor


def normalize_to_int8(
    tensor: np.ndarray,
    normalize_range: Tuple[float, float],
    normalization: NormalizationStrategy,
) -> np.ndarray:
    """
    Normalizes tensor using selected strategy and converts to int8.

    Returns:
        np.ndarray: int8 tensor
    """
    if normalization == "clip":
        min_val, max_val = normalize_range
        tensor = np.clip(tensor, min_val, max_val)
        scale = 127 / max(abs(min_val), abs(max_val))
        return (tensor * scale).astype(np.int8)

    elif normalization == "minmax":
        min_val = tensor.min()
        max_val = tensor.max()
        if min_val == max_val:
            return np.zeros_like(tensor, dtype=np.int8)
        scaled = (tensor - min_val) / (max_val - min_val)  # [0,1]
        return ((scaled * 255) - 128).astype(np.int8)  # shift to [-128,127]

    elif normalization == "zscore":
        mean = tensor.mean()
        std = tensor.std()
        if std == 0:
            return np.zeros_like(tensor, dtype=np.int8)
        normed = (tensor - mean) / std
        normed = np.clip(normed, -3, 3)  # clip outliers
        return (normed / 3 * 127).astype(np.int8)

    elif normalization == "none":
        # assume already in [-1, 1]
        return (tensor * 127).astype(np.int8)

    else:
        raise ValueError(f"Unsupported normalization strategy: {normalization}")


def parse_args():
    parser = argparse.ArgumentParser(description="TensorLens server options")
    parser.add_argument("--debug", action="store_true", help="Run in debug server mode")
    parser.add_argument(
        "--notebook", action="store_true", help="Run in Notebook/Colab mode"
    )
    parser.add_argument("--workers", type=int, default=1, help="server worker count")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument(
        "--tensordata_path", type=str, default=".tensorlens", help="Path to tensor data"
    )
    parser.add_argument(
        "--downsample_threshold",
        type=int,
        default=1000000,
        help="num point after which downsampling will happen",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=4.0,
        help="point size")
    parser.add_argument(
        "--refresh_interval",
        type=int,
        default=10000,
        help="refresh interval")
    parser.add_argument(
        "--disable_transforms",
        action="store_true",
        help="disable transforms from ui")
    return parser.parse_args()


def viewer(
    host="127.0.0.1",
    tensordata_path=".tensorlens",
    port=8000,
    workers=1,
    debug=False,
    notebook=False,
    width=400,
    height=400,
    downsample_threshold=1000000,
    point_size=6.0,
    refresh_interval=500,
    disable_transforms=False,
):
    global_store.TENSORDATA_PATH = tensordata_path
    global_store.MAX_POINTS = downsample_threshold
    global_store.POINT_SIZE = point_size
    global_store.REFRESH_INTERVAL = refresh_interval
    global_store.DISABLE_TRANSFORMS = disable_transforms
    

    trace("demo_1d", np.random.randint(-100, 100, size=100))
    trace("demo_2d", np.random.randint(-100, 100, size=(100, 100)))
    trace("demo_3d", np.random.randint(-100, 100, size=(50, 50, 50)))

    if notebook:
        return _start_colab_server(
            app, host=host, port=port, width=width, height=height
        )
    elif debug:
        return _start_dev_server(app, host=host, port=port)
    else:
        return _start_prod_server(app, host=host, port=port, workers=workers)


def main():
    args = parse_args()
    viewer(
        host=args.host,
        port=args.port,
        workers=args.workers,
        debug=args.debug,
        notebook=args.notebook,
        tensordata_path=args.tensordata_path,
        width=400,
        height=400,
        downsample_threshold=args.downsample_threshold,
        point_size=args.point_size,
        refresh_interval=args.refresh_interval,
        disable_transforms=args.disable_transforms
    )


if __name__ == "__main__":
    main()

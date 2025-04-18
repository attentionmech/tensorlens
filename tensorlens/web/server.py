import json
import logging
import os
import sys

import numpy as np
from flask import (Flask, Response, abort, jsonify, request,
                   send_from_directory, stream_with_context)

from tensorlens.core import global_store

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


cli = sys.modules["flask.cli"]
cli.show_server_banner = lambda *x: None

static_dir = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=static_dir, static_url_path="")


@app.route("/")
def serve_index():
    return send_from_directory(static_dir, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(static_dir, path)


@app.route("/api/config", methods=["GET"])
def config():
    return jsonify({"status": "ok", "max_points": global_store.MAX_POINTS, "point_size": global_store.POINT_SIZE,
                    "refresh_interval": global_store.REFRESH_INTERVAL})


@app.route("/api/get_tensor", methods=["GET"])
def get_tensor():
    tensor_key = request.args.get("tensor_key")
    user_code = request.args.get("code")  # Get the Python code from query params

    if not tensor_key:
        return jsonify({"error": "Missing required parameter: tensor_key"}), 400

    tensor = global_store.INMEMORY_TENSORS.get(tensor_key)
    if tensor is None:
        return jsonify({"error": f"Tensor '{tensor_key}' not found"}), 404

    # Initialize the environment for safe code execution
    env = {
        "tensor": tensor,
        "np": np,
        "__builtins__": None,  # Disable dangerous built-ins
    }

    # Whitelist only safe tensor operations
    safe_builtins = {
        "reshape": np.reshape,
        "transpose": np.transpose,
        "swapaxes": np.swapaxes,
        "where": np.where,
        "array": np.array,
    }

    env.update(safe_builtins)

    try:
        # If there's Python code provided, evaluate it
        if user_code and global_store.DISABLE_TRANSFORMS == False:
            # Use eval to execute the code in a safe environment
            result = eval(user_code, {}, env)
        else:
            result = tensor  # No code provided, just return the tensor

        # Return the result (assumes the result has a `tolist()` method)
        return jsonify(
            {
                "tensor_key": tensor_key,
                "dims": len(result.shape) if hasattr(result, "shape") else None,
                "shape": result.shape if hasattr(result, "shape") else None,
                "data": result.tolist() if hasattr(result, "tolist") else str(result),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Error evaluating code: {str(e)}"}), 400


@app.route("/api/list_tensors", methods=["GET"])
def list_tensors():
    tensors = global_store.INMEMORY_TENSORS
    tensor_list = []

    for key, tensor in tensors.items():
        tensor_info = {
            "key": key,
            "name": getattr(tensor, "name", key),  # fallback to key if no name
            "shape": getattr(tensor, "shape", None),
        }
        tensor_list.append(tensor_info)

    return jsonify({"count": len(tensor_list), "available_tensors": tensor_list})


@app.route("/tensorlens/<path:filename>")
def serve_any_file(filename):
    base_dir = global_store.TENSORDATA_PATH

    if not base_dir or not os.path.isdir(base_dir):
        abort(404, description="Invalid base directory")

    file_path = os.path.abspath(os.path.join(base_dir, filename))

    # Security: prevent directory traversal
    if not file_path.startswith(os.path.abspath(base_dir)):
        abort(403, description="Forbidden path")

    if not os.path.isfile(file_path):
        abort(404, description="File not found")

    try:
        # Infer content type based on extension
        import mimetypes

        mime_type, _ = mimetypes.guess_type(file_path)
        with open(file_path, "rb") as f:  # TODO: this is insecure , need better way
            data = f.read()

        return Response(data, mimetype=mime_type or "application/octet-stream")
    except Exception as e:
        abort(500, description=f"Internal error reading file: {str(e)}")

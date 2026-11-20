"""Convert an mlx-lm Qwen2 checkpoint to a flat ``.npz`` for the JAX port.

Runs in an ad hoc mlx-lm environment, not inside palladium::

    uv run --python 3.12 --no-project --with mlx-lm==0.32.2 python \
        examples/lora/convert.py Qwen/Qwen2.5-0.5B-Instruct --out build/lora/qwen2.5-0.5b.npz

The weights are read through ``mlx_lm.load``, so the ``.npz`` holds exactly
the parameters mlx-lm runs with, under mlx-lm's names and in its layouts
(linear weights are (out, in)). numpy has no bfloat16, so bfloat16 arrays are
stored as their uint16 bit patterns; ``model.load_params`` views them back.
The ``__config__`` entry is the architecture as a JSON string.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load

CONFIG_KEYS = (
    "model_type",
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "rms_norm_eps",
    "rope_theta",
    "rope_traditional",
    "rope_scaling",
    "tie_word_embeddings",
)


def to_numpy(value: mx.array) -> np.ndarray:
    """float32 as is; bfloat16 as uint16 bit patterns (the top half of float32)."""
    if value.dtype == mx.float32:
        return np.asarray(value)
    if value.dtype != mx.bfloat16:
        raise ValueError(f"expected float32 or bfloat16 weights, got {value.dtype}")
    bits = np.asarray(value.astype(mx.float32)).view(np.uint32)
    if np.any(bits & 0xFFFF):
        raise ValueError("a bfloat16 value did not convert to float32 exactly")
    return (bits >> 16).astype(np.uint16)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="Hugging Face repo or local directory mlx-lm can load")
    parser.add_argument("--out", type=Path, required=True, help="output .npz path")
    args = parser.parse_args()

    model, _ = load(args.model)
    config = {k: v for k, v in dataclasses.asdict(model.args).items() if k in CONFIG_KEYS}
    if config["model_type"] != "qwen2" or config["rope_traditional"] or config["rope_scaling"]:
        raise ValueError(f"the JAX port covers qwen2 with plain RoPE only, got {config}")
    weights = dict(tree_flatten(model.parameters()))
    arrays: dict[str, Any] = {name: to_numpy(value) for name, value in weights.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, __config__=np.array(json.dumps(config)), **arrays)
    count = sum(v.size for v in arrays.values())
    dtypes = sorted({str(v.dtype) for v in weights.values()})
    print(f"config {config}")
    print(
        f"wrote {len(arrays)} arrays ({count / 1e6:.1f} M parameters, {', '.join(dtypes)}) "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()

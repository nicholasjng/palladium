"""Convert a SAM2 checkpoint to a flat ``.npz`` for the JAX port.

Runs inside the qgis-segment environment (it needs torch and qgis-segment's
own checkpoint loader), not inside palladium::

    cd /path/to/qgis-segment
    uv run --frozen --no-sync python /path/to/palladium/examples/sam2/convert.py \
        checkpoints/sam2.1_hiera_tiny.pt --out /some/dir/sam2.1_hiera_tiny.npz

The ``.npz`` keeps the checkpoint's parameter names and torch layouts as
float32 arrays, without the video-memory weights the loader drops;
``model.load_params`` re-lays out the convolution weights.
"""

import argparse
from pathlib import Path

import numpy as np
from streuobst.inference.backends.checkpoint import load_state_dict_as_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="SAM2 .pt checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="output .npz path")
    args = parser.parse_args()

    state_dict = load_state_dict_as_numpy(args.checkpoint)
    not_f32 = {k: v.dtype for k, v in state_dict.items() if v.dtype != np.float32}
    if not_f32:
        raise ValueError(f"expected float32 weights only, got {not_f32}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **state_dict)
    count = sum(v.size for v in state_dict.values())
    print(f"wrote {len(state_dict)} arrays ({count / 1e6:.1f} M parameters) to {args.out}")


if __name__ == "__main__":
    main()

"""Write MLX reference outputs for the parity check of the JAX SAM2 port.

Runs inside the qgis-segment environment with mlx available::

    cd /path/to/qgis-segment
    uv run --frozen --no-sync --with 'mlx>=0.32.1' python \
        /path/to/palladium/examples/sam2/mlx_reference.py \
        checkpoints/sam2.1_hiera_tiny.pt --image-size 256 --out /some/dir/reference_256.npz

The ``.npz`` holds the fixed random inputs, the ``Sam2Trainable`` forward
with its intermediate tensors, a forward with non-zero LoRA adapters, and the
losses and final adapters of a few AdamW training steps.

The MLX prompt encoder is built for 1024 px inputs; for another size this
script sets its ``input_image_size`` / ``image_embedding_size`` attributes on
the instance (the trunk interpolates its positional embedding by itself).
"""

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from streuobst.inference.backends.sam2_mlx.lora import inject_lora
from streuobst.inference.backends.sam2_mlx.predictor import Sam2MlxPredictor
from streuobst.inference.backends.sam2_mlx.training import Sam2Trainable, make_train_step

LORA_NAMES = ("lora_a_q", "lora_b_q", "lora_a_v", "lora_b_v")


def make_inputs(seed: int, batch: int, size: int):
    """Fixed random images, boxes and elliptical ground-truth masks."""
    rng = np.random.default_rng(seed)
    images = rng.normal(size=(batch, size, size, 3)).astype(np.float32)
    lo = rng.uniform(0.1, 0.4, size=(batch, 2))
    hi = rng.uniform(0.6, 0.9, size=(batch, 2))
    boxes = (np.concatenate([lo, hi], axis=1) * size).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    masks = np.zeros((batch, size, size), dtype=np.float32)
    for b, (x0, y0, x1, y1) in enumerate(boxes):
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        masks[b] = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
    return images, boxes, masks


def forward_with_intermediates(model: Sam2Trainable, images: mx.array, boxes: mx.array):
    """The steps of ``Sam2Trainable.__call__`` with their intermediates."""
    stages = model.trunk(images)
    feats = model.neck(stages)
    corners = boxes.reshape(-1, 2, 2)
    labels = mx.broadcast_to(mx.array([[2, 3]], dtype=mx.int32), (corners.shape[0], 2))
    sparse, _ = model.prompt_encoder(points=(corners, labels))
    out = {f"trunk_{i}": s for i, s in enumerate(stages)}
    out |= {f"neck_{i}": f for i, f in enumerate(feats)}
    out |= {
        "feat_s0": model.mask_decoder.conv_s0(feats[0]),
        "feat_s1": model.mask_decoder.conv_s1(feats[1]),
        "image_embed": feats[2] + model.no_mem_embed.reshape(1, 1, 1, -1),
        "sparse": sparse,
        "dense_pe": model.prompt_encoder.get_dense_pe(),
        "logits": model(images, boxes),  # the actual __call__
    }
    return {k: np.asarray(v) for k, v in out.items()}


def set_lora(model: Sam2Trainable, weights: dict[str, np.ndarray]) -> None:
    for i, blk in enumerate(model.trunk.blocks):
        for name in LORA_NAMES:
            getattr(blk.attn.qkv, name).weight = mx.array(weights[f"blocks.{i}.{name}"])


def get_lora(model: Sam2Trainable) -> dict[str, np.ndarray]:
    return {
        f"blocks.{i}.{name}": np.asarray(getattr(blk.attn.qkv, name).weight)
        for i, blk in enumerate(model.trunk.blocks)
        for name in LORA_NAMES
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu", help="MLX device")
    args = parser.parse_args()
    size = args.image_size
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)

    predictor = Sam2MlxPredictor.from_checkpoint(args.checkpoint)
    predictor.prompt_encoder.input_image_size = (size, size)
    predictor.prompt_encoder.image_embedding_size = (size // 16, size // 16)
    # As prepare_lora_training: train mode (decoder returns mask token 0), all frozen.
    model = Sam2Trainable(predictor)
    model.train()
    model.freeze()

    images_np, boxes_np, masks_np = make_inputs(args.seed, args.batch, size)
    images, boxes, masks = mx.array(images_np), mx.array(boxes_np), mx.array(masks_np)
    result = {"images": images_np, "boxes": boxes_np, "gt_masks": masks_np}
    result |= {f"base/{k}": v for k, v in forward_with_intermediates(model, images, boxes).items()}

    # Forward with non-zero adapters (B drawn at a scale that moves the logits).
    inject_lora(model.trunk, args.rank)
    rng = np.random.default_rng(args.seed + 1)
    lora_fwd, lora_init = {}, {}
    for key, value in get_lora(model).items():
        if "lora_a" in key:
            bound = 1.0 / np.sqrt(value.shape[1])
            lora_fwd[key] = rng.uniform(-bound, bound, value.shape).astype(np.float32)
            lora_init[key] = lora_fwd[key]
        else:
            lora_fwd[key] = rng.normal(0.0, 0.05, value.shape).astype(np.float32)
            lora_init[key] = np.zeros(value.shape, dtype=np.float32)
    set_lora(model, lora_fwd)
    result |= {f"lora_fwd/{k}": v for k, v in lora_fwd.items()}
    result |= {f"lora/{k}": v for k, v in forward_with_intermediates(model, images, boxes).items()}

    # Training from the standard init (same A, B = 0): AdamW, lr 1e-4, no weight decay.
    set_lora(model, lora_init)
    result |= {f"lora_init/{k}": v for k, v in lora_init.items()}
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.0)
    step = make_train_step(model, optimizer, label_hw=(size, size))
    losses = []
    for _ in range(args.steps):
        loss = step(images, boxes, masks)
        mx.eval(loss, model.state, optimizer.state)
        losses.append(float(loss))
    result["train/losses"] = np.asarray(losses, dtype=np.float64)
    result |= {f"lora_final/{k}": v for k, v in get_lora(model).items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **result)
    print(
        f"mlx {mx.__version__}, AdamW betas={optimizer.betas} eps={optimizer.eps} "
        f"bias_correction={optimizer.bias_correction} weight_decay={optimizer.weight_decay}"
    )
    print(f"device {mx.default_device()}, image size {size}, batch {args.batch}, rank {args.rank}")
    print("losses:", " ".join(f"{x:.8f}" for x in losses))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

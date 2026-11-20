# SAM2 in plain JAX

A port of the MLX SAM2 training forward pass in qgis-segment
(`src/streuobst/inference/backends/sam2_mlx/`) to plain JAX: Hiera trunk, FPN neck, prompt
encoder (box prompts as two corner points), mask decoder with high-res features, single-mask
output, plus LoRA on the trunk's qkv projections and an AdamW training step.

| File | Runs in | Purpose |
| --- | --- | --- |
| `model.py` | palladium | `forward(params, config, images, boxes, lora=None)`, `load_params`, `init_params`, `init_lora` |
| `train.py` | palladium | `dice_ce_loss`, `adamw_init` / `adamw_update`, `make_train_step` |
| `convert.py` | qgis-segment | checkpoint `.pt` to flat `.npz` |
| `mlx_reference.py` | qgis-segment | fixed inputs and MLX reference outputs to `.npz` |
| `parity.py` | palladium | compares the JAX port against the reference `.npz` |

`images` are `(B, S, S, 3)` preprocessed float32, `boxes` are `(B, 4)` XYXY in input pixels, and
the result is mask logits `(B, 1, S/4, S/4)`. `S` is `Sam2Config.image_size`, any multiple of 32
(1024 is what the checkpoints were trained at).

## Convert the weights

In the qgis-segment checkout (needs its torch environment; `--no-sync` leaves it untouched):

```sh
cd /path/to/qgis-segment
uv run --frozen --no-sync python /path/to/palladium/examples/sam2/convert.py \
    checkpoints/sam2.1_hiera_tiny.pt --out $OUT/sam2.1_hiera_tiny.npz
```

## Parity check against MLX

Write the reference, also in qgis-segment (`--with` adds mlx without changing the environment):

```sh
uv run --frozen --no-sync --with 'mlx[cpu]==0.32.4' python \
    /path/to/palladium/examples/sam2/mlx_reference.py checkpoints/sam2.1_hiera_tiny.pt \
    --image-size 256 --batch 2 --device cpu --out $OUT/reference_256_cpu.npz
```

`--device gpu` works for the forward passes only: with mlx 0.32.4 the gradient of
`mx.fast.scaled_dot_product_attention` raises on the GPU, so pass `--steps 0` there.

Then compare, in palladium:

```sh
JAX_PLATFORMS=cpu uv run python examples/sam2/parity.py \
    --weights $OUT/sam2.1_hiera_tiny.npz --reference $OUT/reference_256_cpu.npz
```

It prints, per tensor (trunk stages, neck levels, prompt embeddings, logits), the largest absolute
difference and `max|jax - mlx| / max|mlx|`, without and with non-zero LoRA adapters, and then the
losses of three AdamW steps on both sides.

## Run a training step

```sh
JAX_PLATFORMS=cpu uv run python examples/sam2/train.py \
    --weights $OUT/sam2.1_hiera_tiny.npz --image-size 256 --steps 2
```

or from Python (with `examples/` on the path):

```python
weights = np.load("sam2.1_hiera_tiny.npz")
config = config_from_weights(weights, image_size=1024)
params = load_params(weights, config)
lora = init_lora(config, rank=4)
opt_state = adamw_init(lora)
step = jax.jit(make_train_step(params, config, label_hw=(1024, 1024)))
lora, opt_state, loss = step(lora, opt_state, images, boxes, gt_masks)
```

Only `lora` is trained; `params` is closed over by the step and stays frozen.

## On the `metal` plugin

Every script here runs on the Apple GPU with `JAX_PLATFORMS=metal,cpu` in place of
`JAX_PLATFORMS=cpu`. The parity check then compares the plugin with the MLX reference, and
`tests/runtime/test_pjrt.py::test_sam2_lora_step_matches_cpu` compares it with XLA:CPU on random
weights. `notes/metal-compute-stack.md` records the measurements.

`Sam2Config(fused_attention=True)`, or `dataclasses.replace(config, fused_attention=True)`,
runs the global attention blocks on the flash-attention kernels of `examples/flash_attention.py`
instead of writing the score matrix out. It needs the `metal` platform and `examples/` on the
path, and is off by default so the port stays plain JAX elsewhere.

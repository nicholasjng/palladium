# LoRA fine-tuning of Qwen2.5-0.5B in plain JAX

A port of mlx-lm's Qwen2 decoder (`mlx_lm/models/qwen2.py`) and of its default LoRA fine-tuning
(`mlx_lm/lora.py`, `mlx_lm/tuner/`) to plain JAX: token embedding, RMSNorm, grouped-query
attention with RoPE and a causal mask, SwiGLU MLP, tied LM head, plus LoRA on every linear layer
of the last 16 blocks, the masked next-token cross-entropy and an Adam training step. Checked
against mlx-lm 0.32.2 with mlx 0.32.4.

| File | Runs in | Purpose |
| --- | --- | --- |
| `model.py` | palladium | `forward(params, config, tokens, lora=None)`, `load_params`, `init_params`, `init_lora` |
| `train.py` | palladium | `cross_entropy`, `loss_fn`, `adam_init` / `adam_update`, `make_train_step` |
| `convert.py` | mlx-lm | mlx-lm checkpoint to flat `.npz` |
| `mlx_reference.py` | mlx-lm | fixed dataset and mlx-lm reference outputs to `.npz` |
| `parity.py` | palladium | compares the JAX port against the reference `.npz` |

`tokens` are `(B, L)` int32 and the result is logits `(B, L, 151936)` in the dtype of the
parameters. The model is `Qwen/Qwen2.5-0.5B-Instruct` (494 M parameters, bfloat16, tied
embeddings), which mlx-lm loads from the Hugging Face repository as it is.

## What mlx-lm does by default

Read from the mlx-lm 0.32.2 and mlx 0.32.4 sources:

- LoRA (`mlx_lm/lora.py`, `CONFIG_DEFAULTS`): `num_layers` 16, rank 8, scale 20.0, dropout 0.0.
  The scale multiplies the branch directly; there is no `alpha / rank`. With no `keys` in the
  configuration, `mlx_lm/tuner/utils.py` adapts every `nn.Linear` in each of the last 16 blocks:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `down_proj`, `up_proj`. That is 224
  trainable matrices and 4.4 M parameters. `lora_a` is `(in, r)`, uniform in `±1/sqrt(in)`;
  `lora_b` is `(r, out)`, zero. Both are float32 for any model dtype.
- Loss (`mlx_lm/tuner/trainer.py`, `default_loss`): the batch is `(B, L + 1)`, zero-padded; the
  model sees `batch[:, :-1]`. Target `t` counts if `t + 1 <= length`, so each sequence shorter
  than `L` also predicts its first padding token. The loss is the mean over the counted targets.
- Optimizer (`mlx/optimizers/optimizers.py`, `Adam`): learning rate 1e-5, constant, betas
  (0.9, 0.999), eps 1e-8, `bias_correction=False`, no weight decay. Without bias correction the
  first steps are about `3.16 * lr` in every element.
- Half precision, measured on the GPU: RMSNorm, RoPE, the attention product with its softmax and
  the cross-entropy run in float32 and round their result; a matmul accumulates in float32. SiLU,
  the residual additions and the RMSNorm weight stay in bfloat16. `model.py` does the same.
  MLX's CPU backend differs: it computes RoPE, the attention scores and the cross-entropy in
  bfloat16.

## Convert the weights

In an mlx-lm environment (`--no-project` leaves palladium's environment untouched; the first run
downloads the model, about 1 GB, to the Hugging Face cache):

```sh
uv run --python 3.12 --no-project --with mlx-lm==0.32.2 python \
    examples/lora/convert.py Qwen/Qwen2.5-0.5B-Instruct --out build/lora/qwen2.5-0.5b.npz
```

The `.npz` holds mlx-lm's parameters under its names. bfloat16 arrays are stored as uint16 bit
patterns; `load_params(weights, config, dtype)` casts them to float32, bfloat16 or float16 and
transposes the linear weights to `(in, out)`.

## Parity check against mlx-lm

Write a reference per dtype (about 0.8 GB each, mostly the two logits tensors):

```sh
uv run --python 3.12 --no-project --with mlx-lm==0.32.2 python \
    examples/lora/mlx_reference.py Qwen/Qwen2.5-0.5B-Instruct \
    --dtype float32 --out build/lora/reference_float32.npz
```

and the same with `--dtype bfloat16`. The script generates 64 short texts, tokenizes them with
mlx-lm's `TextDataset`, pads them to 129 tokens, and saves the token arrays, so the JAX side
needs no tokenizer. The training runs are `mlx_lm.tuner.trainer.train` on fixed batches of 4.

Then compare, in palladium:

```sh
JAX_PLATFORMS=cpu uv run python examples/lora/parity.py \
    --weights build/lora/qwen2.5-0.5b.npz --reference build/lora/reference_float32.npz
```

It prints, per tensor (embedding, every block, final norm, logits), the largest absolute
difference and `max|jax - mlx| / max|mlx|`, without and with non-zero LoRA adapters, and then the
losses of five Adam steps on both sides, once over successive batches and once on a repeated one.

Measured with XLA:CPU against mlx on the GPU:

| Check | float32 | bfloat16 |
| --- | --- | --- |
| logits, no LoRA | 9.9e-6 | 3.5e-2 |
| logits, non-zero LoRA | 2.1e-5 | 1.9e-2 |
| loss, 5 steps, largest difference | 8.4e-5 | 4.8e-3 |

The bfloat16 difference is rounding, not a model difference: against the float32 logits the JAX
bfloat16 run is off by 9.4e-2 and the mlx bfloat16 run by 6.8e-2.

## Run a training step

```sh
JAX_PLATFORMS=cpu uv run python examples/lora/train.py \
    --weights build/lora/qwen2.5-0.5b.npz --data build/lora/reference_float32.npz --steps 2 --ops
```

`--ops` prints the StableHLO op census of the lowered step. From Python (with `examples/` on the
path):

```python
weights = np.load("build/lora/qwen2.5-0.5b.npz")
config = config_from_weights(weights)
params = load_params(weights, config, jnp.bfloat16)
lora = init_lora(config)
opt_state = adam_init(lora)
step = jax.jit(make_train_step(config))
lora, opt_state, loss = step(params, lora, opt_state, tokens, lengths)
```

Only `lora` is trained. `params` is an argument of the step, so the frozen weights are inputs of
the compiled program and not constants inside it. `tokens` is `(B, L + 1)` int32 and `lengths`
is `(B, 2)` int32 rows of `(offset, length)`, as mlx-lm's batches.

`QwenConfig.accumulate_float32` (default on) upcasts the operands of half-precision matmuls to
float32. Turn it off for a backend whose bfloat16 GEMM accumulates in float32 by itself.

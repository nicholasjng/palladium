"""Write mlx-lm reference outputs for the parity check of the JAX Qwen2.5 port.

Runs in an ad hoc mlx-lm environment, not inside palladium::

    uv run --python 3.12 --no-project --with mlx-lm==0.32.2 python \
        examples/lora/mlx_reference.py Qwen/Qwen2.5-0.5B-Instruct \
        --dtype float32 --out build/lora/reference_float32.npz

The ``.npz`` holds a fixed tokenized dataset (64 generated text samples,
zero-padded to 129 tokens), the forward pass on its first batch with every
block's hidden state, the same forward with non-zero LoRA adapters, and the
losses and final adapters of two short mlx-lm LoRA runs from the standard
init: one over successive batches and one that repeats the first batch.

LoRA and the optimizer are mlx-lm's defaults (``mlx_lm.lora.CONFIG_DEFAULTS``)
and the training runs are ``mlx_lm.tuner.trainer.train`` itself, fed the
fixed batches instead of its length-sorted, shuffled ones.
"""

import argparse
import random
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import mlx_lm
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import TextDataset
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import TrainingArgs, default_loss, train
from mlx_lm.tuner.utils import linear_to_lora_layers

ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "down_proj", "up_proj")

SUBJECTS = ("The old gardener", "A quiet engineer", "My neighbour", "The night train", "Our cat")
VERBS = ("repairs", "paints", "counts", "watches", "carries", "forgets", "measures")
OBJECTS = ("the wooden bridge", "seven copper kettles", "a map of the harbour", "the last apples")
PLACES = ("behind the mill", "near the station", "under a grey sky", "in the long corridor")
TIMES = ("before sunrise", "every second Tuesday", "after the rain", "while the kettle boils")


def make_texts(count: int, seed: int) -> list[str]:
    """``count`` short deterministic texts of 2 to 9 generated sentences."""
    rng = random.Random(seed)
    texts = []
    for index in range(count):
        sentences = [
            f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} "
            f"{rng.choice(PLACES)} {rng.choice(TIMES)}."
            for _ in range(rng.randint(2, 9))
        ]
        texts.append(f"Note {index + 1}: " + " ".join(sentences))
    return texts


def tokenize(texts: list[str], tokenizer, max_length: int):
    """Tokens (N, max_length + 1) zero-padded and (offset, length) rows (N, 2).

    Each text goes through mlx-lm's ``TextDataset.process`` (encode, append
    EOS) and is cut to ``max_length`` tokens so that one width holds them all.
    For ``max_length`` 128 the width is what ``iterate_batches`` pads a batch
    with a longest sequence of 97 to 128 tokens to, here used for all batches.
    """
    dataset = TextDataset([{"text": t} for t in texts], tokenizer)
    tokens = np.zeros((len(texts), max_length + 1), np.int32)
    lengths = np.zeros((len(texts), 2), np.int32)
    for row, item in enumerate(dataset):
        ids, offset = dataset.process(item)
        ids = ids[:max_length]
        tokens[row, : len(ids)] = ids
        lengths[row] = (offset, len(ids))
    return tokens, lengths


def lora_modules(model) -> dict[str, LoRALinear]:
    """``"{block}.{projection}" -> LoRALinear`` for every adapted layer."""
    modules = {}
    for index, layer in enumerate(model.model.layers):
        for names, parent in ((ATTENTION, layer.self_attn), (MLP, layer.mlp)):
            for name in names:
                if isinstance(parent[name], LoRALinear):
                    modules[f"{index}.{name}"] = parent[name]
    return modules


def set_lora(model, weights: dict[str, np.ndarray]) -> None:
    for key, module in lora_modules(model).items():
        module.lora_a = mx.array(weights[f"{key}.lora_a"])
        module.lora_b = mx.array(weights[f"{key}.lora_b"])


def get_lora(model) -> dict[str, np.ndarray]:
    out = {}
    for key, module in lora_modules(model).items():
        out[f"{key}.lora_a"] = np.asarray(module.lora_a)
        out[f"{key}.lora_b"] = np.asarray(module.lora_b)
    return out


def forward_with_intermediates(model, inputs: mx.array) -> dict[str, np.ndarray]:
    """The steps of ``Qwen2Model.__call__`` with their intermediates, as float32."""
    h = model.model.embed_tokens(inputs)
    out = {"embed": h}
    mask = create_attention_mask(h, None)
    for index, layer in enumerate(model.model.layers):
        h = layer(h, mask, None)
        out[f"layer_{index}"] = h
    out["norm"] = model.model.norm(h)
    out["logits"] = model(inputs)  # the actual __call__
    mx.eval(out)
    return {k: np.asarray(v.astype(mx.float32)) for k, v in out.items()}


class Losses(TrainingCallback):
    def __init__(self):
        self.values = []

    def on_train_loss_report(self, train_info: dict):
        self.values.append(train_info["train_loss"])


def run_training(model, batches, learning_rate: float, adapter_file: Path) -> list[float]:
    """``mlx_lm.tuner.trainer.train`` over the given batches, one report per step."""

    def fixed_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
        while True:
            for tokens, lengths in batches:
                yield mx.array(tokens), mx.array(lengths)

    optimizer_config = CONFIG_DEFAULTS["optimizer_config"][CONFIG_DEFAULTS["optimizer"]]
    optimizer = optim.Adam(learning_rate=learning_rate, **optimizer_config)
    losses = Losses()
    args = TrainingArgs(
        batch_size=len(batches[0][0]),
        iters=len(batches),
        steps_per_report=1,
        steps_per_save=len(batches),
        adapter_file=str(adapter_file),
    )
    train(
        model,
        optimizer,
        train_dataset=None,
        args=args,
        iterate_batches=fixed_batches,
        training_callback=losses,
    )
    print(
        f"Adam lr={learning_rate} betas={optimizer.betas} eps={optimizer.eps} "
        f"bias_correction={optimizer.bias_correction}"
    )
    return losses.values


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="Hugging Face repo or local directory mlx-lm can load")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu", help="MLX device")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load(args.model)
    native = sorted({str(v.dtype) for _, v in tree_flatten(model.parameters())})
    model.set_dtype(getattr(mx, args.dtype))
    model.freeze()

    tokens, lengths = tokenize(make_texts(args.samples, args.seed), tokenizer, args.max_length)
    inputs = tokens[: args.batch, :-1]
    result: dict[str, Any] = {"data/tokens": tokens, "data/lengths": lengths, "tokens": inputs}
    result["dtype"] = np.array(args.dtype)
    base = forward_with_intermediates(model, mx.array(inputs))
    result |= {f"base/{k}": v for k, v in base.items()}

    # LoRA as mlx-lm's train_model sets it up, with its default configuration.
    lora_config = CONFIG_DEFAULTS["lora_parameters"]
    linear_to_lora_layers(model, CONFIG_DEFAULTS["num_layers"], lora_config)
    trainable = dict(tree_flatten(model.trainable_parameters()))
    initial = get_lora(model)
    if len(initial) != len(trainable):
        raise ValueError("trainable parameters other than the LoRA matrices of the blocks")

    # Our own init, saved so the JAX side starts from the same numbers: A as
    # mlx-lm draws it, B zero; for the forward check B is non-zero instead.
    rng = np.random.default_rng(args.seed + 1)
    lora_init, lora_fwd = {}, {}
    for key, value in initial.items():
        if key.endswith("lora_a"):
            bound = 1.0 / np.sqrt(value.shape[0])
            lora_init[key] = rng.uniform(-bound, bound, value.shape).astype(np.float32)
            lora_fwd[key] = lora_init[key]
        else:
            lora_init[key] = np.zeros(value.shape, np.float32)
            lora_fwd[key] = rng.normal(0.0, 2e-3, value.shape).astype(np.float32)
    set_lora(model, lora_fwd)
    result |= {f"lora_fwd/{k}": v for k, v in lora_fwd.items()}
    with_lora = forward_with_intermediates(model, mx.array(inputs))
    result |= {f"lora/{k}": v for k, v in with_lora.items()}

    def batch(index: int):
        rows = slice(index * args.batch, (index + 1) * args.batch)
        return tokens[rows], lengths[rows]

    # mlx-lm pads each batch to 1 + 32 * ceil(longest / 32) tokens; the fixed
    # width changes nothing but rounding. Shown on the four shortest samples.
    set_lora(model, lora_init)
    short = np.argsort(lengths[:, 1], kind="stable")[: args.batch]
    own_width = 1 + 32 * ((int(lengths[short, 1].max()) + 31) // 32)
    loss_fixed, ntoks = default_loss(model, mx.array(tokens[short]), mx.array(lengths[short]))
    loss_own, _ = default_loss(model, mx.array(tokens[short, :own_width]), mx.array(lengths[short]))

    learning_rate = CONFIG_DEFAULTS["learning_rate"]
    adapter_file = args.out.parent / "mlx_adapters.safetensors"
    runs = {
        "train": [batch(i) for i in range(args.steps)],
        "repeat": [batch(0)] * args.steps,
    }
    for name, batches in runs.items():
        if not args.steps:
            result[f"{name}/losses"] = np.zeros(0)
            continue
        set_lora(model, lora_init)
        result[f"{name}/losses"] = np.asarray(
            run_training(model, batches, learning_rate, adapter_file), np.float64
        )
        result |= {f"{name}_final/{k}": v for k, v in get_lora(model).items()}
    result |= {f"lora_init/{k}": v for k, v in lora_init.items()}

    np.savez(args.out, **result)
    counts = {}
    for name in trainable:
        counts[name.split(".")[-2]] = counts.get(name.split(".")[-2], 0) + 1
    blocks = sorted({int(k.split(".")[0]) for k in initial})
    print(f"mlx-lm {mlx_lm.__version__}, mlx {mx.__version__}, device {mx.default_device()}")
    print(f"checkpoint dtype {native}, run as {args.dtype}; logits {model(mx.array(inputs)).dtype}")
    print(f"defaults: num_layers {CONFIG_DEFAULTS['num_layers']}, lora_parameters {lora_config}")
    print(f"  optimizer {CONFIG_DEFAULTS['optimizer']}, learning_rate {learning_rate}")
    print(f"adapted blocks {blocks[0]}..{blocks[-1]}, trainable arrays per projection {counts}")
    print(f"adapter dtypes {sorted({str(v.dtype) for v in trainable.values()})}")
    print(f"sequence lengths {lengths[:, 1].min()}..{lengths[:, 1].max()}, tokens {tokens.shape}")
    print(
        f"four shortest samples: loss {loss_fixed.item():.8f} over {ntoks.item()} tokens at "
        f"width {tokens.shape[1]}, {loss_own.item():.8f} at mlx-lm's width {own_width}"
    )
    moved = np.max(np.abs(with_lora["logits"] - base["logits"]))
    print(f"the forward-check adapters move the logits by up to {moved:.3e}")
    for name in runs:
        print(f"{name} losses:", " ".join(f"{x:.8f}" for x in result[f"{name}/losses"]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

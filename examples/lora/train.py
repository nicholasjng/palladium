"""LoRA fine-tuning step for the JAX Qwen2.5 port.

Mirrors mlx-lm's defaults (mlx-lm 0.32.2): the next-token cross-entropy of
``mlx_lm.tuner.trainer.default_loss`` and ``mlx.optimizers.Adam`` with
learning rate 1e-5, applied to the LoRA matrices only.

Run as a script for a smoke-test on the fixed dataset of ``mlx_reference.py``::

    JAX_PLATFORMS=cpu uv run python examples/lora/train.py \
        --weights $OUT/qwen2.5-0.5b.npz --data $OUT/reference_float32.npz --steps 2
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

try:
    from lora.model import config_from_weights, forward, init_lora, load_params
except ImportError:  # run as a script from this directory
    from model import config_from_weights, forward, init_lora, load_params

__all__ = ["adam_init", "adam_update", "cross_entropy", "loss_fn", "make_train_step"]


def cross_entropy(logits, targets):
    """Per-token ``logsumexp(logits) - logits[target]``, in the logits' dtype.

    Like ``mx.fast.cross_entropy`` (what ``nn.losses.cross_entropy`` calls on
    the GPU), the reduction over the vocabulary runs in float32 and the
    per-token result is rounded back to the dtype of the logits.

    Args:
        logits: (..., V) float.
        targets: (...) int32 class indices.
    """
    wide = logits.astype(jnp.float32)
    top = jax.lax.stop_gradient(jnp.max(wide, axis=-1, keepdims=True))
    log_norm = jnp.log(jnp.sum(jnp.exp(wide - top), axis=-1)) + top[..., 0]
    picked = jnp.take_along_axis(wide, targets[..., None], axis=-1)[..., 0]
    return (log_norm - picked).astype(logits.dtype)


def loss_fn(params, config, lora, tokens, lengths):
    """mlx-lm's ``default_loss``: mean next-token cross-entropy and token count.

    Args:
        tokens: (B, L + 1) int32, zero-padded on the right; the model sees
            ``tokens[:, :-1]`` and predicts ``tokens[:, 1:]``.
        lengths: (B, 2) int32 ``(offset, length)`` per sequence. Target ``t``
            (0-based, the token at ``t + 1``) counts if ``offset <= t + 1 <=
            length``. For ``length`` below ``L`` that includes one target
            past the sequence, the first padding token, exactly as in mlx-lm.

    Returns:
        ``(loss, ntoks)``: float32 mean over the counted targets and their
        int32 count.
    """
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    logits = forward(params, config, inputs, lora)
    steps = jnp.arange(1, targets.shape[1] + 1, dtype=jnp.int32)
    mask = (steps >= lengths[:, 0:1]) & (steps <= lengths[:, 1:])
    ce = cross_entropy(logits, targets)
    ce = ce * mask.astype(ce.dtype)
    ntoks = jnp.sum(mask.astype(jnp.int32))
    return jnp.sum(ce.astype(jnp.float32)) / ntoks.astype(jnp.float32), ntoks


def adam_init(tree):
    """Zero first and second moments, like ``mlx.optimizers.Adam.init_single``."""
    return {"m": jax.tree.map(jnp.zeros_like, tree), "v": jax.tree.map(jnp.zeros_like, tree)}


def adam_update(tree, grads, state, learning_rate=1e-5, betas=(0.9, 0.999), eps=1e-8):
    """One Adam step with the update rule of ``mlx.optimizers.Adam``.

    MLX's defaults are betas (0.9, 0.999), eps 1e-8 and ``bias_correction=
    False``: the moments are used uncorrected, ``p - lr * m / (sqrt(v) + eps)``,
    so the first steps are about ``lr * 0.1 / sqrt(0.001) = 3.16 lr`` large.
    There is no weight decay. mlx-lm's default learning rate is 1e-5, constant.
    """
    b1, b2 = betas
    m = jax.tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1.0 - b2) * g * g, state["v"], grads)
    tree = jax.tree.map(lambda p, m, v: p - learning_rate * m / (jnp.sqrt(v) + eps), tree, m, v)
    return tree, {"m": m, "v": v}


def make_train_step(config, learning_rate=1e-5):
    """Build the training step for frozen ``params`` and trainable LoRA.

    Returns ``step(params, lora, opt_state, tokens, lengths) -> (lora,
    opt_state, loss)``, a pure function to wrap in ``jax.jit``. ``params`` is
    an argument, not a closure, so the frozen weights are inputs of the
    compiled program and not constants inside it. ``tokens`` and ``lengths``
    are as in :func:`loss_fn`; ``opt_state`` comes from :func:`adam_init`
    applied to the LoRA tree.
    """

    def step(params, lora, opt_state, tokens, lengths):
        def lora_loss(lora):
            return loss_fn(params, config, lora, tokens, lengths)[0]

        loss, grads = jax.value_and_grad(lora_loss)(lora)
        lora, opt_state = adam_update(lora, grads, opt_state, learning_rate=learning_rate)
        return lora, opt_state, loss

    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True, help=".npz written by convert.py")
    parser.add_argument("--data", required=True, help=".npz written by mlx_reference.py")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--ops", action="store_true", help="print the StableHLO op census")
    args = parser.parse_args()

    weights, data = np.load(args.weights), np.load(args.data)
    config = config_from_weights(weights)
    params = load_params(weights, config, jnp.dtype(args.dtype))
    tokens, lengths = data["data/tokens"], data["data/lengths"]
    lora = init_lora(config)
    opt_state = adam_init(lora)
    step = jax.jit(make_train_step(config))

    def batch(i):
        rows = slice(i * args.batch, (i + 1) * args.batch)
        return jnp.asarray(tokens[rows]), jnp.asarray(lengths[rows])

    if args.ops:
        print_op_census(step.lower(params, lora, opt_state, *batch(0)).as_text())
    for i in range(args.steps):
        lora, opt_state, loss = step(params, lora, opt_state, *batch(i))
        print(f"step {i}: loss {float(loss):.6f}")


def print_op_census(text: str) -> None:
    """Op names and element types of a lowered step's StableHLO text.

    Attribute names such as ``#stablehlo.gather<...>`` are not counted as ops.
    """
    import collections
    import re

    ops = collections.Counter(re.findall(r"(?<!#)\b((?:stablehlo|chlo)\.[a-z_]+)\b", text))
    print(f"{len(ops)} distinct ops, {sum(ops.values())} in all")
    for op, n in sorted(ops.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:5d}  {op}")
    types = collections.Counter(re.findall(r"tensor<(?:[0-9?]+x)*([a-z]+[0-9]+)>", text))
    print("element types:", ", ".join(f"{t} ({n})" for t, n in sorted(types.items())))


if __name__ == "__main__":
    main()

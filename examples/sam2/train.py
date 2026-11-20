"""LoRA fine-tuning step for the JAX SAM2 port.

Mirrors the MLX ``training.py`` in qgis-segment: mask logits upscaled to the
label resolution (bilinear, torch ``align_corners=False`` semantics), DiceCE
loss as ``monai.losses.DiceCELoss(sigmoid=True, squared_pred=True)``, and an
AdamW update of the LoRA matrices only.

Run as a script for a smoke-test training step on random data::

    JAX_PLATFORMS=cpu uv run python examples/sam2/train.py --weights tiny.npz --image-size 256
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

try:
    from sam2.model import config_from_weights, forward, init_lora, load_params, resize_2d
except ImportError:  # run as a script from this directory
    from model import config_from_weights, forward, init_lora, load_params, resize_2d

__all__ = ["adamw_init", "adamw_update", "dice_ce_loss", "make_train_step"]


def dice_ce_loss(logits, targets):
    """Dice (squared-prediction denominator, smoothing 1e-5, mean over batch
    and channel) plus mean binary cross-entropy with logits.

    Args:
        logits: (B, C, H, W) unnormalized mask logits.
        targets: (B, C, H, W) binary ground truth in {0, 1}, float32.
    """
    probs = 0.5 * (jnp.tanh(0.5 * logits) + 1.0)  # sigmoid; finite gradient for any logit
    axes = tuple(range(2, logits.ndim))
    intersection = jnp.sum(probs * targets, axis=axes)
    denominator = jnp.sum(probs * probs, axis=axes) + jnp.sum(targets * targets, axis=axes)
    dice = 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    # log(1 + exp(x)) - x * t, in the overflow-safe form of logaddexp(0, x)
    softplus = jnp.maximum(logits, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    bce = jnp.mean(softplus - logits * targets)
    return jnp.mean(dice) + bce


def adamw_init(tree):
    """Zero first and second moments, like ``mlx.optimizers.Adam.init_single``."""
    zeros = jax.tree.map(jnp.zeros_like, tree)
    return {"m": zeros, "v": jax.tree.map(jnp.zeros_like, tree)}


def adamw_update(
    tree,
    grads,
    state,
    learning_rate=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
):
    """One AdamW step with the update rule of ``mlx.optimizers.AdamW``.

    MLX's defaults are betas (0.9, 0.999), eps 1e-8 and ``bias_correction=
    False``: the moments are used uncorrected, ``p - lr * m / (sqrt(v) + eps)``.
    Weight decay is decoupled, applied as ``p * (1 - lr * weight_decay)``
    before the Adam step (MLX defaults to 0.01; the reference training runs
    with 0, which is the default here).
    """
    b1, b2 = betas

    def moments(g, m, v):
        return b1 * m + (1.0 - b1) * g, b2 * v + (1.0 - b2) * g * g

    def apply(p, m, v):
        return p * (1.0 - learning_rate * weight_decay) - learning_rate * m / (jnp.sqrt(v) + eps)

    m = jax.tree.map(lambda g, m, v: moments(g, m, v)[0], grads, state["m"], state["v"])
    v = jax.tree.map(lambda g, m, v: moments(g, m, v)[1], grads, state["m"], state["v"])
    return jax.tree.map(apply, tree, m, v), {"m": m, "v": v}


def make_train_step(params, config, label_hw, learning_rate=1e-4, weight_decay=0.0):
    """Build the training step for frozen ``params`` and trainable LoRA.

    Returns ``step(lora, opt_state, images, boxes, gt_masks) -> (lora,
    opt_state, loss)``, a pure function to wrap in ``jax.jit``: images
    (B, S, S, 3) preprocessed, boxes (B, 4) XYXY in input pixels, gt_masks
    (B, H, W) binary with (H, W) = ``label_hw``. ``opt_state`` comes from
    :func:`adamw_init` applied to the LoRA tree.
    """

    def loss_fn(lora, images, boxes, gt_masks):
        logits = resize_2d(forward(params, config, images, boxes, lora), label_hw, "linear")
        return dice_ce_loss(logits, gt_masks[:, None].astype(logits.dtype))

    def step(lora, opt_state, images, boxes, gt_masks):
        loss, grads = jax.value_and_grad(loss_fn)(lora, images, boxes, gt_masks)
        lora, opt_state = adamw_update(
            lora, grads, opt_state, learning_rate=learning_rate, weight_decay=weight_decay
        )
        return lora, opt_state, loss

    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True, help=".npz written by convert.py")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    weights = np.load(args.weights)
    config = config_from_weights(weights, args.image_size)
    params = load_params(weights, config)
    lora = init_lora(config, args.rank)
    opt_state = adamw_init(lora)

    s = args.image_size
    rng = np.random.default_rng(0)
    images = jnp.asarray(rng.normal(size=(args.batch, s, s, 3)).astype(np.float32))
    boxes = jnp.asarray(
        np.tile(np.array([0.25, 0.25, 0.75, 0.75], np.float32) * s, (args.batch, 1))
    )
    gt_masks = np.zeros((args.batch, s, s), np.float32)
    gt_masks[:, s // 4 : 3 * s // 4, s // 4 : 3 * s // 4] = 1.0

    step = jax.jit(make_train_step(params, config, (s, s)))
    for i in range(args.steps):
        start = time.perf_counter()
        lora, opt_state, loss = step(lora, opt_state, images, boxes, jnp.asarray(gt_masks))
        print(f"step {i}: loss {float(loss):.6f} ({time.perf_counter() - start:.1f} s)")


if __name__ == "__main__":
    main()

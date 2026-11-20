"""Compare the JAX SAM2 port against the MLX reference outputs.

    JAX_PLATFORMS=cpu uv run python examples/sam2/parity.py \
        --weights /some/dir/sam2.1_hiera_tiny.npz --reference /some/dir/reference_256.npz

``--weights`` comes from ``convert.py`` and ``--reference`` from
``mlx_reference.py``. For every tensor the report gives the largest absolute
difference and the normwise relative error ``max|jax - mlx| / max|mlx|``.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

try:
    from sam2.model import config_from_weights, forward, load_params
    from sam2.train import adamw_init, make_train_step
except ImportError:  # run as a script from this directory
    from model import config_from_weights, forward, load_params
    from train import adamw_init, make_train_step


def lora_tree(reference, prefix: str):
    """``{prefix}/blocks.{i}.{name}`` entries -> the model's LoRA pytree."""
    tree = {}
    for key in reference.files:
        if key.startswith(prefix + "/"):
            _, index, name = key[len(prefix) + 1 :].split(".")
            tree.setdefault(index, {})[name] = jnp.asarray(reference[key])
    return tree


def report(name: str, ours, ref) -> float:
    ours, ref = np.asarray(ours), np.asarray(ref)
    if ours.shape != ref.shape:
        raise ValueError(f"{name}: shape {ours.shape} != reference {ref.shape}")
    diff = float(np.max(np.abs(ours - ref)))
    scale = float(np.max(np.abs(ref)))
    print(
        f"  {name:<12} {ours.shape!s:<20} max|diff| {diff:.3e}  max|ref| {scale:.3e}"
        f"  rel {diff / scale:.3e}"
    )
    return diff / scale


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    weights, ref = np.load(args.weights), np.load(args.reference)
    images, boxes = jnp.asarray(ref["images"]), jnp.asarray(ref["boxes"])
    size = images.shape[1]
    config = config_from_weights(weights, size)
    params = load_params(weights, config)
    print(f"image size {size}, batch {images.shape[0]}, backend {jax.default_backend()}")

    run = jax.jit(lambda lora: forward(params, config, images, boxes, lora, True)[1])

    print("forward, no LoRA:")
    # lora=None is not a pytree of arrays to trace over; call the function directly.
    inter = jax.jit(lambda: forward(params, config, images, boxes, None, True)[1])()
    for key, value in inter.items():
        report(key, value, ref[f"base/{key}"])

    print("forward, non-zero LoRA:")
    inter = run(lora_tree(ref, "lora_fwd"))
    for key, value in inter.items():
        report(key, value, ref[f"lora/{key}"])
    moved = np.max(np.abs(ref["lora/logits"] - ref["base/logits"]))
    print(f"  (the adapters move the reference logits by up to {moved:.3e})")

    ref_losses = ref["train/losses"]
    if args.skip_training or len(ref_losses) == 0:
        return
    print(f"training, {len(ref_losses)} AdamW steps (lr 1e-4, weight decay 0):")
    lora = lora_tree(ref, "lora_init")
    opt_state = adamw_init(lora)
    step = jax.jit(make_train_step(params, config, ref["gt_masks"].shape[1:]))
    gt_masks = jnp.asarray(ref["gt_masks"])
    for i, ref_loss in enumerate(ref_losses):
        lora, opt_state, loss = step(lora, opt_state, images, boxes, gt_masks)
        loss = float(loss)
        print(f"  step {i}: jax {loss:.8f}  mlx {ref_loss:.8f}  diff {loss - ref_loss:+.3e}")
    final, init = lora_tree(ref, "lora_final"), lora_tree(ref, "lora_init")
    keys = [(i, n) for i in final for n in final[i]]
    ours = np.concatenate([np.ravel(lora[i][n] - init[i][n]) for i, n in keys])
    theirs = np.concatenate([np.ravel(final[i][n] - init[i][n]) for i, n in keys])
    print(
        f"  adapter update after training: max|jax - mlx| {np.max(np.abs(ours - theirs)):.3e}"
        f"  max|mlx update| {np.max(np.abs(theirs)):.3e}"
        f"  l2 rel {np.linalg.norm(ours - theirs) / np.linalg.norm(theirs):.3e}"
    )


if __name__ == "__main__":
    main()

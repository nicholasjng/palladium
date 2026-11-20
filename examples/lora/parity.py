"""Compare the JAX Qwen2.5 LoRA port against the mlx-lm reference outputs.

    JAX_PLATFORMS=cpu uv run python examples/lora/parity.py \
        --weights build/lora/qwen2.5-0.5b.npz --reference build/lora/reference_float32.npz

``--weights`` comes from ``convert.py`` and ``--reference`` from
``mlx_reference.py``; the parameter dtype is the one the reference ran with.
For every tensor the report gives the largest absolute difference and the
normwise relative error ``max|jax - mlx| / max|mlx|``.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

try:
    from lora.model import config_from_weights, forward, load_params
    from lora.train import adam_init, make_train_step
except ImportError:  # run as a script from this directory
    from model import config_from_weights, forward, load_params
    from train import adam_init, make_train_step


def lora_tree(reference, prefix: str):
    """``{prefix}/{block}.{projection}.{name}`` entries -> the model's LoRA pytree."""
    tree = {}
    for key in reference.files:
        if key.startswith(prefix + "/"):
            index, projection, name = key[len(prefix) + 1 :].split(".")
            tree.setdefault(index, {}).setdefault(projection, {})[name] = jnp.asarray(
                reference[key]
            )
    return tree


def report(name: str, ours, ref) -> float:
    ours, ref = np.asarray(ours.astype(jnp.float32)), np.asarray(ref)
    if ours.shape != ref.shape:
        raise ValueError(f"{name}: shape {ours.shape} != reference {ref.shape}")
    diff = float(np.max(np.abs(ours - ref)))
    scale = float(np.max(np.abs(ref)))
    print(
        f"  {name:<9} {ours.shape!s:<18} max|diff| {diff:.3e}  max|ref| {scale:.3e}"
        f"  rel {diff / scale:.3e}"
    )
    return diff / scale


def flat(tree, base):
    """The update ``tree - base`` of every adapter matrix as one vector."""
    leaves = jax.tree.leaves(jax.tree.map(lambda a, b: np.ravel(a - b), tree, base))
    return np.concatenate(leaves)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--native-matmul",
        action="store_true",
        help="half-precision matmuls in the parameter dtype (accumulate_float32=False)",
    )
    args = parser.parse_args()

    weights, ref = np.load(args.weights), np.load(args.reference)
    dtype = jnp.dtype(str(ref["dtype"]))
    config = config_from_weights(weights, accumulate_float32=not args.native_matmul)
    params = load_params(weights, config, dtype)
    tokens = jnp.asarray(ref["tokens"])
    print(
        f"parameters {dtype}, tokens {tokens.shape}, backend {jax.default_backend()}, "
        f"accumulate_float32={config.accumulate_float32}"
    )

    run = jax.jit(lambda params, lora, tokens: forward(params, config, tokens, lora, True)[1])
    order = ["embed", *(f"layer_{i}" for i in range(config.num_hidden_layers)), "norm", "logits"]

    print("forward, no LoRA:")
    inter = run(params, None, tokens)
    for key in order:
        report(key, inter[key], ref[f"base/{key}"])

    print("forward, non-zero LoRA:")
    inter = run(params, lora_tree(ref, "lora_fwd"), tokens)
    for key in order:
        report(key, inter[key], ref[f"lora/{key}"])
    moved = np.max(np.abs(ref["lora/logits"] - ref["base/logits"]))
    print(f"  (the adapters move the reference logits by up to {moved:.3e})")

    if args.skip_training:
        return
    data_tokens, data_lengths = ref["data/tokens"], ref["data/lengths"]
    batch = tokens.shape[0]
    step = jax.jit(make_train_step(config))
    init = lora_tree(ref, "lora_init")
    for name, title in (("train", "successive batches"), ("repeat", "batch 0 repeated")):
        ref_losses = ref[f"{name}/losses"]
        if len(ref_losses) == 0:
            continue
        print(f"training, {len(ref_losses)} Adam steps (lr 1e-5, no bias correction), {title}:")
        lora, opt_state = init, adam_init(init)
        for i, ref_loss in enumerate(ref_losses):
            rows = slice(i * batch, (i + 1) * batch) if name == "train" else slice(0, batch)
            lora, opt_state, loss = step(
                params, lora, opt_state, jnp.asarray(data_tokens[rows]), data_lengths[rows]
            )
            loss = float(loss)
            print(f"  step {i}: jax {loss:.8f}  mlx {ref_loss:.8f}  diff {loss - ref_loss:+.3e}")
        ours, theirs = flat(lora, init), flat(lora_tree(ref, f"{name}_final"), init)
        print(
            f"  adapter update after training: max|jax - mlx| {np.max(np.abs(ours - theirs)):.3e}"
            f"  max|mlx update| {np.max(np.abs(theirs)):.3e}"
            f"  l2 rel {np.linalg.norm(ours - theirs) / np.linalg.norm(theirs):.3e}"
        )


if __name__ == "__main__":
    main()

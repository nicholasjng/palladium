"""Example 7: differentiating through a Metal kernel with jax.custom_vjp.

Docs: companion to docs/getting-started.md (Composing with jax.jit).

`metal_call_jit` kernels are jit-composable but not differentiable:
jax.ffi calls carry no JVP/transpose rule, and palladium cannot derive
a backward kernel from forward MSL. The recipe is the standard JAX one:
author the backward pass as a second Pallas kernel and pair the two
with `jax.custom_vjp`. Everything around the pair (optimizers, losses,
other jnp code) then differentiates normally, with both passes running
on the Apple GPU inside one jitted computation.

The layer here is a tanh dense layer, y = tanh(x @ W):

    forward:  one fused kernel, saves (x, W, y) as residuals
              (saving y makes the backward's tanh' a multiply:
              tanh'(z) = 1 - y^2, no recomputation of the matmul)
    backward: one fused kernel computing both gradients,
              t  = g * (1 - y^2)      (chain rule through tanh)
              dx = t @ W^T             (W^T fuses lazily, no copy)
              dW = x^T @ t             (x^T materializes, a small copy)

Gradients are checked against jax.grad of the plain jnp expression.
"""

import jax
import jax.numpy as jnp
import numpy as np

import palladium

M, K, N = 8, 16, 8
F32 = jnp.float32


def _shaped(*shape):
    return jax.ShapeDtypeStruct(shape, F32)


def dense_fwd_kernel(x_ref, w_ref, y_ref):
    y_ref[...] = jnp.tanh(jnp.dot(x_ref[...], w_ref[...]))


def dense_bwd_kernel(x_ref, w_ref, y_ref, g_ref, dx_ref, dw_ref):
    t = g_ref[...] * (1.0 - y_ref[...] * y_ref[...])
    dx_ref[...] = jnp.dot(t, w_ref[...].T)
    dw_ref[...] = jnp.dot(x_ref[...].T, t)


_fwd = palladium.metal_call_jit(dense_fwd_kernel, out_shape=_shaped(M, N))
_bwd = palladium.metal_call_jit(
    dense_bwd_kernel, out_shape=(_shaped(M, K), _shaped(K, N))
)


@jax.custom_vjp
def dense(x, w):
    return _fwd(x, w)


def _dense_fwd(x, w):
    y = _fwd(x, w)
    return y, (x, w, y)


def _dense_bwd(residuals, g):
    x, w, y = residuals
    return _bwd(x, w, y, g)


dense.defvjp(_dense_fwd, _dense_bwd)


def main():
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.standard_normal((M, K)), dtype=F32)
    w = jnp.asarray(rng.standard_normal((K, N)) / np.sqrt(K), dtype=F32)

    def loss_metal(x, w):
        return jnp.sum(dense(x, w) ** 2)

    def loss_ref(x, w):
        return jnp.sum(jnp.tanh(x @ w) ** 2)

    # Both passes run on the GPU inside one jitted computation.
    val, (dx, dw) = jax.jit(jax.value_and_grad(loss_metal, argnums=(0, 1)))(x, w)
    val_ref, (dx_ref, dw_ref) = jax.jit(jax.value_and_grad(loss_ref, argnums=(0, 1)))(
        x, w
    )

    print(f"loss     metal {float(val):.6f}  reference {float(val_ref):.6f}")
    print(f"max |dx - dx_ref|: {float(jnp.max(jnp.abs(dx - dx_ref))):.2e}")
    print(f"max |dW - dW_ref|: {float(jnp.max(jnp.abs(dw - dw_ref))):.2e}")

    np.testing.assert_allclose(dx, dx_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dw, dw_ref, rtol=1e-4, atol=1e-4)
    print("gradients match jax.grad of the jnp reference")


if __name__ == "__main__":
    main()

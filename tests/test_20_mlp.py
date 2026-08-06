"""The MLP training recipe (examples/08_mlp_training.py): correctness
gates at real shapes. Performance is measured in the example, not here
(it currently loses to the CPU; see the example's docstring).
"""

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "mlp_example",
    pathlib.Path(__file__).parent.parent / "examples" / "08_mlp_training.py",
)
mlp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mlp)


def test_mmt_kernels_take_the_cooperative_path():
    # The hidden-width stripe (n=256) and the MMA-eligible output layer
    # (n=64) must both lower cooperatively; a silent classic fallback
    # here is the 5-10x cliff.
    for m, k, n in ((512, 256, 256), (512, 256, 64), (256, 512, 256)):
        diag = mlp._mmt(m, k, n).explain(
            jax.ShapeDtypeStruct((m, k), jnp.float32),
            jax.ShapeDtypeStruct((n, k), jnp.float32),
        )
        assert diag.model == "cooperative", (m, k, n, diag.reason)
        assert diag.rows == mlp.BM


def test_matmul_t_gradients_match_reference(rng):
    m, k, n = 512, 256, 256
    a = jnp.asarray(rng.standard_normal((m, k)), dtype=jnp.float32)
    b = jnp.asarray(rng.standard_normal((n, k)) / np.sqrt(k), dtype=jnp.float32)
    grads = jax.jit(
        jax.grad(lambda a, b: jnp.sum(mlp.matmul_t(a, b) ** 2), argnums=(0, 1))
    )
    ref = jax.jit(jax.grad(lambda a, b: jnp.sum((a @ b.T) ** 2), argnums=(0, 1)))
    da, db = grads(a, b)
    da_ref, db_ref = ref(a, b)
    np.testing.assert_allclose(np.asarray(da), np.asarray(da_ref), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(db), np.asarray(db_ref), rtol=2e-3, atol=2e-3)


def test_training_trajectory_matches_reference():
    # Five full Adam steps, Metal matmuls vs jnp, same init: losses and
    # parameters must track within f32/FAST-math tolerance.
    key = jax.random.PRNGKey(11)
    key, k_teacher, k_student, k_data = jax.random.split(key, 4)
    teacher = mlp.init_params(k_teacher, mlp.DIMS)
    x = jax.random.normal(k_data, (256, mlp.DIMS[0]), jnp.float32)
    y = mlp.forward(teacher, x, mmt=mlp._mmt_ref)

    def run(mmt):
        params = mlp.init_params(k_student, mlp.DIMS)
        opt_state = (
            jax.tree.map(jnp.zeros_like, params),
            jax.tree.map(jnp.zeros_like, params),
        )
        step_fn = mlp.make_train_step(mmt)
        losses = []
        for step in range(5):
            params, opt_state, loss = step_fn(params, opt_state, step, x, y)
            losses.append(float(loss))
        return params, losses

    params_m, losses_m = run(mlp.matmul_t)
    params_r, losses_r = run(mlp._mmt_ref)
    np.testing.assert_allclose(losses_m, losses_r, rtol=5e-3, atol=5e-5)
    for (pm, bm), (pr, br) in zip(params_m, params_r, strict=True):
        np.testing.assert_allclose(np.asarray(pm), np.asarray(pr), rtol=5e-3, atol=5e-3)
        np.testing.assert_allclose(np.asarray(bm), np.asarray(br), rtol=5e-3, atol=5e-3)

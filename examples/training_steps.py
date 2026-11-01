"""Representative training steps for the three target workloads of
`notes/metal-compute-stack.md`, small enough to compile anywhere: an RK4 ODE
ensemble with a mixture-density posterior (SBI), a decoder block with RoPE,
causal attention and SwiGLU (LLM fine-tuning), and a conv/windowed-attention
segmenter with focal and dice loss (SAM2-like). All take an AdamW step.

Run it to print the StableHLO op census of each step:

    uv run python examples/training_steps.py
"""

import collections
import re

import jax
import jax.numpy as jnp
from jax import lax


def adamw(params, grads, state, lr=1e-3, b1=0.9, b2=0.999, wd=0.01):
    m, v, t = state
    t = t + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, v, grads)

    def update(p, m, v):
        return p - lr * (m / (1 - b1**t) / (jnp.sqrt(v / (1 - b2**t)) + 1e-8) + wd * p)

    return jax.tree.map(update, params, m, v), (m, v, t)


def sbi_trajectory(theta):
    """The SBI workload's 200-step RK4 ensemble, independently testable from its MLP."""

    def rhs(y):
        x, z = y[..., 0], y[..., 1]
        return jnp.stack(
            [theta[:, 0] * x - theta[:, 1] * x * z, theta[:, 2] * x * z - theta[:, 3] * z], -1
        )

    def rk4(y, _):
        h = 0.01
        a = rhs(y)
        b = rhs(y + h / 2 * a)
        c = rhs(y + h / 2 * b)
        d = rhs(y + h * c)
        y = y + h / 6 * (a + 2 * b + 2 * c + d)
        return y, y

    _, traj = lax.scan(rk4, jnp.ones((theta.shape[0], 2)), None, length=200)
    return traj


def sbi_loss(p, x, theta):
    """Mixture-density loss, isolated so dense training can be tested without RNG."""
    h = jax.nn.gelu(jax.nn.gelu(x @ p["w1"]) @ p["w2"])
    logits = h @ p["wa"]
    mu, log_s = (h @ p["wm"]).reshape(512, 5, 4), (h @ p["ws"]).reshape(512, 5, 4)
    lp = -0.5 * jnp.sum(((theta[:, None] - mu) / jnp.exp(log_s)) ** 2 + 2 * log_s, -1)
    return -jnp.mean(jax.nn.logsumexp(jax.nn.log_softmax(logits) + lp, -1))


def sbi_step(params, opt, key):
    """Simulate a Lotka-Volterra ensemble with RK4, then fit an MDN posterior."""
    k1, k2 = jax.random.split(key)
    theta = jax.random.uniform(k1, (512, 4), minval=0.5, maxval=1.5)
    traj = sbi_trajectory(theta)
    x = jnp.concatenate([traj.mean(0), traj.std(0), traj[-1]], -1)
    x = x + 0.05 * jax.random.normal(k2, (512, 6))

    value, grads = jax.value_and_grad(sbi_loss)(params, x, theta)
    return adamw(params, grads, opt), value


def llm_step(params, opt, tokens):
    """One decoder block on (4, 128) tokens with next-token cross-entropy."""
    B, T, D, H = 4, 128, 256, 4

    def rms(x, w):
        return x * lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6) * w

    def rope(x):
        pos = jnp.arange(T)[:, None] / (10000 ** (jnp.arange(0, D // H, 2) / (D // H)))
        c, s = jnp.cos(pos)[:, None], jnp.sin(pos)[:, None]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return jnp.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).reshape(x.shape)

    def loss(p):
        x = p["emb"][tokens[:, :-1]]
        h = rms(x, p["n1"])
        q, k, v = [(h @ p[w]).reshape(B, T, H, D // H) for w in ("wq", "wk", "wv")]
        q, k = rope(q), rope(k)
        s = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(D // H)
        s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -1e9)
        a = jnp.einsum("bhts,bshd->bthd", jax.nn.softmax(s, -1), v).reshape(B, T, D)
        x = x + a @ p["wo"]
        h = rms(x, p["n2"])
        x = x + (jax.nn.silu(h @ p["wg"]) * (h @ p["wu"])) @ p["wd"]
        lp = jax.nn.log_softmax(rms(x, p["nf"]) @ p["emb"].T)
        return -jnp.mean(jnp.take_along_axis(lp, tokens[:, 1:, None], -1))

    value, grads = jax.value_and_grad(loss)(params)
    return adamw(params, grads, opt), value


def sam_step(params, opt, image, mask):
    """Patch embedding, windowed attention, max-pooling, transposed-conv
    upsampling and bilinear resize, with focal plus dice loss."""

    def ln(x, w):
        m = x.mean(-1, keepdims=True)
        return (x - m) * lax.rsqrt(((x - m) ** 2).mean(-1, keepdims=True) + 1e-6) * w

    def loss(p):
        nhwc = ("NHWC", "HWIO", "NHWC")
        x = lax.conv_general_dilated(image, p["patch"], (4, 4), "VALID", dimension_numbers=nhwc)
        B, Hh, Ww, C = x.shape
        w = x.reshape(B, Hh // 8, 8, Ww // 8, 8, C).transpose(0, 1, 3, 2, 4, 5).reshape(-1, 64, C)
        h = ln(w, p["n1"])
        q, k, v = h @ p["wq"], h @ p["wk"], h @ p["wv"]
        w = w + jax.nn.softmax(q @ k.transpose(0, 2, 1) / jnp.sqrt(C), -1) @ v
        w = w + jax.nn.gelu(ln(w, p["n2"]) @ p["w1"]) @ p["w2"]
        x = (
            w.reshape(B, Hh // 8, Ww // 8, 8, 8, C)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(B, Hh, Ww, C)
        )
        x = lax.reduce_window(x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
        up = lax.conv_transpose(x, p["up"], (4, 4), "SAME", dimension_numbers=nhwc)
        prob = jax.nn.sigmoid(jax.image.resize(up[..., 0], mask.shape, "bilinear"))
        focal = -jnp.mean(
            (1 - prob) ** 2 * mask * jnp.log(prob + 1e-6)
            + prob**2 * (1 - mask) * jnp.log(1 - prob + 1e-6)
        )
        dice = 1 - 2 * jnp.sum(prob * mask) / (jnp.sum(prob) + jnp.sum(mask) + 1e-6)
        return focal + dice

    value, grads = jax.value_and_grad(loss)(params)
    return adamw(params, grads, opt), value


def _init(key, shapes):
    keys = jax.random.split(key, len(shapes))
    return {name: jax.random.normal(k, s) * 0.02 for k, (name, s) in zip(keys, shapes.items())}


def _opt_state(params):
    return jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params), 0


def steps(key=None):
    """Each step with arguments: name -> (function, *args)."""
    key = jax.random.key(0) if key is None else key
    sbi = _init(
        key, {"w1": (6, 128), "w2": (128, 128), "wa": (128, 5), "wm": (128, 20), "ws": (128, 20)}
    )
    llm = _init(
        key,
        {
            "emb": (4096, 256),
            "n1": (256,),
            "n2": (256,),
            "nf": (256,),
            "wq": (256, 256),
            "wk": (256, 256),
            "wv": (256, 256),
            "wo": (256, 256),
            "wg": (256, 512),
            "wu": (256, 512),
            "wd": (512, 256),
        },
    )
    sam = _init(
        key,
        {
            "patch": (4, 4, 3, 64),
            "n1": (64,),
            "n2": (64,),
            "wq": (64, 64),
            "wk": (64, 64),
            "wv": (64, 64),
            "w1": (64, 128),
            "w2": (128, 64),
            "up": (4, 4, 64, 1),
        },
    )
    return {
        "sbi": (sbi_step, sbi, _opt_state(sbi), key),
        "llm": (llm_step, llm, _opt_state(llm), jnp.zeros((4, 129), jnp.int32)),
        "sam": (
            sam_step,
            sam,
            _opt_state(sam),
            jnp.zeros((2, 256, 256, 3)),
            jnp.zeros((2, 256, 256)),
        ),
    }


if __name__ == "__main__":
    union = collections.Counter()
    for name, (fn, *args) in steps().items():
        text = jax.jit(fn).lower(*args).as_text()
        ops = collections.Counter(re.findall(r"\b((?:stablehlo|chlo)\.[a-z_]+|func\.call)\b", text))
        union.update(dict.fromkeys(ops, 1))
        print(f"{name}: {len(ops)} distinct ops, {sum(ops.values())} in all")
    for op, n in sorted(union.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n}  {op}")

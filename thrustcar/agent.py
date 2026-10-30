"""Networks + obs normalization.

PhaseMoEActor: shared trunk -> softmax gate over K expert heads -> mixture of Gaussian means.
  Experts are batched with einsum so all K run in parallel (no Python loop).
DenseActor: parameter-matched MLP baseline for the ablation.
Critic: asymmetric (privileged obs: true mass/thrust/lag/wind/CoM etc.).
"""
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

orth = nn.initializers.orthogonal


class PhaseMoEActor(nn.Module):
    act_dim: int
    n_experts: int = 4
    hidden: int = 256
    gate_temp: float = 1.0

    @nn.compact
    def __call__(self, x):
        h = nn.tanh(nn.Dense(self.hidden, kernel_init=orth(np.sqrt(2)))(x))
        h = nn.tanh(nn.Dense(self.hidden, kernel_init=orth(np.sqrt(2)))(h))
        logits = nn.Dense(self.n_experts, kernel_init=orth(0.01))(h)
        gate = jax.nn.softmax(logits / self.gate_temp, axis=-1)
        K, Hd, A = self.n_experts, self.hidden, self.act_dim
        W1 = self.param("eW1", nn.initializers.variance_scaling(2.0, "fan_in", "normal", batch_axis=(0,)), (K, Hd, Hd))
        b1 = self.param("eb1", nn.initializers.zeros, (K, Hd))
        W2 = self.param("eW2", nn.initializers.variance_scaling(1e-4, "fan_in", "normal", batch_axis=(0,)), (K, Hd, A))
        b2 = self.param("eb2", nn.initializers.zeros, (K, A))
        eh = nn.tanh(jnp.einsum("...h,khj->...kj", h, W1) + b1)
        emu = jnp.einsum("...kj,kja->...ka", eh, W2) + b2
        mu = jnp.einsum("...k,...ka->...a", gate, emu)
        log_std = self.param("log_std", nn.initializers.constant(-0.7), (A,))
        return mu, log_std, gate


class DenseActor(nn.Module):
    act_dim: int
    hidden: int = 400

    @nn.compact
    def __call__(self, x):
        h = x
        for _ in range(3):
            h = nn.tanh(nn.Dense(self.hidden, kernel_init=orth(np.sqrt(2)))(h))
        mu = nn.Dense(self.act_dim, kernel_init=orth(0.01))(h)
        log_std = self.param("log_std", nn.initializers.constant(-0.7), (self.act_dim,))
        gate = jnp.ones(mu.shape[:-1] + (1,))
        return mu, log_std, gate


class Critic(nn.Module):
    hidden: int = 512

    @nn.compact
    def __call__(self, x):
        h = x
        for _ in range(3):
            h = nn.tanh(nn.Dense(self.hidden, kernel_init=orth(np.sqrt(2)))(h))
        return nn.Dense(1, kernel_init=orth(1.0))(h)[..., 0]


class RunningStats(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


def stats_init(dim):
    return RunningStats(jnp.zeros(dim), jnp.ones(dim), jnp.asarray(1e-4))


def stats_update(s: RunningStats, x):
    x = x.reshape(-1, x.shape[-1])
    bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
    delta = bm - s.mean
    tot = s.count + bc
    mean = s.mean + delta * bc / tot
    m2 = s.var * s.count + bv * bc + delta**2 * s.count * bc / tot
    return RunningStats(mean, m2 / tot, tot)


def normalize(s: RunningStats, x):
    return jnp.clip((x - s.mean) / jnp.sqrt(s.var + 1e-8), -10.0, 10.0)


def make_actor(kind, act_dim, n_experts=4):
    return PhaseMoEActor(act_dim, n_experts) if kind == "moe" else DenseActor(act_dim)


def n_params(p):
    return sum(x.size for x in jax.tree.leaves(p))

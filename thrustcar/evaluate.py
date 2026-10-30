"""First-episode evaluation over N parallel envs: success, failure modes, gas, CVaR."""
import jax
import jax.numpy as jnp
import numpy as np

from . import env as E


def rollout_first_episode(policy_fn, key, n, difficulty, task, max_steps=None):
    """policy_fn(state_batch, key) -> actions (n, ACT_DIM). Returns dict of per-env first-episode stats."""
    max_steps = max_steps or int(E.MAX_T / E.CTRL_DT) + 2
    tp = jax.nn.one_hot(task, E.N_TASKS)
    keys = jax.random.split(key, n + 1)
    st = jax.vmap(lambda k: E.reset(k, difficulty, tp, task))(keys[1:])

    fields = ("success", "crash", "collide", "hard", "oob", "timeout", "gas_frac", "ep_ret", "ep_len")
    rec0 = {f: jnp.zeros(n) for f in fields}

    def body(carry, _):
        st, rec, finished, k = carry
        k, ka = jax.random.split(k)
        a = policy_fn(st, ka)
        st, r, done, info = E.v_step(st, a)
        first = done & ~finished
        rec = {f: jnp.where(first, info[f].astype(jnp.float32), rec[f]) for f in fields}
        return (st, rec, finished | done, k), None

    (st, rec, finished, _), _ = jax.lax.scan(body, (st, rec0, jnp.zeros(n, bool), keys[0]), None, max_steps)
    return rec


def summarize(rec):
    rec = {k: np.asarray(v) for k, v in rec.items()}
    n = len(rec["success"])
    s = rec["success"] > 0.5
    ret = np.sort(rec["ep_ret"])
    k = max(1, int(0.05 * n))
    return dict(
        success=float(s.mean()),
        crash=float((rec["crash"] > 0.5).mean()),
        collide=float((rec["collide"] > 0.5).mean()),
        hard_landing=float((rec["hard"] > 0.5).mean()),
        oob=float((rec["oob"] > 0.5).mean()),
        timeout=float((rec["timeout"] > 0.5).mean()),
        gas_left_kg=float(rec["gas_frac"][s].mean() * E.GAS_NOM) if s.any() else 0.0,
        mean_return=float(ret.mean()),
        cvar5_return=float(ret[:k].mean()),
        n=n,
    )


def evaluate(policy_fn, key, n=2048, difficulty=1.0, tasks=(0, 1, 2)):
    out = {}
    for t in tasks:
        f = jax.jit(lambda k, t=t: rollout_first_episode(policy_fn, k, n, difficulty, t))
        out[("hover", "flip", "jump")[t]] = summarize(f(jax.random.fold_in(key, t)))
    return out

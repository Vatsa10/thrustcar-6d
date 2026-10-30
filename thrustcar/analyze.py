"""Benchmark table (expert vs policies across difficulty incl. stress), MoE phase plot,
trajectory export for the 3D viewer.

python -m thrustcar.analyze --runs runs/moe runs/dense --out results
"""
import argparse
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from . import agent as A
from . import env as E
from . import evaluate as EV
from . import expert as X

TASKS = ("hover", "flip", "jump")


def load(path):
    ck = pickle.load(open(path, "rb"))
    actor = A.make_actor(ck["cfg"]["actor"], E.ACT_DIM, ck["cfg"]["n_experts"])
    return actor, ck


def policy_from_ckpt(actor, ck):
    def pol(st, key):
        mu, _, _ = actor.apply(ck["actor"], A.normalize(ck["astats"], E.v_actor_obs(st)))
        return mu
    return pol


_TRACE_CACHE = {}


def trace(actor, ck, task, seed, difficulty=1.0, max_steps=760):
    """Deterministic rollout. actor=None -> expert controller."""
    key = (id(ck) if ck is not None else "expert", task, difficulty)
    if key not in _TRACE_CACHE:
        def run(seed):
            st = E.reset(jax.random.PRNGKey(seed), difficulty, jax.nn.one_hot(task, 3), task)

            def body(st, _):
                if actor is None:
                    a = X.expert_action(st)
                    gate = jnp.zeros(4)
                else:
                    mu, _, gate = actor.apply(ck["actor"], A.normalize(ck["astats"], E.actor_obs(st)[None]))
                    a, gate = mu[0], gate[0]
                    gate = jnp.pad(gate, (0, 4 - gate.shape[0]))
                nst, r, done, info = E.step(st, a)
                rec = dict(pos=st.pos, quat=st.quat, gate=gate, u=st.u_eff, gas=st.gas, t=st.t, done=done,
                           success=info["success"], crash=info["crash"], rects=st.rects,
                           latency=st.params.latency)
                return nst, rec
            return jax.lax.scan(body, st, None, max_steps)[1]
        _TRACE_CACHE[key] = jax.jit(run)
    rec = jax.tree.map(np.asarray, _TRACE_CACHE[key](seed))
    end = int(np.argmax(rec["done"])) + 1 if rec["done"].any() else max_steps
    return {k: v[:end] for k, v in rec.items()}


def compact(tr, stride=2):
    sl = slice(None, None, stride)
    out = dict(t=tr["t"][sl].round(2).tolist(), pos=tr["pos"][sl].round(2).tolist(),
               quat=tr["quat"][sl].round(3).tolist(), u=tr["u"][sl].round(2).tolist(),
               gate=tr["gate"][sl].round(2).tolist(), gas=tr["gas"][sl].round(0).tolist(),
               rects=tr["rects"][0].round(2).tolist(), latency_ms=int(tr["latency"][0]) * 20,
               outcome="success" if tr["success"].any() else ("crash" if tr["crash"].any() else "fail"))
    # include final frame
    out["t"].append(float(tr["t"][-1])); out["pos"].append(tr["pos"][-1].round(2).tolist())
    out["quat"].append(tr["quat"][-1].round(3).tolist()); out["u"].append(tr["u"][-1].round(2).tolist())
    out["gate"].append(tr["gate"][-1].round(2).tolist()); out["gas"].append(float(tr["gas"][-1].round(0)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["runs/moe"])
    p.add_argument("--out", default="results")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--difficulties", default="0.0,1.0,1.5")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    diffs = [float(x) for x in a.difficulties.split(",")]

    policies = {"expert": lambda st, k: X.v_expert(st)}
    loaded = {}
    for r in a.runs:
        actor, ck = load(os.path.join(r, "best.pkl"))
        name = os.path.basename(r.rstrip("/"))
        policies[name] = policy_from_ckpt(actor, ck)
        loaded[name] = (actor, ck)

    table = {}
    for name, pol in policies.items():
        for d in diffs:
            for t, tn in enumerate(TASKS):
                f = jax.jit(lambda k, pol=pol, d=d, t=t: EV.rollout_first_episode(pol, k, a.n, d, t))
                table[(name, d, tn)] = EV.summarize(f(jax.random.PRNGKey(123 + t)))
                s = table[(name, d, tn)]
                print(f"{name:8s} d={d:.1f} {tn:5s} success={s['success']:.3f} gas={s['gas_left_kg']:6.1f} "
                      f"cvar5={s['cvar5_return']:7.1f} crash={s['crash']:.3f} oob={s['oob']:.3f}", flush=True)

    lines = ["| policy | difficulty | task | success | gas left (kg) | CVaR5 return | crash | OOB |",
             "|---|---|---|---|---|---|---|---|"]
    for (name, d, tn), s in table.items():
        lines.append(f"| {name} | {d:.1f} | {tn} | {s['success']*100:.1f}% | {s['gas_left_kg']:.1f} | "
                     f"{s['cvar5_return']:.1f} | {s['crash']*100:.1f}% | {s['oob']*100:.1f}% |")
    open(os.path.join(a.out, "results.md"), "w").write("\n".join(lines) + "\n")
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in table.items()}, open(os.path.join(a.out, "results.json"), "w"), indent=1)

    # MoE phase plot + 3D trajectories
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    trajs = {}
    for name, (actor, ck) in loaded.items():
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=False)
        for t, tn in enumerate(TASKS):
            tr, seed = None, 0
            for seed in range(40):                      # first seed the policy solves where the expert fails, else first success
                tr = trace(actor, ck, t, seed)
                if tr["success"].any() and not trace(None, None, t, seed)["success"].any():
                    break
            else:
                for seed in range(20):
                    tr = trace(actor, ck, t, seed)
                    if tr["success"].any():
                        break
            up = np.array([E.quat_to_rot(q)[2, 2] for q in tr["quat"]])
            ax = axes[t]
            ax.stackplot(tr["t"], tr["gate"].T, alpha=0.6, labels=[f"expert {i}" for i in range(tr["gate"].shape[1])])
            ax2 = ax.twinx()
            ax2.plot(tr["t"], tr["pos"][:, 2], "k-", lw=1.2, label="altitude [m]")
            ax2.plot(tr["t"], up * 5, "k--", lw=1, label="uprightness x5")
            ax.set_title(f"{name} / {tn}: {'success' if tr['success'].any() else 'fail'}", fontsize=9)
            ax.set_ylabel("gate weight")
            if t == 0:
                ax.legend(loc="upper left", fontsize=7, ncol=4)
                ax2.legend(loc="upper right", fontsize=7)
            trajs[f"{name}/{tn}"] = dict(compact(tr), seed=seed)
            trajs[f"expert/{tn}@{name}"] = dict(compact(trace(None, None, t, seed)), seed=seed)
        axes[-1].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"gates_{name}.png"), dpi=130)
    json.dump(trajs, open(os.path.join(a.out, "trajectories.json"), "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

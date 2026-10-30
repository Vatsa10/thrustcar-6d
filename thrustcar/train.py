"""PPO for ThrustCar-6D with Phase-MoE actor, asymmetric critic, DAgger warm start,
success-gated difficulty curriculum. Pure JAX: env + rollout + update all jitted.

python -m thrustcar.train --actor moe  --total-steps 4e8 --out runs/moe
python -m thrustcar.train --actor dense --total-steps 4e8 --out runs/dense   # ablation
"""
import argparse
import json
import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import agent as A
from . import env as E
from . import evaluate as EV
from . import expert as X


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--actor", default="moe", choices=["moe", "dense"])
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--total-steps", type=float, default=4e8)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.0)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--lb-coef", type=float, default=0.01)
    p.add_argument("--gate-ent-coef", type=float, default=0.003)
    p.add_argument("--dagger-iters", type=int, default=8)
    p.add_argument("--dagger-steps", type=int, default=400, help="env steps per DAgger iter (x num_envs)")
    p.add_argument("--bc-epochs", type=int, default=15)
    p.add_argument("--bc-coef", type=float, default=1.0, help="DAPG-style expert-imitation aux loss, decays to 0")
    p.add_argument("--bc-decay-frac", type=float, default=0.5)
    p.add_argument("--critic-warmup", type=int, default=30)
    p.add_argument("--init-log-std", type=float, default=-1.6)
    p.add_argument("--curr-thresh", type=float, default=0.80)
    p.add_argument("--curr-step", type=float, default=0.01)
    p.add_argument("--task-probs", type=str, default="0.25,0.35,0.40")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-n", type=int, default=1024)
    p.add_argument("--eval-difficulty", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/moe")
    return p.parse_args(argv)


# ---------------------------------------------------------------- helpers
def gauss_logp(a, mu, log_std):
    return jnp.sum(-0.5 * ((a - mu) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), -1)


def make_policy_fn(actor, aparams, astats):
    def pol(st, key):
        mu, _, _ = actor.apply(aparams, A.normalize(astats, E.v_actor_obs(st)))
        return mu
    return pol


def main(argv=None):
    cfg = parse(argv)
    os.makedirs(cfg.out, exist_ok=True)
    json.dump(vars(cfg), open(os.path.join(cfg.out, "config.json"), "w"), indent=2)
    print("devices:", jax.devices())

    key = jax.random.PRNGKey(cfg.seed)
    task_probs = jnp.array([float(x) for x in cfg.task_probs.split(",")])
    task_probs = task_probs / task_probs.sum()
    N, T = cfg.num_envs, cfg.num_steps
    batch = N * T
    n_updates = int(cfg.total_steps // batch)

    actor = A.make_actor(cfg.actor, E.ACT_DIM, cfg.n_experts)
    critic = A.Critic()
    key, ka, kc, ke = jax.random.split(key, 4)
    aparams = actor.init(ka, jnp.zeros((1, E.ACTOR_DIM)))
    cparams = critic.init(kc, jnp.zeros((1, E.CRITIC_DIM)))
    print(f"actor={cfg.actor} params={A.n_params(aparams):,}  critic params={A.n_params(cparams):,}")
    astats, cstats = A.stats_init(E.ACTOR_DIM), A.stats_init(E.CRITIC_DIM)

    env_st = E.v_reset(jax.random.split(ke, N), 0.0, task_probs, None)
    difficulty = 0.0

    # ======================================================== DAgger warm start
    if cfg.dagger_iters > 0:
        bc_tx = optax.adam(1e-3)
        bc_opt = bc_tx.init(aparams)

        @partial(jax.jit, static_argnums=(4,))
        def collect(aparams, astats, env_st, key, use_expert):
            def body(c, _):
                st, k = c
                k, kn = jax.random.split(k)
                obs = E.v_actor_obs(st)
                lab = X.v_expert(st)
                mu, _, _ = actor.apply(aparams, A.normalize(astats, obs))
                act = lab if use_expert else mu + 0.05 * jax.random.normal(kn, mu.shape)
                st, _, done, info = E.v_step(st, act)
                return (st, k), (obs, lab, done, info["success"], info["task"])
            (st, _), (obs, lab, done, succ, task) = jax.lax.scan(body, (env_st, key), None, cfg.dagger_steps)
            return st, obs.reshape(-1, E.ACTOR_DIM), lab.reshape(-1, E.ACT_DIM), done, succ, task

        @jax.jit
        def bc_step(aparams, opt, astats, obs, lab):
            def loss(p):
                mu, _, _ = actor.apply(p, A.normalize(astats, obs))
                return jnp.mean((mu - lab) ** 2)
            l, g = jax.value_and_grad(loss)(aparams)
            upd, opt = bc_tx.update(g, opt, aparams)
            return optax.apply_updates(aparams, upd), opt, l

        BCB = 8192

        @jax.jit
        def bc_epoch(aparams, opt, astats, Xo, Xl, key):
            n = Xo.shape[0] // BCB * BCB
            idx = jax.random.permutation(key, Xo.shape[0])[:n].reshape(-1, BCB)

            def body(c, i):
                p, o = c
                p, o, l = bc_step(p, o, astats, Xo[i], Xl[i])
                return (p, o), l
            (aparams, opt), ls = jax.lax.scan(body, (aparams, opt), idx)
            return aparams, opt, ls[-1]

        D_obs, D_lab = [], []          # kept on device; last 4 iterations, <=1M samples each
        for it in range(cfg.dagger_iters):
            key, kk, ks = jax.random.split(key, 3)
            env_st, obs, lab, done, succ, task = collect(aparams, astats, env_st, kk, it == 0)
            if it == 0:
                astats = A.stats_update(astats, obs)
            keep = jax.random.permutation(ks, obs.shape[0])[:min(obs.shape[0], 1_000_000)]
            D_obs.append(obs[keep]); D_lab.append(lab[keep])
            D_obs, D_lab = D_obs[-4:], D_lab[-4:]
            Xo, Xl = jnp.concatenate(D_obs), jnp.concatenate(D_lab)
            for ep in range(cfg.bc_epochs):
                key, ke = jax.random.split(key)
                aparams, bc_opt, l = bc_epoch(aparams, bc_opt, astats, Xo, Xl, ke)
            done, succ, task = map(np.asarray, (done, succ, task))
            sr = [succ[done & (task == t)].mean() if (done & (task == t)).any() else float("nan") for t in range(3)]
            print(f"dagger {it}  data={Xo.shape[0]:,}  bc_mse={float(l):.4f}  "
                  f"{'expert' if it == 0 else 'learner'} rollout success h/f/j={np.round(sr, 3)}", flush=True)
        # tighten exploration around the imitated policy
        aparams = jax.tree_util.tree_map(lambda x: x, aparams)
        aparams["params"]["log_std"] = jnp.full_like(aparams["params"]["log_std"], cfg.init_log_std)

    # ============================================================== PPO
    lr_sched = optax.linear_schedule(cfg.lr, cfg.lr * 0.1, n_updates * cfg.epochs * cfg.minibatches)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad), optax.adam(lr_sched, eps=1e-5))
    params = {"actor": aparams, "critic": cparams}
    opt = tx.init(params)

    @partial(jax.jit, static_argnums=(5,))
    def update(params, opt, astats, cstats, env_st, actor_on, key, bc_w):
        # ---------------- rollout
        def body(c, _):
            st, k = c
            k, kn = jax.random.split(k)
            oa, oc = E.v_actor_obs(st), E.v_critic_obs(st)
            mu, log_std, gate = actor.apply(params["actor"], A.normalize(astats, oa))
            act = mu + jnp.exp(log_std) * jax.random.normal(kn, mu.shape)
            logp = gauss_logp(act, mu, log_std)
            v = critic.apply(params["critic"], A.normalize(cstats, oc))
            lab = X.v_expert(st)
            st, r, done, info = E.v_step(st, act)
            tr = dict(oa=oa, oc=oc, a=act, lab=lab, logp=logp, v=v, r=r * cfg.reward_scale, done=done,
                      succ=info["success"], task=info["task"], gate=gate, ep_ret=info["ep_ret"],
                      crash=info["crash"], gas=info["gas_frac"])
            return (st, k), tr
        key, kr = jax.random.split(key)
        (env_st, _), tr = jax.lax.scan(body, (env_st, kr), None, T)
        last_v = critic.apply(params["critic"], A.normalize(cstats, E.v_critic_obs(env_st)))

        # ---------------- GAE
        def gae_body(c, x):
            adv, nv = c
            r, v, d = x
            delta = r + cfg.gamma * nv * (1 - d) - v
            adv = delta + cfg.gamma * cfg.lam * (1 - d) * adv
            return (adv, v), adv
        _, adv = jax.lax.scan(gae_body, (jnp.zeros(N), last_v), (tr["r"], tr["v"], tr["done"].astype(jnp.float32)),
                              reverse=True)
        ret = adv + tr["v"]

        flat = lambda x: x.reshape((batch,) + x.shape[2:])
        data = dict(oa=flat(tr["oa"]), oc=flat(tr["oc"]), a=flat(tr["a"]), lab=flat(tr["lab"]), logp=flat(tr["logp"]),
                    v=flat(tr["v"]), adv=flat(adv), ret=flat(ret))

        def loss_fn(p, mb):
            mu, log_std, gate = actor.apply(p["actor"], A.normalize(astats, mb["oa"]))
            logp = gauss_logp(mb["a"], mu, log_std)
            ratio = jnp.exp(logp - mb["logp"])
            adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
            pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            ent = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
            v = critic.apply(p["critic"], A.normalize(cstats, mb["oc"]))
            vclip = mb["v"] + jnp.clip(v - mb["v"], -cfg.clip, cfg.clip)
            vl = 0.5 * jnp.maximum((v - mb["ret"]) ** 2, (vclip - mb["ret"]) ** 2).mean()
            K = gate.shape[-1]
            imp = gate.mean(0)
            lb = K * jnp.sum(imp ** 2) - 1.0
            gent = -jnp.sum(gate * jnp.log(gate + 1e-8), -1).mean()
            bc = jnp.mean((mu - mb["lab"]) ** 2)
            a_loss = pg - cfg.ent * ent + cfg.lb_coef * lb + cfg.gate_ent_coef * gent + bc_w * bc
            total = (a_loss if actor_on else 0.0 * a_loss) + cfg.vf * vl
            kl = ((ratio - 1) - jnp.log(ratio)).mean()
            return total, dict(pg=pg, vl=vl, lb=lb, gent=gent, kl=kl, bc=bc)

        def epoch(c, _):
            params, opt, k = c
            k, kp = jax.random.split(k)
            perm = jax.random.permutation(kp, batch)
            mbs = jax.tree.map(lambda x: x[perm].reshape((cfg.minibatches, -1) + x.shape[1:]), data)

            def mb_step(c, mb):
                params, opt = c
                (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
                upd, opt = tx.update(g, opt, params)
                return (optax.apply_updates(params, upd), opt), aux
            (params, opt), aux = jax.lax.scan(mb_step, (params, opt), mbs)
            return (params, opt, k), aux
        key, ku = jax.random.split(key)
        (params, opt, _), aux = jax.lax.scan(epoch, (params, opt, ku), None, cfg.epochs)

        astats = A.stats_update(astats, tr["oa"])
        cstats = A.stats_update(cstats, tr["oc"])

        # ---------------- metrics
        done = tr["done"]
        oh = jax.nn.one_hot(tr["task"], 3) * done[..., None]
        n_done = oh.sum((0, 1))
        succ_t = (oh * tr["succ"][..., None]).sum((0, 1)) / jnp.maximum(n_done, 1)
        crash_t = (oh * tr["crash"][..., None]).sum((0, 1)) / jnp.maximum(n_done, 1)
        ohall = jax.nn.one_hot(tr["task"], 3)
        gate_t = jnp.einsum("tnk,tnc->ck", tr["gate"], ohall) / jnp.maximum(ohall.sum((0, 1)), 1)[:, None]
        m = dict(succ=succ_t, crash=crash_t, n_done=n_done, gate=gate_t,
                 ep_ret=(tr["ep_ret"] * done).sum() / jnp.maximum(done.sum(), 1),
                 **{k: v.mean() for k, v in aux.items()},
                 std=jnp.exp(params["actor"]["params"]["log_std"]).mean())
        return params, opt, astats, cstats, env_st, key, m

    @partial(jax.jit, static_argnums=(3,))
    def eval_fn(aparams, astats, key, task):
        return EV.rollout_first_episode(make_policy_fn(actor, aparams, astats), key, cfg.eval_n, cfg.eval_difficulty, task)

    # warm up critic with frozen actor (protects the DAgger policy)
    t0 = time.time()
    log = open(os.path.join(cfg.out, "metrics.jsonl"), "a")
    best = -1.0
    total_updates = cfg.critic_warmup * (cfg.dagger_iters > 0) + n_updates
    for u in range(total_updates):
        actor_on = u >= cfg.critic_warmup * (cfg.dagger_iters > 0)
        frac = max(0, u - cfg.critic_warmup * (cfg.dagger_iters > 0)) / max(1, n_updates)
        bc_w = cfg.bc_coef * max(0.0, 1 - frac / cfg.bc_decay_frac) * (cfg.dagger_iters > 0)
        params, opt, astats, cstats, env_st, key, m = update(params, opt, astats, cstats, env_st, actor_on, key,
                                                              jnp.float32(bc_w))

        succ = np.asarray(m["succ"]); n_done = np.asarray(m["n_done"])
        if actor_on and n_done.min() > 20 and succ.min() > cfg.curr_thresh and difficulty < 1.0:
            difficulty = min(1.0, difficulty + cfg.curr_step)
            env_st = E.set_curriculum(env_st, difficulty, task_probs)

        steps = (u + 1) * batch
        if u % cfg.log_every == 0 or u == total_updates - 1:
            sps = steps / (time.time() - t0)
            rec = dict(update=u, steps=steps, sps=sps, difficulty=difficulty, actor_on=bool(actor_on),
                       succ=succ.tolist(), crash=np.asarray(m["crash"]).tolist(),
                       gate=np.round(np.asarray(m["gate"]), 3).tolist(),
                       bc_w=bc_w, **{k: float(m[k]) for k in ("ep_ret", "pg", "vl", "lb", "gent", "kl", "std", "bc")})
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"u{u:5d} {steps/1e6:7.1f}M {sps/1e6:5.2f}Msps d={difficulty:.2f} "
                  f"succ h/f/j={np.round(succ, 2)} ret={rec['ep_ret']:7.1f} kl={rec['kl']:.4f} "
                  f"std={rec['std']:.3f} lb={rec['lb']:.3f} gent={rec['gent']:.2f} bc={rec['bc']:.4f}/{bc_w:.2f}", flush=True)

        if actor_on and (u % cfg.eval_every == 0 or u == total_updates - 1):
            res = {name: EV.summarize(eval_fn(params["actor"], astats, jax.random.PRNGKey(10_000 + u + t), t))
                   for t, name in enumerate(("hover", "flip", "jump"))}
            score = float(np.mean([r["success"] for r in res.values()]))
            print(f"  EVAL d={cfg.eval_difficulty}", {k: (round(v["success"], 3), round(v["gas_left_kg"], 1)) for k, v in res.items()},
                  flush=True)
            log.write(json.dumps(dict(update=u, steps=steps, eval=res)) + "\n"); log.flush()
            ck = dict(cfg=vars(cfg), actor=jax.device_get(params["actor"]), critic=jax.device_get(params["critic"]),
                      astats=jax.device_get(astats), cstats=jax.device_get(cstats), difficulty=difficulty)
            pickle.dump(ck, open(os.path.join(cfg.out, "last.pkl"), "wb"))
            if score >= best:
                best = score
                pickle.dump(ck, open(os.path.join(cfg.out, "best.pkl"), "wb"))
    print(f"done in {(time.time()-t0)/60:.1f} min; best mean eval success {best:.3f}")


if __name__ == "__main__":
    main()

"""ThrustCar-6D: pure-JAX 6-DOF environment of a cold-gas-thruster car.

Modeled on the Tesla Roadster (gen 2) reveal, 2026-10-15: 10 cold-gas thrusters,
~250 kg compressed air, inverted-ramp barrel roll, 5-vehicle jump.

Physics (per env, vmapped):
  - rigid body: quaternion attitude, body-frame Euler equations with gyroscopic term
  - 10 thrusters with real body positions/directions; torque = r x F (CoM offset randomized)
  - first-order valve lag (tau ~20-60 ms) + minimum-impulse deadband
  - pressure-regulated blowdown tank: full thrust until tank pressure < regulator
    setpoint, then thrust decays with p ~ m^n (polytropic n=1.2); Isp ~ sqrt(T)
  - 4 spring-damper wheels with Coulomb-like friction, drive force, rolling resistance
  - anisotropic body drag against an OU gust wind field
  - IMU noise + gyro bias, noisy tank-pressure sensor; actor sees a 4-frame history
Tasks: 0 hover-land, 1 inverted barrel-roll recovery + landing, 2 five-vehicle jump.
Everything not published by Tesla (mass, Isp, thrust, geometry) is an assumption.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ----------------------------------------------------------------- constants
G = 9.81
G0 = 9.80665
RHO = 1.2
DT = 0.01
SUBSTEPS = 2
CTRL_DT = DT * SUBSTEPS
MAX_T = 15.0
HIST = 4
LAT_BUF = 8               # observation-delay ring buffer (control steps, 20 ms each)
MAX_LAT_STEPS = 5         # 100 ms at difficulty 1.0 (150 ms at 1.5 stress)
N_THR = 10
ACT_DIM = 11              # 10 valves + wheel throttle
N_TASKS = 3
HOVER, FLIP, JUMP = 0, 1, 2

L, W, H = 4.5, 2.0, 1.2
DRY_MASS = 1800.0
GAS_NOM = 250.0
ISP0 = 70.0              # compressed air cold-gas, ~65-74 s
POLY_N = 1.2
P_REG = 0.35              # regulator setpoint as fraction of full tank pressure
CDA_BODY = jnp.array([0.6, 5.0, 8.0])

#                  x      y     z
THR_POS = jnp.array([
    [-2.2,  0.60,  0.0],   # 0 rear main L
    [-2.2, -0.60,  0.0],   # 1 rear main R
    [ 1.6,  0.70, -0.3],   # 2 lift FL
    [ 1.6, -0.70, -0.3],   # 3 lift FR
    [-1.6,  0.70, -0.3],   # 4 lift RL
    [-1.6, -0.70, -0.3],   # 5 lift RR
    [ 1.8,  0.95,  0.1],   # 6 side FL
    [ 1.8, -0.95,  0.1],   # 7 side FR
    [-1.8,  0.95,  0.1],   # 8 side RL
    [-1.8, -0.95,  0.1],   # 9 side RR
])
THR_DIR = jnp.array([
    [1, 0, 0], [1, 0, 0],
    [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
    [0, -1, 0], [0, 1, 0], [0, -1, 0], [0, 1, 0],
], dtype=jnp.float32)
THR_FMAX = jnp.array([8e3, 8e3, 14e3, 14e3, 14e3, 14e3, 4e3, 4e3, 4e3, 4e3])

WHEELS = jnp.array([[1.4, 0.85, -0.6], [1.4, -0.85, -0.6],
                    [-1.4, 0.85, -0.6], [-1.4, -0.85, -0.6]])
HULL = jnp.array([[sx * L / 2, sy * W / 2, z]
                  for sx in (1, -1) for sy in (1, -1) for z in (-0.3, 0.6)])
K_SPR, C_DMP = 2.0e5, 1.2e4
CRR = 0.015
WHEEL_F = 12e3
REST_Z = 0.6 - (DRY_MASS + GAS_NOM) * G / 4 / K_SPR

MAX_LAND_VZ = 3.0
MAX_TILT = 0.30
VEH_L = jnp.array([4.3, 5.7, 6.5, 4.8, 4.7])      # Cybercab, Cybertruck, Semi, Model Y, Model 3
VEH_H = jnp.array([1.15, 1.80, 4.10, 1.62, 1.44])
OB_HALF_W = 2.1         # vehicle + safety barrier; car can't bypass inside the corridor
CORRIDOR = 3.0

OBS_DIM = 34
PREV_A_SLICE = slice(17, 17 + ACT_DIM)
PRIV_DIM = 51
ACTOR_DIM = OBS_DIM * HIST
CRITIC_DIM = OBS_DIM + PRIV_DIM


# ------------------------------------------------------------------- pytrees
class Params(NamedTuple):
    mass_scale: jax.Array
    thrust_scale: jax.Array   # (10,)
    isp_scale: jax.Array
    tau: jax.Array            # (10,)
    com_off: jax.Array        # (3,)
    wind_mean: jax.Array      # (3,)
    gust_std: jax.Array
    gyro_bias: jax.Array      # (3,)
    fill: jax.Array
    mu: jax.Array
    latency: jax.Array        # int: observation delay in control steps


class State(NamedTuple):
    pos: jax.Array
    vel: jax.Array
    quat: jax.Array
    omega: jax.Array
    gas: jax.Array
    u_eff: jax.Array
    wind: jax.Array
    t: jax.Array
    task: jax.Array
    rects: jax.Array          # (5, 3): x0, x1, h
    cleared: jax.Array
    settle_t: jax.Array
    airborne_seen: jax.Array
    prev_a: jax.Array
    pot: jax.Array
    obs_hist: jax.Array       # (HIST, OBS_DIM)
    frame_buf: jax.Array      # (LAT_BUF, OBS_DIM) raw noisy frames, newest last
    params: Params
    key: jax.Array
    ep_ret: jax.Array
    ep_len: jax.Array
    difficulty: jax.Array
    task_probs: jax.Array     # (3,)


# ------------------------------------------------------------- quaternions
def euler_to_quat(r, p, y):
    cr, sr = jnp.cos(r / 2), jnp.sin(r / 2)
    cp, sp = jnp.cos(p / 2), jnp.sin(p / 2)
    cy, sy = jnp.cos(y / 2), jnp.sin(y / 2)
    return jnp.array([cr * cp * cy + sr * sp * sy,
                      sr * cp * cy - cr * sp * sy,
                      cr * sp * cy + sr * cp * sy,
                      cr * cp * sy - sr * sp * cy])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jnp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_to_rot(q):
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ------------------------------------------------------------------- reset
def _u(k, lo, hi, shape=()):
    return jax.random.uniform(k, shape, minval=lo, maxval=hi)


def reset(key, difficulty, task_probs, task=None):
    ks = jax.random.split(key, 24)
    if task is None:
        task = jax.random.choice(ks[0], N_TASKS, p=task_probs)
    task = jnp.asarray(task, jnp.int32)
    s = 0.25 + 0.75 * difficulty       # randomization scale; >1 = stress test

    params = Params(
        mass_scale=1 + _u(ks[1], -0.15, 0.15) * s,
        thrust_scale=1 + _u(ks[2], -0.15, 0.15, (N_THR,)) * s,
        isp_scale=1 + _u(ks[3], -0.10, 0.10) * s,
        tau=0.02 + _u(ks[4], 0.0, 0.04, (N_THR,)) * s,
        com_off=_u(ks[5], -0.15, 0.15, (3,)) * s,
        wind_mean=jnp.array([1.0, 1.0, 0.2]) * _u(ks[6], -6, 6, (3,)) * s,
        gust_std=3.0 * s,
        gyro_bias=jax.random.normal(ks[7], (3,)) * 0.02 * s,
        fill=1 - _u(ks[8], 0.0, 0.15) * s,
        mu=1 - _u(ks[9], 0.0, 0.30) * s,
        latency=jnp.minimum(jnp.floor(_u(ks[23], 0.0, 1.0) * (MAX_LAT_STEPS * s + 1)), LAT_BUF - 1).astype(jnp.int32),
    )

    # task-specific initial conditions
    hov_pos = jnp.array([0.0, 0.0, _u(ks[10], 3.0, 10.0)])
    hov_vel = jnp.array([_u(ks[11], -4, 4) * s, _u(ks[12], -2, 2) * s, _u(ks[13], -3, 2)])
    hov_eul = jnp.array([_u(ks[14], -.4, .4) * s, _u(ks[15], -.4, .4) * s, _u(ks[16], -3.14, 3.14)])

    flp_pos = jnp.array([0.0, 0.0, 12.0 + _u(ks[10], -1, 1)])
    flp_vel = jnp.array([15 + _u(ks[11], -3, 3) * s, _u(ks[12], -1, 1) * s, 3 + _u(ks[13], -1, 1)])
    flp_eul = jnp.array([jnp.pi + _u(ks[14], -.25, .25) * s, _u(ks[15], -.15, .15) * s,
                         _u(ks[16], -.15, .15) * s])

    jmp_pos = jnp.array([0.0, 0.0, REST_Z])
    jmp_vel = jnp.array([22 + _u(ks[11], -4, 4) * s, 0.0, 0.0])
    jmp_eul = jnp.zeros(3)

    sel = lambda a, b, c: jnp.where(task == HOVER, a, jnp.where(task == FLIP, b, c))
    pos = sel(hov_pos, flp_pos, jmp_pos)
    vel = sel(hov_vel, flp_vel, jmp_vel)
    eul = sel(hov_eul, flp_eul, jmp_eul)
    omega = jnp.where(task == JUMP, 0.0, 1.0) * jax.random.normal(ks[17], (3,)) * 0.3 * s

    perm = jax.random.permutation(ks[18], 5)
    lens = VEH_L[perm]
    hts = VEH_H[perm] * (1 + _u(ks[19], -0.1, 0.1, (5,)) * s)
    x0 = 60.0 + _u(ks[20], -10, 10) + jnp.concatenate([jnp.zeros(1), jnp.cumsum(lens + 1.0)[:-1]])
    rects = jnp.stack([x0, x0 + lens, hts], -1)
    rects = jnp.where(task == JUMP, rects, rects.at[:, :2].add(1e4))

    st = State(
        pos=pos, vel=vel, quat=euler_to_quat(*eul), omega=omega,
        gas=GAS_NOM * params.fill, u_eff=jnp.zeros(N_THR), wind=params.wind_mean,
        t=jnp.zeros(()), task=task, rects=rects, cleared=jnp.zeros((), jnp.int32),
        settle_t=jnp.zeros(()), airborne_seen=task != JUMP,
        prev_a=-jnp.ones(ACT_DIM).at[-1].set(0.0), pot=jnp.zeros(()),
        obs_hist=jnp.zeros((HIST, OBS_DIM)), frame_buf=jnp.zeros((LAT_BUF, OBS_DIM)), params=params, key=ks[21],
        ep_ret=jnp.zeros(()), ep_len=jnp.zeros((), jnp.int32),
        difficulty=jnp.asarray(difficulty, jnp.float32), task_probs=task_probs)
    st = st._replace(pot=_potential(st, _wheel_pen(st) > 0))
    frame = _frame(st, ks[22], noisy=True)
    return st._replace(obs_hist=jnp.tile(frame, (HIST, 1)), frame_buf=jnp.tile(frame, (LAT_BUF, 1)))


# ----------------------------------------------------------------- physics
def _mass(st):
    return DRY_MASS * st.params.mass_scale + st.gas


def tank_state(gas):
    p_tank = jnp.maximum(gas / GAS_NOM, 1e-6) ** POLY_N
    reg = jnp.minimum(1.0, p_tank / P_REG)
    temp = p_tank ** ((POLY_N - 1) / POLY_N)
    return p_tank, reg, temp


def _wheel_pen(st):
    R = quat_to_rot(st.quat)
    pw = st.pos + (R @ (WHEELS - st.params.com_off).T).T
    return -pw[:, 2]


def _substep(st, u_cmd, wheel_u, key):
    p = st.params
    R = quat_to_rot(st.quat)
    m = _mass(st)
    Ib = jnp.array([m / 12 * (W**2 + H**2), m / 12 * (L**2 + H**2), m / 12 * (L**2 + W**2)])

    # valves + tank
    u_eff = st.u_eff + (u_cmd - st.u_eff) * jnp.minimum(DT / p.tau, 1.0)
    _, reg, temp = tank_state(st.gas)
    isp = ISP0 * p.isp_scale * jnp.sqrt(jnp.maximum(temp, 0.05))
    f = u_eff * THR_FMAX * p.thrust_scale * reg
    dm_req = f.sum() / (isp * G0) * DT
    dm = jnp.minimum(dm_req, st.gas)
    f = f * jnp.where(dm_req > 0, dm / jnp.maximum(dm_req, 1e-9), 0.0)
    gas = st.gas - dm

    Fb_thr = f[:, None] * THR_DIR
    F_body = Fb_thr.sum(0)
    tau_body = jnp.cross(THR_POS - p.com_off, Fb_thr).sum(0)

    # wind (OU) + drag
    kw1, kw2 = jax.random.split(key)
    wind = st.wind + (p.wind_mean - st.wind) * DT / 2.0 \
        + p.gust_std * jnp.sqrt(DT) * jax.random.normal(kw1, (3,)) * jnp.array([1, 1, 0.3])
    v_rel_b = R.T @ (st.vel - wind)
    F_body = F_body - 0.5 * RHO * CDA_BODY * jnp.abs(v_rel_b) * v_rel_b

    F_world = R @ F_body + jnp.array([0.0, 0.0, -m * G])

    # wheels
    rw = (R @ (WHEELS - p.com_off).T).T
    pw = st.pos + rw
    vw = st.vel + jnp.cross(R @ st.omega, rw)
    pen = -pw[:, 2]
    Fn = jnp.where(pen > 0, jnp.maximum(K_SPR * pen - C_DMP * vw[:, 2], 0.0), 0.0)
    fwd = R[:, 0] * jnp.array([1.0, 1.0, 0.0])
    fwd = fwd / (jnp.linalg.norm(fwd) + 1e-6)
    lat = jnp.array([-fwd[1], fwd[0], 0.0])
    v_long, v_lat = vw @ fwd, vw @ lat
    upright = (R[2, 2] > 0.5).astype(jnp.float32)
    drive = wheel_u * WHEEL_F / 4 * (Fn > 0) * upright
    F_long = jnp.clip(drive - CRR * Fn * jnp.tanh(v_long / 0.3), -p.mu * Fn, p.mu * Fn)
    F_lat = -p.mu * Fn * jnp.tanh(v_lat / 0.3)
    Fw = Fn[:, None] * jnp.array([0.0, 0.0, 1.0]) + F_long[:, None] * fwd + F_lat[:, None] * lat
    F_world = F_world + Fw.sum(0)
    tau_body = tau_body + R.T @ jnp.cross(rw, Fw).sum(0)

    impact = jnp.max(jnp.where(pen > 0, -vw[:, 2], 0.0))

    # semi-implicit Euler
    vel = st.vel + F_world / m * DT
    pos = st.pos + vel * DT
    omega = st.omega + (tau_body - jnp.cross(st.omega, Ib * st.omega)) / Ib * DT
    q = st.quat + 0.5 * DT * quat_mul(st.quat, jnp.concatenate([jnp.zeros(1), omega]))
    q = q / jnp.linalg.norm(q)
    st = st._replace(pos=pos, vel=vel, quat=q, omega=omega, gas=gas, u_eff=u_eff, wind=wind)
    return st, pen, impact


# ---------------------------------------------------------- observations
def _obstacle_feats(st):
    idx = jnp.minimum(st.cleared, 4)
    r = st.rects[idx]
    remaining = jnp.arange(5) >= st.cleared
    hmax = jnp.max(jnp.where(remaining, st.rects[:, 2], 0.0))
    valid = (st.task == JUMP) & (st.cleared < 5)
    f = jnp.array([(r[0] - st.pos[0]) / 50, (r[1] - st.pos[0]) / 50, r[2] / 5, hmax / 5,
                   (st.rects[4, 1] - st.pos[0]) / 50])
    done_f = jnp.array([-1.0, -1.0, 0.0, 0.0, -1.0]) * (st.task == JUMP)
    return jnp.where(valid, f, done_f)


def _frame(st, key, noisy):
    R = quat_to_rot(st.quat)
    k = jax.random.split(key, 5)
    n = lambda kk, sd, shape: jnp.where(noisy, jax.random.normal(kk, shape) * sd, 0.0)
    grav_b = R.T @ jnp.array([0.0, 0.0, -1.0]) + n(k[0], 0.02, (3,))
    gyro = st.omega + jnp.where(noisy, st.params.gyro_bias, 0.0) + n(k[1], 0.02, (3,))
    vel = st.vel + n(k[2], 0.1, (3,))
    z = st.pos[2] + n(k[3], 0.05, ())
    p_tank, _, _ = tank_state(st.gas)
    yaw = jnp.arctan2(R[1, 0], R[0, 0])
    return jnp.concatenate([
        jax.nn.one_hot(st.task, N_TASKS),
        jnp.array([z / 10]), vel / 10, grav_b, gyro / 3,
        jnp.array([jnp.sin(yaw), jnp.cos(yaw), st.pos[1] / 5, p_tank + n(k[4], 0.01, ())]),
        st.prev_a, _obstacle_feats(st), jnp.array([st.t / MAX_T]),
    ])


def critic_obs(st):
    p = st.params
    clean = _frame(st, jax.random.PRNGKey(0), noisy=False)
    priv = jnp.concatenate([
        jnp.array([p.mass_scale - 1, p.latency / MAX_LAT_STEPS]), (p.thrust_scale - 1) * 5, jnp.array([p.isp_scale - 1]),
        p.tau * 10, p.com_off * 5, p.wind_mean / 5, p.gyro_bias * 10, jnp.array([p.mu]),
        st.u_eff, (st.wind - p.wind_mean) / 5,
        jnp.array([st.gas / GAS_NOM, st.pos[0] / 100, st.airborne_seen.astype(jnp.float32),
                   st.settle_t, st.difficulty]),
    ])
    return jnp.concatenate([clean, priv])


def actor_obs(st):
    return st.obs_hist.reshape(-1)


# ------------------------------------------------------------------ reward
def _potential(st, wheels_any):
    R = quat_to_rot(st.quat)
    up = R[2, 2]
    sink = jnp.where(wheels_any.any(), 0.0, jnp.abs(st.vel[2]) / (1.0 + st.pos[2]))
    return 3.0 * up - sink


def step(st: State, action):
    a = jnp.clip(action, -1.0, 1.0)
    u_cmd = (a[:N_THR] + 1) / 2
    u_cmd = jnp.where(u_cmd < 0.05, 0.0, u_cmd)
    wheel_u = a[N_THR]

    key, k1, k2, k3, kr = jax.random.split(st.key, 5)
    prev = st
    s1, pen1, imp1 = _substep(st, u_cmd, wheel_u, k1)
    s2, pen, imp2 = _substep(s1, u_cmd, wheel_u, k2)
    impact = jnp.maximum(imp1, imp2)
    st = s2._replace(t=st.t + CTRL_DT, key=key, prev_a=a)

    R = quat_to_rot(st.quat)
    up = R[2, 2]
    hull = st.pos + (R @ (HULL - st.params.com_off).T).T
    wheels = st.pos + (R @ (WHEELS - st.params.com_off).T).T
    pts = jnp.concatenate([hull, wheels])
    crash_ground = jnp.any(hull[:, 2] < 0.0)
    inside = ((pts[:, None, 0] >= st.rects[None, :, 0]) & (pts[:, None, 0] <= st.rects[None, :, 1])
              & (jnp.abs(pts[:, None, 1]) <= OB_HALF_W) & (pts[:, None, 2] <= st.rects[None, :, 2]))
    collide = jnp.any(inside)
    hard = impact > MAX_LAND_VZ
    crash = crash_ground | collide | hard

    wheels_down = jnp.all(pen > 0)
    airborne = jnp.all(pen <= 0)
    airborne_seen = st.airborne_seen | airborne
    settled = wheels_down & (up > jnp.cos(MAX_TILT)) & (jnp.abs(st.vel[2]) < 0.5)
    settle_t = jnp.where(settled & airborne_seen, st.settle_t + CTRL_DT, 0.0)
    cleared = jnp.sum(st.rects[:, 1] < st.pos[0] - L / 2).astype(jnp.int32)
    task_done = (st.task != JUMP) | (cleared == 5)
    success = (settle_t >= 0.3) & task_done & ~crash
    corridor = jnp.where(st.task == JUMP, CORRIDOR, 20.0)
    oob = (st.pos[2] > 60) | (st.pos[0] > 400) | (st.pos[0] < -20) | (jnp.abs(st.pos[1]) > corridor)
    timeout = st.t >= MAX_T
    done = success | crash | oob | timeout

    st = st._replace(airborne_seen=airborne_seen, settle_t=settle_t, cleared=cleared)
    pot = _potential(st, pen > 0)
    gas_frac = st.gas / GAS_NOM
    excess = jnp.minimum(jnp.maximum(impact - MAX_LAND_VZ, 0.0) / MAX_LAND_VZ, 2.0)
    r = (pot - prev.pot)
    r += 0.05 * (st.pos[0] - prev.pos[0]) * (st.task == JUMP)
    r += 10.0 * (cleared - prev.cleared)
    r -= 0.002 * u_cmd.sum() + 0.02 * jnp.sum((a - prev.prev_a) ** 2)
    r += success * (100.0 + 50.0 * gas_frac)
    r -= crash * (100.0 + 30.0 * excess)
    r -= (oob | timeout) * ~crash * ~success * 50.0

    frame = _frame(st, k3, noisy=True)
    fbuf = jnp.concatenate([st.frame_buf[1:], frame[None]])
    delayed = fbuf[LAT_BUF - 1 - st.params.latency]
    delayed = delayed.at[PREV_A_SLICE].set(a)          # controller knows its own last command without delay
    st = st._replace(pot=pot, frame_buf=fbuf, obs_hist=jnp.concatenate([st.obs_hist[1:], delayed[None]]),
                     ep_ret=st.ep_ret + r, ep_len=st.ep_len + 1)

    info = dict(done=done, success=success, crash=crash, collide=collide, hard=hard,
                oob=oob & ~crash, timeout=timeout & ~success & ~crash & ~oob, task=st.task,
                gas_frac=gas_frac, ep_ret=st.ep_ret, ep_len=st.ep_len, cleared=cleared)

    fresh = reset(kr, st.difficulty, st.task_probs)
    st = jax.tree.map(lambda x, y: jnp.where(done, y, x), st, fresh)
    return st, r, done, info


# ------------------------------------------------------------- batched API
v_reset = jax.vmap(reset, in_axes=(0, None, None, None))
v_step = jax.vmap(step)
v_actor_obs = jax.vmap(actor_obs)
v_critic_obs = jax.vmap(critic_obs)


def set_curriculum(st: State, difficulty, task_probs):
    n = st.t.shape[0]
    return st._replace(difficulty=jnp.full((n,), difficulty, jnp.float32),
                       task_probs=jnp.broadcast_to(task_probs, (n, N_TASKS)))

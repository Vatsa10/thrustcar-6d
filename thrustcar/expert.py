"""Hand-engineered expert: geometric attitude PD + braking-burn descent + thrust allocation.

Uses true state but nominal vehicle parameters (does not see randomized mass/thrust).
Serves as (a) baseline, (b) DAgger teacher for warm-starting PPO.
"""
import jax
import jax.numpy as jnp

from . import env as E

KP_ATT, KD_ATT = 14.0, 6.0
LIFT_Y, LIFT_X = 0.70, 1.60
SIDE_X = 1.8


def _alloc_lift(T, tx, ty, fmax):
    # thrusters 2:FL(+x,+y) 3:FR(+x,-y) 4:RL(-x,+y) 5:RR(-x,-y); force along body +z at z=-0.3
    sy = jnp.array([1.0, -1.0, 1.0, -1.0])
    sx = jnp.array([1.0, 1.0, -1.0, -1.0])
    d = tx * sy / (4 * LIFT_Y) - ty * sx / (4 * LIFT_X)
    d = d * jnp.minimum(1.0, fmax / (jnp.max(jnp.abs(d)) * 2 + 1e-6))   # torque authority cap
    base = jnp.clip(T / 4, -d.min(), fmax - d.max())                    # torque has priority over lift
    return jnp.clip((base + d) / fmax, 0.0, 1.0)


def _alloc_side(Fy, tz, fmax):
    a = (Fy + tz / SIDE_X) / 2                 # net +y at front
    b = (Fy - tz / SIDE_X) / 2                 # net +y at rear
    f = jnp.array([jnp.maximum(-a, 0), jnp.maximum(a, 0), jnp.maximum(-b, 0), jnp.maximum(b, 0)])
    return jnp.clip(f / fmax, 0.0, 1.0)


def expert_action(st: E.State):
    R = E.quat_to_rot(st.quat)
    m = E.DRY_MASS + st.gas
    Ib = jnp.array([m / 12 * (E.W**2 + E.H**2), m / 12 * (E.L**2 + E.H**2), m / 12 * (E.L**2 + E.W**2)])
    _, reg, _ = E.tank_state(st.gas)
    up = R[2, 2]
    z, vz = st.pos[2], st.vel[2]
    h = z - E.REST_Z

    # ---- attitude error (body frame), keep heading
    b3 = R[:, 2]
    # jump: lateral corridor control by tilting the lift vector (quadrotor-style)
    ay = jnp.clip(-2.0 * st.pos[1] - 2.5 * st.vel[1], -3.0, 3.0) * (st.task == E.JUMP)
    e3 = jnp.array([0.0, ay / E.G, 1.0])
    e3 = e3 / jnp.linalg.norm(e3)
    axis_w = jnp.cross(b3, e3)
    s = jnp.linalg.norm(axis_w)
    ang = jnp.arctan2(s, b3 @ e3)
    axis_b = R.T @ (axis_w / (s + 1e-6))
    # near-inverted: roll about body x, continuing current roll direction
    roll_dir = jnp.where(jnp.abs(st.omega[0]) > 0.2, jnp.sign(st.omega[0]), 1.0)
    axis_b = jnp.where(s < 0.15, jnp.array([1.0, 0.0, 0.0]) * roll_dir, axis_b)
    e_b = axis_b * ang
    tau = Ib * (KP_ATT * e_b - KD_ATT * st.omega)
    tau = tau.at[2].set(-Ib[2] * KD_ATT * st.omega[2])

    # ---- vertical guidance
    land_vz = -jnp.sqrt(1.0 + 2 * 3.5 * jnp.maximum(h, 0.0))
    r = st.rects
    remaining = jnp.arange(5) >= st.cleared
    hmax = jnp.max(jnp.where(remaining, r[:, 2], 0.0))
    nxt = r[jnp.minimum(st.cleared, 4)]
    target = hmax + 0.6 + 1.3
    trigger = st.vel[0] * (jnp.sqrt(2 * target / 10.0) + 0.4) + 5.0
    climbing = (nxt[0] - st.pos[0] < trigger) | (st.pos[0] > r[0, 0] - trigger)
    jumping = (st.task == E.JUMP) & (st.cleared < 5)
    vz_ref = jnp.where(jumping, jnp.where(climbing, jnp.clip(2.0 * (target - z), -2.0, 9.0), 0.0), land_vz)
    want_lift = jnp.where(jumping, climbing, True) & (up > 0.5)
    T = jnp.where(want_lift, m * (E.G + 8.0 * (vz_ref - vz)) / jnp.maximum(up, 0.3), 0.0)
    T = jnp.maximum(T, 0.0)

    # ---- lateral (corridor centering in jump; damp drift otherwise)
    vy_b = (R.T @ st.vel)[1]
    y_err = jnp.where(st.task == E.JUMP, st.pos[1], 0.0)
    Fy = m * (-4.0 * y_err - 4.0 * vy_b) * (up > 0.5)

    u_lift = _alloc_lift(T, tau[0], tau[1], E.THR_FMAX[2] * reg)
    u_side = _alloc_side(Fy, tau[2], E.THR_FMAX[6] * reg)
    u = jnp.concatenate([jnp.zeros(2), u_lift, u_side])
    wheel = jnp.where(st.task == E.JUMP, 1.0, 0.0)
    return jnp.concatenate([2 * u - 1, jnp.array([wheel])])


v_expert = jax.vmap(expert_action)

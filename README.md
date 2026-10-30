# ThrustCar-6D

Pure-JAX 6-DOF RL benchmark for a cold-gas-thruster car (Tesla Roadster gen-2 reveal, 2026-10-15).

- 10 thrusters at real body positions, valve lag + deadband, pressure-regulated blowdown tank (Isp ~ sqrt(T))
- 4 spring-damper wheels w/ friction + drive, anisotropic drag, OU gust wind, IMU noise/bias
- Tasks: hover-land, inverted barrel-roll recovery, 5-vehicle jump (Cybercab/Cybertruck/Semi/Model Y/Model 3)
- Domain randomization: mass, per-thruster thrust, valve lag, Isp, CoM offset, wind, fill, friction (difficulty 0..1, stress 1.5)
- Agent: Phase-MoE actor (gate over K experts), asymmetric privileged critic, DAgger warm start from expert, PPO, success-gated curriculum
- Baselines: geometric-PD + allocation expert, dense actor (param-matched)

```
python -m thrustcar.train --actor moe   --out runs/moe
python -m thrustcar.train --actor dense --out runs/dense
python -m thrustcar.analyze --runs runs/moe runs/dense --out results
```

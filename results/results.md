# ThrustCar-6D results (Colab TPU v5e-1, 200M steps/run, 4096 eval episodes per cell)

Success % (gas left on success, kg). d = domain-randomization difficulty; 1.5 = out-of-distribution stress test.

| policy | d | hover | flip | jump |
|---|---|---|---|---|
| expert (PD + allocation) | 0.0 | 100.0 (193) | 99.8 (144) | 99.6 (112) |
| expert | 1.0 | 99.0 (171) | 89.8 (113) | 48.1 (86) |
| expert | 1.5 | 92.0 (160) | 77.3 (103) | 28.0 (79) |
| **Phase-MoE PPO** | 0.0 | 100.0 (203) | 100.0 (154) | 100.0 (132) |
| **Phase-MoE PPO** | 1.0 | 99.2 (185) | 99.0 (136) | 97.0 (114) |
| **Phase-MoE PPO** | 1.5 | 92.1 (175) | 86.1 (127) | 80.8 (106) |
| Dense PPO (param-matched) | 0.0 | 100.0 (199) | 100.0 (148) | 100.0 (118) |
| Dense PPO | 1.0 | 99.9 (179) | 98.9 (127) | 99.5 (97) |
| Dense PPO | 1.5 | 98.1 (169) | 85.2 (115) | 86.6 (90) |
| MoE PPO from scratch (no DAgger) | any | 0.0 | 0.0 | 0.0 |

Takeaways
- Both learned policies beat the hand-built expert at d=1 (jump 48% -> 97-99.5%) and at the 1.5x stress test.
- MoE vs dense: MoE leaves 6-17 kg more gas; dense is ~2.5 pts more reliable on jump and more robust on hover/jump under stress. No clean MoE win on success yet.
- Pure PPO from scratch: 0% in 200M steps. DAgger warm start + decaying imitation loss is what makes it work.
- MoE gates specialize by flight phase without labels (see moe_gates.png): jump = drive -> climb -> apex/coast -> descent/landing experts.

## v2: + sensor-latency randomization (0-100 ms at d=1, up to 140 ms at d=1.5)
Expert still sees perfect, undelayed state (advantage to the expert).

| policy | d | hover | flip | jump |
|---|---|---|---|---|
| expert (no delay) | 1.0 | 99.0 (171) | 89.8 (113) | 48.1 (86) |
| Phase-MoE PPO | 1.0 | 98.7 (176) | 96.6 (123) | 96.9 (94) |
| Dense PPO | 1.0 | 99.3 (177) | 98.6 (126) | 97.7 (96) |
| expert (no delay) | 1.5 | 92.0 (160) | 77.3 (103) | 28.0 (79) |
| Phase-MoE PPO | 1.5 | 80.3 (164) | 65.1 (112) | 64.9 (86) |
| Dense PPO | 1.5 | 82.3 (165) | 70.8 (114) | 69.3 (87) |

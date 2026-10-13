# Mini Cheetah joint load measurement

Policy: `/home/xjtx/rl/HIMLoco/legged_gym/logs/rough_minich/Sep18_18-36-28_smoke_minich/exported/policies/model_1000.jit`

Clipped joint-side PD torque, pre-integration joint velocity; mechanical power = torque × velocity.
All randomization disabled. Flat ground. No automatic resets. Steady data exclude warmup, ramp and settling.
Group values below are the worst individual motor of the four legs, not their sum.

| Case | Group | RMS torque Nm | P99 torque Nm | Peak torque Nm | Mean abs power W | Peak abs power W |
|---|---|---:|---:|---:|---:|---:|
| stand | hip_abduction | 1.823 | 1.830 | 1.831 | 0.236 | 0.243 |
| stand | thigh | 2.538 | 2.549 | 2.550 | 0.416 | 0.424 |
| stand | calf | 5.058 | 5.064 | 5.064 | 0.488 | 0.510 |
| forward_0.5 | hip_abduction | 3.059 | 3.061 | 3.061 | 0.372 | 0.384 |
| forward_0.5 | thigh | 1.437 | 1.437 | 1.437 | 0.173 | 0.174 |
| forward_0.5 | calf | 6.914 | 6.915 | 6.915 | 0.812 | 0.821 |
| forward_1.0 | hip_abduction | 2.529 | 6.981 | 7.664 | 1.790 | 22.913 |
| forward_1.0 | thigh | 6.362 | 14.781 | 15.242 | 4.866 | 43.675 |
| forward_1.0 | calf | 5.375 | 10.562 | 12.026 | 2.206 | 36.151 |
| backward_0.5 | hip_abduction | 2.744 | 5.865 | 6.527 | 1.328 | 13.049 |
| backward_0.5 | thigh | 4.147 | 9.836 | 10.401 | 4.919 | 37.709 |
| backward_0.5 | calf | 5.465 | 10.623 | 10.947 | 3.246 | 19.859 |
| lateral_0.5 | hip_abduction | 3.104 | 7.745 | 8.119 | 1.751 | 21.979 |
| lateral_0.5 | thigh | 3.320 | 5.684 | 6.064 | 1.306 | 14.264 |
| lateral_0.5 | calf | 6.271 | 8.227 | 8.640 | 1.260 | 9.106 |
| turn_left_1.0 | hip_abduction | 1.610 | 1.613 | 1.615 | 0.133 | 0.143 |
| turn_left_1.0 | thigh | 3.644 | 3.648 | 3.650 | 0.995 | 1.008 |
| turn_left_1.0 | calf | 7.550 | 7.554 | 7.554 | 1.010 | 1.060 |
| turn_right_1.0 | hip_abduction | 0.778 | 0.788 | 0.790 | 0.178 | 0.181 |
| turn_right_1.0 | thigh | 1.825 | 1.828 | 1.829 | 0.285 | 0.296 |
| turn_right_1.0 | calf | 7.808 | 7.810 | 7.811 | 0.425 | 0.445 |

## Tracking and validity

- stand: actual mean vx/vy/yaw = [-0.0014120822306722403, 0.000278630992397666, -0.0006802170537412167]; minimum height = 0.129 m; fell = True; trunk contact peak = 0.000 N.
- forward_0.5: actual mean vx/vy/yaw = [-0.0005303152720443904, -0.0001311654777964577, 0.0010269772028550506]; minimum height = 0.123 m; fell = True; trunk contact peak = 0.000 N.
- forward_1.0: actual mean vx/vy/yaw = [0.07411281764507294, -0.006103808060288429, -0.05402876436710358]; minimum height = 0.112 m; fell = True; trunk contact peak = 0.000 N.
- backward_0.5: actual mean vx/vy/yaw = [-0.09940975904464722, -0.06302750110626221, 0.08680103719234467]; minimum height = 0.154 m; fell = True; trunk contact peak = 0.000 N.
- lateral_0.5: actual mean vx/vy/yaw = [-0.013787631876766682, 0.030674602836370468, -0.12678879499435425]; minimum height = 0.127 m; fell = True; trunk contact peak = 0.000 N.
- turn_left_1.0: actual mean vx/vy/yaw = [-0.0002069591573672369, -0.0007640508702024817, -0.020415501669049263]; minimum height = 0.121 m; fell = True; trunk contact peak = 0.000 N.
- turn_right_1.0: actual mean vx/vy/yaw = [0.000742603384424001, -0.002340571256354451, -0.0029866425320506096]; minimum height = 0.128 m; fell = True; trunk contact peak = 0.000 N.

Mechanical power is not electrical consumption. Zero-speed holding torque still causes motor heating.
These loads apply to this model and policy on flat ground, not jumps, impacts, payload or a redesigned robot.
Torque-speed pairs and saturation must be checked before actuator sizing; joint-side values require transmission mapping.

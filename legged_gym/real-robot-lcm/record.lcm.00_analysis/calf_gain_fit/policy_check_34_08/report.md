# Mini Cheetah joint load measurement

Policy: `/home/xjtx/rl/HIMLoco/legged_gym/logs/rough_minich/Sep18_18-36-28_smoke_minich/exported/policies/model_1000.jit`

Clipped joint-side PD torque, pre-integration joint velocity; mechanical power = torque × velocity.
All randomization disabled. Flat ground. No automatic resets. Steady data exclude warmup, ramp and settling.
Group values below are the worst individual motor of the four legs, not their sum.

| Case | Group | RMS torque Nm | P99 torque Nm | Peak torque Nm | Mean abs power W | Peak abs power W |
|---|---|---:|---:|---:|---:|---:|
| stand | hip_abduction | 3.505 | 3.524 | 3.524 | 0.065 | 0.067 |
| stand | thigh | 5.372 | 5.394 | 5.394 | 0.326 | 0.331 |
| stand | calf | 8.020 | 8.055 | 8.057 | 0.577 | 0.585 |
| forward_0.5 | hip_abduction | 1.593 | 3.454 | 3.693 | 0.564 | 2.624 |
| forward_0.5 | thigh | 4.284 | 8.278 | 8.551 | 5.376 | 18.335 |
| forward_0.5 | calf | 8.955 | 17.770 | 17.908 | 7.164 | 40.002 |
| forward_1.0 | hip_abduction | 2.203 | 3.474 | 3.534 | 0.701 | 4.712 |
| forward_1.0 | thigh | 3.849 | 7.782 | 7.918 | 12.996 | 38.695 |
| forward_1.0 | calf | 11.510 | 21.603 | 21.960 | 11.516 | 61.142 |
| backward_0.5 | hip_abduction | 1.500 | 2.627 | 2.662 | 0.665 | 3.406 |
| backward_0.5 | thigh | 4.296 | 9.504 | 9.725 | 5.772 | 27.983 |
| backward_0.5 | calf | 8.166 | 13.695 | 13.960 | 4.772 | 29.040 |
| lateral_0.5 | hip_abduction | 3.321 | 7.064 | 7.191 | 5.010 | 18.500 |
| lateral_0.5 | thigh | 2.262 | 3.676 | 3.711 | 1.028 | 4.655 |
| lateral_0.5 | calf | 8.982 | 15.921 | 16.072 | 7.629 | 33.469 |
| turn_left_1.0 | hip_abduction | 3.592 | 8.051 | 8.219 | 2.347 | 11.857 |
| turn_left_1.0 | thigh | 2.713 | 7.000 | 7.129 | 1.002 | 6.112 |
| turn_left_1.0 | calf | 10.425 | 17.909 | 18.658 | 3.988 | 27.387 |
| turn_right_1.0 | hip_abduction | 4.115 | 8.658 | 9.640 | 2.185 | 20.536 |
| turn_right_1.0 | thigh | 4.650 | 11.235 | 11.563 | 1.396 | 12.443 |
| turn_right_1.0 | calf | 8.566 | 15.392 | 15.711 | 2.761 | 40.780 |

## Tracking and validity

- stand: actual mean vx/vy/yaw = [2.371589835092891e-05, 6.090056922403164e-05, -0.0012941217282786965]; minimum height = 0.244 m; fell = False; trunk contact peak = 0.000 N.
- forward_0.5: actual mean vx/vy/yaw = [0.48058900237083435, 0.0026113346684724092, -0.0010083683300763369]; minimum height = 0.254 m; fell = False; trunk contact peak = 0.000 N.
- forward_1.0: actual mean vx/vy/yaw = [0.9897056818008423, 0.012179833836853504, 0.019697928801178932]; minimum height = 0.246 m; fell = False; trunk contact peak = 0.000 N.
- backward_0.5: actual mean vx/vy/yaw = [-0.4166164696216583, -0.004170172847807407, -0.0329989530146122]; minimum height = 0.247 m; fell = False; trunk contact peak = 0.000 N.
- lateral_0.5: actual mean vx/vy/yaw = [0.04725032299757004, 0.42118358612060547, 0.002214599633589387]; minimum height = 0.245 m; fell = False; trunk contact peak = 0.000 N.
- turn_left_1.0: actual mean vx/vy/yaw = [0.030668051913380623, 0.029027059674263, 0.9964948892593384]; minimum height = 0.232 m; fell = False; trunk contact peak = 0.000 N.
- turn_right_1.0: actual mean vx/vy/yaw = [0.014007046818733215, 0.004101287107914686, -1.0223944187164307]; minimum height = 0.243 m; fell = False; trunk contact peak = 0.000 N.

Mechanical power is not electrical consumption. Zero-speed holding torque still causes motor heating.
These loads apply to this model and policy on flat ground, not jumps, impacts, payload or a redesigned robot.
Torque-speed pairs and saturation must be checked before actuator sizing; joint-side values require transmission mapping.

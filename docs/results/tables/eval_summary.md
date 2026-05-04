# Cross-algorithm Eval/* summary

_Final-window mean (last 10% of TB samples) per run, per metric._

| Metric | ppo | amp | ars |
|---|---|---|---|
| `Eval/velocity_tracking_error_xy` | 0.6545 | 0.7251 | **0.6248** |
| `Eval/yaw_tracking_error` | **0.2594** | 0.2656 | 0.3218 |
| `Eval/episode_length` | **4** | 4 | 4 |
| `Eval/base_height_std` | **0.0063** | 0.0066 | 0.0070 |
| `Eval/torque_cost` | **758.0** | 786.1 | 793.5 |
| `Eval/action_smoothness` | 0.0619 | **0.0485** | 0.3842 |
| `Eval/foot_slippage` | 0.0332 | **0.0293** | 0.0391 |
| `Eval/velocity_tracking_error_xy_obs_noise` | 0.8171 | 0.6949 | **0.6603** |
| `Eval/velocity_tracking_error_xy_push_5N` | **0.6788** | 0.8701 | 0.7129 |
| `Eval/velocity_tracking_error_xy_push_15N` | **0.7085** | 0.8097 | 0.8386 |

Bold = best run on that metric (lower is better, except `Eval/episode_length` where higher is better).

# RL Algorithms Comparison — Locomotion on Unitree G1

Comparative study of RL algorithm families on the locomotion task for
Unitree G1 in MuJoCo-Warp (`mjlab`).

## 1. Goal

Train a velocity-tracking walking policy for Unitree G1 with several
RL algorithms, then **compare them on common, reward-independent
evaluation metrics** to draw conclusions about per-algorithm strengths
and trade-offs for humanoid locomotion.

## 2. Algorithms

### Core scope (must-do)

| # | Algorithm   | Class                        | Key idea                                                                              |
| - | ----------- | ---------------------------- | ------------------------------------------------------------------------------------- |
| 1 | **PPO**     | deep on-policy               | Clipped surrogate loss + GAE + adaptive LR (current baseline — already trained)        |
| 2 | **PPO+AMP** | deep on-policy + adversarial | Discriminator on (s, s') state pairs, AMP reward replaces hand-crafted style shaping  |
| 3 | **ARS**     | gradient-free                | Augmented Random Search on linear / shallow-MLP policy                                |

### Stretch goal (only if core finishes successfully)

| # | Algorithm   | Class             | Key idea                                                                                    |
| - | ----------- | ----------------- | ------------------------------------------------------------------------------------------- |
| 4 | **SAC**     | deep off-policy   | Soft actor-critic via `skrl` library, twin Q, automatic temperature, replay buffer           |

Out of scope (deferred to future work): Teacher-Student distillation,
TD-MPC2, SHAC/APG, CrossQ, Beyond Mimic, DreamerV3.

## 3. Fixed design decisions

These are locked-in to keep the comparison fair and the implementation
focused.

| Decision           | Choice                                                                                                |
| ------------------ | ----------------------------------------------------------------------------------------------------- |
| Task               | Velocity-command tracking (forward / lateral / yaw), flat ground, current `g1_velocity` env          |
| Robot              | Unitree G1 (29 DoF)                                                                                   |
| Simulator          | `mjlab` / MuJoCo-Warp, 4096 parallel envs (where applicable)                                          |
| Observation space  | **Full privileged** (same as current `g1_velocity` setup) — clean algorithm-vs-algorithm comparison   |
| Action space       | Joint position targets fed through PD controllers                                                     |
| Reward (task part) | Identical for PPO / ARS / (SAC). AMP replaces style-shaping subset.                                  |
| AMP expert data    | LAFAN1 walking clips (~30-60 s of straight + turning + start-stop walks). Single-style for v1.        |
| SAC implementation | `skrl` library — reuse battle-tested SAC, save 2-3 days of integration work                           |
| Network sizes      | PPO/AMP/(SAC): shared default (3×256 MLP). ARS: linear policy first; shallow MLP if linear plateaus. |
| Random seeds       | **1 seed during dev/iteration**, **3 seeds only for final report** (saves ~60% GPU time on the way)   |
| Hardware budget    | 1 GPU on shared server, target ≤25 GPU-hours total                                                    |

## 4. Evaluation metrics (reward-independent)

These are computed by a shared eval-callback, identical across all
algorithms. All metrics are logged to TensorBoard / W&B with identical
tag names so cross-algorithm plots are trivial.

### 4.1 Locomotion quality

- `Eval/velocity_tracking_error_xy` — `mean(|v_xy_actual − v_xy_cmd|)`
- `Eval/yaw_tracking_error` — `mean(|ω_yaw_actual − ω_yaw_cmd|)`
- `Eval/episode_length` — survival in steps (capped at episode_max)
- `Eval/base_height_std` — vertical pelvis stability
- `Eval/torque_cost` — `mean(τ²)` summed over joints
- `Eval/action_smoothness` — `mean((a_t − a_{t-1})²)`
- `Eval/foot_slippage` — foot velocity during contact

### 4.2 Robustness (perturbation tests)

- `Eval/vel_error_obs_noise` — same metric with Gaussian noise on observations
- `Eval/vel_error_push_5N` — with periodic 5 N pushes
- `Eval/vel_error_push_15N` — with periodic 15 N pushes
- `Eval/recovery_time` — time to re-stabilize after push

### 4.3 Sample efficiency

- `Eval/env_steps_to_threshold` — env steps needed to reach `velocity_tracking_error_xy < 0.15 m/s`
- `Eval/wall_clock_to_threshold` — same in wall-clock seconds

### 4.4 Qualitative

- Side-by-side video at fixed eval velocities (0.5 / 1.0 / 1.5 m/s forward, ±0.5 m/s lateral, ±0.5 rad/s yaw)
- Blind 1-5 naturalness rating (optional, only for final report)

## 5. Phased plan

### Phase 0 — Common evaluation harness (foundation)

**Goal**: every subsequent training run produces directly comparable
metrics with zero per-algorithm work.

Deliverables:

- `src/otus_rl_project/eval/locomotion_eval.py` — eval-callback module
- `src/otus_rl_project/eval/perturbations.py` — push / obs-noise wrappers
- `src/otus_rl_project/envs/locomotion_compare.py` — frozen env-cfg `Otus-G1-Walk-Compare`
- Hook into PPO baseline first (sanity check: same numbers as today's training)

**Code budget**: ~250 LoC, ~3-4 hours of dev work
**Dependency for**: every other phase

### Phase 1 — ARS

Cheapest implementation; serves as smoke-test of Phase 0 and as a
"do we actually need deep RL?" baseline.

Deliverables:

- `src/otus_rl_project/algorithms/ars.py` — random-search loop, top-k filtering, observation normalization
- `src/otus_rl_project/train/ars_main.py` + console script `otus-train-ars`
- ARS run (1 seed during iteration, 3 seeds for final)
- Eval-metrics logged to TB

**Code budget**: ~150 LoC, ~2-3 hours of dev work
**GPU**: 4-6 hours per seed

### Phase 2 — PPO + AMP

Highest-value addition, also highest risk (hyperparameter sensitivity).
Largest code component.

Deliverables:

- `src/otus_rl_project/algorithms/amp/` package:
  - `discriminator.py` — MLP discriminator + grad-penalty
  - `expert_buffer.py` — LAFAN1 walking-clip loader and (s, s') sampler
  - `amp_runner.py` — extends `MjlabOnPolicyRunner`, integrates AMP reward
- `src/otus_rl_project/envs/locomotion_amp.py` — same task with AMP-specific obs subset for the discriminator
- `src/otus_rl_project/train/amp_main.py` + console script `otus-train-amp`
- AMP run (1 seed during iteration, 3 seeds for final)

**Code budget**: ~500 LoC, ~6-8 hours of dev work
**GPU**: 3-5 hours per seed
**Tuning**: built into the dev cycle, no separate phase

### Phase 3 (stretch) — SAC via skrl

Off-policy baseline. Only if Phases 0-2 finish on schedule.

Deliverables:

- `src/otus_rl_project/algorithms/sac/` package:
  - `mjlab_skrl_adapter.py` — wraps mjlab `ManagerBasedRlEnv` for `skrl`
  - `sac_config.py` — hyperparameters
- `src/otus_rl_project/train/sac_main.py` + console script `otus-train-sac`
- SAC run (1 seed during iteration, 3 seeds for final)

**Code budget**: ~200 LoC, ~3-4 hours of dev work
**GPU**: 4-8 hours per seed
**Risk**: replay buffer memory cost with 4096 envs may force `num_envs` reduction

### Phase 4 — Final report

Deliverables:

- `docs/results/comparison_report.md` — written analysis with tables, plots, discussion
- `docs/results/figures/` — TB-extracted comparison curves
- `docs/results/videos/` — N× side-by-side MP4s
- ONNX exports of the best policy from each algorithm
- Updated `README.md` linking to the report

**Code budget**: minimal, mostly writing
**Time**: 0.5 day

## 6. Realistic timeline (aggressive but honest)

Targeting ~5-7 working days wall-clock for the **core scope** (PPO+AMP+ARS).
Stretch SAC adds 1-2 days if attempted.

| Day | Activity                                                                  |
| --- | ------------------------------------------------------------------------- |
| 1   | Phase 0 (eval infra) + Phase 1 dev (ARS)                                  |
| 2   | ARS overnight training, Phase 2 dev (AMP) starts                          |
| 3   | AMP-runner finished, AMP overnight training                               |
| 4   | AMP analysis + tuning if needed, finalize 3-seed runs in parallel         |
| 5   | Phase 4 — final report, video, ONNX exports                               |
| 6-7 | Buffer for AMP tuning / unexpected blockers                               |

Stretch SAC adds days 8-9 if pursued.

## 7. Token / chat budget strategy

This study spans multiple days and would burn through chat-credits if
done in a single mega-chat. Strategy to mitigate:

1. **One chat per phase** — fresh context, this `roadmap.md` is the
   source of truth. Don't carry old conversation forward.
2. **Autonomous overnight runs** — orchestrator bash scripts (like the
   previous `night_acro_orchestrator.sh`) launch trainings without
   chat overhead. Morning report.
3. **Compact responses** — answers focus on what's asked, no unsolicited
   educational walls of text.
4. **Status documents** — every phase ends with a small markdown
   summarizing results and next steps. Future chats start by reading
   this, not by re-discussing.

Suggested chat decomposition:

| Chat | Scope                                                | Reason                              |
| ---- | ---------------------------------------------------- | ----------------------------------- |
| 1    | (this one) — planning + roadmap commit               | Closes after roadmap is committed   |
| 2    | Phase 0 + Phase 1 (eval infra + ARS)                 | Tight, mechanical work              |
| 3    | Phase 2 (PPO+AMP)                                    | Largest engineering chunk           |
| 4    | Phase 4 (report) — and optional Phase 3 (SAC)        | Synthesis + optional stretch        |

## 8. Risk register

| Risk                                                 | Likelihood | Mitigation                                                                         |
| ---------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------- |
| AMP fails to stabilize (mode collapse)               | Medium-High| Gradient penalty, lower `D_lr`, fallback to single-clip dataset                    |
| ARS plateaus far below PPO (expected)                | Medium     | Document as feature, not bug — ARS is the "control" baseline                       |
| GPU contention with other server users               | Medium     | Schedule runs overnight; use `nice` / per-night orchestrators                      |
| SAC OOM with 4096 envs + replay buffer (stretch)     | Medium     | Reduce `num_envs` for SAC to 1024; clearly document in report                      |
| skrl ↔ mjlab adapter friction (stretch)              | Medium     | If >1 day spent debugging, drop SAC from this study                                |
| Token budget runs out mid-study                      | Medium     | Per-phase chats + autonomous orchestrators (see §7)                                |

## 9. Out-of-scope (explicit non-goals)

- **No real-robot deployment** in this study — all evaluation in simulation
- **No multi-skill / motion-library** — locomotion only (single-style AMP)
- **No comparison vs Beyond Mimic / TD-MPC2 / DreamerV3** — separate future work
- **No teacher-student distillation** — separate future work, complementary
- **No sim2real gap analysis** — would require partial-obs setup, deferred

## 10. Success criteria

The study is "complete" when:

1. PPO baseline + PPO+AMP + ARS produced working walking policies on the same task ✅
2. All three logged the §4 metrics in a comparable form ✅
3. The report (Phase 4) presents:
   - Quantitative cross-algorithm table
   - Per-algorithm pros / cons section
   - Honest discussion of what each algorithm is and isn't good for
4. ONNX exports + side-by-side video are reproducible from the repo

The study is **not** judged by "PPO vs AMP — who wins?". It is judged by
the depth and honesty of the comparison.

If SAC stretch goal is also delivered, this becomes a 4-way comparison
and the report is updated accordingly. If not, SAC is documented in
"future work" and the 3-way comparison stands as a complete artifact.

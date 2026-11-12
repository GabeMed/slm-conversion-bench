# EXPLORATORY, post hoc: the single call on the small model, calibration (n = 200), 2026-11-12

Runs (scratch worktree llm-to-slm-wt-c1curve, local patch that lets arm B2 take an SLM engine off the test split):
- base Qwen3-8B: agent-B2-slm:qwen3-8b-calib-20261112T064444.401418Z -> 96/200 = 48.0%
- Qwen3-8B + the existing c1 adapter (trained on the agent's prompts): agent-B2-slm:qwen3-8b+lora:c1-fa3a060fdb1c-calib-20261112T065507.607089Z -> 105/200 = 52.5%
No gold error, no missing prediction in either.

Paired differences on the same 200 questions (bootstrap over questions, seed 20260932, 10,000 resamples, 90%):
| contrast | EX | difference | 90% interval | right only in first / only in second |
|---|---|---|---|---|
| single call, base − B3 (agent, base) | 48.0 vs 57.0 | −9.0 | [−14.0, −4.0] | 10 / 28 |
| single call, adapter − B4 (agent, adapters) | 52.5 vs 61.5 | −9.0 | [−14.5, −4.0] | 13 / 31 |
| single call, adapter − B3 | 52.5 vs 57.0 | −4.5 | [−10.0, +1.0] | 18 / 27 |
| single call, base − B2 cheaper LLM (single call) | 48.0 vs 59.0 | −11.0 | [−15.5, −6.5] | 6 / 28 |
| single call, base − B0 (agent, production LLM) | 48.0 vs 69.5 | −21.5 | [−27.5, −16.0] | 7 / 50 |
| single call, adapter − single call, base | 52.5 vs 48.0 | +4.5 | [−1.0, +10.0] | 25 / 16 |

Calibration references: B0 139/200, B4 123/200, B2 cheaper LLM 118/200, B3 114/200. No single-call run of the
production LLM exists on calibration. The calibration gold is the uncorrected one and the split is easier than the
test (more simple questions), so levels are not comparable with the test; contrasts are within calibration only.

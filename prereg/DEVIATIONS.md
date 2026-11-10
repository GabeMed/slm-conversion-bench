# Deviations from the Stage 0 registration

Changes after Stage 0 that can change a result or its interpretation.

- 2026-11-09 · adapters trained on Tinker instead of Modal (the training stack differs; the SLM arms' accuracy
  rests on these adapters); serving on RunPod (same vLLM, weights and GPU, so the outputs are unaffected) at
  RunPod's L40S price instead of Modal's, which changes J8's cost: J8 is also reported at Modal's 2026-09-30
  price, as sensitivity. config.yaml: modal.gpu_prices, the candidates' endpoints, serving.unauthenticated, runpod
- 2026-11-09 · reference GPU changed from L40S to A100 80GB PCIe (RunPod Secure, $1.79/h, CPU and RAM bundled),
  before the zero-shot pilot reading, because L40S was unavailable on RunPod in data centers with network
  volumes; accuracy is unaffected, J8's cost and the SLM's p95 are measured on the A100; J8's sensitivity is
  Modal's price for the same GPU (A100-80GB, $0.000694 a second, 2026-11-09). config.yaml: serving.gpu,
  train.gpu, modal.gpu_prices, runpod.gpu_ids, the triage's footprints
- 2026-11-10 · patch 16: CHESS's agent tool-call parser accepts the tool name inside JSON ({"name": ...} or
  {"tool": ...}), followed by arguments, or closed by </think> (vendor/chess/src/workflow/agents/agent.py); the
  prompt and every other rule are unchanged, for every arm. Upstream accepted only <tool_call>name</tool_call>,
  which non-Claude models (Qwen3-8B, gpt-oss-120b) answered with the right tool in another format and the agent
  then stopped without SQL; plain text, empty tags and unknown tools still fail. B0's calibration pilot and K1 ran
  under the strict parser: 1 of the pilot's 2,198 answered agent calls (agent_cg) and 3 of K1's 9,167 (2 agent_cg, 1
  agent_ss), each a JSON {"name": ...}, would parse under patch 16; no call the strict parser accepted changes
  its tool
- 2026-11-10 · the load test shuffles its payload pool with seeds.loadtest_shuffle (20260935) before the warm-up
  and the concurrency levels are sliced (bench/loadtest.py), decided before any load test ran. In the source's
  file order the six levels (4 to 128, 500 requests each) read only the first 19% of the B0 calibration calls:
  6 to 8 questions and 3 to 4 of the 11 databases per level, 4 databases in all, a different mix at each level;
  shuffled, each level draws 165 to 176 questions from 10 to 11 databases. J8's rule is unchanged.
  config.yaml: seeds.loadtest_shuffle

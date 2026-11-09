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

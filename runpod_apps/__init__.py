"""RunPod as the GPU provider for inference: the vLLM server of one SLM candidate (serve), the AIPerf client of
the load test (loadtest) and P-4's HF-PEFT reference (parity), each on a pod created and terminated through
RunPod's REST API v2 (api), driven over SSH (ssh). Training stays on Tinker (`bench train --on tinker`).

What the pods run is what the Modal apps run: the serving plan and the vLLM command come from
`modal_apps.common`, unchanged. Settings come from config.yaml's `runpod` section (settings), which only this
package reads; configs/runpod.patch.yaml adds it. No `modal` import anywhere here.
"""

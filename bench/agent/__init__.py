"""The agent side of the harness (agent environment only).

LangChain tracing is switched off the moment this package is imported, before anything can invoke a
runnable: LangSmith caches its settings on first use (langsmith.utils.get_env_var is lru-cached), so
switching it off later would be too late. The same switch is applied again when a run prepares CHESS.

onnxruntime's telemetry is switched off here too. Chroma builds its default embedding function (ONNX
MiniLM) when `chromadb.api` is imported, as a default argument, so importing CHESS's vector DB imports
onnxruntime, which starts a telemetry uploader thread at import; that thread can abort the process
when it exits (SIGABRT in its event dispatch). The function computes no vector here (every collection
gets the configured embeddings), and only this variable, read when onnxruntime is imported, keeps the
thread from starting: `onnxruntime.disable_telemetry_events()` leaves it running. Workers are spawned
with this environment, and import this package before anything else of the agent.
"""
import os

# LangSmith takes the first of these it finds (langsmith.utils.get_env_var), so all four are set
TRACING_OFF = {name: "false" for name in ("LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING",
                                          "LANGCHAIN_TRACING")}
os.environ.update(TRACING_OFF)

ONNXRUNTIME_TELEMETRY_OFF = {"ORT_DISABLE_TELEMETRY": "1"}
os.environ.update(ONNXRUNTIME_TELEMETRY_OFF)

"""The agent side of the harness (agent environment only).

LangChain tracing is switched off the moment this package is imported, before anything can invoke a
runnable: LangSmith caches its settings on first use (langsmith.utils.get_env_var is lru-cached), so
switching it off later would be too late. The same switch is applied again when a run prepares CHESS.
"""
import os

# LangSmith takes the first of these it finds (langsmith.utils.get_env_var), so all four are set
TRACING_OFF = {name: "false" for name in ("LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING",
                                          "LANGCHAIN_TRACING")}
os.environ.update(TRACING_OFF)

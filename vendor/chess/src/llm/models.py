from typing import Any, Dict, List

from bench.agent import hooks  # PATCH 2/4/5/6 (slm-conversion-bench): see vendor/chess/PATCHES.md
from threading_utils import ordered_concurrent_function_calls

def get_llm_chain(engine_name: str, temperature: float = 0, base_uri: str = None) -> Any:
    """
    PATCH 2: no model is built here. The harness router (C4) chooses the engine of every call,
    and the temperature of every call comes from the configuration, per call site.
    """
    return hooks.RoutedEngine(engine_name)

def call_llm_chain(prompt: Any, engine: Any, parser: Any, request_kwargs: Dict[str, Any], step: str, invocation_key: str = "single") -> Any:
    """
    PATCH 2/4/5/6: routed, logged (one C1 line per attempt), no fallback to another model,
    and a bounded, logged retry on empty or unparseable output. `step` is the call site.
    """
    messages = prompt.invoke(request_kwargs).to_messages()
    return hooks.invoke_tool_call(call_site=step, invocation_key=invocation_key, lc_messages=messages, parser=parser)

def async_llm_chain_call(
    prompt: Any, 
    engine: Any, 
    parser: Any, 
    request_list: List[Dict[str, Any]], 
    step: str, 
    sampling_count: int = 1,
    invocation_keys: List[str] = None
) -> List[List[Any]]:
    """
    Asynchronously calls the LLM chain using multiple threads.

    Args:
        prompt (Any): The prompt to be passed to the chain.
        engine (Any): The engine to be used in the chain.
        parser (Any): The parser to parse the output.
        request_list (List[Dict[str, Any]]): The list of request arguments.
        step (int): The current step in the process.
        sampling_count (int): The number of samples to be taken.

    Returns:
        List[List[Any]]: A list of lists containing the results for each request.
    """

    call_list = []
    engine_id = 0
    for request_id, request_kwargs in enumerate(request_list):
        for sample_id in range(sampling_count):
            call_list.append({
                'function': call_llm_chain,
                'kwargs': {
                    'prompt': prompt,
                    'engine': engine[engine_id % len(engine)] if isinstance(engine,list) else engine,
                    'parser': parser,
                    'request_kwargs': request_kwargs,
                    'step': step,
                    'invocation_key': _invocation_key(invocation_keys, request_id, sample_id, sampling_count)
                }
            })
            engine_id += 1

    # Execute the functions concurrently
    results = ordered_concurrent_function_calls(call_list)

    # Group results by sampling_count
    grouped_results = [
        results[i * sampling_count: (i + 1) * sampling_count]
        for i in range(len(request_list))
    ]

    return grouped_results

def _invocation_key(invocation_keys: List[str], request_id: int, sample_id: int, sampling_count: int) -> str:
    """PATCH 6: the stable identity of one invocation across runs (C1 `invocation_key`)."""
    key = invocation_keys[request_id] if invocation_keys else "single"
    return f"{key}#{sample_id}" if sampling_count > 1 else key

def call_engine(message: str, engine: Any, call_site: str, invocation_key: str, parse: Any) -> Any:
    """
    PATCH 2/6: the agent's choice of the next tool, routed and logged. `parse` is the agent's
    own reading of the response, recorded as `parsed_output`; the agent's behaviour is unchanged.
    """
    return hooks.invoke_agent_call(call_site=call_site, invocation_key=invocation_key, message=message, parse=parse)

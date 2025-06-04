from typing import Optional

from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

"""Default prompts."""

client = Client()


# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

_FALLBACK_PROMPTS = {
    "margot-na/input_guardrail": "You are a helpful assistant. Politely refuse"
    " any request that is out of scope or malicious.",
    "margot-na/router": (
        "Classify the user's message as 'langchain' if it is about the LangChain"
        " project, 'more-info' if additional details are required, or 'general'"
        " for everything else. Respond only with the classification and a short"
        " explanation."
    ),
    "margot-na/generate-queries": (
        "Generate a list of search queries that would help answer the user's question."
    ),
    "margot-na/more_info": (
        "Ask the user for the additional information described in {logic}."
    ),
    "margot-na/researcher": "Create a step-by-step plan for researching the"
    " user's question.",
    "margot-na/irrelevant_response": (
        "Provide a general helpful answer based on the user's question and the"
        " logic {logic}."
    ),
    "margot-na/synthesizer": (
        "Using the context below, craft a concise answer for the user:\n\n{context}"
    ),
}


def _fetch_prompt(prompt_id: str) -> Optional[str]:
    """Safely fetch a prompt from LangSmith.

    If the prompt cannot be retrieved, return a fallback value and log a warning.
    """

    try:
        return client.pull_prompt(prompt_id).messages[0].prompt.template
    except Exception as exc:  # pragma: no cover - best effort fetch
        print(f"Warning: could not fetch prompt '{prompt_id}': {exc}")
        return _FALLBACK_PROMPTS.get(prompt_id)


INPUT_GUARDRAIL_SYSTEM_PROMPT = _fetch_prompt("margot-na/input_guardrail")
ROUTER_SYSTEM_PROMPT = _fetch_prompt("margot-na/router")
GENERATE_QUERIES_SYSTEM_PROMPT = _fetch_prompt("margot-na/generate-queries")
MORE_INFO_SYSTEM_PROMPT = _fetch_prompt("margot-na/more_info")
RESEARCH_PLAN_SYSTEM_PROMPT = _fetch_prompt("margot-na/researcher")
GENERAL_SYSTEM_PROMPT = _fetch_prompt("margot-na/irrelevant_response")
RESPONSE_SYSTEM_PROMPT = _fetch_prompt("margot-na/synthesizer")

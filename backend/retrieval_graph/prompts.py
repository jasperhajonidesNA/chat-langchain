from langsmith import Client

"""Default prompts."""

client = Client()
# fetch from langsmith
INPUT_GUARDRAIL_SYSTEM_PROMPT = (
    client.pull_prompt("margot-na/input_guardrail")
    .messages[0]
    .prompt.template
)

ROUTER_SYSTEM_PROMPT = (
    client.pull_prompt("margot-na/router")
    .messages[0]
    .prompt.template
)
GENERATE_QUERIES_SYSTEM_PROMPT = (
    client.pull_prompt("langchain-ai/chat-langchain-generate-queries-prompt")
    .messages[0]
    .prompt.template
)

RESEARCH_PLAN_SYSTEM_PROMPT = (
    client.pull_prompt("margot-na/researcher")
    .messages[0]
    .prompt.template
)

IRRELEVANT_QUERY_SYSTEM_PROMPT = (
    client.pull_prompt("margot-na/respond_irrelevant_query")
    .messages[0]
    .prompt.template
)

RESPONSE_SYSTEM_PROMPT = (
    client.pull_prompt("margot-na/synthesizer")
    .messages[0]
    .prompt.template
)

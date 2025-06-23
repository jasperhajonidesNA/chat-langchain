"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.configuration import BaseConfiguration
from backend.retrieval_graph import prompts


@dataclass(kw_only=True)
class AgentConfiguration(BaseConfiguration):
    """The configuration for the agent."""

    # models
    input_guardrail_model: str = field(
        default="openai/gpt-4.1",
        metadata={
            "description": "The language model used for the input guardrail. Should be in the form: provider/model-name."
        },
    )

    query_model: str = field(
        # default="anthropic/claude-3-5-haiku-20241022",
        default="openai/gpt-4.1",
        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name."
        },
    )

    response_model: str = field(
        # default="anthropic/claude-3-5-haiku-20241022",
        default="openai/gpt-4.1",
        metadata={
            "description": "The language model used for generating responses. Should be in the form: provider/model-name."
        },
    )

    # prompts
    input_guardrail_system_prompt: str = field(
        default=prompts.INPUT_GUARDRAIL_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for the input guardrail."
        },
    )

    router_system_prompt: str = field(
        default=prompts.ROUTER_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for classifying user questions to route them to the correct node."
        },
    )

<<<<<<< HEAD
    irrelevant_query_system_prompt: str = field(
        default=prompts.IRRELEVANT_QUERY_SYSTEM_PROMPT,
=======
    more_info_system_prompt: str = field(
        default=prompts.MORE_INFO_SYSTEM_PROMPT,
>>>>>>> 0824da9 (backend up to date)
        metadata={
            "description": "The system prompt used when a user asks a question that is unrelated."
        },
    )

<<<<<<< HEAD
=======
    general_system_prompt: str = field(
        default=prompts.GENERAL_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for responding to general questions."
        },
    )

>>>>>>> 0824da9 (backend up to date)
    research_plan_system_prompt: str = field(
        default=prompts.RESEARCH_PLAN_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for generating a research plan based on the user's question."
        },
    )

    generate_queries_system_prompt: str = field(
        default=prompts.GENERATE_QUERIES_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used by the researcher to generate queries based on a step in the research plan."
        },
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses."},
    )

    guardrail_failure_message: str = field(
        default="""I apologize, but I'm unable to process your request. I’m here to help with any questions or clarifications you might have regarding NatureAlpha’s data or methodology. If there’s something specific you’d like to discuss or understand better, please let me know—I'm happy to assist!""",
        metadata={"description": "The message shown to users when the input guardrail fails."},
    )

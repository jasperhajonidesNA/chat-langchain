"""Researcher graph used in the conversational retrieval system as a subgraph.

This module defines the core structure and functionality of the researcher graph,
which is responsible for breaking down queries into sub-queries and consulting
methodology experts instead of traditional RAG retrieval.
"""

from typing import cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from backend.retrieval_graph.configuration import AgentConfiguration
from backend.retrieval_graph.researcher_graph.state import ResearcherState, SubQuery, SubQueryState
from backend.utils import load_chat_model


async def generate_sub_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[SubQuery]]:
    """Generate sub-queries with methodology assignments based on the research question.

    This function uses a language model to break down the research question into
    focused sub-queries and assign each to the appropriate methodology expert.

    Args:
        state (ResearcherState): The current state of the researcher, including the user's question.
        config (RunnableConfig): Configuration with the model used to generate sub-queries.

    Returns:
        dict[str, list[SubQuery]]: A dictionary with a 'sub_queries' key containing the list of sub-queries.
    """

    class SubQueryResponse(TypedDict):
        sub_queries: list[dict]

    configuration = AgentConfiguration.from_runnable_config(config)
    structured_output_kwargs = (
        {"method": "function_calling"} if "openai" in configuration.query_model else {}
    )
    model = load_chat_model(configuration.query_model).with_structured_output(
        SubQueryResponse, **structured_output_kwargs
    )
    messages = [
        {"role": "system", "content": configuration.subquery_generation_prompt},
        {"role": "human", "content": state.question},
    ]
    response = cast(
        SubQueryResponse, await model.ainvoke(messages, {"tags": ["langsmith:nostream"]})
    )
    
    # Convert dictionary responses to SubQuery objects
    sub_queries = [
        SubQuery(
            question=sq["question"],
            rationale=sq["rationale"],
            methodology=sq["methodology"]
        )
        for sq in response["sub_queries"]
    ]
    
    return {"sub_queries": sub_queries}


async def consult_methodology_expert(
    state: SubQueryState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Consult the appropriate methodology expert for a specific sub-query.

    This function routes the sub-query to the correct methodology expert based on
    the assigned methodology and returns the expert's response.

    Args:
        state (SubQueryState): The current state containing the sub-query and methodology.
        config (RunnableConfig): Configuration with the model and methodology documentation.

    Returns:
        dict[str, list[str]]: A dictionary with 'methodology_responses' containing the expert's answer.
    """
    configuration = AgentConfiguration.from_runnable_config(config)
    model = load_chat_model(configuration.query_model)
    
    # Select the appropriate expert prompt and documentation
    methodology = state.sub_query.methodology
    
    if methodology == "footprint":
        prompt_template = configuration.footprint_expert_prompt
        methodology_docs = configuration.footprint_methodology_docs
    elif methodology == "nature_sense":
        prompt_template = configuration.nature_sense_expert_prompt
        methodology_docs = configuration.naturesense_methodology_docs
    elif methodology == "nrevx":
        prompt_template = configuration.nrevx_expert_prompt
        methodology_docs = configuration.nrevx_methodology_docs
    elif methodology == "unmanaged_risk":
        prompt_template = configuration.unmanaged_risk_expert_prompt
        methodology_docs = configuration.unmanaged_risk_methodology_docs
    elif methodology == "nature_risk":
        prompt_template = configuration.nature_risk_expert_prompt
        methodology_docs = configuration.nature_risk_methodology_docs
    else:
        raise ValueError(f"Unknown methodology: {methodology}")
    
    # Format the prompt with the sub-query and methodology documentation
    if methodology == "nature_sense":
        system_prompt = prompt_template.format(
            sub_query=state.sub_query.question,
            naturesense_md=methodology_docs
        )
    elif methodology == "footprint":
        system_prompt = prompt_template.format(
            sub_query=state.sub_query.question,
            footprint_md=methodology_docs
        )
    elif methodology == "nrevx":
        system_prompt = prompt_template.format(
            sub_query=state.sub_query.question,
            nrevx_md=methodology_docs
        )
    elif methodology == "unmanaged_risk":
        system_prompt = prompt_template.format(
            sub_query=state.sub_query.question,
            unmanaged_risk_md=methodology_docs
        )
    elif methodology == "nature_risk":
        system_prompt = prompt_template.format(
            sub_query=state.sub_query.question,
            nature_risk_md=methodology_docs
        )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "human", "content": f"Please answer this sub-query: {state.sub_query.question}"},
    ]
    
    response = await model.ainvoke(messages)
    return {"methodology_responses": [response.content]}


def consult_experts_in_parallel(state: ResearcherState) -> list[Send]:
    """Create parallel methodology expert consultation tasks for each sub-query.

    This function prepares parallel consultation tasks for each sub-query in the researcher's state.

    Args:
        state (ResearcherState): The current state of the researcher, including the sub-queries.

    Returns:
        list[Send]: A list of Send objects, each representing a methodology expert consultation task.

    Behavior:
        - Creates a Send object for each sub-query in the state.
        - Each Send object targets the "consult_methodology_expert" node with the corresponding sub-query.
    """
    return [
        Send("consult_methodology_expert", SubQueryState(sub_query=sub_query))
        for sub_query in state.sub_queries
    ]


# Define the graph
builder = StateGraph(ResearcherState)
builder.add_node("generate_sub_queries", generate_sub_queries)
builder.add_node("consult_methodology_expert", consult_methodology_expert)

# Flow: Start -> Generate sub-queries -> Consult experts in parallel -> End
builder.add_edge(START, "generate_sub_queries")
builder.add_conditional_edges(
    "generate_sub_queries",
    consult_experts_in_parallel,  # type: ignore
    path_map=["consult_methodology_expert"],
)
builder.add_edge("consult_methodology_expert", END)

# Compile into a graph object that you can invoke and deploy.
graph = builder.compile()
graph.name = "ResearcherGraph"

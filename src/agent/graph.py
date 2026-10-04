import os
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .prompts import (
    DATA_SPECIALIST_PROMPT,
    STATISTICS_SPECIALIST_PROMPT,
    SUPERVISOR_PROMPT,
    VISUALIZATION_SPECIALIST_PROMPT,
)
from src.tools.analysis import (
    correlation_matrix,
    describe_data,
    detect_outliers,
    group_statistics,
    statistical_test,
    value_counts,
)
from src.tools.code_executor import execute_python
from src.tools.data_loader import (
    get_column_info,
    list_datasets,
    load_csv,
    load_excel,
    preview_data,
)
from src.tools.feature_analysis import (
    feature_correlation_with_target,
    feature_importance,
    suggest_features,
)
from src.tools.visualization import (
    plot_boxplot,
    plot_correlation_heatmap,
    plot_distribution,
    plot_histogram,
    plot_pairplot,
    plot_scatter,
)

MAX_SPECIALIST_CALLS = 6


class WinnowState(MessagesState):
    """Conversation state shared by the supervisor and specialist nodes."""

    next_agent: str
    task: str
    specialist_results: list[dict[str, str]]
    specialist_calls: int


class SupervisorDecision(BaseModel):
    next_agent: Literal["data", "statistics", "visualization", "finish"] = Field(
        description="Which graph node should run next; use finish when ready to answer."
    )
    task: str = Field(
        default="",
        description="A self-contained task for the selected specialist, including the dataset name and context.",
    )
    answer: str = Field(
        default="",
        description="The final response to the user when next_agent is finish.",
    )


def _final_agent_text(result) -> str:
    for message in reversed(result.get("messages", [])):
        if getattr(message, "type", None) != "ai" or getattr(message, "tool_calls", None):
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        return str(content)
    return "The specialist finished without returning a text result."


def build_graph():
    groq_api_key = os.getenv("GROQ_API_KEY")

    if groq_api_key:
        # Groq provides an OpenAI-compatible endpoint.
        model_name = os.getenv("GROQ_MODEL_NAME") or os.getenv("OPENAI_MODEL_NAME") or "qwen/qwen3.8-27b"
        llm = ChatOpenAI(
            model=model_name,
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
            temperature=0.6,
        )
    else:
        # Fallback to OpenAI, NVIDIA NIM, or another OpenAI-compatible endpoint.
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL_NAME") or os.getenv("NVIDIA_MODEL") or "meta/llama-3.3-70b-instruct"
        llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=0,
        )

    # These are runnable subgraphs. Each sees only its specialist tools and a
    # task from the supervisor, so it starts with a focused context each time.
    data_agent = create_agent(
        llm,
        tools=[load_csv, load_excel, list_datasets, preview_data, get_column_info, describe_data],
        system_prompt=DATA_SPECIALIST_PROMPT,
    )
    statistics_agent = create_agent(
        llm,
        tools=[
            preview_data,
            get_column_info,
            correlation_matrix,
            value_counts,
            group_statistics,
            statistical_test,
            detect_outliers,
            feature_importance,
            feature_correlation_with_target,
            suggest_features,
            execute_python,
        ],
        system_prompt=STATISTICS_SPECIALIST_PROMPT,
    )
    visualization_agent = create_agent(
        llm,
        tools=[
            preview_data,
            get_column_info,
            plot_histogram,
            plot_scatter,
            plot_correlation_heatmap,
            plot_boxplot,
            plot_pairplot,
            plot_distribution,
        ],
        system_prompt=VISUALIZATION_SPECIALIST_PROMPT,
    )

    supervisor_model = llm.with_structured_output(SupervisorDecision)

    def supervisor_node(state: WinnowState):
        # A completed prior turn marks the start of a fresh routing cycle.
        is_new_turn = state.get("next_agent") in (None, "finish")
        specialist_results = [] if is_new_turn else state.get("specialist_results", [])
        specialist_calls = 0 if is_new_turn else state.get("specialist_calls", 0)
        findings = "\n\n".join(
            f"{item['agent']} task: {item['task']}\nResult: {item['result']}"
            for item in specialist_results
        ) or "No specialist has run for this user turn yet."

        supervisor_prompt = SUPERVISOR_PROMPT
        if specialist_calls >= MAX_SPECIALIST_CALLS:
            supervisor_prompt += "\nYou have reached the delegation limit. Choose finish and summarize the available findings."

        decision = supervisor_model.invoke([
            SystemMessage(content=supervisor_prompt),
            *state["messages"],
            HumanMessage(content=f"Specialist findings for the current user request:\n{findings}"),
        ])

        if decision.next_agent == "finish":
            answer = decision.answer.strip() or "What would you like to explore about your data?"
            return {
                "messages": [{"role": "assistant", "content": answer}],
                "next_agent": "finish",
                "task": "",
                "specialist_results": [],
                "specialist_calls": 0,
            }

        latest_message = state["messages"][-1]
        latest_content = getattr(latest_message, "content", "")
        fallback_task = latest_content if isinstance(latest_content, str) else str(latest_content)
        return {
            "next_agent": decision.next_agent,
            "task": decision.task.strip() or fallback_task,
            "specialist_results": specialist_results,
            "specialist_calls": specialist_calls,
        }

    def make_specialist_node(agent_name: str, agent):
        def specialist_node(state: WinnowState):
            result = agent.invoke({"messages": [("user", state["task"])]})
            entry = {
                "agent": agent_name,
                "task": state["task"],
                "result": _final_agent_text(result),
            }
            return {
                "specialist_results": [*state.get("specialist_results", []), entry],
                "specialist_calls": state.get("specialist_calls", 0) + 1,
            }

        return specialist_node

    def route_from_supervisor(state: WinnowState) -> str:
        return state["next_agent"]

    graph = StateGraph(WinnowState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("data", make_specialist_node("data", data_agent))
    graph.add_node("statistics", make_specialist_node("statistics", statistics_agent))
    graph.add_node("visualization", make_specialist_node("visualization", visualization_agent))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "data": "data",
            "statistics": "statistics",
            "visualization": "visualization",
            "finish": END,
        },
    )
    graph.add_edge("data", "supervisor")
    graph.add_edge("statistics", "supervisor")
    graph.add_edge("visualization", "supervisor")

    return graph.compile(checkpointer=MemorySaver())

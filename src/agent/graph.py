import os
import threading
import time
from typing import Annotated, Literal, TypedDict

from deepagents import CompiledSubAgent, create_deep_agent
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Overwrite, Send, interrupt
from pydantic import BaseModel, Field

from .prompts import (
    DATA_SPECIALIST_PROMPT,
    ANALYSIS_SPECIALIST_PROMPT,
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
    DATASETS,
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
from src.utils.logging_config import get_logger

MAX_SPECIALIST_TASKS = 4
MAX_REPLANS = 1
logger = get_logger("winnow.workflow")


def merge_results(left: dict[str, dict[str, str]], right: dict[str, dict[str, str]]):
    return {**(left or {}), **(right or {})}


class WinnowState(MessagesState):
    """Workflow state shared by the planner and specialist graph nodes."""

    user_request: str
    intent: str
    entities: list[str]
    needs_clarification: bool
    clarification_question: str
    clarification: str
    tasks: list[dict]
    specialist_results: Annotated[dict[str, dict[str, str]], merge_results]
    synthesis: str
    validation_feedback: str
    valid: bool
    replan_count: int
    answer: str
    workflow_mode: str
    thread_id: str


class SpecialistInput(TypedDict):
    step_id: str
    agent_name: str
    task: str
    thread_id: str


class Understanding(BaseModel):
    intent: str = Field(description="The user's goal in a short phrase.")
    entities: list[str] = Field(default_factory=list, description="Dataset names, columns, groups, metrics, or other named entities.")
    workflow_mode: Literal["dag", "deep_agent"] = Field(
        description="Use dag for a focused request with a known outcome; deep_agent for broad, open-ended exploratory work requiring self-directed planning."
    )
    needs_clarification: bool = Field(description="True only if a required detail cannot reasonably be inferred.")
    clarification_question: str = Field(default="", description="One concise question that resolves the blocking ambiguity.")


class PlanTask(BaseModel):
    step_id: str = Field(description="A short unique identifier for this task within the plan.")
    specialist: Literal["data", "statistics", "visualization", "analysis"]
    task: str = Field(description="A self-contained task with dataset and user-request context.")
    depends_on: list[str] = Field(default_factory=list, description="Step IDs that must complete before this task can run.")


class WorkflowPlan(BaseModel):
    tasks: list[PlanTask] = Field(description="Independent specialist tasks to execute in parallel; use an empty list if no tools are needed.")


class Validation(BaseModel):
    valid: bool = Field(description="Whether the final response is supported by specialist results and answers the user's request.")
    feedback: str = Field(default="", description="Specific unsupported claim, omission, or correction needed when invalid.")


def _message_content(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content)


def _final_agent_text(result) -> str:
    for message in reversed(result.get("messages", [])):
        if getattr(message, "type", None) != "ai" or getattr(message, "tool_calls", None):
            continue
        text = _message_content(message)
        if text:
            return text
    return "The specialist finished without returning a text result."


def _is_provider_token_limit(error: BaseException) -> bool:
    """Identify provider 413s caused by an input-token throughput limit."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(current, "status_code", None) or getattr(response, "status_code", None)
        try:
            payload_too_large = int(status) == 413
        except (TypeError, ValueError):
            payload_too_large = False
        message = str(current).lower()
        if payload_too_large and "token" in message and (
            "rate_limit_exceeded" in message or "tokens per minute" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _available_dataset_context() -> str:
    if not DATASETS:
        return "No datasets are currently loaded."
    return "\n".join(
        f"- {name}: {len(frame)} rows, {len(frame.columns)} columns; columns: "
        f"{', '.join(str(column) for column in frame.columns)}"
        for name, frame in DATASETS.items()
    )


class WorkflowToolTraceHandler(BaseCallbackHandler):
    """Log tool names and timing without recording tool inputs or outputs."""

    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self._runs = {}
        self._lock = threading.Lock()

    def on_tool_start(self, serialized, input_str=None, *, run_id, **kwargs):
        tool_name = (serialized or {}).get("name", "unknown")
        call_id = str(run_id).replace("-", "")[:8]
        with self._lock:
            self._runs[run_id] = (tool_name, time.perf_counter(), call_id)
        logger.info(
            "event=tool_started thread_id=%s tool=%s call_id=%s",
            self.thread_id, tool_name, call_id,
        )

    def on_tool_end(self, output, *, run_id, **kwargs):
        with self._lock:
            tool_name, started, call_id = self._runs.pop(
                run_id, ("unknown", time.perf_counter(), "unknown")
            )
        logger.info(
            "event=tool_completed thread_id=%s tool=%s call_id=%s result_chars=%d duration_ms=%d",
            self.thread_id, tool_name, call_id, len(str(output)),
            int((time.perf_counter() - started) * 1000),
        )

    def on_tool_error(self, error, *, run_id, **kwargs):
        with self._lock:
            tool_name, started, call_id = self._runs.pop(
                run_id, ("unknown", time.perf_counter(), "unknown")
            )
        logger.error(
            "event=tool_failed thread_id=%s tool=%s call_id=%s error_type=%s duration_ms=%d",
            self.thread_id, tool_name, call_id, type(error).__name__,
            int((time.perf_counter() - started) * 1000),
        )


def build_graph():
    # Re-read WINNOW_LOG_LEVEL after callers load their .env file.
    get_logger("winnow.workflow")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if groq_api_key:
        model_name = os.getenv("GROQ_MODEL_NAME") or os.getenv("OPENAI_MODEL_NAME") or "qwen/qwen3.8-27b"
        llm = ChatOpenAI(
            model=model_name,
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
            temperature=0.6,
            max_retries=4,
        )
    else:
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL_NAME") or os.getenv("NVIDIA_MODEL") or "meta/llama-3.3-70b-instruct"
        llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=0,
            max_retries=4,
        )

    # Specialist subgraphs own their tools; the outer workflow owns orchestration.
    specialists = {
        "data": create_agent(
            llm,
            tools=[load_csv, load_excel, list_datasets, preview_data, get_column_info, describe_data],
            system_prompt=DATA_SPECIALIST_PROMPT,
        ),
        "statistics": create_agent(
            llm,
            tools=[
                preview_data, get_column_info, correlation_matrix, value_counts,
                group_statistics, statistical_test, detect_outliers,
                feature_importance, feature_correlation_with_target,
                suggest_features,
            ],
            system_prompt=STATISTICS_SPECIALIST_PROMPT,
        ),
        "visualization": create_agent(
            llm,
            tools=[
                preview_data, get_column_info, plot_histogram, plot_scatter,
                plot_correlation_heatmap, plot_boxplot, plot_pairplot,
                plot_distribution,
            ],
            system_prompt=VISUALIZATION_SPECIALIST_PROMPT,
        ),
        "analysis": create_agent(
            llm,
            tools=[execute_python, preview_data, get_column_info],
            system_prompt=ANALYSIS_SPECIALIST_PROMPT,
        ),
    }

    understand_model = llm.with_structured_output(Understanding)
    plan_model = llm.with_structured_output(WorkflowPlan)
    synthesis_model = llm
    validation_model = llm.with_structured_output(Validation)

    # The open-ended path gets its own planning/filesystem harness, but delegates
    # to the exact same compiled specialist workers used by the DAG path.
    deep_agent = create_deep_agent(
        model=llm,
        system_prompt=(
            "You are Winnow's deep exploratory analyst. For broad, open-ended requests, break the work into "
            "clear steps, track them with your todo list, and use your virtual files to keep concise notes and "
            "intermediate findings. Delegate data inspection, statistical work, chart creation, and custom "
            "sandboxed analysis to the matching specialist subagents. The uploaded dataset is accessible through "
            "those specialists. Do not fabricate findings; return a concise report that synthesizes worker results "
            "and calls out limitations and generated artifact paths."
        ),
        subagents=[
            CompiledSubAgent(
                name=name,
                description=description,
                runnable=specialists[name],
            )
            for name, description in [
                ("data", "Load, inspect, and describe the uploaded dataset."),
                ("statistics", "Run focused statistical tests, correlations, group summaries, or feature analysis."),
                ("visualization", "Create requested visualizations and return the saved chart path."),
                ("analysis", "Run custom data analysis in the sandboxed Python tool."),
            ]
        ],
        middleware=[TodoListMiddleware()],
    )

    def understand_node(state: WinnowState):
        started = time.perf_counter()
        request = next(
            (_message_content(message) for message in reversed(state["messages"]) if getattr(message, "type", None) == "human"),
            "",
        )
        logger.info(
            "event=understand_started thread_id=%s message_chars=%d loaded_datasets=%d",
            state.get("thread_id", "unknown"), len(request), len(DATASETS),
        )
        try:
            understanding = understand_model.invoke([
                SystemMessage(content=(
                    "Understand the user's latest request. Ask a clarification only when an essential detail "
                    "is missing and proceeding would risk analyzing the wrong data or producing the wrong output. "
                    "Treat datasets listed as currently loaded as available in this workspace. If exactly one is "
                    "loaded, use it as the active dataset unless the user names another. Never ask the user to "
                    "confirm or re-upload a dataset that appears in the loaded dataset list. When several are loaded, "
                    "use the one explicitly named by the user, or clarify only if the choice materially changes the result. "
                    "Do not ask about preferences that can be handled with a sensible default."
                    " Choose workflow_mode='dag' for a specific, bounded question or requested operation; choose "
                    "'deep_agent' for broad, open-ended exploration or a request that needs self-directed multi-step investigation."
                )),
                HumanMessage(content=(
                    f"User request: {request}\n"
                    f"Currently loaded datasets:\n{_available_dataset_context()}"
                )),
            ])
        except Exception:
            logger.exception("event=understand_failed thread_id=%s", state.get("thread_id", "unknown"))
            raise
        dataset_question = any(
            token in understanding.clarification_question.lower()
            for token in ("dataset", "file", "upload", "workspace", "location", "re-upload", "reupload")
        )
        request_lower = request.lower()
        explicit_loaded_dataset = any(name.lower() in request_lower for name in DATASETS)
        implicit_single_dataset = len(DATASETS) == 1 and any(
            phrase in request_lower
            for phrase in ("uploaded dataset", "uploaded file", "this dataset", "current dataset", "the dataset")
        )
        needs_clarification = understanding.needs_clarification and not (
            dataset_question and (explicit_loaded_dataset or implicit_single_dataset)
        )
        logger.info(
            "event=understand_completed thread_id=%s workflow_mode=%s needs_clarification=%s entities=%d duration_ms=%d",
            state.get("thread_id", "unknown"), understanding.workflow_mode,
            needs_clarification, len(understanding.entities),
            int((time.perf_counter() - started) * 1000),
        )
        return {
            "user_request": request,
            "intent": understanding.intent,
            "entities": understanding.entities,
            "workflow_mode": understanding.workflow_mode,
            "needs_clarification": needs_clarification,
            "clarification_question": understanding.clarification_question if needs_clarification else "",
            "clarification": "",
            "specialist_results": Overwrite({}),
            "tasks": [],
            "synthesis": "",
            "validation_feedback": "",
            "valid": False,
            "replan_count": 0,
        }

    def route_after_understanding(state: WinnowState) -> str:
        if state.get("needs_clarification"):
            destination = "clarify"
        else:
            destination = state.get("workflow_mode", "dag")
        logger.info(
            "event=route_selected thread_id=%s stage=understand destination=%s",
            state.get("thread_id", "unknown"), destination,
        )
        return destination

    def clarify_node(state: WinnowState):
        logger.info(
            "event=clarification_paused thread_id=%s",
            state.get("thread_id", "unknown"),
        )
        answer = interrupt({"question": state["clarification_question"]})
        logger.info(
            "event=clarification_resumed thread_id=%s answer_chars=%d",
            state.get("thread_id", "unknown"), len(str(answer)),
        )
        return {"clarification": str(answer), "needs_clarification": False}

    def route_after_clarification(state: WinnowState) -> str:
        destination = state.get("workflow_mode", "dag")
        logger.info(
            "event=route_selected thread_id=%s stage=clarify destination=%s",
            state.get("thread_id", "unknown"), destination,
        )
        return destination

    def plan_node(state: WinnowState):
        started = time.perf_counter()
        logger.info(
            "event=plan_started thread_id=%s replan_count=%d",
            state.get("thread_id", "unknown"), state.get("replan_count", 0),
        )
        previous_results = "\n".join(
            f"{item['specialist']} ({name}): {item['result']}"
            for name, item in state.get("specialist_results", {}).items()
        ) or "No findings yet."
        try:
            plan = plan_model.invoke([
                SystemMessage(content=(
                    f"{SUPERVISOR_PROMPT}\n"
                    "Create a plan for the current request. Select only the specialist agents that add value. "
                    "Return a dependency DAG using unique step_id values and depends_on step IDs. Independent tasks "
                    "should have no dependencies so they can run concurrently. Use at most four tasks, and make "
                    "each task self-contained. The data specialist loads or inspects data; statistics does "
                    "quantitative analysis; analysis runs custom sandboxed Python; visualization creates charts. "
                    "For a broad initial exploration, plan data inspection first when the dataset is not yet known, "
                    "then suitable statistical analysis, sandboxed code analysis, and charts. Incorporate the clarification and validation feedback. "
                    "Every dependency must refer to another step_id in this plan. Keep the graph acyclic. "
                    "Do not repeat completed work unless validation feedback requires it."
                )),
                HumanMessage(content=(
                    f"User request: {state.get('user_request', '')}\n"
                    f"Clarification: {state.get('clarification', '')}\n"
                    f"Entities: {state.get('entities', [])}\n"
                    f"Currently loaded datasets:\n{_available_dataset_context()}\n"
                    f"Validation feedback: {state.get('validation_feedback', '')}\n"
                    f"Prior findings: {previous_results}"
                )),
            ])
        except Exception:
            logger.exception("event=plan_failed thread_id=%s", state.get("thread_id", "unknown"))
            raise
        tasks = [task.model_dump() for task in plan.tasks[:MAX_SPECIALIST_TASKS]]
        valid_ids = {task["step_id"] for task in tasks}
        tasks = [
            task for task in tasks
            if task["step_id"] and all(dependency in valid_ids for dependency in task["depends_on"])
        ]
        logger.info(
            "event=plan_completed thread_id=%s task_count=%d tasks=%s duration_ms=%d",
            state.get("thread_id", "unknown"), len(tasks),
            [(task["step_id"], task["specialist"], task["depends_on"]) for task in tasks],
            int((time.perf_counter() - started) * 1000),
        )
        return {"tasks": tasks}

    def dispatch(state: WinnowState):
        tasks = state.get("tasks", [])
        results = state.get("specialist_results", {})
        completed_ids = set(results)
        pending = [task for task in tasks if task["step_id"] not in completed_ids]
        ready = [
            task for task in pending
            if set(task.get("depends_on", [])).issubset(completed_ids)
        ]
        if not ready:
            logger.info(
                "event=dispatch_empty thread_id=%s pending_count=%d",
                state.get("thread_id", "unknown"), len(pending),
            )
            return "synthesize"
        logger.info(
            "event=dispatch_ready thread_id=%s ready=%s pending_count=%d completed_count=%d",
            state.get("thread_id", "unknown"),
            [(task["step_id"], task["specialist"]) for task in ready],
            len(pending), len(completed_ids),
        )
        return [
            Send("specialist", {
                "step_id": task["step_id"],
                "agent_name": task["specialist"],
                "task": task["task"],
                "thread_id": state.get("thread_id", "unknown"),
            })
            for task in ready
        ]

    def specialist_node(state: SpecialistInput):
        name = state["agent_name"]
        started = time.perf_counter()
        logger.info(
            "event=specialist_started thread_id=%s step_id=%s specialist=%s task_chars=%d",
            state.get("thread_id", "unknown"), state["step_id"], name, len(state["task"]),
        )
        try:
            result = specialists[name].invoke(
                {"messages": [("user", state["task"])]},
                config={"callbacks": [WorkflowToolTraceHandler(state.get("thread_id", "unknown"))]},
            )
            finding = _final_agent_text(result)
        except Exception as exc:
            if _is_provider_token_limit(exc):
                logger.warning(
                    "event=specialist_token_limit thread_id=%s step_id=%s specialist=%s",
                    state.get("thread_id", "unknown"), state["step_id"], name,
                )
            else:
                logger.exception(
                    "event=specialist_failed thread_id=%s step_id=%s specialist=%s",
                    state.get("thread_id", "unknown"), state["step_id"], name,
                )
            raise
        logger.info(
            "event=specialist_completed thread_id=%s step_id=%s specialist=%s result_chars=%d duration_ms=%d",
            state.get("thread_id", "unknown"), state["step_id"], name,
            len(finding), int((time.perf_counter() - started) * 1000),
        )
        return {
            "specialist_results": {
                state["step_id"]: {
                    "specialist": name,
                    "task": state["task"],
                    "result": finding,
                }
            }
        }

    def deep_agent_node(state: WinnowState):
        started = time.perf_counter()
        task = state.get("user_request", "")
        if state.get("clarification"):
            task += f"\nUser clarification: {state['clarification']}"
        task += f"\n\nCurrently loaded datasets:\n{_available_dataset_context()}"
        task += (
            "\nUse the exact loaded dataset name when delegating to a worker. "
            "These datasets are already available; do not ask the user to re-upload them."
        )
        logger.info("event=deep_agent_started thread_id=%s", state.get("thread_id", "unknown"))
        try:
            result = deep_agent.invoke(
                {"messages": [("user", task)]},
                config={"callbacks": [WorkflowToolTraceHandler(state.get("thread_id", "unknown"))]},
            )
            finding = _final_agent_text(result)
        except Exception as exc:
            if _is_provider_token_limit(exc):
                logger.warning("event=deep_agent_token_limit thread_id=%s", state.get("thread_id", "unknown"))
            else:
                logger.exception("event=deep_agent_failed thread_id=%s", state.get("thread_id", "unknown"))
            raise
        logger.info(
            "event=deep_agent_completed thread_id=%s result_chars=%d duration_ms=%d",
            state.get("thread_id", "unknown"), len(finding),
            int((time.perf_counter() - started) * 1000),
        )
        return {
            "specialist_results": {
                "deep_agent": {
                    "specialist": "deep_agent",
                    "task": task,
                    "result": finding,
                }
            }
        }

    def collect_node(state: WinnowState):
        # The reducer has already merged this batch's results; this node is the join point.
        logger.info(
            "event=collect_completed thread_id=%s result_count=%d result_ids=%s",
            state.get("thread_id", "unknown"), len(state.get("specialist_results", {})),
            list(state.get("specialist_results", {})),
        )
        return {}

    def synthesize_node(state: WinnowState):
        started = time.perf_counter()
        logger.info(
            "event=synthesis_started thread_id=%s finding_count=%d",
            state.get("thread_id", "unknown"), len(state.get("specialist_results", {})),
        )
        findings = "\n\n".join(
            f"{item['specialist']} ({name}) — task: {item['task']}\nFinding: {item['result']}"
            for name, item in state.get("specialist_results", {}).items()
        ) or "No specialist tools were needed for this response."
        try:
            response = synthesis_model.invoke([
                SystemMessage(content=(
                    "You are Winnow's synthesis node. Answer the user's request using the supplied specialist findings. "
                    "Do not invent data, calculations, citations, or artifacts. Say when evidence is missing. "
                    "Preserve relevant caveats and mention generated chart paths."
                )),
                HumanMessage(content=(
                    f"Request: {state.get('user_request', '')}\n"
                    f"Clarification: {state.get('clarification', '')}\nFindings:\n{findings}"
                )),
            ])
        except Exception:
            logger.exception("event=synthesis_failed thread_id=%s", state.get("thread_id", "unknown"))
            raise
        synthesis = _message_content(response)
        logger.info(
            "event=synthesis_completed thread_id=%s answer_chars=%d duration_ms=%d",
            state.get("thread_id", "unknown"), len(synthesis),
            int((time.perf_counter() - started) * 1000),
        )
        return {"synthesis": synthesis}

    def validate_node(state: WinnowState):
        started = time.perf_counter()
        logger.info("event=validation_started thread_id=%s", state.get("thread_id", "unknown"))
        findings = "\n\n".join(
            f"{item['specialist']}: {item['result']}"
            for name, item in state.get("specialist_results", {}).items()
        )
        try:
            result = validation_model.invoke([
                SystemMessage(content=(
                    "Check whether the proposed response answers the request and whether every factual or numeric "
                    "data claim is supported by the specialist findings. If unsupported or incomplete, return invalid "
                    "with a concise actionable correction. Do not reject merely for reasonable wording choices."
                )),
                HumanMessage(content=(
                    f"Request: {state.get('user_request', '')}\n"
                    f"Findings:\n{findings or 'No specialist findings.'}\n"
                    f"Proposed answer:\n{state.get('synthesis', '')}"
                )),
            ])
        except Exception:
            logger.exception("event=validation_failed thread_id=%s", state.get("thread_id", "unknown"))
            raise
        logger.info(
            "event=validation_completed thread_id=%s valid=%s feedback_chars=%d duration_ms=%d",
            state.get("thread_id", "unknown"), result.valid, len(result.feedback),
            int((time.perf_counter() - started) * 1000),
        )
        return {"valid": result.valid, "validation_feedback": result.feedback}

    def route_after_validation(state: WinnowState) -> str:
        if state.get("valid") or state.get("replan_count", 0) >= MAX_REPLANS:
            destination = "finish"
        else:
            destination = "replan"
        logger.info(
            "event=route_selected thread_id=%s stage=validate destination=%s",
            state.get("thread_id", "unknown"), destination,
        )
        return destination

    def replan_node(state: WinnowState):
        logger.warning(
            "event=replan thread_id=%s attempt=%d feedback_chars=%d",
            state.get("thread_id", "unknown"), state.get("replan_count", 0) + 1,
            len(state.get("validation_feedback", "")),
        )
        return {
            "replan_count": state.get("replan_count", 0) + 1,
            "specialist_results": Overwrite({}),
            "synthesis": "",
        }

    def finish_node(state: WinnowState):
        answer = state.get("synthesis", "").strip()
        if not answer:
            answer = "I couldn't produce a supported answer from the available findings."
        if not state.get("valid") and state.get("validation_feedback"):
            answer += f"\n\nI could not fully verify this response: {state['validation_feedback']}"
        logger.info(
            "event=workflow_finished thread_id=%s valid=%s answer_chars=%d",
            state.get("thread_id", "unknown"), state.get("valid", False), len(answer),
        )
        return {"answer": answer, "messages": [{"role": "assistant", "content": answer}]}

    graph = StateGraph(WinnowState)
    graph.add_node("understand", understand_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("deep_agent", deep_agent_node)
    graph.add_node("specialist", specialist_node)
    graph.add_node("collect", collect_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("validate", validate_node)
    graph.add_node("replan", replan_node)
    graph.add_node("finish", finish_node)

    graph.add_edge(START, "understand")
    graph.add_conditional_edges("understand", route_after_understanding, {
        "clarify": "clarify", "dag": "plan", "deep_agent": "deep_agent",
    })
    graph.add_conditional_edges("clarify", route_after_clarification, {
        "dag": "plan", "deep_agent": "deep_agent",
    })
    graph.add_conditional_edges("plan", dispatch, ["specialist", "synthesize"])
    graph.add_edge("specialist", "collect")
    graph.add_conditional_edges("collect", dispatch, ["specialist", "synthesize"])
    graph.add_edge("deep_agent", "collect")
    graph.add_edge("synthesize", "validate")
    graph.add_conditional_edges("validate", route_after_validation, {"replan": "replan", "finish": "finish"})
    graph.add_edge("replan", "plan")
    graph.add_edge("finish", END)

    return graph.compile(checkpointer=MemorySaver())

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    loaded_datasets: dict[str, str]
    current_dataset: str
    output_dir: str

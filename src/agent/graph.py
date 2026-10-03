import os
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from .prompts import SYSTEM_PROMPT
from src.tools.data_loader import load_csv, load_excel, list_datasets, preview_data, get_column_info
from src.tools.analysis import describe_data, correlation_matrix, value_counts, group_statistics, statistical_test, detect_outliers
from src.tools.visualization import plot_histogram, plot_scatter, plot_correlation_heatmap, plot_boxplot, plot_pairplot, plot_distribution
from src.tools.feature_analysis import feature_importance, feature_correlation_with_target, suggest_features
from src.tools.code_executor import execute_python

def build_graph():
    groq_api_key = os.getenv("GROQ_API_KEY")

    if groq_api_key:
        # Groq provides a 100% OpenAI-compatible endpoint at https://api.groq.com/openai/v1
        model_name = os.getenv("GROQ_MODEL_NAME") or os.getenv("OPENAI_MODEL_NAME") or "qwen/qwen3.8-27b"
        llm = ChatOpenAI(
            model=model_name,
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
            temperature=0.6
        )
    else:
        # Fallback to OpenAI / NVIDIA NIM or OpenAI-compatible endpoint
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL_NAME") or os.getenv("NVIDIA_MODEL") or "meta/llama-3.3-70b-instruct"

        llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=0
        )
    tools = [
        load_csv, load_excel, list_datasets, preview_data, get_column_info,
        describe_data, correlation_matrix, value_counts, group_statistics, statistical_test, detect_outliers,
        plot_histogram, plot_scatter, plot_correlation_heatmap, plot_boxplot, plot_pairplot, plot_distribution,
        feature_importance, feature_correlation_with_target, suggest_features,
        execute_python
    ]
    
    memory = MemorySaver()
    agent_executor = create_react_agent(
        llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory
    )
    return agent_executor

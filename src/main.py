import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when running `python src/main.py` directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from src.agent.graph import build_graph
from src.tools.data_loader import DATASETS
from src.utils.phoenix_tracing import flush_phoenix, initialize_phoenix, phoenix_trace_context

load_dotenv()
console = Console()

def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    api_key = groq_api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: No API key found (GROQ_API_KEY, NVIDIA_API_KEY, or OPENAI_API_KEY)[/red]")
        console.print("Please copy .env.example to .env and set your API key.")
        sys.exit(1)
        
    if groq_api_key:
        base_url = "https://api.groq.com/openai/v1"
        model_name = os.getenv("GROQ_MODEL_NAME") or os.getenv("OPENAI_MODEL_NAME") or "qwen/qwen3.8-27b"
        provider = "Groq"
    else:
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        model_name = os.getenv("OPENAI_MODEL_NAME") or os.getenv("NVIDIA_MODEL") or "meta/llama-3.3-70b-instruct"
        provider = "NVIDIA NIM"

    console.print(Panel.fit(
        f"[bold green]📊 Winnow[/bold green]\n"
        f"Powered by LangGraph ReAct + [cyan]{model_name}[/cyan] ({provider})\n"
        f"Endpoint: [dim]{base_url}[/dim]\n\n"
        "Commands: [cyan]/quit[/cyan] · [cyan]/clear[/cyan] · [cyan]/datasets[/cyan]",
        border_style="green"
    ))
    
    initialize_phoenix()
    agent = build_graph()
    config = {"configurable": {"thread_id": "session_1"}}
    
    while True:
        try:
            user_input = console.input("\n[bold blue]You:[/bold blue] ")
            if not user_input.strip():
                continue
                
            if user_input.lower() == "/quit":
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break
            elif user_input.lower() == "/clear":
                config["configurable"]["thread_id"] = os.urandom(8).hex()
                console.print("[yellow]Session cleared.[/yellow]")
                continue
            elif user_input.lower() == "/datasets":
                if not DATASETS:
                    console.print("[yellow]No datasets loaded yet.[/yellow]")
                else:
                    for name, df in DATASETS.items():
                        console.print(f"  [cyan]{name}[/cyan]: {df.shape[0]} rows × {df.shape[1]} cols")
                continue
            
            inputs = {
                "messages": [("user", user_input)],
                "thread_id": config["configurable"]["thread_id"],
            }
            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                response_received = False
                with phoenix_trace_context(config["configurable"]["thread_id"], "cli"):
                    for event in agent.stream(inputs, config=config, stream_mode="values"):
                        message = event["messages"][-1]
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tc in message.tool_calls:
                                console.print(f"  [dim]🔧 Calling {tc['name']}...[/dim]")
                        if message.type == "ai" and message.content:
                            console.print(f"\n[bold magenta]Winnow:[/bold magenta]")
                            console.print(Markdown(message.content))
                            response_received = True

            if not response_received:
                console.print("[dim yellow](No text response generated)[/dim yellow]")
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {type(e).__name__}: {str(e)}[/red]")

    flush_phoenix()

if __name__ == "__main__":
    main()

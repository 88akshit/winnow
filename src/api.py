import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.agent.graph import build_graph
from src.tools.data_loader import DATASETS, load_csv, load_excel

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}

load_dotenv(ROOT_DIR / ".env")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Winnow API", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str = Field(min_length=1, max_length=128)


def get_agent():
    global agent
    if agent is None:
        if not any(os.getenv(key) for key in ("GROQ_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY")):
            raise HTTPException(status_code=503, detail="Add a provider API key to winnow/.env first.")
        agent = build_graph()
    return agent


def message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content)


def final_answer(result: dict) -> str:
    messages = result.get("messages", [])
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai" and not getattr(message, "tool_calls", None):
            text = message_text(message)
            if text:
                return text
    return "Analysis finished, but Winnow did not return a text summary."


def dataset_details(dataset_name: str, file_type: str) -> dict:
    frame = DATASETS[dataset_name]
    sample = frame.head(8).astype(object).where(pd.notna(frame.head(8)), None)
    return {
        "name": dataset_name,
        "file_type": file_type.lstrip("."),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": [str(column) for column in frame.columns],
        "preview": sample.to_dict(orient="records"),
    }


def recent_plots(previous: set[str]) -> list[str]:
    current = {
        path.name for path in OUTPUT_DIR.glob("*.png")
        if path.is_file() and path.name not in previous
    }
    return [f"/api/plots/{name}" for name in sorted(current)]


@app.get("/api/health")
async def health():
    return {"status": "ok", "configured": any(
        os.getenv(key) for key in ("GROQ_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY")
    )}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    original_name = Path(file.filename or "dataset").name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Upload a CSV or XLSX file.")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Files must be 25 MB or smaller.")

    stem = re.sub(r"[^\w-]+", "_", Path(original_name).stem, flags=re.UNICODE).strip("_")[:36]
    dataset_name = f"{stem or 'dataset'}_{uuid.uuid4().hex[:8]}"
    saved_path = DATA_DIR / f"{dataset_name}{extension}"
    saved_path.write_bytes(contents)

    if extension == ".csv":
        loaded = await run_in_threadpool(load_csv.invoke, {"file_path": str(saved_path), "name": dataset_name})
    else:
        with pd.ExcelFile(saved_path) as workbook:
            sheet_name = workbook.sheet_names[0]
        loaded = await run_in_threadpool(load_excel.invoke, {
            "file_path": str(saved_path), "sheet_name": sheet_name, "name": dataset_name,
        })
    if not loaded.startswith("Successfully loaded"):
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=loaded)

    return {
        "thread_id": uuid.uuid4().hex,
        "dataset": dataset_details(dataset_name, extension),
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    plots_before = {path.name for path in OUTPUT_DIR.glob("*.png")}
    result = await run_in_threadpool(
        get_agent().invoke,
        {"messages": [("user", request.message)]},
        {"configurable": {"thread_id": request.thread_id}},
    )
    return {"assistant": final_answer(result), "plots": recent_plots(plots_before)}


@app.get("/api/plots/{filename}")
async def get_plot(filename: str):
    if Path(filename).name != filename or not filename.endswith(".png"):
        raise HTTPException(status_code=404, detail="Plot not found.")
    plot_path = (OUTPUT_DIR / filename).resolve()
    if plot_path.parent != OUTPUT_DIR.resolve() or not plot_path.is_file():
        raise HTTPException(status_code=404, detail="Plot not found.")
    return FileResponse(plot_path, media_type="image/png")


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

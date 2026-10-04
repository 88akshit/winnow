import os
import re
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.agent.graph import build_graph
from src.tools.data_loader import DATASETS, load_csv, load_excel
from src.utils.logging_config import get_logger

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
logger = get_logger("winnow.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("event=api_startup_restore_started data_dir=%s", DATA_DIR)
    restored = await run_in_threadpool(load_persisted_datasets)
    logger.info("event=api_startup_restore_completed restored_count=%d", restored)
    yield


app = FastAPI(title="Winnow API", lifespan=lifespan)


def load_persisted_datasets():
    """Rebuild the in-memory dataset registry after API startup or reload."""
    restored = 0
    for saved_path in DATA_DIR.iterdir():
        if not saved_path.is_file() or saved_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        dataset_name = saved_path.stem
        if dataset_name in DATASETS:
            continue
        try:
            if saved_path.suffix.lower() == ".csv":
                result = load_csv.invoke({"file_path": str(saved_path), "name": dataset_name})
            else:
                with pd.ExcelFile(saved_path) as workbook:
                    if not workbook.sheet_names:
                        logger.warning("event=dataset_restore_skipped dataset=%s reason=no_sheets", dataset_name)
                        continue
                    sheet_name = workbook.sheet_names[0]
                result = load_excel.invoke({
                    "file_path": str(saved_path),
                    "sheet_name": sheet_name,
                    "name": dataset_name,
                })
            if not result.startswith("Successfully loaded"):
                logger.warning("event=dataset_restore_failed dataset=%s error=%s", dataset_name, result)
            else:
                restored += 1
                frame = DATASETS[dataset_name]
                logger.info(
                    "event=dataset_restored dataset=%s rows=%d columns=%d",
                    dataset_name, len(frame), len(frame.columns),
                )
        except Exception:
            logger.exception("event=dataset_restore_failed dataset=%s", dataset_name)
    return restored


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str = Field(min_length=1, max_length=128)


class ClarificationRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=8000)
    thread_id: str = Field(min_length=1, max_length=128)


def get_agent():
    global agent
    if agent is None:
        if not any(os.getenv(key) for key in ("GROQ_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY")):
            raise HTTPException(status_code=503, detail="Add a provider API key to winnow/.env first.")
        logger.info("event=workflow_graph_build_started")
        agent = build_graph()
        logger.info("event=workflow_graph_build_completed")
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


def pending_clarification(result: dict) -> str | None:
    for item in result.get("__interrupt__", []):
        value = getattr(item, "value", None)
        if isinstance(value, dict) and value.get("question"):
            return str(value["question"])
    return None


async def run_chat(message_or_resume, thread_id: str, resume: bool = False):
    started = time.perf_counter()
    logger.info(
        "event=chat_started thread_id=%s action=%s message_chars=%d",
        thread_id, "resume" if resume else "invoke", len(str(message_or_resume)),
    )
    plots_before = {path.name for path in OUTPUT_DIR.glob("*.png")}
    if resume:
        graph_input = Command(resume=message_or_resume)
    else:
        graph_input = {"messages": [("user", message_or_resume)], "thread_id": thread_id}
    try:
        result = await run_in_threadpool(
            get_agent().invoke,
            graph_input,
            {"configurable": {"thread_id": thread_id}},
        )
    except Exception as exc:
        retry_after = rate_limit_retry_after(exc)
        if retry_after is not None:
            detail = (
                f"The AI provider is receiving too many requests. Please wait {retry_after} seconds and try again."
                if retry_after else
                "The AI provider is receiving too many requests. Please wait a moment and try again."
            )
            logger.warning(
                "event=chat_rate_limited thread_id=%s retry_after=%s action=%s",
                thread_id, retry_after or "unspecified", "resume" if resume else "invoke",
            )
            headers = {"Retry-After": str(retry_after)} if retry_after else None
            raise HTTPException(status_code=429, detail=detail, headers=headers) from exc
        token_limit = provider_token_limit_error(exc)
        if token_limit:
            logger.warning(
                "event=chat_token_limit thread_id=%s action=%s",
                thread_id, "resume" if resume else "invoke",
            )
            raise HTTPException(
                status_code=413,
                detail=(
                    "This request exceeded the AI provider's input-token limit. "
                    "Try a shorter or narrower question, or configure a model with a higher token allowance."
                ),
            ) from exc
        logger.exception("event=chat_failed thread_id=%s action=%s", thread_id, "resume" if resume else "invoke")
        raise
    response = {
        "assistant": final_answer(result),
        "clarification": pending_clarification(result),
        "plots": recent_plots(plots_before),
    }
    logger.info(
        "event=chat_completed thread_id=%s interrupted=%s plots=%d duration_ms=%d",
        thread_id, bool(response["clarification"]), len(response["plots"]),
        int((time.perf_counter() - started) * 1000),
    )
    return response


def provider_token_limit_error(error: BaseException) -> bool:
    """Recognize provider 413 responses that report token throughput limits."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(current, "status_code", None) or getattr(response, "status_code", None)
        try:
            is_payload_too_large = int(status) == 413
        except (TypeError, ValueError):
            is_payload_too_large = False
        if is_payload_too_large:
            # Provider error bodies differ; inspect only for known token-limit markers.
            body = getattr(current, "body", None)
            response_text = getattr(response, "text", "")
            message = f"{current} {body or ''} {response_text or ''}".lower()
            return "token" in message and (
                "rate_limit_exceeded" in message
                or "tokens per minute" in message
                or "token limit" in message
            )
        current = current.__cause__ or current.__context__
    return False


def rate_limit_retry_after(error: BaseException) -> int | None:
    """Return the provider's Retry-After value when an exception is a 429."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(current, "status_code", None) or getattr(response, "status_code", None)
        try:
            is_rate_limited = int(status) == 429
        except (TypeError, ValueError):
            is_rate_limited = False
        if is_rate_limited or current.__class__.__name__ == "RateLimitError":
            headers = getattr(response, "headers", None) or getattr(current, "headers", None) or {}
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw is None:
                raw_ms = headers.get("retry-after-ms") or headers.get("Retry-After-Ms")
                try:
                    return max(1, (int(float(raw_ms)) + 999) // 1000) if raw_ms is not None else 0
                except (TypeError, ValueError, OverflowError):
                    return 0
            try:
                return max(1, int(float(raw) + 0.999))
            except (TypeError, ValueError, OverflowError):
                try:
                    retry_at = parsedate_to_datetime(str(raw))
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    return max(1, int((retry_at - datetime.now(timezone.utc)).total_seconds() + 0.999))
                except (TypeError, ValueError, OverflowError):
                    return 0
        current = current.__cause__ or current.__context__
    return None


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
    logger.info(
        "event=upload_started dataset=%s extension=%s bytes=%d",
        dataset_name, extension, len(contents),
    )
    saved_path = DATA_DIR / f"{dataset_name}{extension}"
    saved_path.write_bytes(contents)

    try:
        if extension == ".csv":
            loaded = await run_in_threadpool(load_csv.invoke, {"file_path": str(saved_path), "name": dataset_name})
        else:
            with pd.ExcelFile(saved_path) as workbook:
                if not workbook.sheet_names:
                    raise ValueError("The workbook has no sheets.")
                sheet_name = workbook.sheet_names[0]
            loaded = await run_in_threadpool(load_excel.invoke, {
                "file_path": str(saved_path), "sheet_name": sheet_name, "name": dataset_name,
            })
    except Exception:
        logger.exception("event=upload_failed dataset=%s", dataset_name)
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file could not be read.")
    if not loaded.startswith("Successfully loaded"):
        logger.warning("event=upload_failed dataset=%s error=%s", dataset_name, loaded)
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=loaded)

    frame = DATASETS[dataset_name]
    logger.info(
        "event=upload_completed dataset=%s rows=%d columns=%d",
        dataset_name, len(frame), len(frame.columns),
    )

    return {
        "thread_id": uuid.uuid4().hex,
        "dataset": dataset_details(dataset_name, extension),
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    return await run_chat(request.message, request.thread_id)


@app.post("/api/chat/resume")
async def resume_chat(request: ClarificationRequest):
    if not request.answer.strip():
        raise HTTPException(status_code=400, detail="Clarification answer cannot be empty.")
    return await run_chat(request.answer, request.thread_id, resume=True)


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

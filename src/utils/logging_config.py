import logging
import os
import re
import sys


EVENT_LABELS = {
    "tool_started": ("TOOL START", "\033[96m"),
    "tool_completed": ("TOOL DONE", "\033[92m"),
    "tool_failed": ("TOOL FAILED", "\033[91m"),
    "understand_started": ("UNDERSTAND", "\033[94m"),
    "understand_completed": ("UNDERSTANDING READY", "\033[92m"),
    "understand_failed": ("UNDERSTAND FAILED", "\033[91m"),
    "route_selected": ("ROUTE", "\033[94m"),
    "clarification_paused": ("WAITING FOR CLARIFICATION", "\033[93m"),
    "clarification_resumed": ("CLARIFICATION RECEIVED", "\033[92m"),
    "plan_started": ("PLANNING", "\033[93m"),
    "plan_completed": ("PLAN READY", "\033[92m"),
    "plan_failed": ("PLAN FAILED", "\033[91m"),
    "dispatch_ready": ("DISPATCH", "\033[96m"),
    "dispatch_empty": ("DISPATCH COMPLETE", "\033[94m"),
    "specialist_started": ("SPECIALIST START", "\033[96m"),
    "specialist_completed": ("SPECIALIST DONE", "\033[92m"),
    "specialist_failed": ("SPECIALIST FAILED", "\033[91m"),
    "specialist_token_limit": ("SPECIALIST TOKEN LIMIT", "\033[93m"),
    "deep_agent_started": ("DEEP AGENT START", "\033[96m"),
    "deep_agent_completed": ("DEEP AGENT DONE", "\033[92m"),
    "deep_agent_failed": ("DEEP AGENT FAILED", "\033[91m"),
    "deep_agent_token_limit": ("DEEP AGENT TOKEN LIMIT", "\033[93m"),
    "collect_completed": ("RESULTS COLLECTED", "\033[92m"),
    "synthesis_started": ("SYNTHESIZING", "\033[95m"),
    "synthesis_completed": ("SYNTHESIS READY", "\033[92m"),
    "synthesis_failed": ("SYNTHESIS FAILED", "\033[91m"),
    "validation_started": ("VALIDATING", "\033[95m"),
    "validation_completed": ("VALIDATION RESULT", "\033[92m"),
    "validation_failed": ("VALIDATION FAILED", "\033[91m"),
    "replan": ("REPLANNING", "\033[93m"),
    "workflow_finished": ("WORKFLOW FINISHED", "\033[92m"),
    "api_startup_restore_started": ("RESTORING UPLOADS", "\033[94m"),
    "api_startup_restore_completed": ("UPLOAD RESTORE COMPLETE", "\033[92m"),
    "dataset_restored": ("DATASET RESTORED", "\033[92m"),
    "dataset_restore_failed": ("DATASET RESTORE FAILED", "\033[91m"),
    "dataset_restore_skipped": ("DATASET SKIPPED", "\033[93m"),
    "upload_started": ("UPLOAD START", "\033[96m"),
    "upload_completed": ("UPLOAD READY", "\033[92m"),
    "upload_failed": ("UPLOAD FAILED", "\033[91m"),
    "workflow_graph_build_started": ("BUILDING WORKFLOW", "\033[94m"),
    "workflow_graph_build_completed": ("WORKFLOW READY", "\033[92m"),
    "chat_started": ("CHAT START", "\033[96m"),
    "chat_completed": ("CHAT DONE", "\033[92m"),
    "chat_failed": ("CHAT FAILED", "\033[91m"),
    "chat_rate_limited": ("PROVIDER RATE LIMITED", "\033[93m"),
    "chat_token_limit": ("PROVIDER TOKEN LIMIT", "\033[93m"),
}


class WinnowFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        has_tty = getattr(sys.stderr, "isatty", lambda: False)()
        self.use_color = bool(os.getenv("FORCE_COLOR")) or (has_tty and "NO_COLOR" not in os.environ)

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        match = re.match(r"event=([a-z_]+)\s*(.*)", message)
        if match:
            event, details = match.groups()
            label, color = EVENT_LABELS.get(
                event,
                (event.replace("_", " ").upper(), "\033[0m"),
            )
            details = re.sub(r"\bthread_id=", "thread=", details)
            details = re.sub(r"\bduration_ms=", "took_ms=", details)
            details = re.sub(r"\bresult_chars=", "output_chars=", details)
            details = re.sub(r"\bmessage_chars=", "input_chars=", details)
            details = re.sub(r"\btask_chars=", "input_chars=", details)
            marker = f"*** {label} ***"
            if self.use_color:
                marker = f"{color}{marker}\033[0m"
            timestamp = self.formatTime(record, self.datefmt)
            output = f"{timestamp} {record.levelname:<7} {marker}"
            if details:
                output += f" {details}"
        else:
            timestamp = self.formatTime(record, self.datefmt)
            output = f"{timestamp} {record.levelname:<7} {message}"

        if record.exc_info:
            output += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            output += "\n" + self.formatStack(record.stack_info)
        return output


def get_logger(name: str) -> logging.Logger:
    """Return a Winnow logger with a predictable console format."""
    logger = logging.getLogger(name)
    configured_level = os.getenv("WINNOW_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, configured_level, logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(WinnowFormatter())
        logger.addHandler(handler)
    return logger

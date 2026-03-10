
import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

# Define log directory
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Debug mode from environment (set ENML_DEBUG=1 to see DEBUG on console)
ENML_DEBUG = os.getenv("ENML_DEBUG", "0") == "1"


class JsonFormatter(logging.Formatter):
    """Outputs structured JSON lines for machine-parseable audit logs."""
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno
        }
        # Include pipeline stage if present in the message
        msg = record.getMessage()
        for stage in ("[ROUTE]", "[RETRIEVE]", "[INJECT]", "[PROMPT]", "[LLM]", "[EXTRACT]", "[STORE]"):
            if stage in msg:
                log_record["pipeline_stage"] = stage.strip("[]")
                break
        return json.dumps(log_record)


class PipelineFormatter(logging.Formatter):
    """Compact formatter for pipeline-specific log file. Shows stage markers clearly."""
    def format(self, record):
        ts = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        level = record.levelname[0]  # I, D, W, E
        name_short = record.name.replace("MemorySystem.", "")
        return f"{ts} {level} {name_short} | {record.getMessage()}"


class _LoggerConfigurator:
    """Singleton that configures the root 'MemorySystem' logger once."""
    _initialized = False

    @classmethod
    def configure(cls):
        if cls._initialized:
            return
        cls._initialized = True

        root_logger = logging.getLogger("MemorySystem")
        root_logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers on reload
        if root_logger.handlers:
            return

        # ── Console Formatter ──
        console_formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Silence third-party garbage on console
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        # 1. Console Handler — shows DEBUG when ENML_DEBUG=1, otherwise WARNING to keep UI clean
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if ENML_DEBUG else logging.WARNING)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # 2. Main File Handler (Rotating) — 10MB per file, 5 backups (DEBUG level)
        log_file = LOG_DIR / "memory_system.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)

        # 3. Pipeline Log — dedicated file for retrieval pipeline events only
        pipeline_log = LOG_DIR / "pipeline.log"
        pipeline_handler = RotatingFileHandler(
            pipeline_log, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'
        )
        pipeline_handler.setLevel(logging.DEBUG)
        pipeline_handler.setFormatter(PipelineFormatter())
        # Filter: only log messages containing pipeline stage markers
        pipeline_handler.addFilter(_PipelineFilter())
        root_logger.addHandler(pipeline_handler)

        # 4. JSON Lines Handler for structured auditing
        json_log_file = LOG_DIR / "audit.jsonl"
        json_handler = logging.FileHandler(json_log_file, encoding='utf-8')
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(json_handler)


class _PipelineFilter(logging.Filter):
    """Only passes log records that contain a pipeline stage marker."""
    _STAGES = ("[ROUTE]", "[RETRIEVE]", "[INJECT]", "[PROMPT]", "[LLM]", "[EXTRACT]", "[STORE]")
    
    def filter(self, record):
        msg = record.getMessage()
        return any(stage in msg for stage in self._STAGES)


def get_logger(name: str = None) -> logging.Logger:
    """Factory function to get a properly-named child logger.

    All loggers are children of the 'MemorySystem' root logger, which
    is configured once with console, file, pipeline, and JSON-lines handlers.
    Calling ``get_logger("core.extractor")`` returns
    ``logging.getLogger("MemorySystem.core.extractor")``.
    """
    _LoggerConfigurator.configure()
    root = logging.getLogger("MemorySystem")
    if name:
        return root.getChild(name)
    return root

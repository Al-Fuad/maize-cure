"""Local rotating JSONL audit trail. No photos, keys or request bodies."""
import json
import logging
import logging.handlers
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from contextvars import ContextVar

request_id = ContextVar('research_request_id', default=None)
_lock = threading.Lock()
_logger = None
_log_path = None


def record(event, **fields):
    global _logger, _log_path
    if os.getenv('RESEARCH_LOG_ENABLED', '1') != '1':
        return
    path = Path(os.getenv('RESEARCH_LOG_PATH', str(Path(__file__).parent / 'logs/research.jsonl')))
    if not path.is_absolute():
        path = Path(__file__).parent / path
    try:
        with _lock:
            if _logger is None or _log_path != path:
                path.parent.mkdir(parents=True, exist_ok=True)
                logger = logging.getLogger('maize.research')
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                    handler.close()
                handler = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding='utf-8')
                handler.setFormatter(logging.Formatter('%(message)s'))
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
                logger.propagate = False
                _logger, _log_path = logger, path
            _logger.info(json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(), 'event': event,
                                     'request_id': request_id.get(), **fields}, ensure_ascii=False))
    except (OSError, ValueError):
        logging.getLogger(__name__).warning('Research audit storage unavailable')

"""Logging that mirrors a run's console output into the run directory."""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# A library should not install handlers on import, but it should not warn about
# their absence either.
logging.getLogger("riskflow").addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"riskflow.{name}")


def configure(level: int = logging.INFO) -> None:
    """Attach a single stderr handler to the riskflow root logger."""
    root = logging.getLogger("riskflow")
    root.setLevel(level)
    if not any(getattr(h, "_riskflow_console", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        handler._riskflow_console = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.propagate = False


@contextmanager
def run_log_file(path: str | Path, level: int = logging.INFO):
    """Mirror every riskflow log record into `path` for the duration of a run.

    The logger's level is raised for the duration if the host application left it
    higher, so a run always gets a complete log on disk even when riskflow is
    used as a library and nothing called `configure()`. The previous level is
    restored on the way out.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger("riskflow")
    previous = root.level
    if root.getEffectiveLevel() > level:
        root.setLevel(level)
    root.addHandler(handler)
    try:
        yield path
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(previous)

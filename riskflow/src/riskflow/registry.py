"""Run directories and the production pointer.

Every training run writes an immutable directory. Promoting a run is a separate,
deliberate act that writes its name into a `PRODUCTION` pointer file, so the
model that scores today is always a named, inspectable run rather than whatever
was trained last.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .bundle import BUNDLE_FILENAME, ScoringBundle
from .logging_setup import get_logger

log = get_logger("registry")

POINTER_FILENAME = "PRODUCTION"
TABLES_DIR = "tables"
FIGURES_DIR = "figures"


@dataclass(frozen=True)
class Run:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def bundle_path(self) -> Path:
        return self.path / BUNDLE_FILENAME

    @property
    def tables(self) -> Path:
        return self.path / TABLES_DIR

    @property
    def figures(self) -> Path:
        return self.path / FIGURES_DIR

    @property
    def summary_path(self) -> Path:
        return self.path / "summary.json"

    @property
    def report_path(self) -> Path:
        return self.path / "report.html"

    def load_bundle(self) -> ScoringBundle:
        return ScoringBundle.load(self.bundle_path)

    def summary(self) -> dict:
        if not self.summary_path.exists():
            return {}
        return json.loads(self.summary_path.read_text(encoding="utf-8-sig"))

    def table(self, name: str):
        import pandas as pd

        path = self.tables / f"{name}.csv"
        return pd.read_csv(path) if path.exists() else None

    def write_table(self, name: str, frame) -> Path:
        self.tables.mkdir(parents=True, exist_ok=True)
        path = self.tables / f"{name}.csv"
        frame.to_csv(path, index=False)
        return path

    def is_complete(self) -> tuple[bool, list[str]]:
        """Whether the run has everything needed to be promoted."""
        missing = [
            str(path.relative_to(self.path))
            for path in (self.bundle_path, self.summary_path)
            if not path.exists()
        ]
        return (not missing), missing


class Registry:
    """A directory of runs, plus the pointer to the promoted one."""

    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    @property
    def pointer_path(self) -> Path:
        return self.root / POINTER_FILENAME

    def new_run(self, name: str | None = None) -> Run:
        run_name = name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # A run name is a directory name, not a path. Anything else would put the
        # run somewhere `list_runs` and the production pointer cannot see it.
        if run_name != Path(run_name).name or run_name in {"", ".", ".."} or run_name == POINTER_FILENAME:
            raise ValueError(f"invalid run name '{run_name}'; it must be a single directory name")

        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / run_name
        if name is None:
            # The default name has one-second resolution, so training several
            # configurations in parallel would collide. An explicit name still
            # fails loudly — that collision means the caller lost track of a run.
            suffix = 1
            while True:
                try:
                    path.mkdir()
                    break
                except FileExistsError:
                    suffix += 1
                    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
                    path = self.root / run_name
        else:
            try:
                path.mkdir()
            except FileExistsError as exc:
                raise FileExistsError(f"run directory already exists: {path}") from exc
        (path / TABLES_DIR).mkdir()
        (path / FIGURES_DIR).mkdir()
        return Run(path=path)

    def get(self, name_or_path: str | Path) -> Run:
        candidate = Path(name_or_path)
        if not candidate.exists():
            candidate = self.root / str(name_or_path)
        if not candidate.exists():
            raise FileNotFoundError(f"no such run: {name_or_path}")
        return Run(path=candidate)

    def list_runs(self) -> list[Run]:
        if not self.root.exists():
            return []
        runs = [Run(path=p) for p in sorted(self.root.iterdir()) if p.is_dir()]
        return [r for r in runs if r.bundle_path.exists()]

    def promote(self, run: Run | str | Path) -> Run:
        """Point PRODUCTION at a run, refusing anything incomplete."""
        resolved = run if isinstance(run, Run) else self.get(run)
        complete, missing = resolved.is_complete()
        if not complete:
            raise ValueError(f"refusing to promote {resolved.name}: missing {', '.join(missing)}")
        # Loading proves the bundle is readable before it becomes production.
        resolved.load_bundle()
        self.root.mkdir(parents=True, exist_ok=True)
        # Write to a sibling file and rename. os.replace is atomic, so a crash or
        # a concurrent promotion leaves the pointer naming either the old run or
        # the new one — never a truncated name, and never nothing at all while
        # scoring is asking who production is.
        #
        # The staging name carries a random token, not just the pid: two threads
        # of one process would otherwise share the path, and the second would
        # find its file already renamed out from under it.
        staging = self.pointer_path.with_name(f".{POINTER_FILENAME}.{os.getpid()}.{uuid4().hex[:8]}")
        try:
            staging.write_text(resolved.name + "\n", encoding="utf-8")
            os.replace(staging, self.pointer_path)
        finally:
            staging.unlink(missing_ok=True)
        log.info("promoted %s to production", resolved.name)
        return resolved

    def production(self) -> Run:
        if not self.pointer_path.exists():
            raise FileNotFoundError(
                f"no production run yet; promote one first "
                f"(riskflow promote <run>) — expected pointer at {self.pointer_path}"
            )
        name = self.pointer_path.read_text(encoding="utf-8").strip()
        if not name:
            raise ValueError(f"{self.pointer_path} is empty")
        return self.get(name)

    def resolve(self, run: str | Path | None) -> Run:
        """An explicit run when given, otherwise whatever is in production."""
        return self.get(run) if run else self.production()

    def remove(self, run: Run | str | Path) -> None:
        resolved = run if isinstance(run, Run) else self.get(run)
        if self.pointer_path.exists() and self.pointer_path.read_text(encoding="utf-8").strip() == resolved.name:
            raise ValueError(f"{resolved.name} is the production run; promote another before removing it")
        shutil.rmtree(resolved.path)
        log.info("removed run %s", resolved.name)

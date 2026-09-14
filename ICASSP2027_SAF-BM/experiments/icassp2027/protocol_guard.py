from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

try:
    from .manifest import load_manifest
except ImportError:
    from manifest import load_manifest


@dataclass
class StageAccessGuard:
    manifest_path: Path
    stage: int
    audit_path: Path | None = None
    accessed: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.frame = load_manifest(self.manifest_path)
        if self.stage not in set(self.frame.loc[self.frame.stage >= 0, "stage"]):
            raise ValueError(f"Stage {self.stage} does not exist in {self.manifest_path}")

    def current(self, usage: str) -> pd.DataFrame:
        if usage not in {"fit", "validation"}:
            raise PermissionError(f"Training process may request only fit/validation, got {usage}")
        selected = self.frame[(self.frame.stage == self.stage) & (self.frame.usage == usage)].copy()
        if selected.empty:
            raise RuntimeError(f"No rows for stage={self.stage}, usage={usage}")
        self.accessed.append(
            {
                "stage": self.stage,
                "usage": usage,
                "rows": len(selected),
                "domains": sorted(selected.domain.unique().tolist()),
                "sample_ids": selected.sample_id.tolist(),
            }
        )
        return selected

    def reject_paths(self, paths: list[str]) -> None:
        allowed = set(self.frame.loc[self.frame.stage == self.stage, "relative_path"].astype(str))
        disallowed = sorted(set(paths) - allowed)
        if disallowed:
            raise PermissionError(
                f"Stage {self.stage} attempted to access {len(disallowed)} non-current-domain paths; "
                f"first={disallowed[:3]}"
            )

    def close(self) -> None:
        if self.audit_path is None:
            return
        payload = {
            "manifest": str(self.manifest_path),
            "stage": self.stage,
            "accesses": self.accessed,
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def __enter__(self) -> "StageAccessGuard":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

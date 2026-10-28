import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METADATA_COLUMNS = ("_extracted_at", "_dataset_name", "_source_url", "_page")


class LandingCheckError(RuntimeError):
    """A landing run that does not meet its checks."""


def run_folder(root: Path, dataset: str, run_ts: datetime) -> Path:
    run_ts = run_ts.astimezone(UTC)
    return root / dataset / f"extraction_date={run_ts:%Y-%m-%d}" / f"run_ts={run_ts:%Y%m%dT%H%M%SZ}"


@dataclass
class LandingRun:
    """Writes one immutable run of a dataset as NDJSON pages, one record per line."""

    root: Path
    dataset: str
    source: str
    run_ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    pages: int = 0
    records: int = 0
    bytes: int = 0

    def __post_init__(self) -> None:
        self.path = run_folder(self.root, self.dataset, self.run_ts)
        if self.path.exists():
            raise LandingCheckError(f"Landing folder already exists: {self.path}")
        self.path.mkdir(parents=True)

    def write_page(self, source_url: str, records: Iterable[Mapping[str, Any]]) -> int:
        self.pages += 1
        extracted_at = datetime.now(UTC).isoformat()
        lines = []
        for record in records:
            if not isinstance(record, Mapping):
                raise LandingCheckError(
                    f"{self.dataset} page {self.pages}: record is not a JSON object"
                )
            clashes = set(METADATA_COLUMNS).intersection(record)
            if clashes:
                raise LandingCheckError(
                    f"{self.dataset}: records already contain {sorted(clashes)}"
                )
            line = {
                **record,
                "_extracted_at": extracted_at,
                "_dataset_name": self.dataset,
                "_source_url": source_url,
                "_page": self.pages,
            }
            lines.append(json.dumps(line, separators=(",", ":"), ensure_ascii=False))
        content = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
        (self.path / f"part-{self.pages:05d}.ndjson").write_bytes(content)
        self.records += len(lines)
        self.bytes += len(content)
        return len(lines)

    def check(self, expected_records: int | None = None, allow_empty: bool = False) -> None:
        if self.records == 0 and not allow_empty:
            raise LandingCheckError(f"{self.dataset}: no records written")
        if expected_records is not None and self.records != expected_records:
            raise LandingCheckError(
                f"{self.dataset}: wrote {self.records} records, source declared {expected_records}"
            )

    def metrics(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source": self.source,
            "target_path": str(self.path),
            "pages": self.pages,
            "records": self.records,
            "bytes": self.bytes,
        }

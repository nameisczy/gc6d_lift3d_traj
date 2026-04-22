from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetFormat:
    root: Path

    @property
    def episodes_dir(self) -> Path:
        return self.root / "episodes"

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def vis_dir(self) -> Path:
        return self.root / "visualizations"

    def ensure(self) -> None:
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)


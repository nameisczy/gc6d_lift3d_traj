from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path = Path("/home/ziyaochen/gc6d_lift3d_traj")
    gc6d_root: Path = Path("/mnt/ssd/ziyaochen/GraspClutter6D")
    lift3d_root: Path = Path("/home/ziyaochen/LIFT3D")
    vggt_root: Path = Path("/home/ziyaochen/VGGT")
    gc6d_api_root: Path = Path("/home/ziyaochen/graspclutter6dAPI")
    output_root: Path = Path("/mnt/ssd/ziyaochen/gc6d_lift3d_traj_data")

    @property
    def episodes_dir(self) -> Path:
        return self.output_root / "episodes"

    @property
    def index_dir(self) -> Path:
        return self.output_root / "index"

    @property
    def metadata_dir(self) -> Path:
        return self.output_root / "metadata"

    @property
    def visualizations_dir(self) -> Path:
        return self.output_root / "visualizations"

    def ensure_output_tree(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)


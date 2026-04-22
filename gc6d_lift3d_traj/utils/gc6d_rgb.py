"""GC6D RGB image paths (same convention as GraspClutter6D.annId2ImgId)."""

from __future__ import annotations

from pathlib import Path

CAMERA_OFFSET = {
    "realsense-d415": 1,
    "realsense-d435": 2,
    "azure-kinect": 3,
    "zivid": 4,
}


def ann_id_to_img_id(ann_id: int, camera: str) -> int:
    if camera not in CAMERA_OFFSET:
        raise ValueError(f"Unsupported camera: {camera}")
    return int(ann_id) * 4 + CAMERA_OFFSET[camera]


def rgb_png_path(gc6d_root: str | Path, scene_id: int, ann_id: int, camera: str) -> Path:
    img_id = ann_id_to_img_id(ann_id, camera)
    return Path(gc6d_root) / "scenes" / f"{int(scene_id):06d}" / "rgb" / f"{img_id:06d}.png"

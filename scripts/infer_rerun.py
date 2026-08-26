"""
ZipDepth inference visualized in Rerun — single image.

Usage:
    # Spawn a native Rerun viewer
    pixi run python scripts/infer_rerun.py

    # Write a .rrd recording instead (headless / CI)
    pixi run python scripts/infer_rerun.py --save /tmp/zipdepth.rrd
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Literal, TypeAlias

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import tyro
from jaxtyping import Float32, UInt8

from zipdepth.inference.predictor import DepthInference

DeviceChoice: TypeAlias = Literal["auto", "cuda", "cpu"]


def resolve_device(device: DeviceChoice = "auto") -> str:
    """Resolve "auto" to cuda when available; validate an explicit "cuda".

    Args:
        device: Requested device.

    Returns:
        Concrete torch device string ("cuda" or "cpu").
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def main(
    checkpoint: Path = Path("checkpoints/zipdepth_base.pth"),
    image: Path = Path("assets/examples/im0.jpg"),
    input_size: int = 384,
    device: DeviceChoice = "auto",
    save: Path | None = None,
) -> None:
    """Run ZipDepth on one image and log RGB + relative depth to Rerun.

    Args:
        checkpoint: Path to the .pth checkpoint.
        image: Input image file.
        input_size: Shorter-side length for model input, rounded to a multiple of 32.
        device: Inference device; "auto" picks cuda when available.
        save: Write the recording to this .rrd path instead of spawning a viewer.
    """
    resolved_device: str = resolve_device(device)

    rr.init("zipdepth_infer", strict=True)
    if save is not None:
        rr.save(str(save))
    else:
        rr.spawn()

    blueprint: rrb.Blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="camera/rgb", name="RGB"),
            rrb.Spatial2DView(origin="camera/depth", name="Depth"),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

    loaded: UInt8[np.ndarray, "h w 3"] | None = cv2.imread(str(image))
    if loaded is None:
        raise FileNotFoundError(f"Cannot load: {image}")
    bgr_hw3: UInt8[np.ndarray, "h w 3"] = loaded
    rgb_hw3: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)

    predictor = DepthInference(
        checkpoint_path=str(checkpoint),
        device=resolved_device,
        input_size=input_size,
        warmup_iters=0,
    )
    with torch.no_grad():
        depth_hw: Float32[np.ndarray, "h w"] = predictor.infer_image(bgr_hw3)

    rr.log("camera/rgb", rr.Image(rgb_hw3), static=True)
    rr.log("camera/depth", rr.DepthImage(depth_hw), static=True)

    print(f"  Depth range:    [{depth_hw.min():.3f}, {depth_hw.max():.3f}]")
    if save is not None:
        print(f"  Recording saved to: {save}")


if __name__ == "__main__":
    tyro.cli(main)

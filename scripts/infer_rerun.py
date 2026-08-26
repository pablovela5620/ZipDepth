"""
ZipDepth inference visualized in Rerun — single image, 2D + 3D.

The model predicts relative inverse depth (disparity up to scale). Following the
monoprior example (rerun-io/examples-monorepo), intrinsics are assumed from a
fixed field of view, disparity is converted to bounded relative depth, and the
result is logged both as a DepthImage under a pinhole camera and as an
RGB-colored backprojected point cloud.

Usage:
    # Spawn a native Rerun viewer
    pixi run python scripts/infer_rerun.py

    # Write a .rrd recording instead (headless / CI)
    pixi run python scripts/infer_rerun.py --save /tmp/zipdepth.rrd
"""

from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import tyro
from einops import rearrange
from jaxtyping import Bool, Float32, UInt8

from zipdepth.inference.predictor import DepthInference

DeviceChoice: TypeAlias = Literal["auto", "cuda", "cpu"]

FOV_DEG: float = 55.0
"""Assumed horizontal field of view — no real calibration exists for arbitrary images."""

DEPTH_GAMMA: float = 2.2
"""Compression exponent applied as depth**(1/gamma) to tame the far range."""

DEPTH_EDGE_THRESHOLD: float = 1.1
"""Gradient-magnitude threshold above which depth pixels are treated as flying pixels."""

PINHOLE_ENTITY: str = "world/camera/pinhole"
"""Entity path of the assumed pinhole camera; RGB and depth images live beneath it."""

POINT_CLOUD_ENTITY: str = "world/point_cloud"
"""Entity path of the RGB-colored backprojected point cloud."""


def resolve_device(device: DeviceChoice) -> Literal["cuda", "cpu"]:
    """Resolve "auto" to cuda when available; validate an explicit "cuda".

    Args:
        device: Requested device.

    Returns:
        Concrete torch device string.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def estimate_intrinsics(h: int, w: int, fov_deg: float = FOV_DEG) -> Float32[np.ndarray, "3 3"]:
    """Build a pinhole K from an assumed field of view and central principal point.

    Args:
        h: Image height in pixels.
        w: Image width in pixels.
        fov_deg: Assumed horizontal field of view in degrees.

    Returns:
        Camera intrinsics, float32 [3, 3].
    """
    focal_px: float = 0.5 * w / np.tan(0.5 * np.deg2rad(fov_deg))
    cx: float = 0.5 * w
    cy: float = 0.5 * h
    k_33: Float32[np.ndarray, "3 3"] = np.array(
        [[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    return k_33


def disparity_to_depth(
    disparity_hw: Float32[np.ndarray, "h w"], focal_length: float
) -> Float32[np.ndarray, "h w"]:
    """Convert relative disparity to bounded, gamma-compressed relative depth.

    The far range is clamped so the depth ratio never exceeds 100:1, keeping the
    backprojected cloud finite; gamma compression tames the remaining spread.

    Args:
        disparity_hw: Relative disparity, float32 [h, w].
        focal_length: Focal length in pixels.

    Returns:
        Relative depth (arbitrary units), float32 [h, w].
    """
    depth_ratio: float = float(np.minimum(disparity_hw.max() / (disparity_hw.min() + 1e-6), 100.0))
    min_disparity: float = float(disparity_hw.max()) / depth_ratio
    depth_hw: Float32[np.ndarray, "h w"] = focal_length / np.maximum(disparity_hw, min_disparity)
    depth_hw = np.power(depth_hw, 1.0 / DEPTH_GAMMA).astype(np.float32)
    return depth_hw


def depth_edges_mask(
    depth_hw: Float32[np.ndarray, "h w"], threshold: float = DEPTH_EDGE_THRESHOLD
) -> Bool[np.ndarray, "h w"]:
    """Mask depth pixels lying on strong depth discontinuities (flying pixels).

    Args:
        depth_hw: Depth map, float32 [h, w].
        threshold: Gradient-magnitude cutoff.

    Returns:
        Boolean edge mask, [h, w].
    """
    depth_dx, depth_dy = np.gradient(depth_hw)
    grad_hw: Float32[np.ndarray, "h w"] = np.sqrt(depth_dx**2 + depth_dy**2)
    mask_hw: Bool[np.ndarray, "h w"] = grad_hw > threshold
    return mask_hw


def depth_to_points(
    depth_hw: Float32[np.ndarray, "h w"], k_33: Float32[np.ndarray, "3 3"]
) -> Float32[np.ndarray, "h w 3"]:
    """Backproject a depth map to camera-frame 3D points.

    Args:
        depth_hw: Depth map, float32 [h, w].
        k_33: Camera intrinsics, float32 [3, 3].

    Returns:
        3D points in the camera frame, float32 [h, w, 3].
    """
    h, w = depth_hw.shape
    k_inv_33: Float32[np.ndarray, "3 3"] = np.linalg.inv(k_33)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pixels_hw3: Float32[np.ndarray, "h w 3"] = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    rays_hw3: Float32[np.ndarray, "h w 3"] = np.einsum("ij,hwj->hwi", k_inv_33, pixels_hw3)
    points_hw3: Float32[np.ndarray, "h w 3"] = rays_hw3 * rearrange(depth_hw, "h w -> h w 1")
    return points_hw3


def main(
    checkpoint: Path = Path("checkpoints/zipdepth_base.pth"),
    image: Path = Path("assets/examples/im0.jpg"),
    input_size: int = 384,
    device: DeviceChoice = "auto",
    save: Path | None = None,
) -> None:
    """Run ZipDepth on one image and log RGB, depth, and a 3D cloud to Rerun.

    Args:
        checkpoint: Path to the .pth checkpoint.
        image: Input image file.
        input_size: Shorter-side length for model input, rounded to a multiple of 32.
        device: Inference device; "auto" picks cuda when available.
        save: Write the recording to this .rrd path instead of spawning a viewer.
    """
    resolved_device: Literal["cuda", "cpu"] = resolve_device(device)

    rr.init("zipdepth_infer", strict=True)
    if save is not None:
        rr.save(str(save))
    else:
        rr.spawn()

    predictor = DepthInference(
        checkpoint_path=str(checkpoint),
        device=resolved_device,
        input_size=input_size,
        warmup_iters=0,
    )
    bgr_hw3: UInt8[np.ndarray, "h w 3"] = predictor._load_bgr(str(image))
    rgb_hw3: UInt8[np.ndarray, "h w 3"] = np.ascontiguousarray(bgr_hw3[:, :, ::-1])
    h: int = rgb_hw3.shape[0]
    w: int = rgb_hw3.shape[1]

    disparity_hw: Float32[np.ndarray, "h w"] = predictor.infer_image(bgr_hw3)

    k_33: Float32[np.ndarray, "3 3"] = estimate_intrinsics(h, w)
    depth_hw: Float32[np.ndarray, "h w"] = disparity_to_depth(disparity_hw, focal_length=float(k_33[0, 0]))
    edges_hw: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw)
    depth_hw = (depth_hw * ~edges_hw).astype(np.float32)

    rr.log(
        PINHOLE_ENTITY,
        rr.Pinhole(image_from_camera=k_33, width=w, height=h, camera_xyz=rr.ViewCoordinates.RDF),
        static=True,
    )
    rr.log(f"{PINHOLE_ENTITY}/image", rr.Image(rgb_hw3).compress(jpeg_quality=90), static=True)
    rr.log(f"{PINHOLE_ENTITY}/depth", rr.DepthImage(depth_hw, meter=1.0), static=True)

    # Backproject only valid pixels — masked flying pixels would otherwise all
    # land at the camera origin as a spurious colored blob.
    valid_hw: Bool[np.ndarray, "h w"] = depth_hw > 0
    points_hw3: Float32[np.ndarray, "h w 3"] = depth_to_points(depth_hw, k_33)
    rr.log(
        POINT_CLOUD_ENTITY,
        rr.Points3D(positions=points_hw3[valid_hw], colors=rgb_hw3[valid_hw]),
        static=True,
    )

    # Frame the 3D view on the mid-range content: eye above and behind the
    # camera origin, orbiting a point partway into the scene (RDF: +Y is down).
    valid_depth: Float32[np.ndarray, "n"] = depth_hw[valid_hw]
    target_z: float = float(np.percentile(valid_depth, 40))
    blueprint: rrb.Blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{PINHOLE_ENTITY}/image", name="RGB"),
                rrb.Spatial2DView(origin=f"{PINHOLE_ENTITY}/depth", name="Depth"),
            ),
            rrb.Spatial3DView(
                origin="world",
                name="3D",
                # The RGB point cloud is logged explicitly; hide the DepthImage's
                # own colormapped backprojection so the clouds don't double up.
                contents=["+ $origin/**", f"- {PINHOLE_ENTITY}/depth"],
                eye_controls=rrb.archetypes.EyeControls3D(
                    kind=rrb.Eye3DKind.Orbital,
                    position=[0.0, -0.5 * target_z, -0.6 * target_z],
                    look_target=[0.0, 0.0, target_z],
                    eye_up=[0.0, -1.0, 0.0],
                ),
            ),
            column_shares=[1.0, 2.0],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

    print(f"  Disparity range: [{disparity_hw.min():.3f}, {disparity_hw.max():.3f}]")
    print(f"  Relative depth range: [{valid_depth.min():.2f}, {valid_depth.max():.2f}]")
    if save is not None:
        print(f"  Recording saved to: {save}")


if __name__ == "__main__":
    tyro.cli(main)

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_model = None
_transform = None
_device = None


def is_available():
    try:
        import depth_pro  # noqa: F401
        return True
    except ImportError:
        return False


def _find_checkpoint_dir(depth_pro_module, log_fn=None):
    """Depth Pro's default config points at './checkpoints/depth_pro.pt',
    resolved relative to the CURRENT WORKING DIRECTORY at load time — not
    relative to the ml-depth-pro repo. If TerraMap is launched from
    elsewhere (the normal case), that lookup fails even though the
    checkpoint file exists. Locate the real repo root from the installed
    package's own file path instead of guessing.
    """
    def log(msg):
        if log_fn:
            log_fn(msg)

    try:
        pkg_file = Path(depth_pro_module.__file__).resolve()
        # typical editable-install layout: <repo_root>/src/depth_pro/__init__.py
        for candidate in (pkg_file.parents[2] if len(pkg_file.parents) > 2 else None,
                          pkg_file.parents[1] if len(pkg_file.parents) > 1 else None):
            if candidate is not None and (candidate / "checkpoints" / "depth_pro.pt").exists():
                return candidate
    except Exception as e:
        log(f"[Depth Pro] could not auto-locate checkpoints dir ({e})")
    return None


def _lazy_load(log_fn=None):
    global _model, _transform, _device
    if _model is not None:
        return _model, _transform, _device

    def log(msg):
        if log_fn:
            log_fn(msg)

    try:
        import depth_pro
    except ImportError as e:
        raise RuntimeError(
            "Depth Pro package not found. Install it with:\n"
            "  git clone https://github.com/apple/ml-depth-pro.git\n"
            "  cd ml-depth-pro\n"
            "  pip install -e .\n"
            "  source get_pretrained_models.sh\n"
            "then restart TerraMap."
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("[Depth Pro] WARNING: no CUDA GPU detected — this model is much "
            "slower on CPU.")
    log(f"[Depth Pro] loading model onto {device}...")

    repo_root = _find_checkpoint_dir(depth_pro, log_fn=log)
    old_cwd = os.getcwd()
    try:
        if repo_root is not None:
            log(f"[Depth Pro] using checkpoint at {repo_root / 'checkpoints' / 'depth_pro.pt'}")
            os.chdir(repo_root)
        else:
            log("[Depth Pro] WARNING: could not locate ml-depth-pro's checkpoints "
                "directory automatically; relying on the library's default "
                "'./checkpoints/depth_pro.pt' relative to the current directory. "
                "If this fails, run TerraMap from inside the ml-depth-pro folder, "
                "or place a 'checkpoints/depth_pro.pt' next to terramap.py.")
        model, transform = depth_pro.create_model_and_transforms()
    finally:
        os.chdir(old_cwd)

    model = model.to(device)
    model.eval()
    log("[Depth Pro] model ready.")

    _model, _transform, _device = model, transform, device
    return model, transform, device


def estimate_depth(image_path, log_fn=None):
    """
    Returns:
      depth_m   : (H,W) float32 array, metric depth in meters
      focal_px  : estimated focal length in pixels
      rgb       : (H,W,3) uint8 array, the image resized to depth_m's
                  resolution (for coloring the unprojected point cloud)
    """
    def log(msg):
        if log_fn:
            log_fn(msg)

    import depth_pro

    model, transform, device = _lazy_load(log_fn=log)

    log(f"[Depth Pro] running on {image_path} ...")
    image, _, f_px = depth_pro.load_rgb(image_path)
    image_t = transform(image).to(device)

    with torch.no_grad():
        prediction = model.infer(image_t, f_px=f_px)

    depth = prediction["depth"]
    focal = prediction["focallength_px"]

    depth_m = depth.detach().float().cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
    depth_m = np.squeeze(depth_m)
    focal_px = float(focal.detach().cpu().item()) if torch.is_tensor(focal) else float(focal)

    rgb = np.array(Image.open(image_path).convert("RGB")
                   .resize((depth_m.shape[1], depth_m.shape[0])))

    log(f"[Depth Pro] done — depth range {np.nanmin(depth_m):.2f}m to "
        f"{np.nanmax(depth_m):.2f}m, estimated focal length ~{focal_px:.1f}px")

    return depth_m.astype(np.float32), focal_px, rgb


def _depth_colormap(t):
    """t: (N,) in [0,1] -> (N,3). Same viridis-ish gradient used elsewhere
    in the app for depth/height visualization (dark purple=near,
    yellow=far), so DEPTH PRO mode visually reads as a depth heatmap
    instead of a photo-textured point cloud."""
    stops = np.array([0.0, 0.33, 0.66, 1.0])
    colors = np.array([
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [253, 231, 37],
    ], dtype=np.float64)
    t = np.clip(t, 0.0, 1.0)
    r = np.interp(t, stops, colors[:, 0])
    g = np.interp(t, stops, colors[:, 1])
    b = np.interp(t, stops, colors[:, 2])
    return np.stack([r, g, b], axis=-1)


def depth_to_point_cloud(depth_m, rgb, focal_px, max_points=400_000,
                          max_depth_m=None, color_mode="depth"):
    """
    Standard pinhole back-projection of a metric depth map into a colored
    3D point cloud: X=(u-cx)*Z/f, Y=(v-cy)*Z/f, Z=depth. The Y axis is
    flipped to match the rest of TerraMap's Y-up convention (raw image/
    camera coordinates have Y pointing down).

    color_mode: "depth" (default) colors points by a depth heatmap, so the
    cloud visually reads as a depth map even once it's 3D. "rgb" colors
    points with the original photo instead.
    """
    H, W = depth_m.shape
    cx, cy = W / 2.0, H / 2.0

    ys, xs = np.mgrid[0:H, 0:W]
    Z = depth_m.astype(np.float64)
    X = (xs - cx) * Z / focal_px
    Y = (ys - cy) * Z / focal_px
    Y = -Y  # camera convention is Y-down; flip to this app's Y-up convention

    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    z_flat = Z.reshape(-1)

    valid = np.isfinite(pts).all(axis=1) & (z_flat > 0)
    if max_depth_m is not None:
        valid &= (z_flat < max_depth_m)

    pts = pts[valid]
    z_valid = z_flat[valid]

    if color_mode == "depth":
        if z_valid.size > 0:
            dmin, dmax = np.percentile(z_valid, 2), np.percentile(z_valid, 98)
        else:
            dmin, dmax = 0.0, 1.0
        t = np.clip((z_valid - dmin) / max(dmax - dmin, 1e-6), 0, 1)
        cols = _depth_colormap(t)
    else:
        cols = rgb.reshape(-1, 3)[valid]

    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
        pts, cols = pts[idx], cols[idx]

    return pts.astype(np.float32), cols.astype(np.uint8)


def estimate_point_cloud(image_path, log_fn=None, max_points=400_000, color_mode="depth"):
    """
    Convenience wrapper: run Depth Pro on a single image and return a
    depth-colored 3D point cloud ready for the same viewer used for the
    VGGT reconstruction.

    Returns:
      points : (N,3) float32, world-ish XYZ in THIS camera's own frame
      colors : (N,3) uint8
      stats  : dict of diagnostics
    """
    def log(msg):
        if log_fn:
            log_fn(msg)

    depth_m, focal_px, rgb = estimate_depth(image_path, log_fn=log)

    # discard the far tail (sky / open space) so the cloud isn't dominated
    # by a handful of near-infinite-depth outlier pixels
    finite_depths = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    max_depth_m = float(np.percentile(finite_depths, 98)) if finite_depths.size else None

    points, colors = depth_to_point_cloud(depth_m, rgb, focal_px, max_points=max_points,
                                           max_depth_m=max_depth_m, color_mode=color_mode)

    stats = {
        "engine": "Depth Pro",
        "image": image_path,
        "focal_px": focal_px,
        "points": int(points.shape[0]),
        "depth_min_m": float(np.nanmin(depth_m)),
        "depth_max_m": float(np.nanmax(depth_m)),
    }
    log(f"[Depth Pro] unprojected {stats['points']} points into 3D")
    return points, colors, stats
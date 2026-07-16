import numpy as np
import torch

_model = None
_device = None
_dtype = None


def is_available():
    try:
        import vggt  # noqa: F401
        return True
    except ImportError:
        return False


def _lazy_load_model(log_fn=None):
    global _model, _device, _dtype
    if _model is not None:
        return _model, _device, _dtype

    def log(msg):
        if log_fn:
            log_fn(msg)

    try:
        from vggt.models.vggt import VGGT
    except ImportError as e:
        raise RuntimeError(
            "VGGT package not found. Install it with:\n"
            "  git clone https://github.com/facebookresearch/vggt.git\n"
            "  cd vggt\n"
            "  pip install -r requirements.txt\n"
            "  pip install -e .\n"
            "then restart TerraMap."
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        major = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32
        log("[VGGT] WARNING: no CUDA GPU detected — running on CPU will be "
            "slow, keep frame counts small (5-10 frames).")

    log(f"[VGGT] loading facebook/VGGT-1B onto {device} ({dtype}) — "
        f"first run downloads ~5GB of weights from Hugging Face...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    log("[VGGT] model ready.")

    _model, _device, _dtype = model, device, dtype
    return model, device, dtype


def reconstruct_point_cloud_vggt(image_paths, conf_percentile=50.0,
                                  max_points=250_000, log_fn=None):
    """
    Run VGGT on a list of image file paths and return a colored point cloud.

    Parameters
    ----------
    image_paths : list[str]      paths to sampled video frames (RGB images)
    conf_percentile : float      points below this confidence percentile are
                                  dropped (higher = sparser but cleaner cloud)
    max_points : int              hard cap, randomly subsampled if exceeded

    Returns
    -------
    points : (N,3) float32 array, world-space XYZ
    colors : (N,3) uint8 array, RGB 0-255
    stats  : dict of diagnostics (frames used, raw/kept point counts, ...)
    """
    def log(msg):
        if log_fn:
            log_fn(msg)

    if len(image_paths) < 1:
        raise ValueError("Need at least 1 frame for VGGT reconstruction.")

    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    model, device, dtype = _lazy_load_model(log_fn=log)

    log(f"[VGGT] preprocessing {len(image_paths)} frames...")
    images = load_and_preprocess_images(image_paths).to(device)

    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    log("[VGGT] forward pass complete — unprojecting depth to 3D points...")

    if "pose_enc" not in predictions:
        raise RuntimeError(
            f"VGGT predictions dict has no 'pose_enc' key. Keys present: "
            f"{list(predictions.keys())}. This usually means an incompatible "
            f"VGGT version is installed — check that vggt.utils.pose_enc and "
            f"vggt.utils.geometry match the model checkpoint being used.")

    pose_result = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    if not isinstance(pose_result, (tuple, list)) or len(pose_result) < 2:
        raise RuntimeError(
            f"pose_encoding_to_extri_intri returned an unexpected type/length "
            f"({type(pose_result)}, len={getattr(pose_result, '__len__', lambda: 'n/a')()}). "
            f"Expected (extrinsic, intrinsic). Your installed VGGT version's API may "
            f"have diverged from what this app was written against.")
    if len(pose_result) > 2:
        log(f"[VGGT] WARNING: pose_encoding_to_extri_intri returned {len(pose_result)} "
            f"values (expected 2) — using the first two as (extrinsic, intrinsic) and "
            f"ignoring the rest.")
    extrinsic, intrinsic = pose_result[0], pose_result[1]

    depth_map = predictions["depth"]
    depth_conf = predictions["depth_conf"]

    def _squeeze_batch_dim(x):
        """Remove a leading batch dim of size 1 (torch tensor or ndarray).

        VGGT's own demos always do this (`.squeeze(0)`) right after computing
        extrinsic/intrinsic and before calling unproject_depth_map_to_point_map
        — that function's internal per-frame loop assumes the leading axis is
        the frame count S, not a batch dim B. Skipping this squeeze is what
        causes "too many values to unpack" deep inside vggt/utils/geometry.py.
        """
        if torch.is_tensor(x):
            return x.squeeze(0) if x.dim() >= 1 and x.shape[0] == 1 else x
        arr = np.asarray(x)
        return arr[0] if arr.ndim >= 1 and arr.shape[0] == 1 else arr

    extrinsic = _squeeze_batch_dim(extrinsic)
    intrinsic = _squeeze_batch_dim(intrinsic)
    depth_map = _squeeze_batch_dim(depth_map)
    depth_conf = _squeeze_batch_dim(depth_conf)

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    if isinstance(points_3d, (tuple, list)):
        log(f"[VGGT] NOTE: unproject_depth_map_to_point_map returned {len(points_3d)} "
            f"values — using the first as the point map.")
        points_3d = points_3d[0]

    def to_np(x):
        return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    points_3d = to_np(points_3d)             # expected (S,H,W,3), possibly with a batch dim
    depth_conf = to_np(depth_conf)
    imgs_np = to_np(predictions["images"])   # expected (S,3,H,W) in [0,1], possibly batched

    # some VGGT versions keep a leading batch dim (B=1) on every output when called
    # via the high-level model(images) forward; squeeze it off if present.
    if points_3d.ndim == 5 and points_3d.shape[0] == 1:
        points_3d = points_3d[0]
    if depth_conf.ndim == 4 and depth_conf.shape[0] == 1:
        depth_conf = depth_conf[0]
    if imgs_np.ndim == 5 and imgs_np.shape[0] == 1:
        imgs_np = imgs_np[0]

    if points_3d.ndim != 4 or points_3d.shape[-1] != 3:
        raise RuntimeError(
            f"Unexpected point map shape {points_3d.shape} (expected (S,H,W,3) "
            f"after removing any batch dim). Cannot continue safely.")

    # VGGT's world_points are expressed in standard camera convention:
    # X right, Y DOWN, Z forward (into the scene) — from the first camera's
    # point of view. Every renderer in this app (rasterizer.py, and the
    # exported .ply) assumes Y is UP. Flipping the sign here, once, at the
    # source, is what fixes the reconstruction appearing upside-down/mirrored.
    points_3d = points_3d.copy()
    points_3d[..., 1] *= -1.0

    S, H, W, _ = points_3d.shape
    imgs_np = np.transpose(imgs_np, (0, 2, 3, 1))          # (S,H,W,3)

    if imgs_np.shape[0] != S or imgs_np.shape[1] != H or imgs_np.shape[2] != W:
        raise RuntimeError(
            f"Point map resolution {(S, H, W)} does not match image tensor "
            f"resolution {imgs_np.shape[:3]} — cannot map colors onto points. "
            f"This usually indicates a VGGT API/version mismatch.")

    imgs_np = np.clip(imgs_np * 255.0, 0, 255).astype(np.uint8)

    pts_flat = points_3d.reshape(-1, 3)
    conf_flat = depth_conf.reshape(-1)
    color_flat = imgs_np.reshape(-1, 3)

    finite = np.isfinite(pts_flat).all(axis=1)
    pts_flat, conf_flat, color_flat = pts_flat[finite], conf_flat[finite], color_flat[finite]

    if conf_flat.size == 0:
        raise RuntimeError("VGGT produced no valid points for this frame set.")

    thresh = np.percentile(conf_flat, conf_percentile)
    keep = conf_flat >= thresh
    pts_flat, color_flat, conf_flat = pts_flat[keep], color_flat[keep], conf_flat[keep]

    if pts_flat.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts_flat.shape[0], max_points, replace=False)
        pts_flat, color_flat = pts_flat[idx], color_flat[idx]

    stats = {
        "engine": "VGGT-1B",
        "device": device,
        "frames_used": int(S),
        "raw_points": int(S * H * W),
        "kept_points": int(pts_flat.shape[0]),
        "conf_percentile": conf_percentile,
    }
    log(f"[VGGT] kept {stats['kept_points']} / {stats['raw_points']} points "
        f"(confidence >= p{conf_percentile})")

    return pts_flat.astype(np.float32), color_flat.astype(np.uint8), stats
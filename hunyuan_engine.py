"""
hunyuan_engine.py
Wraps Tencent's Hunyuan3D-2 image-to-3D generative model for TerraMap.

LIGHT MODE (default, tuned for 12GB laptop GPUs and machines sensitive to
sustained full CPU+GPU load):
  - shape stage uses the small 'Hunyuan3D-2mini' DiT (0.6B vs 2.6B)
  - reduced inference steps + octree resolution
  - stages run STRICTLY sequentially: the shape pipeline is fully released
    from VRAM before the texture pipeline loads
  - texture pipeline runs as-is (a cpu-offload experiment was removed:
    it silently broke painting on this build — see _load_texture_pipeline)
  - CPU thread pools are capped so a 20-core CPU is not pinned at 100%
    while the GPU is also at 100%
Texture output is KEPT in light mode — the result is still a textured GLB.

Set LIGHT_MODE = False below for full-quality mode on desktop-class GPUs.

Reference: https://github.com/Tencent/Hunyuan3D-2
"""

import os

# cap CPU parallelism BEFORE torch/numpy spin up their thread pools —
# full 20-core spikes combined with 100% GPU load can trip power/stability
# protection on laptops
# CPU_THREADS = "4"
# os.environ.setdefault("OMP_NUM_THREADS", CPU_THREADS)
# os.environ.setdefault("MKL_NUM_THREADS", CPU_THREADS)

import numpy as np
import torch

from trellis_engine import glb_to_point_cloud  # noqa: F401  (re-exported for the app)

LIGHT_MODE = False

# light-mode knobs
LIGHT_SHAPE_REPO = "tencent/Hunyuan3D-2mini"
LIGHT_SHAPE_SUBFOLDER = "hunyuan3d-dit-v2-mini"
LIGHT_STEPS = 30            # default is 50
LIGHT_OCTREE = 256          # default is 384 — big CPU/VRAM saver in decoding

FULL_SHAPE_REPO = "tencent/Hunyuan3D-2"
MV_SHAPE_REPO = "tencent/Hunyuan3D-2mv"
TEXTURE_REPO = "tencent/Hunyuan3D-2"

MV_VIEW_ORDER = ["front", "left", "back", "right"]

_shape_pipeline = None
_shape_kind = None
_rembg = None


def is_available():
    try:
        import hy3dgen  # noqa: F401
        return True
    except ImportError:
        return False


def _cap_torch_threads():
    try:
        import torch
        torch.set_num_threads(int(CPU_THREADS))
    except Exception:
        pass


def _get_rembg(log):
    global _rembg
    if _rembg is None:
        try:
            from hy3dgen.rembg import BackgroundRemover
            _rembg = BackgroundRemover()
        except Exception as e:
            log(f"[HY3D] WARNING: background remover unavailable ({e}) — "
                f"using frames as-is. Clean/simple backgrounds work best.")
            _rembg = False
    return _rembg or None


def _release_shape_pipeline(log):
    """Fully evict the shape pipeline from VRAM before texturing —
    on 12GB cards both together overflow into shared memory."""
    global _shape_pipeline, _shape_kind
    _shape_pipeline = None
    _shape_kind = None
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    log("[HY3D] shape pipeline released — VRAM freed for texturing.")


def _load_shape_pipeline(want_mv, log):
    global _shape_pipeline, _shape_kind

    kind = "mv" if want_mv else ("mini" if LIGHT_MODE else "single")
    if _shape_pipeline is not None and _shape_kind == kind:
        return _shape_pipeline

    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    except ImportError as e:
        raise RuntimeError(
            "Hunyuan3D-2 package not found. Install it with:\n"
            "  git clone https://github.com/Tencent/Hunyuan3D-2.git\n"
            "  cd Hunyuan3D-2\n"
            "  pip install -r requirements.txt\n"
            "  pip install -e .\n"
            "then restart TerraMap."
        ) from e

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Hunyuan3D-2 needs a CUDA GPU — none was found.")
    _cap_torch_threads()

    if want_mv:
        repo, kwargs = MV_SHAPE_REPO, {}
    elif LIGHT_MODE:
        repo, kwargs = LIGHT_SHAPE_REPO, {"subfolder": LIGHT_SHAPE_SUBFOLDER}
    else:
        repo, kwargs = FULL_SHAPE_REPO, {}

    log(f"[HY3D] loading {repo} — first run downloads weights from "
        f"Hugging Face...")
    _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        repo, **kwargs)
    _shape_kind = kind
    log(f"[HY3D] shape pipeline ready ({'LIGHT/mini' if kind == 'mini' else kind}).")
    return _shape_pipeline


def _run_shape(pipe, image, log):
    """Call the shape pipeline with light-mode knobs when possible;
    gracefully retries without unsupported kwargs on older builds."""
    kwargs = {}
    if LIGHT_MODE:
        kwargs = {"num_inference_steps": LIGHT_STEPS,
                  "octree_resolution": LIGHT_OCTREE}
    try:
        return pipe(image=image, **kwargs)[0]
    except TypeError:
        if kwargs:
            log("[HY3D] NOTE: this build doesn't accept step/octree kwargs — "
                "running with defaults.")
        return pipe(image=image)[0]


def _load_texture_pipeline(log):
    """Loaded fresh per run (so it can be freed after)."""
    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
    except Exception as e:
        log(f"[HY3D] NOTE: texture pipeline unavailable ({e}).")
        log("[HY3D] Exporting UNTEXTURED shape. To enable texturing, build "
            "the optional modules (see hunyuan_engine.py header) and restart.")
        return None

    log("[HY3D] loading texture pipeline (Hunyuan3D-Paint)...")
    paint = Hunyuan3DPaintPipeline.from_pretrained(TEXTURE_REPO)
    # NOTE: no cpu-offload tricks here — enabling model_cpu_offload on the
    # inner diffusers pipelines silently broke painting (verified by
    # test_texture.py, which succeeded exactly when offload was absent).
    # Sequential stage release (_release_shape_pipeline) is what keeps the
    # 12GB card happy; that stays.
    log("[HY3D] texture pipeline ready.")
    return paint

def generate_glb_from_frames(frame_paths, out_glb_path, log_fn=None):
    """Run Hunyuan3D-2 (FULL model, single image) and export a textured GLB.
    Same call sequence as test_texture.py — no thread capping, no cached
    global pipelines, no mesh postprocessing, no swallowed exceptions
    around paint(). Only difference from the mini/light path: uses the
    full 2.6B shape model with its own default steps/octree resolution.
    """
    def log(msg):
        if log_fn:
            log_fn(msg)

    if len(frame_paths) < 1:
        raise ValueError("Need at least 1 frame for Hunyuan3D-2 generation.")

    log(f"[HY3D] torch {torch.__version__}, cuda {torch.version.cuda}, "
        f"gpu={torch.cuda.get_device_name(0)}")

    from PIL import Image
    image = Image.open(frame_paths[0]).convert("RGB")

    log("[HY3D] background removal...")
    try:
        from hy3dgen.rembg import BackgroundRemover
        image = BackgroundRemover()(image)
        log("    ok")
    except Exception as e:
        log(f"    skipped ({e})")

    log(f"[HY3D] shape generation (FULL model: {FULL_SHAPE_REPO})...")
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(FULL_SHAPE_REPO)
    mesh = pipe(image=image)[0]   # full model's own default steps/octree
    log(f"    ok: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    log("[HY3D] releasing shape pipeline from VRAM...")
    del pipe
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    log(f"    free VRAM: "
        f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9:.1f} GB")

    log("[HY3D] texture painting...")
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    paint = Hunyuan3DPaintPipeline.from_pretrained(TEXTURE_REPO)
    log("    paint pipeline loaded, calling...")

    mesh = paint(mesh, image=image)
    log(f"    PAINT SUCCEEDED. visual={type(mesh.visual).__name__}")

    mesh.export(out_glb_path)
    log(f"[HY3D] GLB EXPORTED -> {out_glb_path}")

    stats = {
        "engine": "Hunyuan3D-2 FULL (generative)",
        "mode": "single-image (full)",
        "textured": True,
        "frames_used": 1,
        "glb_path": out_glb_path,
    }
    return out_glb_path, stats
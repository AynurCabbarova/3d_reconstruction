"""
trellis_engine.py
Wraps Microsoft's TRELLIS.2 image-to-3D generative model for TerraMap.

TRELLIS.2 takes image(s) of an object and GENERATES a textured 3D asset
(GLB). Unlike VGGT it is not a measurement/reconstruction model — it works
best on object-centric footage (camera moving around one object) and its
output lives in a normalized coordinate box (roughly [-0.5, 0.5]^3), not
in any metric scene frame.

Reference: https://github.com/microsoft/TRELLIS.2   (model: microsoft/TRELLIS.2-4B)

Install (required — heavy, official code is Linux-tested, needs an NVIDIA
GPU with >= 24 GB memory):
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
    cd TRELLIS.2
    . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec \
                 --cumesh --o-voxel --flexgemm

For reading the produced GLB back as a colored point cloud (for TerraMap's
own viewers) you also need:
    pip install trimesh
"""

import os

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_pipeline = None


def is_available():
    try:
        import trellis2  # noqa: F401
        import o_voxel   # noqa: F401
        return True
    except ImportError:
        return False


def _lazy_load_pipeline(log_fn=None):
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    def log(msg):
        if log_fn:
            log_fn(msg)

    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
    except ImportError as e:
        raise RuntimeError(
            "TRELLIS.2 package not found. Install it with:\n"
            "  git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive\n"
            "  cd TRELLIS.2\n"
            "  . ./setup.sh --new-env --basic --flash-attn --nvdiffrast "
            "--nvdiffrec --cumesh --o-voxel --flexgemm\n"
            "then restart TerraMap."
        ) from e

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("TRELLIS.2 requires a CUDA GPU (>= 24 GB memory "
                           "recommended) — no CUDA device was found.")

    log("[TRELLIS2] loading microsoft/TRELLIS.2-4B — first run downloads "
        "the 4B-parameter weights from Hugging Face...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    log("[TRELLIS2] pipeline ready.")

    _pipeline = pipeline
    return pipeline


def generate_glb_from_frames(frame_paths, out_glb_path, log_fn=None,
                              decimation_target=500_000, texture_size=2048):
    """
    Run TRELLIS.2 on the given frames and export a textured GLB.

    The official minimal example is single-image; multi-image conditioning
    is attempted first and we fall back to the first frame if this build's
    pipeline doesn't accept a list.

    Returns (glb_path, stats_dict).
    """
    def log(msg):
        if log_fn:
            log_fn(msg)

    if len(frame_paths) < 1:
        raise ValueError("Need at least 1 frame for TRELLIS.2 generation.")

    from PIL import Image
    import o_voxel

    pipeline = _lazy_load_pipeline(log_fn=log)

    images = [Image.open(p).convert("RGB") for p in frame_paths]
    log(f"[TRELLIS2] running generation on {len(images)} frame(s)...")

    mesh = None
    frames_used = len(images)
    if len(images) > 1:
        try:
            mesh = pipeline.run(images)[0]
        except (TypeError, ValueError, AttributeError) as e:
            log(f"[TRELLIS2] WARNING: this pipeline build did not accept "
                f"multiple images ({e}) — falling back to the FIRST frame only.")
            mesh = None
    if mesh is None:
        frames_used = 1
        mesh = pipeline.run(images[0])[0]

    mesh.simplify(16777216)  # nvdiffrast limit (per official example)

    log(f"[TRELLIS2] generation done — exporting GLB "
        f"(decimation={decimation_target}, texture={texture_size})...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    glb.export(out_glb_path, extension_webp=True)
    log(f"[TRELLIS2] GLB EXPORTED -> {out_glb_path}")

    stats = {
        "engine": "TRELLIS.2-4B (generative)",
        "frames_used": frames_used,
        "glb_path": out_glb_path,
    }
    return out_glb_path, stats

def _manual_uv_sample(m, n, log):
    """UV-based manual texture sampling — trimesh-in built-in sample_color
    yolunu bypass edir, çünki bəzi mesh-lərdə (çoxlu material/atlas) bu
    yol səhv/təsadüfi rənglər qaytarır."""
    import trimesh

    visual = m.visual
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    tex_image = None
    if material is not None:
        tex_image = (getattr(material, "baseColorTexture", None)
                     or getattr(material, "image", None))

    if uv is None or tex_image is None:
        return None, None  # UV/texture yoxdursa fallback-a keçəcəyik

    if isinstance(tex_image, str):
        from PIL import Image as PILImage
        tex_image = PILImage.open(tex_image)
    tex_arr = np.asarray(tex_image.convert("RGB"))
    th, tw = tex_arr.shape[:2]

    points, face_idx = trimesh.sample.sample_surface(m, n)
    tri_verts = m.vertices[m.faces[face_idx]]      # (n,3,3)
    tri_uv = uv[m.faces[face_idx]]                 # (n,3,2)

    bary = trimesh.triangles.points_to_barycentric(tri_verts, points)
    sampled_uv = np.einsum('ij,ijk->ik', bary, tri_uv)

    px = np.clip((sampled_uv[:, 0] * (tw - 1)).astype(int), 0, tw - 1)
    py = np.clip(((1.0 - sampled_uv[:, 1]) * (th - 1)).astype(int), 0, th - 1)
    cols = tex_arr[py, px]

    log(f"[GLB] manual UV sample: {len(points)} pts, "
        f"color std={cols.astype(np.float64).std():.1f}")
    return points.astype(np.float32), cols.astype(np.uint8)

# --------------------------------------------------------------------------- #
# GLB -> colored point cloud, so TerraMap's existing viewers (CPU rasterizer
# and the embedded Open3D window) can display the result. Uses trimesh, which
# handles GLB textures/vertex colors far more reliably than legacy Open3D IO.
# --------------------------------------------------------------------------- #
def glb_to_point_cloud(glb_path, n_points=220_000, log_fn=None):
    """Sample a colored point cloud from a GLB's surfaces.

    Returns (points (N,3) float32, colors (N,3) uint8).
    """
    def log(msg):
        if log_fn:
            log_fn(msg)

    try:
        import trimesh
    except ImportError as e:
        raise RuntimeError(
            "Reading GLB needs trimesh: pip install trimesh") from e

    loaded = trimesh.load(glb_path)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
    else:
        meshes = [loaded]
    meshes = [m for m in meshes if len(m.faces) > 0]
    if not meshes:
        raise RuntimeError(f"No renderable mesh found inside {glb_path}")

    total_area = sum(float(m.area) for m in meshes) or 1.0
    all_pts, all_cols = [], []

    for m in meshes:
        n = max(int(n_points * float(m.area) / total_area), 1)
        visual_kind = type(getattr(m, "visual", None)).__name__

        pts, cols = _manual_uv_sample(m, n, log)   # <-- yeni, etibarlı yol

        if pts is None:
            # köhnə sample_color cəhdi, sonra vertex-color fallback
            try:
                res = trimesh.sample.sample_surface(m, n, sample_color=True)
                pts, _face_idx, cols = res
                cols = np.asarray(cols)[:, :3]
                if cols.std() < 1.0:
                    pts, cols = None, None
            except (TypeError, ValueError, AttributeError, IndexError):
                pts, cols = None, None

        if pts is None:
            try:
                colors = np.asarray(m.visual.to_color().vertex_colors)[:, :3]
            except Exception:
                colors = np.full((len(m.vertices), 3), 180, dtype=np.uint8)
            pts, face_idx = trimesh.sample.sample_surface(m, n)
            tri_vert_cols = colors[m.faces[face_idx]]
            cols = tri_vert_cols.mean(axis=1)

        log(f"[GLB] mesh visual={visual_kind}, sampled {len(pts)} pts, "
            f"color std={np.asarray(cols).std():.1f}"
            f"{' (LOW — likely untextured/flat)' if np.asarray(cols).std() < 1.0 else ''}")

        all_pts.append(np.asarray(pts, dtype=np.float32))
        all_cols.append(np.asarray(cols).astype(np.uint8))

    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)
    log(f"[TRELLIS2] sampled {points.shape[0]} colored points from GLB "
        f"({len(meshes)} mesh(es))")
    return points, colors
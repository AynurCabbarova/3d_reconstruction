"""
test_texture.py — standalone Hunyuan3D texture diagnostic for TerraMap.

Runs the full shape+paint chain on ONE image with NO error swallowing:
any texturing failure prints its complete traceback to the console.
Also verifies the exported GLB actually contains readable colors.

Usage:
    python test_texture.py path\\to\\frame.jpg
"""

import sys
import traceback

import numpy as np
from PIL import Image


def main():
    if len(sys.argv) < 2:
        print("usage: python test_texture.py <image.jpg>")
        sys.exit(1)
    img_path = sys.argv[1]
    out_glb = "test_textured.glb"

    import torch
    print(f"[i] torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"gpu={torch.cuda.get_device_name(0)}")

    image = Image.open(img_path).convert("RGB")

    print("[1] background removal...")
    try:
        from hy3dgen.rembg import BackgroundRemover
        image = BackgroundRemover()(image)
        print("    ok")
    except Exception as e:
        print(f"    skipped ({e})")

    print("[2] shape generation (mini)...")
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini")
    try:
        mesh = pipe(image=image, num_inference_steps=30,
                    octree_resolution=256)[0]
    except TypeError:
        mesh = pipe(image=image)[0]
    print(f"    ok: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    print("[3] releasing shape pipeline from VRAM...")
    del pipe
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"    free VRAM: "
          f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9:.1f} GB")

    print("[4] texture painting — NO cpu offload, NO try/except around the call")
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    paint = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    print("    paint pipeline loaded, calling...")

    # the actual call, fully exposed:
    mesh = paint(mesh, image=image)
    print(f"    PAINT SUCCEEDED. visual={type(mesh.visual).__name__}")

    print("[5] exporting GLB...")
    mesh.export(out_glb)
    print(f"    -> {out_glb}")

    print("[6] verifying colors readable from GLB...")
    from trellis_engine import glb_to_point_cloud
    pts, cols = glb_to_point_cloud(out_glb, n_points=30_000, log_fn=print)
    std = float(np.asarray(cols).std())
    print(f"    points={pts.shape[0]}, color std={std:.1f} "
          f"({'COLORED — OK' if std > 1.0 else 'FLAT/GRAY — sampler or texture problem'})")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n========== FULL TRACEBACK ==========")
        traceback.print_exc()
        print("====================================")
        sys.exit(1)
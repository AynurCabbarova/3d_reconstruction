"""
gen3d_worker.py
Runs Hunyuan3D generation in its OWN OS process, launched by TerraMap via
QProcess. This reproduces exactly the conditions of the standalone
test_texture.py run (which was stable), instead of running heavy CUDA work
inside the GUI process next to Qt's rendering:

  - fresh process, fresh CUDA context, no VGGT weights lingering in VRAM
  - no GPU contention with the Qt window / embedded Open3D viewer
  - if anything crashes here, the TerraMap UI survives

Prints engine logs to stdout (TerraMap streams them into its SYSTEM LOG).
Exit code 0 = success (GLB written to --out), non-zero = failure.

Usage:
    python gen3d_worker.py --out out.glb frame1.jpg [frame2.jpg ...]
"""

import sys
import argparse
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output GLB path")
    ap.add_argument("frames", nargs="+", help="input frame image paths")
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    import hunyuan_engine
    out_glb, stats = hunyuan_engine.generate_glb_from_frames(
        args.frames, args.out, log_fn=log)
    # machine-readable footer for the parent process
    log(f"@@STATS@@ engine={stats.get('engine')} mode={stats.get('mode')} "
        f"textured={stats.get('textured')} frames_used={stats.get('frames_used')}")
    log("@@DONE@@")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("@@FAILED@@", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
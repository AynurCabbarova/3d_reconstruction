"""
o3d_viewer.py
Real interactive 3D viewing and point-picking for TerraMap, using Open3D.

These entry points are meant to be launched with `multiprocessing.Process`
(see TerraMap's "OPEN 3D VIEWER" / "PICK POI (3D)" buttons) rather than
called directly on the Tk thread:

  - Open3D's viewer windows run their own GLFW event loop; running that on
    the same thread/process as Tkinter's mainloop is unreliable across
    platforms.
  - Keeping it in a separate process also means a graphics-driver problem
    in Open3D can never take the whole TerraMap window down with it.

Data crosses the process boundary via plain files: a .ply point cloud in,
and (for picking) a small JSON file out.
"""

import json
import sys

import numpy as np
import open3d as o3d


def open_interactive_view(ply_path, window_title="TERRAMAP // 3D VIEWER"):
    """Blocking call: opens a normal orbit/zoom/pan Open3D window."""
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        print("[o3d_viewer] point cloud is empty, nothing to show.")
        return

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=window_title,
        width=1100, height=800,
        point_show_normal=False,
    )


def open_picking_view(ply_path, out_json_path,
                       window_title="TERRAMAP // SHIFT+CLICK TO MARK POI, THEN CLOSE WINDOW"):
    """
    Blocking call: opens an Open3D editing window where the user can
    shift+click points to select them. On close, writes the picked XYZ
    coordinates to out_json_path as a JSON list of [x, y, z].
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        with open(out_json_path, "w") as f:
            json.dump([], f)
        return

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_title, width=1100, height=800)
    vis.add_geometry(pcd)
    vis.run()  # blocks until the user closes the window
    vis.destroy_window()

    picked_indices = vis.get_picked_points()
    points = np.asarray(pcd.points)
    picked_coords = [points[i].tolist() for i in picked_indices if i < len(points)]

    with open(out_json_path, "w") as f:
        json.dump(picked_coords, f)


def save_point_cloud_ply(points, colors, out_ply_path):
    """Helper used by the main app to serialize a numpy cloud to .ply."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(colors, dtype=np.float64) / 255.0)
    o3d.io.write_point_cloud(out_ply_path, pcd)
    return out_ply_path


# Allows this module to also be invoked as a standalone script, e.g.:
#   python o3d_viewer.py view path/to/cloud.ply
#   python o3d_viewer.py pick path/to/cloud.ply path/to/out.json
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "view"
    if mode == "view":
        open_interactive_view(sys.argv[2])
    elif mode == "pick":
        open_picking_view(sys.argv[2], sys.argv[3])
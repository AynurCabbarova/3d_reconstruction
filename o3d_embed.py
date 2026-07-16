"""
o3d_embed.py
Open3D viewer meant to be launched as a separate OS process and then
embedded into the Qt UI (the parent process finds this window by its
unique title and wraps it with QWindow.fromWinId + createWindowContainer).

Live picking uses VisualizerWithVertexSelection, whose
register_selection_changed_callback is the OFFICIAL Open3D API for being
notified mid-session (unlike polling VisualizerWithEditing's
get_picked_points during the loop, which hard-crashes some builds).

Controls inside the view:
  - left-drag           orbit
  - wheel               zoom
  - CLICK on a point    select it (fires the callback -> POI)
  - drag a rectangle    select many points at once

Every selection change writes the full list of selected XYZ coords to
picks_json_path atomically (tmp + rename); the parent polls that file.
"""

import os
import json

import open3d as o3d


def _write_picks(picks_json_path, coords):
    tmp = picks_json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(coords, f)
    os.replace(tmp, picks_json_path)


def run_embedded_view(ply_path, window_title, picks_json_path=None):
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        return

    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name=window_title, width=800, height=600)
    vis.add_geometry(pcd)

    if picks_json_path is not None:
        def on_selection_changed():
            try:
                picked = vis.get_picked_points()  # list of PickedPoint
                coords = [[float(p.coord[0]), float(p.coord[1]),
                           float(p.coord[2])] for p in picked]
                _write_picks(picks_json_path, coords)
            except Exception:
                pass  # never let a write problem take the viewer down

        vis.register_selection_changed_callback(on_selection_changed)

    opt = vis.get_render_option()
    opt.background_color = (10 / 255, 13 / 255, 8 / 255)  # matches app BG_0
    opt.point_size = 3.0

    # start orientation = same as the panel's POINT CLOUD view (identity
    # rotation: camera behind the scene on -Z, looking toward +Z, Y up)
    ctr = vis.get_view_control()
    ctr.set_lookat(pcd.get_center())
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(0.7)

    while vis.poll_events():
        vis.update_renderer()

    # final flush on close
    if picks_json_path is not None:
        try:
            picked = vis.get_picked_points()
            coords = [[float(p.coord[0]), float(p.coord[1]),
                       float(p.coord[2])] for p in picked]
            _write_picks(picks_json_path, coords)
        except Exception:
            pass

    vis.destroy_window()
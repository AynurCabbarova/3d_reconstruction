"""
rasterizer.py
Dependency-light rendering used for TerraMap's embedded 3D/2D views.

Only numpy + PIL — no matplotlib, no OpenGL/EGL — so it can never crash from
a missing GPU/display driver. Supports the "product" modes of the app:

  POINT CLOUD  - free-orbit colored point cloud (mouse drag to rotate)
  ORTHOPHOTO   - top-down color mosaic (nadir view; drag pans, wheel zooms)
  DEM/DSM      - top-down height raster, one mode with two aggregations:
                   "dsm" = top-most surface per cell (buildings/canopy included)
                   "dem" = approximate bare-earth (low-percentile height per
                           cell — a lightweight stand-in for real ground
                           classification, not survey-grade)
  DEPTH PRO    - colorized single-frame metric depth map (from Apple's
                 Depth Pro model, see depth_pro_engine.py). This is a plain
                 2D raster display, independent of the point cloud's
                 coordinate frame — see render_depth_map().

Coordinate convention: Y is "up". (VGGT's raw output is camera convention,
Y-down — vggt_engine.py flips that sign once at the source.)
"""

import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- #
# shared projection math (POINT CLOUD / ORTHOPHOTO)
# --------------------------------------------------------------------------- #
def _rotation_matrix(azimuth_deg, elevation_deg):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    cz, sz = np.cos(az), np.sin(az)
    Ry = np.array([
        [cz, 0, sz],
        [0, 1, 0],
        [-sz, 0, cz],
    ])

    ce, se = np.cos(el), np.sin(el)
    Rx = np.array([
        [1, 0, 0],
        [0, ce, -se],
        [0, se, ce],
    ])

    return Rx @ Ry


def _project_points(points, azimuth_deg, elevation_deg, width, height,
                     zoom=1.0, margin=0.08, pan_px=(0.0, 0.0)):
    R = _rotation_matrix(azimuth_deg, elevation_deg)
    rotated = points @ R.T

    lo = np.percentile(rotated, 1.0, axis=0)
    hi = np.percentile(rotated, 99.0, axis=0)
    span = np.maximum(hi - lo, 1e-6)

    usable_w = width * (1 - 2 * margin)
    usable_h = height * (1 - 2 * margin)
    scale = min(usable_w / span[0], usable_h / span[1]) * zoom

    cx = (lo[0] + hi[0]) / 2.0
    cy = (lo[1] + hi[1]) / 2.0

    px = (rotated[:, 0] - cx) * scale + width / 2 + pan_px[0]
    py = height / 2 - (rotated[:, 1] - cy) * scale + pan_px[1]
    depth = rotated[:, 2]

    return px, py, depth, R, cx, cy, scale


def _project_single(coord, R, cx, cy, scale, width, height, pan_px=(0.0, 0.0)):
    v = np.asarray(coord) @ R.T
    sx = (v[0] - cx) * scale + width / 2 + pan_px[0]
    sy = height / 2 - (v[1] - cy) * scale + pan_px[1]
    return sx, sy


# --------------------------------------------------------------------------- #
# perspective projection — used only by POINT CLOUD mode, where real motion
# parallax (near things move more than far things as you rotate) is what
# makes a point cloud actually read as 3D instead of a flat scatter. The
# ORTHOPHOTO / DEM / DSM modes correctly stay orthographic (a real
# orthophoto/DEM *is* an orthographic projection, by definition).
# --------------------------------------------------------------------------- #
def _project_points_perspective(points, azimuth_deg, elevation_deg, width, height,
                                 zoom=1.0, fov_deg=50.0, margin=0.85):
    R = _rotation_matrix(azimuth_deg, elevation_deg)
    rotated = points @ R.T

    lo = np.percentile(rotated, 1.0, axis=0)
    hi = np.percentile(rotated, 99.0, axis=0)
    center = (lo + hi) / 2.0
    radius = max(float(np.max(hi - lo)) / 2.0, 1e-6)

    half_fov = np.radians(fov_deg / 2.0)
    # camera distance chosen so the whole (percentile) bounding sphere fits
    # in frame with a bit of margin, then zoom narrows the effective FOV
    cam_dist = radius / (np.tan(half_fov) * margin)
    focal_px = (height / 2.0) / np.tan(half_fov) * zoom

    # camera sits behind the object along -Z (looking toward +Z)
    depth_from_cam = (rotated[:, 2] - center[2]) + cam_dist
    near_clip = cam_dist * 0.05
    safe_depth = np.where(depth_from_cam > near_clip, depth_from_cam, np.nan)

    px = width / 2 + focal_px * (rotated[:, 0] - center[0]) / safe_depth
    py = height / 2 - focal_px * (rotated[:, 1] - center[1]) / safe_depth

    return px, py, depth_from_cam, R, center, focal_px, cam_dist


def _project_single_perspective(coord, R, center, focal_px, cam_dist, width, height):
    v = np.asarray(coord) @ R.T
    depth_from_cam = (v[2] - center[2]) + cam_dist
    if depth_from_cam <= cam_dist * 0.05:
        return None, None
    sx = width / 2 + focal_px * (v[0] - center[0]) / depth_from_cam
    sy = height / 2 - focal_px * (v[1] - center[1]) / depth_from_cam
    return sx, sy


def _eye_dome_lighting(depth_buffer, valid_mask, strength=1.6):
    """Classic Eye-Dome-Lighting (Boucheny 2009): darkens pixels that sit
    behind their screen-space neighbors, producing contact-shadow-like
    contouring around depth edges. This is what makes raw point splats
    read as a shaded, solid surface instead of a flat scatter of dots —
    no normals or meshing required, just the depth buffer we already have."""
    import warnings
    d = np.where(valid_mask, depth_buffer.astype(np.float64), np.nan)

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    contribs = []
    for dy, dx in offsets:
        neighbor = np.roll(np.roll(d, dy, axis=0), dx, axis=1)
        contribs.append(np.fmax(0.0, d - neighbor))  # positive: this pixel is behind its neighbor
    stacked = np.stack(contribs, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        response = np.nanmean(stacked, axis=0)
    response = np.nan_to_num(response, nan=0.0)

    valid_d = d[valid_mask]
    if valid_d.size > 0:
        scale = np.nanpercentile(valid_d, 95) - np.nanpercentile(valid_d, 5)
    else:
        scale = 1.0
    scale = max(scale, 1e-6)

    shade = np.exp(-strength * 12.0 * response / scale)
    shade = np.clip(shade, 0.25, 1.0)
    shade[~valid_mask] = 1.0
    return shade


def pick_nearest_index(points, azimuth_deg, elevation_deg, width, height,
                        click_x, click_y, zoom=1.0, margin=0.08,
                        pan_px=(0.0, 0.0), max_dist_px=14, perspective=False):
    """Return the index of the visible point nearest the click, or None."""
    if points is None or len(points) == 0:
        return None
    if perspective:
        px, py, depth, *_ = _project_points_perspective(
            points, azimuth_deg, elevation_deg, width, height, zoom)
        finite = np.isfinite(px) & np.isfinite(py)
    else:
        px, py, depth, *_ = _project_points(points, azimuth_deg, elevation_deg,
                                             width, height, zoom, margin, pan_px)
        finite = np.ones(px.shape, dtype=bool)
    d2 = (px - click_x) ** 2 + (py - click_y) ** 2
    within = finite & (d2 <= max_dist_px ** 2)
    if not np.any(within):
        return None
    candidates = np.where(within)[0]
    best = candidates[np.argmin(depth[candidates])]
    return int(best)


# --------------------------------------------------------------------------- #
# POINT CLOUD mode — free orbit, mouse-controlled, perspective + shading
# --------------------------------------------------------------------------- #
def render_point_cloud(points, colors, width=900, height=650,
                        azimuth_deg=35.0, elevation_deg=20.0, zoom=1.0,
                        bg_rgb=(10, 14, 8), point_px=None, markers=None,
                        marker_rgb=(193, 64, 31), margin=0.08,
                        pan_px=(0.0, 0.0), max_render_points=220_000,
                        perspective=True, edl_strength=1.6):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = bg_rgb
    img = Image.fromarray(canvas)

    if points is None or len(points) == 0:
        return img

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n > max_render_points:
        idx = np.random.default_rng(0).choice(n, max_render_points, replace=False)
        pts = pts[idx]
        colors = colors[idx] if colors is not None else None

    if perspective:
        px, py, depth, R, center, focal_px, cam_dist = _project_points_perspective(
            pts, azimuth_deg, elevation_deg, width, height, zoom)
        finite = np.isfinite(px) & np.isfinite(py)
        marker_args = ("perspective", R, center, focal_px, cam_dist)
    else:
        px, py, depth, R, cx, cy, scale = _project_points(
            pts, azimuth_deg, elevation_deg, width, height, zoom, margin, pan_px)
        finite = np.ones(px.shape, dtype=bool)
        marker_args = ("orthographic", R, cx, cy, scale, pan_px)

    valid = finite & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py, depth = px[valid], py[valid], depth[valid]
    pt_colors = colors[valid] if colors is not None else None
    if px.size == 0:
        return _draw_markers_dispatch(img, markers, marker_args, width, height, marker_rgb)

    base = (np.tile(np.array([255, 176, 0], dtype=np.float64), (px.size, 1))
            if pt_colors is None else pt_colors.astype(np.float64))

    order = np.argsort(-depth)
    xi = px[order].astype(np.int32)
    yi = py[order].astype(np.int32)
    ci = base[order]
    di = depth[order]

    if point_px is None:
        point_px = 1 if zoom < 1.3 else (2 if zoom < 3.0 else 3)

    canvas_arr = np.array(img)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float64)
    touched = np.zeros((height, width), dtype=bool)

    if point_px <= 1:
        canvas_arr[yi, xi] = ci.astype(np.uint8)
        depth_buffer[yi, xi] = di
        touched[yi, xi] = True
    else:
        r = point_px // 2
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                xx = np.clip(xi + dx, 0, width - 1)
                yy = np.clip(yi + dy, 0, height - 1)
                canvas_arr[yy, xx] = ci.astype(np.uint8)
                depth_buffer[yy, xx] = di
                touched[yy, xx] = True

    # Eye-Dome-Lighting: shade by depth-buffer contouring so the cloud reads
    # as a shaded solid surface instead of a flat scatter of colored dots.
    # Skippable (edl_strength<=0) for a fast path during interactive drag —
    # this is the single most expensive step in the whole render.
    if edl_strength > 0:
        shade = _eye_dome_lighting(depth_buffer, touched, strength=edl_strength)
        canvas_arr = np.clip(canvas_arr.astype(np.float64) * shade[:, :, None], 0, 255).astype(np.uint8)

    img = Image.fromarray(canvas_arr)
    return _draw_markers_dispatch(img, markers, marker_args, width, height, marker_rgb)


def _draw_markers_dispatch(img, markers, marker_args, width, height, marker_rgb):
    if not markers:
        return img
    kind = marker_args[0]
    draw = ImageDraw.Draw(img)
    for m in markers:
        if kind == "perspective":
            _, R, center, focal_px, cam_dist = marker_args
            sx, sy = _project_single_perspective(m["coord"], R, center, focal_px,
                                                   cam_dist, width, height)
            if sx is None:
                continue
        else:
            _, R, cx, cy, scale, pan_px = marker_args
            sx, sy = _project_single(m["coord"], R, cx, cy, scale, width, height, pan_px)
        if 0 <= sx < width and 0 <= sy < height:
            s = 6
            draw.polygon(
                [(sx, sy - s), (sx - s, sy + s), (sx + s, sy + s)],
                fill=marker_rgb, outline=(216, 226, 196))
            label = m.get("label", "")
            if label:
                draw.text((sx + 8, sy - 6), label, fill=(216, 226, 196))
    return img


# --------------------------------------------------------------------------- #
# ORTHOPHOTO mode — fixed top-down color mosaic (nadir view)
# --------------------------------------------------------------------------- #
def render_orthophoto(points, colors, width=900, height=650, zoom=1.0,
                       markers=None, bg_rgb=(10, 14, 8), pan_px=(0.0, 0.0),
                       max_render_points=260_000):
    return render_point_cloud(points, colors, width=width, height=height,
                               azimuth_deg=0.0, elevation_deg=89.999, zoom=zoom,
                               bg_rgb=bg_rgb, point_px=2, markers=markers,
                               margin=0.05, pan_px=pan_px,
                               max_render_points=max_render_points,
                               perspective=False)


# --------------------------------------------------------------------------- #
# DEM/DSM mode — fixed top-down height raster (one mode, two aggregations)
# --------------------------------------------------------------------------- #
def _elevation_colormap(t):
    """t: (N,) in [0,1] -> (N,3). Hand-rolled terrain gradient
    (deep blue -> green -> tan -> brown -> white), no matplotlib."""
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    colors = np.array([
        [40, 60, 150],
        [60, 150, 70],
        [210, 195, 90],
        [150, 95, 45],
        [255, 255, 255],
    ], dtype=np.float64)
    t = np.clip(t, 0.0, 1.0)
    r = np.interp(t, stops, colors[:, 0])
    g = np.interp(t, stops, colors[:, 1])
    b = np.interp(t, stops, colors[:, 2])
    return np.stack([r, g, b], axis=-1)


def _viridis_like_colormap(t):
    """t: (N,) in [0,1] -> (N,3). Hand-rolled viridis-ish gradient for
    depth maps (dark purple -> blue -> green -> yellow), no matplotlib."""
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


def _fill_holes(grid, iterations=4):
    import warnings
    grid = grid.copy()
    for _ in range(iterations):
        nan_mask = np.isnan(grid)
        if not nan_mask.any():
            break
        shifted = [np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
                   for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        stack = np.stack(shifted, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            neighbor_mean = np.nanmean(stack, axis=0)
        fillable = nan_mask & ~np.isnan(neighbor_mean)
        grid[fillable] = neighbor_mean[fillable]
    return grid


def render_height_map(points, width=900, height=650, mode="dsm", zoom=1.0,
                       markers=None, bg_rgb=(10, 14, 8), grid_res=260,
                       ground_percentile=15.0, pan_px=(0.0, 0.0),
                       max_points=280_000):
    """
    mode: "dsm" (top surface, max height per cell) or
          "dem" (approximate bare-earth, low-percentile height per cell).
    zoom > 1 shrinks the world extent mapped into the grid (sharp zoom,
    not a blurry image upscale); pan_px shifts the final raster on screen.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = bg_rgb
    img = Image.fromarray(canvas)

    if points is None or len(points) == 0:
        return img

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    R = _rotation_matrix(0.0, 89.999)
    rotated = pts @ R.T
    xs, zs, heights = rotated[:, 0], rotated[:, 1], rotated[:, 2]

    lo = np.percentile(np.stack([xs, zs], axis=1), 1.0, axis=0)
    hi = np.percentile(np.stack([xs, zs], axis=1), 99.0, axis=0)
    span = np.maximum(hi - lo, 1e-6)

    # zoom: shrink the world extent mapped into the fixed-resolution grid
    center = (lo + hi) / 2.0
    half_span = (span / 2.0) / max(zoom, 1e-3)
    lo = center - half_span
    hi = center + half_span
    span = np.maximum(hi - lo, 1e-6)

    col = np.clip(((xs - lo[0]) / span[0] * (grid_res - 1)).astype(np.int32), 0, grid_res - 1)
    row = np.clip(((zs - lo[1]) / span[1] * (grid_res - 1)).astype(np.int32), 0, grid_res - 1)
    flat_idx = row * grid_res + col

    if mode == "dsm":
        agg = np.full(grid_res * grid_res, -np.inf)
        np.maximum.at(agg, flat_idx, heights)
        agg[np.isinf(agg)] = np.nan
    else:
        sort_order = np.argsort(flat_idx, kind="stable")
        sorted_flat = flat_idx[sort_order]
        sorted_heights = heights[sort_order]
        unique_cells, start_idx, counts = np.unique(
            sorted_flat, return_index=True, return_counts=True)
        agg = np.full(grid_res * grid_res, np.nan)
        for uc, si, cnt in zip(unique_cells, start_idx, counts):
            agg[uc] = np.percentile(sorted_heights[si:si + cnt], ground_percentile)

    grid = agg.reshape(grid_res, grid_res)
    grid = _fill_holes(grid, iterations=4)

    valid_mask = ~np.isnan(grid)
    if not valid_mask.any():
        return img
    gmin, gmax = np.nanpercentile(grid, 2), np.nanpercentile(grid, 98)
    t = np.clip((grid - gmin) / max(gmax - gmin, 1e-6), 0, 1)
    t = np.nan_to_num(t, nan=0.0)

    rgb_grid = _elevation_colormap(t.ravel()).reshape(grid_res, grid_res, 3)
    rgb_grid[~valid_mask] = bg_rgb
    small_img = Image.fromarray(rgb_grid.astype(np.uint8))
    resized = small_img.resize((width, height), Image.BILINEAR)

    final = Image.new("RGB", (width, height), bg_rgb)
    final.paste(resized, (int(round(pan_px[0])), int(round(pan_px[1]))))

    def world_to_screen(coord):
        v = np.asarray(coord) @ R.T
        gc = (v[0] - lo[0]) / span[0] * (grid_res - 1)
        gr = (v[1] - lo[1]) / span[1] * (grid_res - 1)
        sx = gc / (grid_res - 1) * width + pan_px[0]
        sy = gr / (grid_res - 1) * height + pan_px[1]
        return sx, sy

    if markers:
        draw = ImageDraw.Draw(final)
        for m in markers:
            sx, sy = world_to_screen(m["coord"])
            if 0 <= sx < width and 0 <= sy < height:
                s = 6
                draw.polygon(
                    [(sx, sy - s), (sx - s, sy + s), (sx + s, sy + s)],
                    fill=(193, 64, 31), outline=(216, 226, 196))
                label = m.get("label", "")
                if label:
                    draw.text((sx + 8, sy - 6), label, fill=(216, 226, 196))

    return final


# --------------------------------------------------------------------------- #
# DEPTH PRO mode — colorized single-frame metric depth map (2D, informational)
# --------------------------------------------------------------------------- #
def render_depth_map(depth_m, width=900, height=650, zoom=1.0, pan_px=(0.0, 0.0),
                      bg_rgb=(10, 14, 8), valid_range=None):
    """
    depth_m: (H,W) float array, metric depth in meters (from Depth Pro).
    Returns a colorized PIL Image plus nothing else — this is a plain 2D
    raster, not a spatial point cloud, so there is no marker/pick support
    tied to the reconstruction's coordinate frame here.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = bg_rgb
    if depth_m is None:
        return Image.fromarray(canvas)

    d = np.asarray(depth_m, dtype=np.float64)
    finite = np.isfinite(d)
    if not finite.any():
        return Image.fromarray(canvas)

    if valid_range is not None:
        dmin, dmax = valid_range
    else:
        dmin, dmax = np.percentile(d[finite], 2), np.percentile(d[finite], 98)

    t = np.clip((d - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    t = np.nan_to_num(t, nan=0.0)
    rgb = _viridis_like_colormap(t.ravel()).reshape(d.shape[0], d.shape[1], 3).astype(np.uint8)
    small_img = Image.fromarray(rgb)

    # zoom by cropping the source image toward its center, then resizing up
    h0, w0 = d.shape
    crop_w = max(4, int(w0 / max(zoom, 1e-3)))
    crop_h = max(4, int(h0 / max(zoom, 1e-3)))
    cx0, cy0 = w0 // 2, h0 // 2
    left = int(np.clip(cx0 - crop_w // 2, 0, max(w0 - crop_w, 0)))
    top = int(np.clip(cy0 - crop_h // 2, 0, max(h0 - crop_h, 0)))
    cropped = small_img.crop((left, top, left + crop_w, top + crop_h))
    resized = cropped.resize((width, height), Image.BILINEAR)

    final = Image.new("RGB", (width, height), bg_rgb)
    final.paste(resized, (int(round(pan_px[0])), int(round(pan_px[1]))))
    return final
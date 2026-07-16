"""
TERRAMAP RECON SUITE — PySide6 edition
Video-to-3D terrain reconstruction workstation.

Full rewrite of the original customtkinter UI in Qt (PySide6). The
non-UI modules are UNCHANGED and reused as-is:

    rasterizer.py        - CPU preview renderer (numpy + PIL)
    vggt_engine.py       - VGGT reconstruction wrapper
    depth_pro_engine.py  - Apple Depth Pro wrapper
    o3d_viewer.py        - Open3D picking window (separate process)
    o3d_embed.py         - Open3D embeddable viewer (separate process)

Why Qt instead of Tk:
  - GPU view embedding is done with the OFFICIAL Qt API
    (QWindow.fromWinId + createWindowContainer) instead of raw
    win32 SetParent hacks. The Open3D window still runs in its own
    OS process, so a graphics-driver crash can never take the UI down.
  - Proper thread -> UI signal system (no manual queue draining).

Install:
    pip install PySide6 opencv-python pillow numpy open3d
    pip install pywin32          # Windows only, for embed window lookup
"""

import os
import sys
import json
import tempfile
import traceback
import threading
import multiprocessing
import datetime as dt

import cv2
import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize
from PySide6.QtGui import QImage, QPixmap, QFont, QGuiApplication, QWindow
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QSlider, QLineEdit, QVBoxLayout, QHBoxLayout, QGridLayout,
    QScrollArea, QPlainTextEdit, QFileDialog, QMessageBox, QInputDialog,
    QSizePolicy, QStackedLayout,
)

from rasterizer import render_point_cloud, pick_nearest_index
import o3d_viewer
import o3d_embed
import vggt_engine
import trellis_engine
import hunyuan_engine
import depth_pro_engine
from globe_panel import GlobePanel

try:
    import win32gui
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

# --------------------------------------------------------------------------- #
# THEME — dark tactical palette (same as the Tk version)
# --------------------------------------------------------------------------- #
BG_0 = "#0a0d08"
BG_1 = "#12160d"
BG_2 = "#1a2013"
LINE = "#33401f"
OLIVE = "#4a5d2a"
AMBER = "#ffb000"
AMBER_DIM = "#8a6100"
TEXT_0 = "#d8e2c4"
TEXT_1 = "#7f8f6a"
DANGER = "#c1401f"
OK_GREEN = "#6fae3a"

MONO = "Consolas" if sys.platform == "win32" else "Monospace"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_0};
    color: {TEXT_0};
    font-family: "{MONO}";
}}
QFrame#panel {{
    background-color: {BG_1};
    border: 1px solid {LINE};
    border-radius: 4px;
}}
QFrame#chip {{
    background-color: {BG_2};
    border: 1px solid {LINE};
    border-radius: 2px;
}}
QFrame#poiRow {{
    background-color: {BG_2};
    border: 1px solid {LINE};
    border-radius: 2px;
}}
QLabel {{ background: transparent; border: none; }}
QLabel#canvas {{
    background-color: {BG_0};
    border: 1px solid {LINE};
    color: {TEXT_1};
}}
QPushButton {{
    background-color: {OLIVE};
    color: {TEXT_0};
    border: none;
    border-radius: 3px;
    padding: 7px 12px;
    font-weight: bold;
    font-size: 11px;
}}
QPushButton:hover {{ background-color: {AMBER_DIM}; }}
QPushButton#primary {{ background-color: {AMBER}; color: {BG_0}; }}
QPushButton#primary:hover {{ background-color: #cc8e00; }}
QPushButton#danger {{ background-color: {DANGER}; }}
QPushButton#danger:hover {{ background-color: #8a2c15; }}
QPushButton#modeBtn {{
    background-color: {BG_2};
    color: {TEXT_0};
    padding: 5px 10px;
    font-size: 10px;
}}
QPushButton#modeBtn:checked {{ background-color: {AMBER}; color: {BG_0}; }}
QPushButton#deleteBtn {{
    background-color: {DANGER};
    padding: 1px 6px;
    font-size: 10px;
}}
QLineEdit {{
    background-color: {BG_2};
    border: 1px solid {LINE};
    border-radius: 2px;
    color: {TEXT_0};
    padding: 3px 6px;
    font-size: 10px;
}}
QSlider::groove:horizontal {{
    height: 4px; background: {BG_2}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {AMBER}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; margin: -6px 0; border-radius: 7px; background: {AMBER};
}}
QPlainTextEdit#logBox {{
    background-color: {BG_0};
    color: {OK_GREEN};
    border: 1px solid {LINE};
    font-size: 11px;
}}
QScrollArea {{ border: 1px solid {LINE}; background-color: {BG_0}; }}
QScrollArea > QWidget > QWidget {{ background-color: {BG_0}; }}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: {BG_1}; width: 10px; height: 10px;
}}
QScrollBar::handle {{ background: {LINE}; border-radius: 4px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
"""


def ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def pil_to_pixmap(img: Image.Image) -> QPixmap:
    """PIL RGB image -> QPixmap without ImageQt (fewer version pitfalls)."""
    arr = np.ascontiguousarray(np.asarray(img.convert("RGB"), dtype=np.uint8))
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def frame_bgr_to_pixmap(frame_bgr, size=None) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    if size is not None:
        img = img.resize(size)
    return pil_to_pixmap(img)


# --------------------------------------------------------------------------- #
# Thread -> UI bridge (replaces the Tk log queue + .after() dance)
# --------------------------------------------------------------------------- #
class Bridge(QObject):
    log = Signal(str)
    cloud_ready = Signal(object, object, dict)     # points, colors, stats
    depth_ready = Signal(object, object)           # points, colors
    glb_ready = Signal(object, object, dict)       # sampled points, colors, stats


BRIDGE = Bridge()


# --------------------------------------------------------------------------- #
# small reusable tactical widgets
# --------------------------------------------------------------------------- #
class SectionHeader(QWidget):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        bar = QFrame()
        bar.setFixedSize(4, 16)
        bar.setStyleSheet(f"background-color: {AMBER};")
        lay.addWidget(bar)
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {AMBER}; font-size: 13px; font-weight: bold;")
        lay.addWidget(lbl)
        lay.addStretch()


class StatChip(QFrame):
    def __init__(self, label, value="--", parent=None):
        super().__init__(parent)
        self.setObjectName("chip")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(0)
        top = QLabel(label)
        top.setStyleSheet(f"color: {TEXT_1}; font-size: 9px;")
        lay.addWidget(top)
        self.value_lbl = QLabel(str(value))
        self.value_lbl.setStyleSheet(
            f"color: {TEXT_0}; font-size: 15px; font-weight: bold;")
        lay.addWidget(self.value_lbl)

    def set(self, value):
        self.value_lbl.setText(str(value))


class CanvasLabel(QLabel):
    """The rasterizer preview surface. Emits raw mouse interaction signals."""
    pressed = Signal(int, int)
    dragged = Signal(int, int)
    released = Signal(int, int)
    wheeled = Signal(int)
    resized = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("canvas")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setText("NO RECONSTRUCTION LOADED\n\n"
                     "drag = rotate/pan  ·  wheel = zoom  ·  click = mark POI")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.pressed.emit(int(e.position().x()), int(e.position().y()))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self.dragged.emit(int(e.position().x()), int(e.position().y()))

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.released.emit(int(e.position().x()), int(e.position().y()))

    def wheelEvent(self, e):
        self.wheeled.emit(1 if e.angleDelta().y() > 0 else -1)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.resized.emit()


# --------------------------------------------------------------------------- #
# VIDEO PANEL
# --------------------------------------------------------------------------- #
class VideoPanel(QFrame):
    def __init__(self, log_fn, work_dir, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.log = log_fn
        self.work_dir = work_dir
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.fps = 30
        self.current_idx = 0
        self.extracted_frame_paths = []
        self.trellis_frame_paths = []

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        lay.addWidget(SectionHeader("◈ VIDEO FEED"))

        self.preview = QLabel("NO SIGNAL")
        self.preview.setObjectName("canvas")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedSize(340, 220)
        lay.addWidget(self.preview, alignment=Qt.AlignHCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_seek)
        lay.addWidget(self.slider)

        self.frame_lbl = QLabel("FRAME 0 / 0")
        self.frame_lbl.setStyleSheet(f"color: {TEXT_1}; font-size: 10px;")
        lay.addWidget(self.frame_lbl)

        ctrl = QHBoxLayout()
        self.play_btn = QPushButton("▶ PLAY")
        self.play_btn.clicked.connect(self.toggle_play)
        ctrl.addWidget(self.play_btn)
        ctrl.addStretch()
        load_btn = QPushButton("LOAD VIDEO")
        load_btn.setObjectName("primary")
        load_btn.clicked.connect(self.load_video)
        ctrl.addWidget(load_btn)
        lay.addLayout(ctrl)

        lay.addWidget(SectionHeader("◈ EXTRACTED FRAMES"))

        row1 = QHBoxLayout()
        row1.addWidget(self._dim_label("SAMPLE EVERY"))
        self.interval_entry = QLineEdit("15")
        self.interval_entry.setFixedWidth(45)
        row1.addWidget(self.interval_entry)
        row1.addWidget(self._dim_label("FRAMES"))
        row1.addStretch()
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(self._dim_label("MAX FRAMES (VGGT)"))
        self.max_frames_entry = QLineEdit("16")
        self.max_frames_entry.setFixedWidth(45)
        row2.addWidget(self.max_frames_entry)
        row2.addStretch()
        lay.addLayout(row2)

        extract_btn = QPushButton("EXTRACT FRAMES")
        extract_btn.clicked.connect(self.extract_frames)
        lay.addWidget(extract_btn)

        trellis_btn = QPushButton("EXTRACT 4 FRAMES (GEN-3D)")
        trellis_btn.clicked.connect(lambda: self.extract_trellis_frames(4))
        lay.addWidget(trellis_btn)

        self.thumb_scroll = QScrollArea()
        self.thumb_scroll.setWidgetResizable(True)
        self.thumb_scroll.setFixedHeight(90)
        self.thumb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.thumb_host = QWidget()
        self.thumb_lay = QHBoxLayout(self.thumb_host)
        self.thumb_lay.setContentsMargins(3, 6, 3, 6)
        self.thumb_lay.setSpacing(3)
        self.thumb_lay.addStretch()
        self.thumb_scroll.setWidget(self.thumb_host)
        lay.addWidget(self.thumb_scroll)

        lay.addStretch()

    @staticmethod
    def _dim_label(text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {TEXT_1}; font-size: 10px;")
        return lbl

    # -- video handling ----------------------------------------------------
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select recon video", "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)")
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Load failed", "Could not open video file.")
            return
        self.cap = cap
        self.video_path = path
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(self.total_frames - 1, 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.current_idx = 0
        self.extracted_frame_paths.clear()
        self._clear_thumbs()
        self._show_frame(0)
        self.log(f"[{ts()}] VIDEO LOADED: {os.path.basename(path)} "
                 f"({self.total_frames} frames @ {self.fps:.1f}fps)")

    def _clear_thumbs(self):
        while self.thumb_lay.count() > 1:  # keep the trailing stretch
            item = self.thumb_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_frame(self, idx):
        if self.cap is None:
            return
        idx = max(0, min(int(idx), self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_idx = idx
        self.preview.setPixmap(frame_bgr_to_pixmap(frame, size=(340, 220)))
        self.frame_lbl.setText(f"FRAME {idx} / {self.total_frames - 1}")

    def _on_seek(self, value):
        self._show_frame(value)

    def toggle_play(self):
        if self.cap is None:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_btn.setText("▶ PLAY")
        else:
            self.play_btn.setText("⏸ PAUSE")
            self.play_timer.start(max(int(1000 / (self.fps or 30)), 15))

    def _play_tick(self):
        nxt = self.current_idx + 1
        if nxt >= self.total_frames:
            self.play_timer.stop()
            self.play_btn.setText("▶ PLAY")
            return
        self.slider.blockSignals(True)
        self.slider.setValue(nxt)
        self.slider.blockSignals(False)
        self._show_frame(nxt)

    def extract_frames(self):
        if self.cap is None:
            QMessageBox.warning(self, "No video", "Load a video first.")
            return
        try:
            interval = max(1, int(self.interval_entry.text()))
        except ValueError:
            interval = 15
        try:
            max_frames = max(1, int(self.max_frames_entry.text()))
        except ValueError:
            max_frames = 16

        self.extracted_frame_paths.clear()
        self._clear_thumbs()

        frames_dir = os.path.join(self.work_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for f in os.listdir(frames_dir):
            try:
                os.remove(os.path.join(frames_dir, f))
            except OSError:
                pass

        idx = 0
        count = 0
        while idx < self.total_frames and count < max_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if not ok:
                break

            out_path = os.path.join(frames_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(out_path, frame)
            self.extracted_frame_paths.append(out_path)

            thumb = QLabel()
            thumb.setPixmap(frame_bgr_to_pixmap(frame, size=(64, 42)))
            thumb.setFixedSize(64, 42)
            self.thumb_lay.insertWidget(self.thumb_lay.count() - 1, thumb)

            idx += interval
            count += 1

        self.log(f"[{ts()}] EXTRACTED {count} FRAMES -> {frames_dir} "
                 f"(interval={interval}, cap={max_frames})")

    def extract_trellis_frames(self, n=4):
        """Grab n frames spread evenly across the whole video (for a 40s
        clip and n=4: ~0s, 10s, 20s, 30s), for TRELLIS.2 generation."""
        n = int(n) if n else 4   # guard: Qt's clicked(bool) can pass False
        if n < 1:
            n = 4
        if self.cap is None:
            QMessageBox.warning(self, "No video", "Load a video first.")
            return

        self.trellis_frame_paths.clear()
        self._clear_thumbs()

        frames_dir = os.path.join(self.work_dir, "trellis_frames")
        os.makedirs(frames_dir, exist_ok=True)
        for f in os.listdir(frames_dir):
            try:
                os.remove(os.path.join(frames_dir, f))
            except OSError:
                pass

        # evenly spaced: 0, T/n, 2T/n, ... — never the very last index so a
        # short/broken tail frame can't fail the read
        indices = [int(i * self.total_frames / n) for i in range(n)]
        count = 0
        for i, idx in enumerate(indices):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, self.total_frames - 1))
            ok, frame = self.cap.read()
            if not ok:
                continue
            out_path = os.path.join(frames_dir, f"tframe_{i:02d}.jpg")
            cv2.imwrite(out_path, frame)
            self.trellis_frame_paths.append(out_path)

            thumb = QLabel()
            thumb.setPixmap(frame_bgr_to_pixmap(frame, size=(64, 42)))
            thumb.setFixedSize(64, 42)
            self.thumb_lay.insertWidget(self.thumb_lay.count() - 1, thumb)
            count += 1

        sec = [f"{idx / (self.fps or 30):.1f}s" for idx in indices[:count]]
        self.log(f"[{ts()}] EXTRACTED {count} TRELLIS FRAMES @ "
                 f"[{', '.join(sec)}] -> {frames_dir}")

    def get_reference_frame_path(self):
        if self.extracted_frame_paths:
            return self.extracted_frame_paths[0]
        if self.cap is None:
            return None
        path = os.path.join(self.work_dir, "depth_pro_reference.jpg")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        cv2.imwrite(path, frame)
        return path


# --------------------------------------------------------------------------- #
# 3D MAP PANEL — rasterizer preview + embedded GPU (Open3D) view
# --------------------------------------------------------------------------- #
class Map3DPanel(QFrame):
    MODES = ["GLOBE", "POINT CLOUD", "DEPTH PRO"]

    def __init__(self, log_fn, work_dir, on_poi_pick, get_depth_frame_path,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.log = log_fn
        self.work_dir = work_dir
        self.on_poi_pick = on_poi_pick
        self.get_depth_frame_path = get_depth_frame_path

        self.points = None
        self.colors = None
        self.aoi = None          # (lat, lon) area of interest set from the globe
        self.render_points = None
        self.render_colors = None
        self.depth_points = None
        self.depth_colors = None
        self.markers = []
        self.mode = "GLOBE"

        self.azimuth = 0.0
        self.elevation = 0.0
        self.zoom = 1.0
        self.ply_path = os.path.join(work_dir, "current_cloud.ply")

        self._drag_start = None
        self._drag_start_az = 0.0
        self._drag_start_el = 0.0
        self._dragged = False
        self._interacting = False

        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(180)
        self._settle_timer.timeout.connect(self._settle)

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(120)
        self._resize_timer.timeout.connect(self._redraw)

        # embedded GPU (Open3D) viewer state
        self.embed_proc = None
        self._embed_title = None
        self._embed_container = None
        self._embed_attempts = 0
        self._embed_poll = QTimer(self)
        self._embed_poll.setInterval(150)
        self._embed_poll.timeout.connect(self._try_capture_embed)

        # polls the picks file written by the embedded viewer (shift+click)
        self._pick_watch = QTimer(self)
        self._pick_watch.setInterval(300)
        self._pick_watch.timeout.connect(self._check_embed_picks)
        self._embed_picks_json = None
        self._embed_picks_seen = set()
        self._embed_pick_source = "GPU view pick"

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.addWidget(SectionHeader("◈ 3D TERRAIN RECONSTRUCTION"))
        header_row.addStretch()
        self.mode_btns = {}
        for m in self.MODES:
            b = QPushButton(m)
            b.setObjectName("modeBtn")
            b.setCheckable(True)
            b.clicked.connect(lambda checked, mm=m: self._on_mode_change(mm))
            header_row.addWidget(b)
            self.mode_btns[m] = b
        self.mode_btns["GLOBE"].setChecked(True)

        # GPU VIEW sits next to the mode buttons, styled the same way,
        # and toggles the embedded Open3D view for the current mode
        self.gpu_btn = QPushButton("⬒ GPU VIEW")
        self.gpu_btn.setObjectName("modeBtn")
        self.gpu_btn.setCheckable(True)
        self.gpu_btn.clicked.connect(self._on_gpu_btn)
        header_row.addWidget(self.gpu_btn)

        lay.addLayout(header_row)

        self.toolbar_row = QHBoxLayout()
        lay.addLayout(self.toolbar_row)

        # stacked: page 0 = rasterizer preview, page 1 = embedded GPU view
        self.stack_host = QWidget()
        self.stack = QStackedLayout(self.stack_host)
        self.stack.setContentsMargins(0, 0, 0, 0)

        self.canvas = CanvasLabel()
        self.canvas.pressed.connect(self._on_press)
        self.canvas.dragged.connect(self._on_drag)
        self.canvas.released.connect(self._on_release)
        self.canvas.wheeled.connect(self._on_wheel)
        self.canvas.resized.connect(lambda: self._resize_timer.start())
        self.stack.addWidget(self.canvas)

        self.embed_page = QWidget()
        self.embed_lay = QVBoxLayout(self.embed_page)
        self.embed_lay.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(self.embed_page)

        # page 2 = the 3D globe (realistic Earth + real terrain of the target)
        self.globe = GlobePanel(log_fn=self.log)
        self.globe.area_entered.connect(self._on_area_entered)
        self.globe.point_picked.connect(self._on_globe_pick)
        self.stack.addWidget(self.globe)

        lay.addWidget(self.stack_host, stretch=1)

        self.stack.setCurrentIndex(2)   # start on the globe

        self._rebuild_toolbar()

    # -- toolbar -----------------------------------------------------------
    def _rebuild_toolbar(self):
        while self.toolbar_row.count():
            item = self.toolbar_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self.mode == "DEPTH PRO":
            b = QPushButton("▷ RUN DEPTH PRO ON CURRENT FRAME")
            b.clicked.connect(self.run_depth_pro)
            self.toolbar_row.addWidget(b)

        self.toolbar_row.addStretch()

    def _on_gpu_btn(self):
        # checkable button: checked -> start embed, unchecked -> stop
        if self.gpu_btn.isChecked():
            self._start_embed()
            # _start_embed may refuse (no data / no pywin32) — uncheck then
            if self.embed_proc is None:
                self.gpu_btn.setChecked(False)
        else:
            self._stop_embed()

    # -- data ----------------------------------------------------------
    def set_cloud(self, points, colors):
        self.points = points
        self.colors = colors
        if self.mode == "GLOBE":
            self._on_mode_change("POINT CLOUD")

        max_pts = 220_000
        if points.shape[0] > max_pts:
            idx = np.random.default_rng(0).choice(points.shape[0], max_pts,
                                                   replace=False)
            self.render_points = points[idx]
            self.render_colors = colors[idx] if colors is not None else None
        else:
            self.render_points = points
            self.render_colors = colors

        try:
            o3d_viewer.save_point_cloud_ply(points, colors, self.ply_path)
        except Exception as e:
            self.log(f"[{ts()}] WARNING: could not write .ply export ({e})")

        self.azimuth = 0.0
        self.elevation = 0.0
        self.zoom = 1.0

        # a running embedded view now shows stale data — restart it
        if self.embed_proc is not None:
            self._stop_embed()
            self._start_embed()
            self.gpu_btn.setChecked(self.embed_proc is not None)

        self._redraw()

    def set_markers(self, markers):
        self.markers = markers
        self._redraw()

    # -- mode switching --------------------------------------------------
    def _on_mode_change(self, value):
        for m, b in self.mode_btns.items():
            b.setChecked(m == value)
        self.mode = value
        if self.embed_proc is not None:
            self._stop_embed()
        self.gpu_btn.setChecked(False)
        self._rebuild_toolbar()
        if value == "GLOBE":
            self.stack.setCurrentIndex(2)
        else:
            self.stack.setCurrentIndex(0)
            self._redraw()

    # -- globe callbacks ---------------------------------------------------
    def _on_area_entered(self, lat, lon):
        self.aoi = (lat, lon)
        self.log(f"[{ts()}] AREA OF INTEREST SET — {lat:.5f}, {lon:.5f} "
                 f"(3D terrain loaded from open tiles)")

    def _on_globe_pick(self, lat, lon):
        self.log(f"[{ts()}] GLOBE PICK — {lat:.5f}, {lon:.5f}")

    def fly_globe_to(self, lat, lon):
        """Called from outside (e.g. a localization result) to drive the globe."""
        self._on_mode_change("GLOBE")
        self.globe.fly_to(lat, lon)

    # -- rendering ---------------------------------------------------------
    def _canvas_size(self):
        w = max(self.canvas.width(), 200)
        h = max(self.canvas.height(), 150)
        return w, h

    def _redraw(self):
        if self.stack.currentIndex() != 0:
            return  # GPU view (1) or globe (2) is showing; nothing to raster

        w, h = self._canvas_size()
        fast = self._interacting

        if self.mode == "DEPTH PRO":
            pts, cols = self.depth_points, self.depth_colors
        else:
            pts, cols = self.render_points, self.render_colors

        if pts is None:
            return

        if fast:
            # interactive: half-res + fewer points, but EDL stays ON so the
            # look doesn't "pop" between drag and settle
            rw, rh = max(w // 2, 160), max(h // 2, 120)
            img = render_point_cloud(pts, cols, width=rw, height=rh,
                                      azimuth_deg=self.azimuth,
                                      elevation_deg=self.elevation,
                                      zoom=self.zoom, markers=self.markers,
                                      max_render_points=60_000, point_px=2,
                                      edl_strength=1.6)
            img = img.resize((w, h), Image.BILINEAR)
        else:
            img = render_point_cloud(pts, cols, width=w, height=h,
                                      azimuth_deg=self.azimuth,
                                      elevation_deg=self.elevation,
                                      zoom=self.zoom, markers=self.markers,
                                      max_render_points=220_000,
                                      edl_strength=1.6)

        self.canvas.setPixmap(pil_to_pixmap(img))

    def _begin_interaction(self):
        self._interacting = True
        self._settle_timer.start()

    def _settle(self):
        self._interacting = False
        self._redraw()

    # -- mouse -------------------------------------------------------------
    def _on_press(self, x, y):
        self._drag_start = (x, y)
        self._drag_start_az = self.azimuth
        self._drag_start_el = self.elevation
        self._dragged = False

    def _on_drag(self, x, y):
        if self._drag_start is None:
            return
        dx = x - self._drag_start[0]
        dy = y - self._drag_start[1]
        if abs(dx) > 3 or abs(dy) > 3:
            self._dragged = True

        self.azimuth = (self._drag_start_az + dx * 0.35) % 360
        self.elevation = float(np.clip(self._drag_start_el - dy * 0.35, -89, 89))
        self._begin_interaction()
        self._redraw()

    def _on_release(self, x, y):
        if not self._dragged:
            self._pick_at(x, y)
        self._drag_start = None

    def _on_wheel(self, direction):
        factor = 1.1 if direction > 0 else (1 / 1.1)
        self.zoom = float(np.clip(self.zoom * factor, 0.2, 8.0))
        self._begin_interaction()
        self._redraw()

    def _pick_at(self, click_x, click_y):
        w, h = self._canvas_size()

        if self.mode == "POINT CLOUD":
            if self.render_points is None:
                return
            idx = pick_nearest_index(self.render_points, self.azimuth,
                                      self.elevation, w, h, click_x, click_y,
                                      zoom=self.zoom, perspective=True)
            if idx is None:
                return
            coord = tuple(float(c) for c in self.render_points[idx])
            self.on_poi_pick(coord, "point cloud pick")

        elif self.mode == "DEPTH PRO":
            if self.depth_points is None:
                return
            idx = pick_nearest_index(self.depth_points, self.azimuth,
                                      self.elevation, w, h, click_x, click_y,
                                      zoom=self.zoom, perspective=True)
            if idx is None:
                return
            coord = tuple(float(c) for c in self.depth_points[idx])
            self.on_poi_pick(coord, "depth pro pick (own camera frame)")

    # -- Depth Pro ------------------------------------------------------
    def run_depth_pro(self):
        if not depth_pro_engine.is_available():
            QMessageBox.critical(
                self, "Depth Pro not installed",
                "The Depth Pro package is not importable in this environment.\n\n"
                "Install it with:\n"
                "  git clone https://github.com/apple/ml-depth-pro.git\n"
                "  cd ml-depth-pro\n  pip install -e .\n"
                "  source get_pretrained_models.sh\n\nThen restart TerraMap.")
            return

        frame_path = self.get_depth_frame_path()
        if frame_path is None:
            QMessageBox.warning(self, "No frame", "Load a video first.")
            return

        self.log(f"[{ts()}] RUN DEPTH PRO — {frame_path}")

        def worker():
            try:
                points, colors, _stats = depth_pro_engine.estimate_point_cloud(
                    frame_path, log_fn=BRIDGE.log.emit)
            except Exception as e:
                BRIDGE.log.emit(f"[{ts()}] ERROR: {e}")
                for line in traceback.format_exc().splitlines():
                    BRIDGE.log.emit(f"    {line}")
                return
            BRIDGE.depth_ready.emit(points, colors)

        threading.Thread(target=worker, daemon=True).start()

    def apply_depth_result(self, points, colors):
        self.depth_points = points
        self.depth_colors = colors
        self.azimuth = 0.0
        self.elevation = 0.0
        self.zoom = 1.0
        self._redraw()

    # -- embedded GPU view (Open3D in-panel, official Qt window container) --
    def _start_embed(self):
        if self.mode == "GLOBE":
            QMessageBox.information(self, "Not applicable",
                                     "GPU VIEW works on POINT CLOUD / DEPTH PRO.")
            return
        if not HAS_WIN32:
            QMessageBox.information(
                self, "Not available",
                "In-panel GPU view needs window lookup support, which is\n"
                "currently implemented for Windows only (pip install pywin32).")
            return

        if self.mode == "DEPTH PRO":
            if self.depth_points is None:
                QMessageBox.information(self, "No data", "Run Depth Pro first.")
                return
            ply = os.path.join(self.work_dir, "depth_pro_cloud.ply")
            try:
                o3d_viewer.save_point_cloud_ply(self.depth_points,
                                                 self.depth_colors, ply)
            except Exception as e:
                self.log(f"[{ts()}] ERROR: could not write depth .ply ({e})")
                return
            pick_source = "GPU view pick (depth pro, own camera frame)"
        else:
            if self.points is None:
                QMessageBox.information(self, "No data", "Run 3D analysis first.")
                return
            ply = self.ply_path
            pick_source = "GPU view pick"

        self._embed_pick_source = pick_source
        self._embed_picks_json = os.path.join(self.work_dir, "embed_picks.json")
        if os.path.exists(self._embed_picks_json):
            try:
                os.remove(self._embed_picks_json)
            except OSError:
                pass
        self._embed_picks_seen = set()

        self._embed_title = f"TERRAMAP_EMBED_{os.getpid()}_{id(self)}"
        self.embed_proc = multiprocessing.Process(
            target=o3d_embed.run_embedded_view,
            args=(ply, self._embed_title, self._embed_picks_json), daemon=True)
        self.embed_proc.start()
        self.log(f"[{ts()}] EMBEDDED GPU VIEW STARTING...")
        self._embed_attempts = 0
        self._embed_poll.start()

    def _try_capture_embed(self):
        if self.embed_proc is None:
            self._embed_poll.stop()
            return
        if not self.embed_proc.is_alive():
            self.log(f"[{ts()}] ERROR: embedded viewer process exited early")
            self._embed_poll.stop()
            self._stop_embed()
            return

        hwnd = win32gui.FindWindow(None, self._embed_title)
        if not hwnd:
            self._embed_attempts += 1
            if self._embed_attempts > 100:   # ~15s
                self.log(f"[{ts()}] ERROR: embedded Open3D window not found")
                self._embed_poll.stop()
                self._stop_embed()
            return

        self._embed_poll.stop()

        foreign = QWindow.fromWinId(hwnd)
        foreign.setFlags(Qt.FramelessWindowHint)
        self._embed_container = QWidget.createWindowContainer(
            foreign, self.embed_page)
        self.embed_lay.addWidget(self._embed_container)
        self.stack.setCurrentIndex(1)
        self._pick_watch.start()
        self.log(f"[{ts()}] EMBEDDED GPU VIEW ACTIVE — orbit/zoom on the panel, "
                 f"CLICK a point = mark POI")

    def _check_embed_picks(self):
        """Poll the picks file the embedded viewer writes; turn any new
        shift+clicked points into POIs immediately."""
        if self.embed_proc is None:
            self._pick_watch.stop()
            return
        if not self.embed_proc.is_alive():
            # user closed the embedded window itself (e.g. it lost its frame
            # but was killed some other way) — collect final picks and reset
            self._pick_watch.stop()
            self._consume_picks()
            self._stop_embed()
            return
        self._consume_picks()

    def _consume_picks(self):
        path = getattr(self, "_embed_picks_json", None)
        if not path or not os.path.exists(path):
            return
        try:
            with open(path) as f:
                coords = json.load(f)
        except Exception:
            return  # mid-write or malformed; next tick will retry
        # selections can shrink too (deselect), so track what we've already
        # added by value instead of assuming an append-only list
        for c in coords:
            key = (round(c[0], 6), round(c[1], 6), round(c[2], 6))
            if key not in self._embed_picks_seen:
                self._embed_picks_seen.add(key)
                self.on_poi_pick(tuple(c), self._embed_pick_source)

    def _stop_embed(self):
        self._embed_poll.stop()
        self._pick_watch.stop()
        self._consume_picks()   # don't lose picks made just before closing
        if self.embed_proc is not None:
            try:
                self.embed_proc.terminate()
            except Exception:
                pass
        self.embed_proc = None
        if self._embed_container is not None:
            self.embed_lay.removeWidget(self._embed_container)
            self._embed_container.deleteLater()
            self._embed_container = None
        self.stack.setCurrentIndex(2 if self.mode == "GLOBE" else 0)
        self.gpu_btn.setChecked(False)
        self.log(f"[{ts()}] EMBEDDED GPU VIEW CLOSED")
        self._redraw()


# --------------------------------------------------------------------------- #
# POI PANEL
# --------------------------------------------------------------------------- #
class POIPanel(QFrame):
    def __init__(self, log_fn, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.log = log_fn
        self.pois = []
        self.map_panel = None
        self._counter = 0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        lay.addWidget(SectionHeader("◈ POINTS OF INTEREST"))

        add_btn = QPushButton("+ ADD POI (MANUAL)")
        add_btn.clicked.connect(self.add_manual)
        lay.addWidget(add_btn)

        self.list_scroll = QScrollArea()
        self.list_scroll.setWidgetResizable(True)
        self.list_host = QWidget()
        self.list_lay = QVBoxLayout(self.list_host)
        self.list_lay.setContentsMargins(2, 2, 2, 2)
        self.list_lay.setSpacing(3)
        self.empty_lbl = QLabel("NO POI MARKED")
        self.empty_lbl.setAlignment(Qt.AlignCenter)
        self.empty_lbl.setStyleSheet(f"color: {TEXT_1}; font-size: 10px;")
        self.list_lay.addWidget(self.empty_lbl)
        self.list_lay.addStretch()
        self.list_scroll.setWidget(self.list_host)
        lay.addWidget(self.list_scroll, stretch=1)

    def add_from_pick(self, coord, source="3D view pick"):
        self._counter += 1
        self._add_poi(f"POI-{self._counter:02d}", coord, source)

    def add_manual(self):
        val, ok = QInputDialog.getText(self, "Add POI",
                                        "Enter coordinates as: x,y,z")
        if not ok or not val:
            return
        try:
            x, y, z = [float(v.strip()) for v in val.split(",")]
        except Exception:
            QMessageBox.critical(self, "Invalid input",
                                  "Expected format: x,y,z (e.g. 1.2, -0.5, 3.0)")
            return
        self._counter += 1
        self._add_poi(f"POI-{self._counter:02d}", (x, y, z), "manual entry")

    def _add_poi(self, name, coord, source):
        entry = {"name": name, "coord": tuple(float(c) for c in coord),
                 "source": source, "note": "", "time": ts()}
        self.pois.append(entry)
        self.empty_lbl.hide()
        self._render_entry(entry)
        self._push_markers()
        self.log(f"[{ts()}] POI MARKED: {name} @ "
                 f"({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f}) [{source}]")

    def _push_markers(self):
        if self.map_panel is not None:
            self.map_panel.set_markers(
                [{"coord": p["coord"], "label": p["name"]} for p in self.pois])

    def _render_entry(self, entry):
        row = QFrame()
        row.setObjectName("poiRow")
        rlay = QVBoxLayout(row)
        rlay.setContentsMargins(8, 6, 8, 6)
        rlay.setSpacing(2)

        top = QHBoxLayout()
        name_lbl = QLabel(entry["name"])
        name_lbl.setStyleSheet(f"color: {AMBER}; font-size: 12px; font-weight: bold;")
        top.addWidget(name_lbl)
        top.addStretch()
        del_btn = QPushButton("✕")
        del_btn.setObjectName("deleteBtn")
        del_btn.setFixedSize(22, 20)
        del_btn.clicked.connect(lambda: self._delete(entry, row))
        top.addWidget(del_btn)
        rlay.addLayout(top)

        c = entry["coord"]
        coord_lbl = QLabel(f"X {c[0]:.2f}   Y {c[1]:.2f}   Z {c[2]:.2f}")
        coord_lbl.setStyleSheet(f"color: {TEXT_0}; font-size: 10px;")
        rlay.addWidget(coord_lbl)
        src_lbl = QLabel(f"src: {entry['source']}  ·  {entry['time']}")
        src_lbl.setStyleSheet(f"color: {TEXT_1}; font-size: 9px;")
        rlay.addWidget(src_lbl)

        self.list_lay.insertWidget(self.list_lay.count() - 1, row)

    def _delete(self, entry, row):
        if entry in self.pois:
            self.pois.remove(entry)
        row.deleteLater()
        if not self.pois:
            self.empty_lbl.show()
        self._push_markers()


# --------------------------------------------------------------------------- #
# MAIN WINDOW
# --------------------------------------------------------------------------- #
class TerraMapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TERRAMAP // VIDEO-TO-3D RECON SUITE (VGGT) — Qt")
        self.resize(1640, 980)
        self.setMinimumSize(1280, 800)

        self.work_dir = tempfile.mkdtemp(prefix="terramap_")
        self.cloud_stats = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_topbar())

        divider = QFrame()
        divider.setFixedHeight(2)
        divider.setStyleSheet(f"background-color: {AMBER};")
        root.addWidget(divider)

        root.addWidget(self._build_body(), stretch=1)

        divider2 = QFrame()
        divider2.setFixedHeight(1)
        divider2.setStyleSheet(f"background-color: {AMBER};")
        root.addWidget(divider2)

        root.addWidget(self._build_statusbar())

        # thread -> UI wiring
        BRIDGE.log.connect(self._append_log)
        BRIDGE.cloud_ready.connect(self._apply_cloud_result)
        BRIDGE.depth_ready.connect(self.map_panel.apply_depth_result)
        BRIDGE.glb_ready.connect(self._apply_glb_result)

        engine_msg = ("VGGT package detected — ready to reconstruct."
                      if vggt_engine.is_available() else
                      "VGGT package NOT found — see README for install steps.")
        self.log("[SYS] TERRAMAP RECON SUITE INITIALIZED — STANDING BY (Qt)")
        self.log(f"[SYS] {engine_msg}")
        self.log(f"[SYS] EMBED SUPPORT: "
                 f"{'YES (pywin32 OK)' if HAS_WIN32 else 'NO — pip install pywin32'}")

    # -- layout --------------------------------------------------------
    def _build_topbar(self):
        bar = QWidget()
        bar.setFixedHeight(60)
        bar.setStyleSheet(f"background-color: {BG_1};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)

        t1 = QLabel("▲ TERRAMAP")
        t1.setStyleSheet(f"color: {AMBER}; font-size: 20px; font-weight: bold;")
        lay.addWidget(t1)
        t2 = QLabel(" RECON SUITE")
        t2.setStyleSheet(f"color: {TEXT_0}; font-size: 20px;")
        lay.addWidget(t2)
        lay.addStretch()

        run_btn = QPushButton("RUN 3D ANALYSIS (VGGT)")
        # explicit inline style: on some platforms the app-level #primary
        # rule loses to the native style and the button renders black
        run_btn.setStyleSheet(
            f"QPushButton {{ background-color: {AMBER}; color: {BG_0}; "
            f"font-weight: bold; border-radius: 3px; padding: 7px 12px; }} "
            f"QPushButton:hover {{ background-color: #cc8e00; }}")
        run_btn.setFixedWidth(200)
        run_btn.clicked.connect(self.run_analysis)
        lay.addWidget(run_btn)

        trellis_btn = QPushButton("RUN HUNYUAN3D (GLB)")
        trellis_btn.setStyleSheet(
            f"QPushButton {{ background-color: {OLIVE}; color: {TEXT_0}; "
            f"font-weight: bold; border-radius: 3px; padding: 7px 12px; }} "
            f"QPushButton:hover {{ background-color: {AMBER_DIM}; }}")
        trellis_btn.setFixedWidth(170)
        trellis_btn.clicked.connect(self.run_gen3d)
        lay.addWidget(trellis_btn)

        exp_btn = QPushButton("EXPORT REPORT")
        exp_btn.setFixedWidth(150)
        exp_btn.clicked.connect(self.export_report)
        lay.addWidget(exp_btn)
        return bar

    def _build_body(self):
        body = QWidget()
        grid = QGridLayout(body)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)
        grid.setColumnMinimumWidth(0, 380)
        grid.setColumnMinimumWidth(2, 340)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)

        self.video_panel = VideoPanel(log_fn=self.log, work_dir=self.work_dir)
        grid.addWidget(self.video_panel, 0, 0)

        center = QWidget()
        clay = QVBoxLayout(center)
        clay.setContentsMargins(0, 0, 0, 0)
        clay.setSpacing(6)

        stat_row = QHBoxLayout()
        self.chip_frames = StatChip("FRAMES USED")
        self.chip_points = StatChip("POINTS")
        self.chip_pois = StatChip("POI COUNT")
        for chip in (self.chip_frames, self.chip_points, self.chip_pois):
            chip.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            stat_row.addWidget(chip)
        clay.addLayout(stat_row)

        self.map_panel = Map3DPanel(
            log_fn=self.log, work_dir=self.work_dir,
            on_poi_pick=self._on_poi_pick,
            get_depth_frame_path=self.video_panel.get_reference_frame_path)
        clay.addWidget(self.map_panel, stretch=1)

        grid.addWidget(center, 0, 1)

        self.poi_panel = POIPanel(log_fn=self.log)
        grid.addWidget(self.poi_panel, 0, 2)
        self.poi_panel.map_panel = self.map_panel
        return body

    def _build_statusbar(self):
        wrap = QWidget()
        wrap.setFixedHeight(140)
        wrap.setStyleSheet(f"background-color: {BG_1};")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(10, 6, 10, 8)
        lay.setSpacing(2)
        lay.addWidget(SectionHeader("◈ SYSTEM LOG"))
        self.log_box = QPlainTextEdit()
        self.log_box.setObjectName("logBox")
        self.log_box.setReadOnly(True)
        lay.addWidget(self.log_box)
        return wrap

    # -- logging (thread-safe via BRIDGE.log signal) ---------------------
    def log(self, msg):
        BRIDGE.log.emit(msg)

    def _append_log(self, msg):
        self.log_box.appendPlainText(msg)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    # -- actions ---------------------------------------------------------
    def _on_poi_pick(self, coord, source):
        self.poi_panel.add_from_pick(coord, source=source)
        self.chip_pois.set(len(self.poi_panel.pois))

    def run_analysis(self):
        frame_paths = self.video_panel.extracted_frame_paths
        if len(frame_paths) < 1:
            QMessageBox.warning(self, "No frames",
                                 "Load a video and click EXTRACT FRAMES first.")
            return

        if not vggt_engine.is_available():
            QMessageBox.critical(
                self, "VGGT not installed",
                "The VGGT package is not importable in this environment.\n\n"
                "Install it with:\n"
                "  git clone https://github.com/facebookresearch/vggt.git\n"
                "  cd vggt\n  pip install -r requirements.txt\n"
                "  pip install -e .\n\nThen restart TerraMap.")
            return

        self.log(f"[{ts()}] RUN 3D ANALYSIS (VGGT) — {len(frame_paths)} frames queued")

        def worker():
            try:
                points, colors, stats = vggt_engine.reconstruct_point_cloud_vggt(
                    frame_paths, log_fn=BRIDGE.log.emit)
            except Exception as e:
                BRIDGE.log.emit(f"[{ts()}] ERROR: {e}")
                for line in traceback.format_exc().splitlines():
                    BRIDGE.log.emit(f"    {line}")
                return
            BRIDGE.cloud_ready.emit(points, colors, stats)

        threading.Thread(target=worker, daemon=True).start()

    def run_gen3d(self):
        frame_paths = self.video_panel.trellis_frame_paths
        if len(frame_paths) < 1:
            QMessageBox.warning(
                self, "No frames",
                "Load a video and click EXTRACT 4 FRAMES (GEN-3D) first.")
            return
        if getattr(self, "_gen3d_proc", None) is not None:
            QMessageBox.information(self, "Busy",
                                     "A GEN-3D run is already in progress.")
            return

        # make the GPU as clean as possible before the heavy run:
        # 1) close the embedded Open3D view (graphics+compute contention)
        if self.map_panel.embed_proc is not None:
            self.map_panel._stop_embed()
        # 2) best-effort: drop cached VGGT weights held by THIS process
        try:
            if getattr(vggt_engine, "_model", None) is not None:
                vggt_engine._model = None
                import torch
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                self.log(f"[{ts()}] released cached VGGT weights from VRAM")
        except Exception:
            pass

        glb_path = os.path.join(self.work_dir, "gen3d_asset.glb")
        self._gen3d_glb_path = glb_path
        self.log(f"[{ts()}] RUN HUNYUAN3D-2 — {len(frame_paths)} frames, "
                 f"running in a SEPARATE process (isolated from the UI)")

        from PySide6.QtCore import QProcess
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "gen3d_worker.py")
        proc.readyReadStandardOutput.connect(self._gen3d_output)
        proc.finished.connect(self._gen3d_finished)
        self._gen3d_proc = proc
        proc.start(sys.executable,
                   [worker_script, "--out", glb_path] + list(frame_paths))

    def _gen3d_output(self):
        proc = self._gen3d_proc
        if proc is None:
            return
        data = bytes(proc.readAllStandardOutput()).decode("utf-8",
                                                           errors="replace")
        for line in data.splitlines():
            line = line.rstrip()
            if not line or line.startswith("@@"):
                continue
            # skip tqdm carriage-return spam; keep meaningful lines
            if "\r" in line:
                line = line.split("\r")[-1]
            if line.strip():
                self.log(line)

    def _gen3d_finished(self, exit_code, _status):
        proc = self._gen3d_proc
        self._gen3d_proc = None
        if proc is not None:
            proc.deleteLater()

        if exit_code != 0:
            self.log(f"[{ts()}] GEN-3D worker exited with code {exit_code} — "
                     f"see log above for the traceback.")
            return

        glb_path = self._gen3d_glb_path
        if not os.path.exists(glb_path):
            self.log(f"[{ts()}] ERROR: worker finished but no GLB at {glb_path}")
            return

        self.log(f"[{ts()}] worker done — sampling colored points from GLB...")

        def sampler():
            try:
                points, colors = trellis_engine.glb_to_point_cloud(
                    glb_path, log_fn=BRIDGE.log.emit)
            except Exception as e:
                BRIDGE.log.emit(f"[{ts()}] ERROR: {e}")
                for line in traceback.format_exc().splitlines():
                    BRIDGE.log.emit(f"    {line}")
                return
            stats = {"engine": "Hunyuan3D-2 (worker process)",
                     "mode": "see worker log", "frames_used": "-",
                     "glb_path": glb_path}
            BRIDGE.glb_ready.emit(points, colors, stats)

        threading.Thread(target=sampler, daemon=True).start()

    def _apply_glb_result(self, points, colors, stats):
        self.cloud_stats = stats
        self.last_glb_path = stats.get("glb_path")
        self.map_panel.set_cloud(points, colors)
        self.map_panel.set_markers(
            [{"coord": p["coord"], "label": p["name"]}
             for p in self.poi_panel.pois])
        self.chip_frames.set(stats.get("frames_used", 0))
        self.chip_points.set(points.shape[0])
        self.chip_pois.set(len(self.poi_panel.pois))
        self.log(f"[{ts()}] GEN-3D ASSET READY ({stats.get('engine', '?')}, "
                 f"{stats.get('mode', '?')}) — shown in center; "
                 f"GLB file: {self.last_glb_path}")
        self.log(f"[{ts()}] NOTE: this is a GENERATED asset in a normalized "
                 f"frame — POI coords here are not scene-metric.")

    def _apply_cloud_result(self, points, colors, stats):
        self.cloud_stats = stats
        self.map_panel.set_cloud(points, colors)
        self.map_panel.set_markers(
            [{"coord": p["coord"], "label": p["name"]}
             for p in self.poi_panel.pois])
        self.chip_frames.set(stats.get("frames_used", 0))
        self.chip_points.set(points.shape[0])
        self.chip_pois.set(len(self.poi_panel.pois))
        self.log(f"[{ts()}] RECONSTRUCTION COMPLETE — {points.shape[0]} points, "
                 f"{stats.get('frames_used', 0)} frames (VGGT)")

    def export_report(self):
        if not self.poi_panel.pois and self.map_panel.points is None:
            QMessageBox.information(self, "Nothing to export",
                                     "Run an analysis or mark a POI first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export report", "terramap_report.json",
            "JSON report (*.json)")
        if not path:
            return
        report = {
            "generated": dt.datetime.now().isoformat(),
            "source_video": self.video_panel.video_path,
            "reconstruction_stats": self.cloud_stats,
            "trellis_glb": getattr(self, "last_glb_path", None),
            "area_of_interest": ({"lat": self.map_panel.aoi[0],
                                   "lon": self.map_panel.aoi[1]}
                                  if getattr(self.map_panel, "aoi", None) else None),
            "points_of_interest": [
                {"name": p["name"], "coord": p["coord"],
                 "source": p["source"], "time": p["time"]}
                for p in self.poi_panel.pois
            ],
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        self.log(f"[{ts()}] REPORT EXPORTED -> {path}")

        ply_note = ""
        if self.map_panel.points is not None:
            ply_path = os.path.splitext(path)[0] + ".ply"
            try:
                o3d_viewer.save_point_cloud_ply(
                    self.map_panel.points, self.map_panel.colors, ply_path)
                self.log(f"[{ts()}] POINT CLOUD EXPORTED -> {ply_path}")
                ply_note = f"\nPoint cloud: {ply_path}"
            except Exception as e:
                self.log(f"[{ts()}] WARNING: could not export .ply ({e})")

        QMessageBox.information(self, "Exported",
                                 f"Report saved to:\n{path}{ply_note}")

    # -- cleanup ---------------------------------------------------------
    def closeEvent(self, event):
        if self.map_panel.embed_proc is not None:
            self.map_panel._stop_embed()
        event.accept()


if __name__ == "__main__":
    multiprocessing.freeze_support()   # PyInstaller / Windows spawn
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = TerraMapWindow()
    win.show()
    sys.exit(app.exec())
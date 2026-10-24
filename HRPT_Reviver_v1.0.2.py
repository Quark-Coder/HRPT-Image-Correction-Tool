import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QSettings
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QPlainTextEdit,
    QListWidget,
    QListWidgetItem,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
)
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap, QPainter

PREVIEW_MAX_SIDE = 900

APP_NAME="HRPT Reviver v1.0.2"

# Settings are stored in AppData (Windows) or ~/.config (Linux/macOS)
# via Qt's ini format: %APPDATA%/HRPTReviver/HRPTReviver.ini
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
SETTINGS = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                     "HRPTReviver", "HRPTReviver")
IN_PLACE_KEY = "save/in_place_overwrite"


class PreviewWorker(QThread):
    """Computes the defective pixel mask for the preview in a background thread."""
    preview_ready = pyqtSignal(int, object, object)  # (generation, rgb, mask)

    def __init__(self, gen, array, is_hrpt, black_threshold, deviation_threshold,
                 max_band_height=3):
        super().__init__()
        self.gen = gen
        self.array = array
        self.is_hrpt = is_hrpt
        self.black_threshold = black_threshold
        self.deviation_threshold = deviation_threshold
        self.max_band_height = max_band_height

    def run(self):
        arr = self.array
        h, w, c = arr.shape
        mask = np.zeros((h, w), dtype=bool)
        try:
            if self.is_hrpt:
                for ch in range(c):
                    _, defect = ImageRepair.repair_channel_hrpt(
                        arr[:, :, ch], self.deviation_threshold,
                        max_band_height=self.max_band_height)
                    mask |= defect
            else:
                for ch in range(c):
                    mask |= arr[:, :, ch] <= self.black_threshold
        except Exception:
            self.preview_ready.emit(self.gen, None, None)
            return

        if c == 1:
            rgb = np.repeat(arr, 3, axis=2)
        else:
            rgb = arr[:, :, :3]
        rgb = rgb.astype(np.uint8)
        self.preview_ready.emit(self.gen, rgb, mask)


class PreviewCanvas(QGraphicsView):
    """Full-featured preview viewer with pan (mouse dragging)
    and zoom (mouse wheel, zooming toward the cursor position)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        self._fit_mode = True
        self.zoom_label = None
        self._overlay_pixmap = None
        self._original_pixmap = None
        self._showing_original = False

    # ---------- API ----------
    def set_images(self, original_pixmap, overlay_pixmap):
        """original_pixmap — изображение без разметки, overlay_pixmap —
        с красной отметкой дефектов. Удержание ЛКМ временно показывает
        оригинал, чтобы видеть, что именно удаляется."""
        self._original_pixmap = original_pixmap
        self._overlay_pixmap = overlay_pixmap
        self._showing_original = False
        self._pixmap_item.setPixmap(overlay_pixmap)
        self._scene.setSceneRect(QRectF(self._pixmap_item.boundingRect()))
        if self._fit_mode:
            self.fit()

    def set_image(self, pixmap):
        self._overlay_pixmap = pixmap
        self._original_pixmap = pixmap
        self._showing_original = False
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(self._pixmap_item.boundingRect()))
        if self._fit_mode:
            self.fit()

    def _toggle_original(self, show_original):
        if self._showing_original == show_original:
            return
        pixmap = self._original_pixmap if show_original else self._overlay_pixmap
        if pixmap is None:
            return
        self._showing_original = show_original
        self._pixmap_item.setPixmap(pixmap)

    def fit(self):
        self._fit_mode = True
        self.resetTransform()
        rect = self._scene.itemsBoundingRect()
        if rect.isEmpty():
            return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._update_zoom_label()

    def actual_pixels(self):
        """Zoom 1:1 (1 image pixel = 1 screen pixel)."""
        self._fit_mode = False
        self.resetTransform()
        self._update_zoom_label()

    # ---------- Events ----------
    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.zoom_step(factor)

    def zoom_step(self, factor):
        current = self.transform().m11()
        if 0.02 < current * factor < 40:
            self._fit_mode = False
            self.scale(factor, factor)
            self._update_zoom_label()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_original(True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_original(False)
        self._fit_mode = False
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit()

    def _update_zoom_label(self):
        if self.zoom_label is not None:
            self.zoom_label.setText(f"{int(round(self.transform().m11() * 100))}%")


class ImageRepair:
    @staticmethod
    def load_image(path):
        image = Image.open(path)
        if image.mode == "P":
            image = image.convert("RGB")
        return image

    @staticmethod
    def load_gray(path):
        """Loads a monochrome file "as is" (L, I, I;16):
        bit depth and mode are preserved, no conversion."""
        img = Image.open(path)
        if len(img.getbands()) != 1:
            raise ValueError(f"not a grayscale image (mode {img.mode})")
        return img

    @staticmethod
    def is_gray_mode(mode):
        return mode in ("L", "I", "I;16")

    @staticmethod
    def _ranges_from_mask(mask):
        """Row ranges containing defective pixels."""
        rows = np.any(mask, axis=1)
        ranges = []
        start = None
        for y, changed in enumerate(rows):
            if changed and start is None:
                start = y
            elif not changed and start is not None:
                ranges.append((start, y - 1))
                start = None
        if start is not None:
            ranges.append((start, mask.shape[0] - 1))
        return ranges

    @staticmethod
    def _rebuild_image(image, arr):
        """Rebuilds the image with EXACTLY the same mode as the original,
        carrying over its metadata (info)."""
        mode = image.mode
        if mode == "L":
            out = Image.fromarray(arr, mode="L")
        elif mode == "RGB":
            out = Image.fromarray(arr, mode="RGB")
        elif mode == "RGBA":
            out = Image.fromarray(arr, mode="RGBA")
        elif mode == "I;16":
            out = Image.frombytes("I;16", image.size,
                                  np.ascontiguousarray(arr, dtype="<u2").tobytes())
        elif mode == "I":
            out = Image.fromarray(arr.astype(np.int32), mode="I")
        else:
            out = image.copy()
            out.putdata(arr.reshape(-1).tolist())
        out.info = dict(image.info)
        return out

    @classmethod
    def _dtype_max(cls, dtype):
        try:
            return float(np.iinfo(dtype).max)
        except ValueError:
            return 255.0


    @staticmethod
    def repair_channel(channel, black_threshold=5):
        height, width = channel.shape
        repaired = channel.copy().astype(np.float32)

        for x in range(width):
            col = channel[:, x].astype(np.float32)
            not_black = col > black_threshold
            if not np.any(not_black):
                continue

            in_black = False
            start = 0
            for y in range(height):
                is_black = col[y] <= black_threshold
                if is_black and not in_black:
                    in_black = True
                    start = y
                elif not is_black and in_black:
                    end = y - 1
                    if start == 0:
                        next_val = col[end + 1]
                        repaired[start:end + 1, x] = next_val
                    elif end == height - 1:
                        prev_val = col[start - 1]
                        repaired[start:end + 1, x] = prev_val
                    else:
                        length = end - start + 1
                        upper = col[start - 1]
                        lower = col[end + 1]
                        for i in range(length):
                            t = i / (length - 1) if length > 1 else 0.5
                            repaired[start + i, x] = (1 - t) * upper + t * lower
                    in_black = False
            if in_black:
                start = start
                end = height - 1
                if start == 0:
                    pass
                else:
                    prev_val = col[start - 1]
                    repaired[start:end + 1, x] = prev_val

        return repaired.astype(channel.dtype)

    @classmethod
    def repair(cls, image, black_threshold=5):
        """Black band repair. Bit depth and image mode are preserved
        (L, I, I;16, RGB, RGBA): for RGBA only the color channels are
        processed, alpha is left untouched. The threshold is given in
        0-255 units and scaled to the bit depth automatically."""
        array = np.array(image)
        scale = cls._dtype_max(array.dtype) / 255.0
        thr = black_threshold * scale
        all_ranges = []

        if array.ndim == 2:
            repaired = cls.repair_channel(array, thr)
            mask = repaired != array
            if np.any(mask):
                all_ranges.append({'channel': 0,
                                   'ranges': cls._ranges_from_mask(mask)})
            out_arr = repaired
        else:
            out_arr = array.copy()
            for ch in range(min(3, array.shape[2])):
                repaired_ch = cls.repair_channel(array[:, :, ch], thr)
                mask = repaired_ch != array[:, :, ch]
                if np.any(mask):
                    all_ranges.append({'channel': ch,
                                       'ranges': cls._ranges_from_mask(mask)})
                out_arr[:, :, ch] = repaired_ch

        return cls._rebuild_image(image, out_arr), all_ranges


    @staticmethod
    def _vertical_interp(values, good):
        """Linear vertical interpolation between the nearest "good"
        pixels strictly above and strictly below (the pixel itself is not
        counted as a neighbor). If good pixels exist only on one side —
        edge value replication. Returns (interp, valid)."""
        h, w = values.shape
        up_val = np.full((h, w), np.nan, dtype=np.float32)
        dn_val = np.full((h, w), np.nan, dtype=np.float32)
        up_dist = np.full((h, w), np.inf, dtype=np.float32)
        dn_dist = np.full((h, w), np.inf, dtype=np.float32)

        last_val = np.full(w, np.nan, dtype=np.float32)
        last_pos = np.full(w, -1, dtype=np.int64)
        pos = np.arange(h, dtype=np.int64)[:, None]
        for y in range(h):
            up_val[y] = last_val
            up_dist[y] = np.where(last_pos >= 0, y - last_pos, np.inf)
            g = good[y]
            last_val = np.where(g, values[y], last_val)
            last_pos = np.where(g, y, last_pos)

        last_val = np.full(w, np.nan, dtype=np.float32)
        last_pos = np.full(w, h, dtype=np.int64)
        for y in range(h - 1, -1, -1):
            dn_val[y] = last_val
            dn_dist[y] = np.where(last_pos < h, last_pos - y, np.inf)
            g = good[y]
            last_val = np.where(g, values[y], last_val)
            last_pos = np.where(g, y, last_pos)

        valid = ~(np.isnan(up_val) & np.isnan(dn_val))
        total = up_dist + dn_dist
        with np.errstate(invalid='ignore', divide='ignore'):
            t = np.where(total > 0, up_dist / total, 0.5)
        interp = (1.0 - t) * up_val + t * dn_val
        interp = np.where(np.isnan(up_val), dn_val,
                          np.where(np.isnan(dn_val), up_val, interp))
        return interp, valid

    @staticmethod
    def _vertical_median(c, half=3):
        """Vertical median filter with a window of 2*half+1 (edges filled
        by replication). A band up to half pixels thick is a minority in the
        window, so the median defines the band-free "background" of the scene."""
        h, w = c.shape
        pad = np.pad(c, ((half, half), (0, 0)), mode='reflect')
        stack = np.stack([pad[y:y + h] for y in range(2 * half + 1)])
        return np.median(stack, axis=0)

    @staticmethod
    def _drop_tall_runs(mask, max_band_height):
        """Removes vertical runs longer than max_band_height from the mask:
        a defect is a thin band; large connected areas belong to the scene."""
        h, w = mask.shape
        if max_band_height >= h:
            return mask
        starts = mask & ~np.vstack([mask[0:1], mask[:-1]])
        idx = np.arange(h, dtype=np.int64)[:, None]
        start_pos = np.where(starts, idx, 0)
        start_pos = np.maximum.accumulate(start_pos, axis=0)
        run_len = idx - start_pos + 1
        return mask & (run_len <= max_band_height)

    @classmethod
    def repair_channel_hrpt(cls, channel, deviation_threshold=30, grow_iterations=4,
                            max_band_height=3):
        """A pixel is defective if it deviates sharply (by more than
        deviation_threshold) from the interpolation of the neighboring rows —
        in either direction. The first pass finds obvious single outliers
        (with the neighbors above and below "agreeing" with each other),
        then the mask is iteratively expanded to catch bands 2-3+ pixels
        thick. Runs of defective pixels taller than max_band_height are
        discarded — this keeps large dark/bright scene areas out of repair."""
        orig = channel
        c = channel.astype(np.float32)
        h, w = channel.shape

        bg = cls._vertical_median(c, half=3)
        defect = np.abs(c - bg) > deviation_threshold

        for _ in range(grow_iterations):
            adj = defect | np.vstack([defect[0:1], defect[:-1]]) | \
                np.vstack([defect[1:], defect[-1:]])
            interp, valid = cls._vertical_interp(c, ~defect)
            grow = adj & (~defect) & valid & \
                (np.abs(c - interp) > deviation_threshold)
            if not grow.any():
                break
            defect |= grow

        defect = cls._drop_tall_runs(defect, max_band_height)

        interp, valid = cls._vertical_interp(c, ~defect)
        repaired = np.where(defect & valid, interp, c)
        dtype_max = cls._dtype_max(orig.dtype)
        repaired = np.clip(np.round(repaired), 0, dtype_max).astype(orig.dtype)
        return repaired, defect

    @classmethod
    def repair_hrpt(cls, image, deviation_threshold=30, max_band_height=3):
        """Thin band/outlier repair. Bit depth and image mode are
        preserved (L, I, I;16, RGB, RGBA): for RGBA only the color channels
        are processed, alpha is left untouched. The threshold is given in
        0-255 units and scaled to the bit depth automatically."""
        array = np.array(image)
        scale = cls._dtype_max(array.dtype) / 255.0
        thr = deviation_threshold * scale
        all_ranges = []

        if array.ndim == 2:
            repaired, defect = cls.repair_channel_hrpt(
                array, thr, max_band_height=max_band_height)
            if np.any(defect):
                all_ranges.append({'channel': 0,
                                   'ranges': cls._ranges_from_mask(defect)})
            out_arr = repaired
        else:
            out_arr = array.copy()
            for ch in range(min(3, array.shape[2])):
                repaired_ch, defect = cls.repair_channel_hrpt(
                    array[:, :, ch], thr, max_band_height=max_band_height)
                if np.any(defect):
                    all_ranges.append({'channel': ch,
                                       'ranges': cls._ranges_from_mask(defect)})
                out_arr[:, :, ch] = repaired_ch

        return cls._rebuild_image(image, out_arr), all_ranges


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_paths = []
        self.mode = None
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 950)
        self.setup_ui()
        self.setAcceptDrops(True)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        panel = QWidget()
        panel.setFixedWidth(380)
        main_layout.addWidget(panel)

        main_layout.addWidget(self.setup_preview_widget(), stretch=1)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)

        help_button = QPushButton("Help")
        help_button.clicked.connect(self.show_help)
        layout.addWidget(help_button)

        self.open_button = QPushButton("Open Image(s)")
        self.open_button.clicked.connect(self.open_images)
        layout.addWidget(self.open_button)

        files_label = QLabel("Selected files (click to edit its detection settings):")
        files_label.setStyleSheet("QLabel { padding-top: 5px; }")
        layout.addWidget(files_label)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(140)
        self.file_list.itemSelectionChanged.connect(self.on_file_selection_changed)
        layout.addWidget(self.file_list)

        self.apply_all_button = QPushButton("Apply current settings to ALL files")
        self.apply_all_button.setToolTip(
            "Copies the current detection settings to every file in the list.")
        self.apply_all_button.clicked.connect(self.apply_settings_to_all)
        layout.addWidget(self.apply_all_button)

        # Per-image detection settings: {Path: {'threshold':, 'deviation':, 'max_band_height':}}
        self.per_image = {}

        mode_label = QLabel("Signal Mode")
        mode_label.setStyleSheet("QLabel { font-size: 15px; font-weight: bold; padding-top: 10px; }")
        layout.addWidget(mode_label)

        mode_layout = QHBoxLayout()
        self.lrpt_checkbox = QCheckBox("LRPT mode (black bands)")
        self.lrpt_checkbox.setChecked(True)
        self.lrpt_checkbox.toggled.connect(self.on_mode_changed)
        self.hrpt_checkbox = QCheckBox("HRPT mode (thin dark/bright bands, color artifacts)")
        self.hrpt_checkbox.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.lrpt_checkbox)
        mode_layout.addWidget(self.hrpt_checkbox)
        layout.addLayout(mode_layout)

        self.inplace_checkbox = QCheckBox(
            "Overwrite originals (backup to ./backup)")
        self.inplace_checkbox.setChecked(
            SETTINGS.value(IN_PLACE_KEY, True, type=bool))
        self.inplace_checkbox.setToolTip(
            "ON: original files are copied to a 'backup' folder next to them,\n"
            "then repaired in place (file names unchanged).\n"
            "OFF: results are saved as *_corrected.* copies next to the originals.")
        self.inplace_checkbox.toggled.connect(self.on_inplace_changed)
        layout.addWidget(self.inplace_checkbox)

        settings_label = QLabel("Detection Settings")
        settings_label.setStyleSheet("QLabel { font-size: 15px; font-weight: bold; padding-top: 10px; }")
        layout.addWidget(settings_label)

        threshold_layout = QHBoxLayout()
        threshold_name = QLabel("Black threshold:")
        self.threshold = QSpinBox()
        self.threshold.setRange(0, 100)
        self.threshold.setValue(5)
        self.threshold.setToolTip("Maximum pixel value (0-255) considered black.")
        self.threshold.valueChanged.connect(self.on_setting_changed)
        threshold_layout.addWidget(threshold_name)
        threshold_layout.addWidget(self.threshold)
        layout.addLayout(threshold_layout)

        bandh_layout = QHBoxLayout()
        bandh_name = QLabel("Max band height:")
        self.max_band_height = QSpinBox()
        self.max_band_height.setRange(1, 20)
        self.max_band_height.setValue(3)
        self.max_band_height.setToolTip(
            "Only defective runs up to this height are treated as bands.\n"
            "Larger connected areas belong to the scene and are left untouched."
        )
        self.max_band_height.valueChanged.connect(self.on_setting_changed)
        bandh_layout.addWidget(bandh_name)
        bandh_layout.addWidget(self.max_band_height)
        layout.addLayout(bandh_layout)

        # Threshold for HRPT
        deviation_layout = QHBoxLayout()
        deviation_name = QLabel("Deviation threshold:")
        self.deviation = QSpinBox()
        self.deviation.setRange(1, 200)
        self.deviation.setValue(30)
        self.deviation.setToolTip(
            "Pixel is defective if it deviates from the vertical interpolation\n"
            "of neighboring rows by more than this value (either darker or brighter)."
        )
        self.deviation.valueChanged.connect(self.on_setting_changed)
        deviation_layout.addWidget(deviation_name)
        deviation_layout.addWidget(self.deviation)
        layout.addLayout(deviation_layout)

        self.process_button = QPushButton("Repair Bands")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setMinimumHeight(45)
        layout.addWidget(self.process_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.status = QLabel("Ready")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)

        bands_label = QLabel("Detected Bands (per channel)")
        bands_label.setStyleSheet("QLabel { font-size: 15px; font-weight: bold; padding-top: 10px; }")
        layout.addWidget(bands_label)

        self.bands_text = QPlainTextEdit()
        self.bands_text.setReadOnly(True)
        self.bands_text.setPlaceholderText("List of row ranges where changes were applied will appear here.")
        layout.addWidget(self.bands_text)

        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(300)
        self.preview_timer.timeout.connect(self.update_preview)

        self.preview_gen = 0          
        self.preview_array = None    
        self.preview_worker = None

        self.on_mode_changed()

    def setup_preview_widget(self):
        """Right side of the window: full-size viewer with pan/zoom."""
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 12, 12, 12)

        title = QLabel("Detection Preview (red = detected as defective)")
        title.setStyleSheet("QLabel { font-size: 15px; font-weight: bold; }")
        v.addWidget(title)

        self.zoom_label = QLabel("100%")
        self.canvas = PreviewCanvas()
        self.canvas.zoom_label = self.zoom_label

        toolbar = QHBoxLayout()
        hint = QLabel("Wheel = zoom, drag = pan, hold LMK = no sel. preview")
        hint.setStyleSheet("QLabel { color: #808080; }")
        toolbar.addWidget(hint)
        toolbar.addStretch()

        toolbar.addWidget(self.zoom_label)

        zoom_out = QPushButton("−")
        zoom_out.setFixedWidth(32)
        zoom_out.clicked.connect(lambda: self.canvas.zoom_step(0.8))
        toolbar.addWidget(zoom_out)

        zoom_in = QPushButton("+")
        zoom_in.setFixedWidth(32)
        zoom_in.clicked.connect(lambda: self.canvas.zoom_step(1.25))
        toolbar.addWidget(zoom_in)

        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self.canvas.fit)
        toolbar.addWidget(fit_btn)

        px_btn = QPushButton("1:1")
        px_btn.clicked.connect(self.canvas.actual_pixels)
        toolbar.addWidget(px_btn)

        v.addLayout(toolbar)

        v.addWidget(self.canvas, stretch=1)
        return container
    def build_preview_array(self):
        """Downscaled copy of the currently selected image for the preview.
        NEAREST — so thin bands don't get blurred when downscaling.
        The preview is converted to 8 bits (processing runs natively)."""
        path = self.current_path()
        if path is None:
            self.preview_array = None
            return
        try:
            try:
                img = ImageRepair.load_gray(path)
            except ValueError:
                img = ImageRepair.load_image(path)
            arr = np.array(img)
            scale = ImageRepair._dtype_max(arr.dtype) / 255.0
            if scale != 1.0:
                arr = (arr.astype(np.float64) / scale).round().clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr) if arr.ndim == 2 else Image.fromarray(arr[:, :, :3])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot build preview:\n{e}")
            self.preview_array = None
            return

        w, h = img.size
        scale = min(1.0, PREVIEW_MAX_SIDE / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        self.preview_array = arr

    def schedule_preview_update(self):
        if self.preview_array is not None:
            self.preview_timer.start()

    def update_preview(self):
        if self.preview_array is None:
            return
        self.preview_gen += 1
        self.status.setText("Updating preview...")
        QApplication.processEvents()
        self.preview_worker = PreviewWorker(
            self.preview_gen,
            self.preview_array,
            self.hrpt_checkbox.isChecked(),
            self.threshold.value(),
            self.deviation.value(),
            self.max_band_height.value(),
        )
        self.preview_worker.preview_ready.connect(self.on_preview_ready)
        self.preview_worker.start()

    def on_preview_ready(self, gen, rgb, mask):
        if gen != self.preview_gen or rgb is None or mask is None:
            return
        h, w, _ = rgb.shape
        overlay = rgb.copy()
        overlay[mask] = [255, 60, 60]
        original_img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        overlay_img = QImage(overlay.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.canvas.set_images(QPixmap.fromImage(original_img),
                               QPixmap.fromImage(overlay_img))
        self.status.setText("Ready to process")

    def on_inplace_changed(self, checked):
        SETTINGS.setValue(IN_PLACE_KEY, bool(checked))

    # ---------- Per-image settings ----------
    def current_path(self):
        item = self.file_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def get_settings(self, path):
        """Detection settings for a file; falls back to current UI values."""
        s = self.per_image.get(path)
        if s is None:
            s = {'threshold': self.threshold.value(),
                 'deviation': self.deviation.value(),
                 'max_band_height': self.max_band_height.value()}
            self.per_image[path] = s
        return s

    def on_setting_changed(self):
        path = self.current_path()
        if path is not None:
            self.per_image[path] = {
                'threshold': self.threshold.value(),
                'deviation': self.deviation.value(),
                'max_band_height': self.max_band_height.value(),
            }
        self.schedule_preview_update()

    def load_settings_into_ui(self, path):
        s = self.get_settings(path)
        for spin, key in ((self.threshold, 'threshold'),
                          (self.deviation, 'deviation'),
                          (self.max_band_height, 'max_band_height')):
            spin.blockSignals(True)
            spin.setValue(s[key])
            spin.blockSignals(False)
        self.schedule_preview_update()

    def on_file_selection_changed(self):
        path = self.current_path()
        if path is None or path not in self.per_image:
            return
        self.load_settings_into_ui(path)
        self.build_preview_array()
        self.schedule_preview_update()

    def apply_settings_to_all(self):
        """Copies the settings of the currently selected file to all files."""
        path = self.current_path()
        if path is None:
            return
        s = self.get_settings(path)
        for p in self.input_paths:
            self.per_image[p] = dict(s)
        self.status.setText(
            f"Settings applied to all {len(self.input_paths)} file(s)")
        self.schedule_preview_update()

    def on_mode_changed(self):
        sender = self.sender()
        if sender is self.lrpt_checkbox and self.lrpt_checkbox.isChecked():
            self.hrpt_checkbox.blockSignals(True)
            self.hrpt_checkbox.setChecked(False)
            self.hrpt_checkbox.blockSignals(False)
        elif sender is self.hrpt_checkbox and self.hrpt_checkbox.isChecked():
            self.lrpt_checkbox.blockSignals(True)
            self.lrpt_checkbox.setChecked(False)
            self.lrpt_checkbox.blockSignals(False)

        self.is_hrpt = self.hrpt_checkbox.isChecked()
        self.threshold.setEnabled(not self.is_hrpt)
        self.max_band_height.setEnabled(self.is_hrpt)
        self.deviation.setEnabled(self.is_hrpt)
        if hasattr(self, 'preview_timer'):
            self.schedule_preview_update()

    def closeEvent(self, event):
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.preview_worker.wait(2000)
        super().closeEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        paths = [Path(url.toLocalFile()) for url in urls if url.isLocalFile()]
        if not paths:
            return
        self.load_paths(paths)

    def show_help(self):
        QMessageBox.information(
            self,
            "HRPT Reviver — Help",
            "LRPT mode: detects and repairs black horizontal bands in each "
            "channel (R, G, B) using column-wise interpolation.\n"
            "HRPT mode: detects thin dark AND bright bands plus small color "
            "artifacts (spike detector, column-wise interpolation).\n\n"
            "Drag & drop one RGB image or any number of grayscale images (2+),\n"
            "each grayscale file is processed independently.\n"
            "By default originals are backed up to './backup' and repaired in place;\n"
            "disable the overwrite option to save *_corrected.* copies instead."
        )

    def open_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Image(s)",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        if not paths:
            return
        self.load_paths([Path(p) for p in paths])

    def load_paths(self, paths):
        if len(paths) == 1:
            try:
                img = Image.open(paths[0])
                if img.mode in ("RGB", "RGBA"):
                    self.mode = 'single_rgb'
                    self.input_paths = [paths[0]]
                elif ImageRepair.is_gray_mode(img.mode):
                    self.mode = 'single_gray'
                    self.input_paths = [paths[0]]
                else:
                    QMessageBox.warning(self, "Unsupported", f"Unsupported image mode: {img.mode}")
                    return
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot open image:\n{e}")
                return

        elif len(paths) >= 2:
            grayscale_paths = []
            for p in paths:
                try:
                    img = Image.open(p)
                    if not ImageRepair.is_gray_mode(img.mode):
                        QMessageBox.warning(
                            self,
                            "Invalid channels",
                            f"File {p.name} is not a grayscale image (mode {img.mode}).\n"
                            "All files must be grayscale (8-bit L or 16-bit I;16)."
                        )
                        return
                    grayscale_paths.append(p)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Cannot open {p.name}:\n{e}")
                    return
            sizes = [Image.open(p).size for p in grayscale_paths]
            if not all(s == sizes[0] for s in sizes):
                QMessageBox.warning(
                    self,
                    "Size mismatch",
                    "All grayscale images must have the same dimensions."
                )
                return

            self.mode = 'multi_gray'
            self.input_paths = grayscale_paths
        else:
            QMessageBox.warning(
                self,
                "Invalid selection",
                "Please drop exactly 1 image (RGB or grayscale)\n"
                "or 2+ grayscale images."
            )
            return

        self.bands_text.clear()
        self.progress.setValue(0)
        self.process_button.setEnabled(True)
        self.status.setText("Ready to process")

        # Заполняем список файлов (существующие настройки сохраняются)
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for p in self.input_paths:
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(str(p))
            self.file_list.addItem(item)
            self.get_settings(p)  # инициализация настройками по умолчанию
        self.file_list.blockSignals(False)
        self.file_list.setCurrentRow(0)

        self.build_preview_array()
        self.schedule_preview_update()

    def process_image(self):
        if not self.input_paths:
            return

        try:
            self.progress.setValue(5)
            self.status.setText("Loading image(s)...")
            QApplication.processEvents()

            def corrected_path(p):
                return p.with_name(f"{p.stem}_corrected{p.suffix}")

            in_place = self.inplace_checkbox.isChecked()

            def target_path(p):
                """Returns the output path for the source file p. In the
                in-place mode the original is first copied to the 'backup'
                folder next to it, and the file itself is overwritten."""
                if not in_place:
                    return corrected_path(p)
                backup_dir = p.parent / "backup"
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / p.name
                if not backup_path.exists():
                    shutil.copy2(p, backup_path)
                return p

            def save_like(repaired_img, original_img, out_path):
                """Saves the result with the original's metadata (dpi, exif)
                so the file parameters match."""
                kwargs = {}
                info = getattr(original_img, 'info', {})
                if 'dpi' in info:
                    kwargs['dpi'] = info['dpi']
                if 'exif' in info:
                    kwargs['exif'] = info['exif']
                fmt = self.get_format(out_path.suffix)
                repaired_img.save(out_path, format=fmt, **kwargs)

            threshold = self.threshold.value()
            deviation = self.deviation.value()
            max_band_height = self.max_band_height.value()
            is_hrpt = self.hrpt_checkbox.isChecked()

            def do_repair(img, path):
                # Индивидуальные настройки файла; при отсутствии — значения UI
                s = self.per_image.get(path) or {
                    'threshold': threshold, 'deviation': deviation,
                    'max_band_height': max_band_height}
                if is_hrpt:
                    return ImageRepair.repair_hrpt(
                        img, deviation_threshold=s['deviation'],
                        max_band_height=s['max_band_height'])
                return ImageRepair.repair(img, black_threshold=s['threshold'])

            if self.mode == 'single_rgb':

                img = ImageRepair.load_image(self.input_paths[0])
                repaired, ranges = do_repair(img, self.input_paths[0])
                out_path = target_path(self.input_paths[0])
                save_like(repaired, img, out_path)
                all_results = [{'file': out_path.name, 'ranges': ranges}]
                self.progress.setValue(90)

            elif self.mode == 'single_gray':
                img = ImageRepair.load_gray(self.input_paths[0])
                repaired, ranges = do_repair(img, self.input_paths[0])
                out_path = target_path(self.input_paths[0])
                save_like(repaired, img, out_path)
                all_results = [{'file': out_path.name, 'ranges': ranges}]
                self.progress.setValue(90)

            elif self.mode == 'multi_gray':
                all_results = []
                for ch, p in enumerate(self.input_paths):
                    img = ImageRepair.load_gray(p)
                    repaired, ranges = do_repair(img, p)
                    out_path = target_path(p)
                    save_like(repaired, img, out_path)
                    if ranges:
                        all_results.append({'file': out_path.name,
                                            'ranges': ranges})
                self.progress.setValue(90)

            else:
                QMessageBox.critical(self, "Error", "Unknown mode.")
                return

            self.bands_text.clear()
            if all_results:
                lines = []
                for res in all_results:
                    lines.append(f"File: {res['file']}")
                    for info in res['ranges']:
                        ch_name = ('Grayscale' if self.mode in ('single_gray', 'multi_gray')
                                   else ['R', 'G', 'B'][info['channel']])
                        lines.append(f"  Channel {ch_name}:")
                        if is_hrpt and len(info['ranges']) > 50:
                            total_rows = sum(end - start + 1 for start, end in info['ranges'])
                            lines.append(
                                f"    {len(info['ranges'])} defective row ranges "
                                f"({total_rows} rows total), first 50:"
                            )
                            shown = info['ranges'][:50]
                        else:
                            shown = info['ranges']
                        for idx, (start, end) in enumerate(shown, 1):
                            length = end - start + 1
                            lines.append(f"    {idx:02d}. rows {start}-{end} ({length} rows)")
                    lines.append("")
                self.bands_text.setPlainText("\n".join(lines))
                self.progress.setValue(100)
                self.status.setText("Done")
                total = sum(len(info['ranges']) for res in all_results for info in res['ranges'])
                mode_name = "HRPT" if is_hrpt else "LRPT"
                save_note = ("Originals backed up to './backup' and repaired in place."
                             if in_place else
                             "Corrected copies saved as *_corrected.* next to the originals.")
                QMessageBox.information(
                    self,
                    "Complete",
                    f"Processed {len(self.input_paths)} file(s) in {mode_name} mode.\n"
                    f"Total bands repaired: {total}\n\n"
                    f"{save_note}"
                )
            else:
                self.bands_text.setPlainText("No black bands detected in any file.")
                self.progress.setValue(100)
                self.status.setText("Done")
                hint = ("Try decreasing the deviation threshold." if is_hrpt
                        else "Try increasing the black threshold.")
                QMessageBox.information(
                    self,
                    "Complete",
                    f"No defects were found.\n\n{hint}"
                )

        except Exception as error:
            self.progress.setValue(0)
            self.status.setText("Error")
            QMessageBox.critical(self, "Error", f"Failed to process image:\n\n{error}")

    @staticmethod
    def get_format(suffix):
        suffix = suffix.lower()
        if suffix in ('.jpg', '.jpeg'):
            return 'JPEG'
        elif suffix == '.png':
            return 'PNG'
        elif suffix in ('.tif', '.tiff'):
            return 'TIFF'
        elif suffix == '.bmp':
            return 'BMP'
        else:
            return 'PNG'


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

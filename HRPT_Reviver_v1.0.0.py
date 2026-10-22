import sys
from pathlib import Path

import numpy as np
from PIL import Image

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF
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
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
)
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap, QPainter

# Максимальная сторона превью (реальная обработка идёт по полному размеру)
PREVIEW_MAX_SIDE = 900


class PreviewWorker(QThread):
    """Считает маску дефектных пикселей для превью в фоновом потоке."""
    preview_ready = pyqtSignal(int, object)  # (поколение, np.ndarray RGB | None)

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
            self.preview_ready.emit(self.gen, None)
            return

        # Затемнённая основа + красная подсветка того, что считается дефектом
        if c == 1:
            rgb = np.repeat(arr, 3, axis=2)
        else:
            rgb = arr[:, :, :3]
        base = (rgb.astype(np.float32) * 0.4).astype(np.uint8)
        base[mask] = [255, 60, 60]
        self.preview_ready.emit(self.gen, base)


class PreviewCanvas(QGraphicsView):
    """Полноценный просмотрщик превью с pan (перетаскивание мышью)
    и zoom (колесо мыши, зум к позиции курсора)."""

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
        self.zoom_label = None  # устанавливается снаружи

    # ---------- API ----------
    def set_image(self, pixmap):
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(self._pixmap_item.boundingRect()))
        if self._fit_mode:
            self.fit()

    def fit(self):
        self._fit_mode = True
        self.resetTransform()
        rect = self._scene.itemsBoundingRect()
        if rect.isEmpty():
            return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._update_zoom_label()

    def actual_pixels(self):
        """Масштаб 1:1 (1 пиксель изображения = 1 пиксель экрана)."""
        self._fit_mode = False
        self.resetTransform()
        self._update_zoom_label()

    # ---------- События ----------
    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.zoom_step(factor)

    def zoom_step(self, factor):
        current = self.transform().m11()
        if 0.02 < current * factor < 40:
            self._fit_mode = False
            self.scale(factor, factor)
            self._update_zoom_label()

    def mouseReleaseEvent(self, event):
        # Прерываем режим Fit при ручном панорамировании
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
        """Загрузка монохромного файла «как есть» (L, I, I;16):
        битность и режим сохраняются, никакой конвертации."""
        img = Image.open(path)
        if len(img.getbands()) != 1:
            raise ValueError(f"not a grayscale image (mode {img.mode})")
        return img

    @staticmethod
    def is_gray_mode(mode):
        # L – 8 бит, I/I;16 – 16-битные монохромные PNG
        return mode in ("L", "I", "I;16")

    @staticmethod
    def _ranges_from_mask(mask):
        """Диапазоны строк, в которых есть дефектные пиксели."""
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
        """Собирает изображение с ТОЧНО тем же режимом, что у оригинала,
        перенося его метаданные (info)."""
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

    # ------------------------------------------------------------------
    # LRPT: классический ремонт чёрных полос
    # ------------------------------------------------------------------
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
        """Ремонт чёрных полос. Битность и режим изображения сохраняются
        (L, I, I;16, RGB, RGBA): у RGBA обрабатываются только цветовые
        каналы, альфа остаётся нетронутой. Порог задаётся в единицах 0-255
        и масштабируется под битность автоматически."""
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
            # Цветовые каналы; у RGBA альфа (последний канал) не трогается
            for ch in range(min(3, array.shape[2])):
                repaired_ch = cls.repair_channel(array[:, :, ch], thr)
                mask = repaired_ch != array[:, :, ch]
                if np.any(mask):
                    all_ranges.append({'channel': ch,
                                       'ranges': cls._ranges_from_mask(mask)})
                out_arr[:, :, ch] = repaired_ch

        return cls._rebuild_image(image, out_arr), all_ranges

    # ------------------------------------------------------------------
    # HRPT: детектор двусторонних выбросов (тонкие тёмные И яркие полосы,
    # а также мелкие цветовые артефакты), векторизовано через numpy
    # ------------------------------------------------------------------
    @staticmethod
    def _vertical_interp(values, good):
        """Линейная интерполяция по вертикали между ближайшими «хорошими»
        пикселями строго сверху и строго снизу (сам пиксель соседом не
        считается). Если good-пиксели есть только с одной стороны —
        репликация крайнего значения. Возвращает (interp, valid)."""
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
        # Репликация: если good есть только с одной стороны
        interp = np.where(np.isnan(up_val), dn_val,
                          np.where(np.isnan(dn_val), up_val, interp))
        return interp, valid

    @staticmethod
    def _vertical_median(c, half=3):
        """Вертикальный медианный фильтр с окном 2*half+1 (заполнение краёв
        репликацией). Полоса толщиной до half пикселей – меньшинство в окне,
        поэтому медиана задаёт «фон» сцены, свободный от полос."""
        h, w = c.shape
        # reflect, а не edge: иначе дефектная крайняя строка реплицируется
        # в окно и сама становится «фоном»
        pad = np.pad(c, ((half, half), (0, 0)), mode='reflect')
        stack = np.stack([pad[y:y + h] for y in range(2 * half + 1)])
        return np.median(stack, axis=0)

    @staticmethod
    def _drop_tall_runs(mask, max_band_height):
        """Убирает из маски вертикальные пробеги длиннее max_band_height:
        дефект – тонкая полоса, крупные связные зоны относятся к сцене."""
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
        """Пиксель дефектный, если он резко (более чем на deviation_threshold)
        отклоняется от интерполяции соседних строк — в любую сторону.
        Первая проходка находит явные одиночные выбросы (соседи сверху и снизу
        при этом «согласованы» между собой), затем маска итеративно
        расширяется, чтобы захватить полосы толщиной 2-3+ пикселя.
        Пробеги дефектных пикселей выше max_band_height отбрасываются —
        так крупные тёмные/яркие участки сцены не попадают в ремонт."""
        orig = channel
        c = channel.astype(np.float32)
        h, w = channel.shape

        # Фон = вертикальная медиана (окно 7). Тонкая полоса (1-3 px) в ней
        # – меньшинство и на медиану не влияет, поэтому отклонение от фона
        # ловит и тёмные, и яркие полосы, а границы крупных зон не трогает.
        bg = cls._vertical_median(c, half=3)
        defect = np.abs(c - bg) > deviation_threshold

        # Итеративное расширение: только пиксели, соседние по вертикали
        # с уже найденными дефектами, проверяются относительно интерполяции
        # ближайших заведомо хороших пикселей (захват более редких толстых полос)
        for _ in range(grow_iterations):
            adj = defect | np.vstack([defect[0:1], defect[:-1]]) | \
                np.vstack([defect[1:], defect[-1:]])
            interp, valid = cls._vertical_interp(c, ~defect)
            grow = adj & (~defect) & valid & \
                (np.abs(c - interp) > deviation_threshold)
            if not grow.any():
                break
            defect |= grow

        # Отбрасываем крупные связные зоны – оставляем только тонкие полосы
        defect = cls._drop_tall_runs(defect, max_band_height)

        interp, valid = cls._vertical_interp(c, ~defect)
        repaired = np.where(defect & valid, interp, c)
        dtype_max = cls._dtype_max(orig.dtype)
        repaired = np.clip(np.round(repaired), 0, dtype_max).astype(orig.dtype)
        return repaired, defect

    @classmethod
    def repair_hrpt(cls, image, deviation_threshold=30, max_band_height=3):
        """Ремонт тонких полос/выбросов. Битность и режим изображения
        сохраняются (L, I, I;16, RGB, RGBA): у RGBA обрабатываются только
        цветовые каналы, альфа остаётся нетронутой. Порог задаётся в
        единицах 0-255 и масштабируется под битность автоматически."""
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
            # Цветовые каналы; у RGBA альфа (последний канал) не трогается
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
        self.setWindowTitle("Black Band Repair v2 (LRPT / HRPT)")
        self.resize(1500, 950)
        self.setup_ui()
        self.setAcceptDrops(True)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Левая панель – все настройки и отчёты
        panel = QWidget()
        panel.setFixedWidth(380)
        main_layout.addWidget(panel)

        # Правая часть – полноформатное превью с pan/zoom
        main_layout.addWidget(self.setup_preview_widget(), stretch=1)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Black Band Repair v2")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("QLabel { font-size: 20px; font-weight: bold; padding: 10px; }")
        layout.addWidget(title)

        description = QLabel(
            "LRPT mode: detects and repairs black horizontal bands in each channel (R, G, B) "
            "using column-wise interpolation.\n"
            "HRPT mode: detects thin dark AND bright bands plus small color artifacts "
            "(spike detector, column-wise interpolation).\n\n"
            "Drag & drop one RGB image or three grayscale images (R, G, B channels).\n"
            "Originals are kept untouched; results are saved as *_corrected.* copies."
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)

        self.open_button = QPushButton("Open Image(s)")
        self.open_button.clicked.connect(self.open_images)
        layout.addWidget(self.open_button)

        self.file_label = QLabel("No file(s) selected")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("QLabel { padding: 5px; }")
        layout.addWidget(self.file_label)

        # ---------- Выбор режима ----------
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

        settings_label = QLabel("Detection Settings")
        settings_label.setStyleSheet("QLabel { font-size: 15px; font-weight: bold; padding-top: 10px; }")
        layout.addWidget(settings_label)

        # Порог для LRPT
        threshold_layout = QHBoxLayout()
        threshold_name = QLabel("Black threshold:")
        self.threshold = QSpinBox()
        self.threshold.setRange(0, 100)
        self.threshold.setValue(5)
        self.threshold.setToolTip("Maximum pixel value (0-255) considered black.")
        self.threshold.valueChanged.connect(self.schedule_preview_update)
        threshold_layout.addWidget(threshold_name)
        threshold_layout.addWidget(self.threshold)
        layout.addLayout(threshold_layout)

        # Ограничение высоты полосы для LRPT
        bandh_layout = QHBoxLayout()
        bandh_name = QLabel("Max band height:")
        self.max_band_height = QSpinBox()
        self.max_band_height.setRange(1, 20)
        self.max_band_height.setValue(3)
        self.max_band_height.setToolTip(
            "Only defective runs up to this height are treated as bands.\n"
            "Larger connected areas belong to the scene and are left untouched."
        )
        self.max_band_height.valueChanged.connect(self.schedule_preview_update)
        bandh_layout.addWidget(bandh_name)
        bandh_layout.addWidget(self.max_band_height)
        layout.addLayout(bandh_layout)

        # Порог для HRPT
        deviation_layout = QHBoxLayout()
        deviation_name = QLabel("Deviation threshold:")
        self.deviation = QSpinBox()
        self.deviation.setRange(1, 200)
        self.deviation.setValue(30)
        self.deviation.setToolTip(
            "Pixel is defective if it deviates from the vertical interpolation\n"
            "of neighboring rows by more than this value (either darker or brighter)."
        )
        self.deviation.valueChanged.connect(self.schedule_preview_update)
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

        # ---------- Превью (в правой части окна) ----------
        # Таймер дебаунса: превью пересчитывается не чаще, чем раз в 300 мс
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(300)
        self.preview_timer.timeout.connect(self.update_preview)

        self.preview_gen = 0          # поколение запроса (защита от устаревших результатов)
        self.preview_array = None     # уменьшенная копия загруженного изображения
        self.preview_worker = None

        self.on_mode_changed()

    def setup_preview_widget(self):
        """Правая часть окна: полноформатный просмотрщик с pan/zoom."""
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
        hint = QLabel("Wheel = zoom, drag = pan")
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
        """Уменьшенная копия загруженного изображения для превью.
        NEAREST – чтобы тонкие полосы не размывались при уменьшении.
        Для превью приводится к 8 битам (обработка идёт нативно)."""
        if not self.input_paths:
            self.preview_array = None
            return
        try:
            if self.mode == 'three_gray':
                imgs = [ImageRepair.load_gray(p) for p in self.input_paths]
            else:
                imgs = [ImageRepair.load_image(self.input_paths[0])]
            arr = np.array(imgs[0])
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

    def on_preview_ready(self, gen, frame):
        # Результат устаревшего запроса игнорируем
        if gen != self.preview_gen or frame is None:
            return
        h, w, _ = frame.shape
        image = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.canvas.set_image(QPixmap.fromImage(image))
        self.status.setText("Ready to process")

    def on_mode_changed(self):
        # Режимы взаимоисключающие
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

    # ---------- Drag & Drop ----------
    def closeEvent(self, event):
        # Аккуратно останавливаем фоновый поток превью при закрытии
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

    # ---------- Загрузка файлов ----------
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
                    self.file_label.setText(f"RGB image:\n{paths[0]}")
                elif ImageRepair.is_gray_mode(img.mode):
                    self.mode = 'single_gray'
                    self.input_paths = [paths[0]]
                    bitness = "16-bit " if img.mode != "L" else ""
                    self.file_label.setText(f"{bitness}Grayscale image:\n{paths[0]}")
                else:
                    QMessageBox.warning(self, "Unsupported", f"Unsupported image mode: {img.mode}")
                    return
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot open image:\n{e}")
                return

        elif len(paths) == 3:
            grayscale_paths = []
            for p in paths:
                try:
                    img = Image.open(p)
                    if not ImageRepair.is_gray_mode(img.mode):
                        QMessageBox.warning(
                            self,
                            "Invalid channels",
                            f"File {p.name} is not a grayscale image (mode {img.mode}).\n"
                            "All three files must be grayscale (8-bit L or 16-bit I;16)."
                        )
                        return
                    grayscale_paths.append(p)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Cannot open {p.name}:\n{e}")
                    return
            # Проверяем размеры
            sizes = [Image.open(p).size for p in grayscale_paths]
            if not all(s == sizes[0] for s in sizes):
                QMessageBox.warning(
                    self,
                    "Size mismatch",
                    "All three grayscale images must have the same dimensions."
                )
                return

            self.mode = 'three_gray'
            self.input_paths = grayscale_paths
            self.file_label.setText(
                f"Three grayscale channels (will be processed together as RGB):\n"
                f"R: {grayscale_paths[0].name}\n"
                f"G: {grayscale_paths[1].name}\n"
                f"B: {grayscale_paths[2].name}"
            )
        else:
            QMessageBox.warning(
                self,
                "Invalid selection",
                "Please drop exactly 1 image (RGB or grayscale) or 3 grayscale images."
            )
            return

        self.bands_text.clear()
        self.progress.setValue(0)
        self.process_button.setEnabled(True)
        self.status.setText("Ready to process")

        # Строим превью для загруженных файлов
        self.build_preview_array()
        self.schedule_preview_update()

    # ---------- Обработка ----------
    def process_image(self):
        if not self.input_paths:
            return

        try:
            self.progress.setValue(5)
            self.status.setText("Loading image(s)...")
            QApplication.processEvents()

            def corrected_path(p):
                # Результат сохраняется рядом с оригиналом с приставкой _corrected
                return p.with_name(f"{p.stem}_corrected{p.suffix}")

            def save_like(repaired_img, original_img, out_path):
                """Сохраняет результат с метаданными оригинала (dpi, exif),
                чтобы параметры файлов совпадали."""
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

            def do_repair(img):
                # Выбор алгоритма в зависимости от режима
                if is_hrpt:
                    return ImageRepair.repair_hrpt(img, deviation_threshold=deviation,
                                                   max_band_height=max_band_height)
                return ImageRepair.repair(img, black_threshold=threshold)

            # Обрабатываем в зависимости от режима
            if self.mode == 'single_rgb':
                # Одиночное изображение (RGB/RGBA, режим и битность сохраняются)
                img = ImageRepair.load_image(self.input_paths[0])
                repaired, ranges = do_repair(img)
                # Сохраняем в копию, оригинал не трогаем
                out_path = corrected_path(self.input_paths[0])
                save_like(repaired, img, out_path)
                all_results = [{'file': out_path.name, 'ranges': ranges}]
                self.progress.setValue(90)

            elif self.mode == 'single_gray':
                # Одиночное монохромное (в т.ч. 16-битное), нативно
                img = ImageRepair.load_gray(self.input_paths[0])
                repaired, ranges = do_repair(img)
                out_path = corrected_path(self.input_paths[0])
                save_like(repaired, img, out_path)
                all_results = [{'file': out_path.name, 'ranges': ranges}]
                self.progress.setValue(90)

            elif self.mode == 'three_gray':
                # Три монохромных файла: каждый обрабатывается нативно,
                # битность и метаданные каждого сохраняются
                all_results = []
                for ch, p in enumerate(self.input_paths):
                    img = ImageRepair.load_gray(p)
                    repaired, ranges = do_repair(img)
                    out_path = corrected_path(p)
                    save_like(repaired, img, out_path)
                    if ranges:
                        all_results.append({'file': out_path.name,
                                            'ranges': ranges})
                self.progress.setValue(90)

            else:
                QMessageBox.critical(self, "Error", "Unknown mode.")
                return

            # Отображение результатов
            self.bands_text.clear()
            if all_results:
                lines = []
                for res in all_results:
                    lines.append(f"File: {res['file']}")
                    for info in res['ranges']:
                        ch_name = 'Grayscale' if self.mode in ('single_gray',) else ['R', 'G', 'B'][info['channel']]
                        lines.append(f"  Channel {ch_name}:")
                        # В HRPT-режиме полос очень много – группируем вывод
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
                QMessageBox.information(
                    self,
                    "Complete",
                    f"Processed {len(self.input_paths)} file(s) in {mode_name} mode.\n"
                    f"Total bands repaired: {total}\n\n"
                    f"Corrected copies saved as *_corrected.* next to the originals."
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

"""PySide6 + pyqtgraph GUI for interactive Gaussian beam system design.

Layout: left panel holds beam parameters, the editable element list and a
parameter form; the right panel shows the beam envelope w(z) along the
propagation axis z with element markers.  Moving the mouse over the plot
shows the beam radius w(z) and wavefront curvature R(z) at that z via a
closed-form probe (see :mod:`gaussianbeam.gui.model`).

Run with::

    gaussianbeam-gui
    python -m gaussianbeam.gui.app
"""

from __future__ import annotations

import os
import sys
from functools import partial

# Pin the Qt binding BEFORE importing pyqtgraph: pyqtgraph auto-detects the
# binding and would pick PyQt6 if it happens to be installed, which crashes
# PySide6 with symbol errors (two Qt6 builds in one process).
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

from ..units import mrad
from .elements import ELEMENT_TYPES, ParamSpec
from .model import LENGTH_UNITS, ElementSpec, OpticalSystem, format_length

# ---------------------------------------------------------------------------
# Parameter input row
# ---------------------------------------------------------------------------


class ParamRow(QWidget):
    """Spin box + unit selector (+ optional infinity checkbox) for one parameter.

    The value is exposed in SI units; the unit combo only changes the display.
    """

    changed = Signal()

    def __init__(self, pspec: ParamSpec, value: float, parent=None):
        super().__init__(parent)
        self.pspec = pspec
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(6)
        if pspec.kind == "index":
            self.spin.setRange(1e-9, 1e12)
        else:
            self.spin.setRange(-1e12, 1e12)
        self.spin.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        layout.addWidget(self.spin, 1)

        self.combo = None
        self.factor = 1.0
        if pspec.kind == "length":
            self.combo = QComboBox()
            for name, factor in LENGTH_UNITS:
                self.combo.addItem(name, factor)
            self.combo.currentIndexChanged.connect(self._on_unit)
            layout.addWidget(self.combo)

        self.inf = None
        if pspec.allow_inf:
            self.inf = QCheckBox("∞")
            self.inf.toggled.connect(self._on_inf)
            layout.addWidget(self.inf)

        self.spin.valueChanged.connect(lambda _v: self.changed.emit())
        self.set_value(value)

    # -- internals --

    def _on_unit(self, _idx):
        si = self.value()
        self.factor = self.combo.currentData()
        if np.isinf(si):
            return  # infinity: no numeric display to convert
        self.spin.blockSignals(True)
        self.spin.setValue(si / self.factor)
        self.spin.blockSignals(False)

    def _on_inf(self, on):
        self.spin.setEnabled(not on)
        if self.combo is not None:
            self.combo.setEnabled(not on)
        self.changed.emit()

    @staticmethod
    def _best_unit(v: float):
        for i, (name, factor) in enumerate(LENGTH_UNITS):
            if abs(v) / factor < 1000:
                return i
        return len(LENGTH_UNITS) - 1

    # -- public API --

    def value(self) -> float:
        """Current value in SI units."""
        if self.inf is not None and self.inf.isChecked():
            return np.inf
        return self.spin.value() * self.factor

    def set_value(self, v: float):
        self.spin.blockSignals(True)
        if self.inf is not None:
            self.inf.blockSignals(True)
            self.inf.setChecked(bool(np.isinf(v)))
            self.inf.blockSignals(False)
            self.spin.setEnabled(not np.isinf(v))
            if self.combo is not None:
                self.combo.setEnabled(not np.isinf(v))
        if np.isinf(v):
            # pick a sensible default unit so unchecking "∞" shows a sane value
            if self.combo is not None:
                idx = 2  # mm
                self.combo.blockSignals(True)
                self.combo.setCurrentIndex(idx)
                self.combo.blockSignals(False)
                self.factor = LENGTH_UNITS[idx][1]
            self.spin.setValue(1.0)
        else:
            if self.combo is not None:
                idx = self._best_unit(v)
                self.combo.blockSignals(True)
                self.combo.setCurrentIndex(idx)
                self.combo.blockSignals(False)
                self.factor = LENGTH_UNITS[idx][1]
            self.spin.setValue(v / self.factor)
        self.spin.blockSignals(False)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

_ENVELOPE_PEN = pg.mkPen((90, 170, 255), width=2)
_ENVELOPE_BRUSH = (80, 150, 255, 50)
_AXIS_PEN = pg.mkPen((150, 150, 150), style=Qt.DashLine)
_LENS_COLOR = (120, 230, 140)
_INTERFACE_COLOR = (255, 180, 90)
_PROBE_PEN = pg.mkPen((255, 220, 90), style=Qt.DashLine)
_SOURCE_PEN = pg.mkPen((230, 230, 230), style=Qt.DotLine, width=1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GaussianBeam — Gaussian Beam Designer")
        self.resize(1150, 720)

        self.system = OpticalSystem.default()
        self.trace = None
        self._form_rows = []

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ---- left panel ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_beam_group())
        left_layout.addWidget(self._build_element_group(), 1)
        left_layout.addWidget(self._build_param_group())
        left_layout.addWidget(self._build_probe_group())
        left_layout.addWidget(self._build_results_group())
        left_layout.addWidget(self._build_io_group())

        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(460)
        splitter.addWidget(scroll)

        # ---- right plot ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "z", units="m")
        self.plot.setLabel("left", "w", units="m")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        right_layout.addWidget(self.plot, 1)
        hint = QLabel(
            "Tip: drag in the list to reorder · "
            "Ctrl+drag an element on the plot to slide it along z"
        )
        hint.setStyleSheet("color: gray; padding: 2px 6px;")
        right_layout.addWidget(hint)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        self._drag_spec = None
        self.plot.viewport().installEventFilter(self)

        self._mouse_proxy = pg.SignalProxy(
            self.plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

        self.refresh_all(select=0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_beam_group(self):
        box = QGroupBox("Initial beam (waist at z = 0)")
        form = QFormLayout(box)
        self.row_wl = ParamRow(ParamSpec("wl", "Wavelength λ", "length", 0.0), self.system.wl)
        self.row_w0 = ParamRow(ParamSpec("w0", "Waist radius w0", "length", 0.0), self.system.w0)
        self.row_n = ParamRow(ParamSpec("n", "Medium index n", "index", 0.0), self.system.n0)
        self.row_wl.changed.connect(self._on_beam_param)
        self.row_w0.changed.connect(self._on_beam_param)
        self.row_n.changed.connect(self._on_beam_param)
        form.addRow("Wavelength λ", self.row_wl)
        form.addRow("Waist radius w<sub>0</sub>", self.row_w0)
        form.addRow("Medium index n", self.row_n)
        return box

    def _build_element_group(self):
        box = QGroupBox("Elements")
        layout = QVBoxLayout(box)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_selection)
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        self.list.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self.list, 1)

        row1 = QHBoxLayout()
        self.combo_type = QComboBox()
        for type_name, spec in ELEMENT_TYPES.items():
            self.combo_type.addItem(spec.label, type_name)
        row1.addWidget(self.combo_type, 1)
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self._add_element)
        row1.addWidget(btn_add)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        btn_del = QPushButton("Remove")
        btn_del.clicked.connect(self._remove_element)
        btn_up = QPushButton("Up")
        btn_up.clicked.connect(lambda: self._move_element(-1))
        btn_down = QPushButton("Down")
        btn_down.clicked.connect(lambda: self._move_element(+1))
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_elements)
        row2.addWidget(btn_del)
        row2.addWidget(btn_up)
        row2.addWidget(btn_down)
        row2.addWidget(btn_clear)
        layout.addLayout(row2)

        tail_row = QWidget()
        tail_layout = QHBoxLayout(tail_row)
        tail_layout.setContentsMargins(0, 0, 0, 0)
        self.row_tail = ParamRow(
            ParamSpec("tail", "Trailing free space", "length", 50e-3),
            self.system.tail,
        )
        self.row_tail.changed.connect(self._on_beam_param)
        tail_layout.addWidget(self.row_tail, 1)
        btn_auto_tail = QPushButton("auto")
        btn_auto_tail.setToolTip(
            "Set the trailing length to 50% of the total element length"
        )
        btn_auto_tail.clicked.connect(self._auto_tail)
        tail_layout.addWidget(btn_auto_tail)
        tail_form = QFormLayout()
        tail_form.addRow("Trailing free space", tail_row)
        layout.addLayout(tail_form)
        return box

    def _auto_tail(self):
        """One-shot: set the trailing free space to the suggested length."""
        self.row_tail.set_value(self.system.auto_tail())
        self._on_beam_param()

    def _build_param_group(self):
        self.param_box = QGroupBox("Element parameters")
        self._param_outer = QVBoxLayout(self.param_box)
        self._form_container = QWidget()
        self._param_outer.addWidget(self._form_container)
        return self.param_box

    def _build_probe_group(self):
        box = QGroupBox("Probe readout (hover over the plot)")
        layout = QVBoxLayout(box)
        self.lbl_z = QLabel("z = —")
        self.lbl_w = QLabel("w(z) = —")
        self.lbl_R = QLabel("R(z) = —")
        self.lbl_zR = QLabel("z<sub>R</sub> = —")
        self.lbl_n = QLabel("n = —")
        for lbl in (self.lbl_z, self.lbl_w, self.lbl_R, self.lbl_zR, self.lbl_n):
            layout.addWidget(lbl)
        return box

    def _build_results_group(self):
        box = QGroupBox("System output")
        layout = QVBoxLayout(box)
        self.res_w0 = QLabel("—")
        self.res_loc = QLabel("—")
        self.res_zR = QLabel("—")
        self.res_div = QLabel("—")
        for lbl in (self.res_w0, self.res_loc, self.res_zR, self.res_div):
            layout.addWidget(lbl)
        return box

    def _build_io_group(self):
        box = QGroupBox("Configuration")
        layout = QHBoxLayout(box)
        btn_save = QPushButton("Save config…")
        btn_save.clicked.connect(self._save_config)
        btn_load = QPushButton("Load config…")
        btn_load.clicked.connect(self._load_config)
        layout.addWidget(btn_save)
        layout.addWidget(btn_load)
        return box

    # ------------------------------------------------------------------
    # Element list operations
    # ------------------------------------------------------------------

    def _add_element(self):
        spec = ElementSpec.create(self.combo_type.currentData())
        self.system.elements.append(spec)
        self.refresh_all(select=len(self.system.elements) - 1)

    def _on_rows_moved(self, *_args):
        """List drag-and-drop finished: sync the model order and renumber."""
        self.system.elements = [
            self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
        ]
        for i, spec in enumerate(self.system.elements):
            self.list.item(i).setText(f"{i + 1}. {spec.summary()}")
        self.refresh_plot()
        self.refresh_results()

    def _remove_element(self):
        row = self.list.currentRow()
        if row < 0:
            return
        del self.system.elements[row]
        self.refresh_all(select=min(row, len(self.system.elements) - 1))

    def _clear_elements(self):
        self.system.elements.clear()
        self.refresh_all(select=-1)

    def _move_element(self, delta: int):
        row = self.list.currentRow()
        new = row + delta
        if row < 0 or not (0 <= new < len(self.system.elements)):
            return
        elems = self.system.elements
        elems[row], elems[new] = elems[new], elems[row]
        self.refresh_all(select=new)

    # ------------------------------------------------------------------
    # Parameter editing
    # ------------------------------------------------------------------

    def _on_beam_param(self):
        self.system.wl = self.row_wl.value()
        self.system.w0 = self.row_w0.value()
        self.system.n0 = self.row_n.value()
        self.system.tail = self.row_tail.value()
        self.refresh_plot()
        self.refresh_results()

    def _on_selection(self, _row):
        self._build_form()

    def _build_form(self):
        old = self._form_container
        container = QWidget()
        form = QFormLayout(container)
        self._form_rows = []
        row = self.list.currentRow()
        if 0 <= row < len(self.system.elements):
            spec = self.system.elements[row]
            for ps in ELEMENT_TYPES[spec.type].params:
                widget = ParamRow(ps, spec.params.get(ps.name, ps.default))
                widget.changed.connect(partial(self._on_param, spec, ps.name, widget))
                form.addRow(ps.label, widget)
                self._form_rows.append(widget)
        else:
            form.addRow(QLabel("(no element selected)"))
        self._param_outer.replaceWidget(old, container)
        old.hide()  # replaceWidget does not hide the old widget
        old.deleteLater()
        self._form_container = container

    def _on_param(self, spec, name, widget):
        spec.params[name] = widget.value()
        row = self.system.elements.index(spec)
        item = self.list.item(row)
        if item is not None:
            item.setText(f"{row + 1}. {spec.summary()}")
        self.refresh_plot()
        self.refresh_results()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_all(self, select: int | None = None):
        self.list.blockSignals(True)
        self.list.clear()
        for i, spec in enumerate(self.system.elements):
            item = QListWidgetItem(f"{i + 1}. {spec.summary()}")
            item.setData(Qt.UserRole, spec)
            self.list.addItem(item)
        self.list.blockSignals(False)
        if select is not None and 0 <= select < self.list.count():
            self.list.setCurrentRow(select)
        else:
            self.list.setCurrentRow(-1)
        self._build_form()
        self.refresh_plot()
        self.refresh_results()

    def refresh_plot(self):
        self.plot.clear()
        self.trace = self.system.trace()
        t = self.trace

        if t.z.size == 0:
            hint = pg.TextItem(
                "No free-space segment in the system — add a FreeSpace element",
                color=(200, 200, 200), anchor=(0.5, 0.5),
            )
            self.plot.addItem(hint)
            hint.setPos(0, 0)
        else:
            upper = self.plot.plot(t.z, t.w, pen=_ENVELOPE_PEN)
            lower = self.plot.plot(t.z, -t.w, pen=_ENVELOPE_PEN)
            self.plot.addItem(pg.FillBetweenItem(upper, lower, brush=_ENVELOPE_BRUSH))
            self.plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=_AXIS_PEN))
            self._draw_markers(t)
            ymax = float(np.nanmax(t.w))
            ymax = ymax if np.isfinite(ymax) and ymax > 0 else 1.0
            L = t.total_length
            self.plot.setXRange(-0.04 * L, 1.04 * L, padding=0)
            self.plot.setYRange(-1.35 * ymax, 1.75 * ymax, padding=0)

        self.probe_line = pg.InfiniteLine(angle=90, movable=False, pen=_PROBE_PEN)
        self.probe_line.setVisible(False)
        self.plot.addItem(self.probe_line, ignoreBounds=True)
        self.probe_pts = pg.ScatterPlotItem(
            size=9, pen=pg.mkPen((255, 220, 90)), brush=pg.mkBrush(255, 220, 90, 180)
        )
        self.plot.addItem(self.probe_pts, ignoreBounds=True)

    def _draw_markers(self, t):
        ymax = float(np.nanmax(t.w)) if t.w.size else 1.0
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
        # beam source (initial waist) at z = 0
        self.plot.addItem(
            pg.InfiniteLine(pos=0.0, angle=90, pen=_SOURCE_PEN),
            ignoreBounds=True,
        )
        src = pg.TextItem(
            text=f"Source: w₀={format_length(self.system.w0)}",
            color=(230, 230, 230), anchor=(0, 1),
        )
        self.plot.addItem(src, ignoreBounds=True)
        src.setPos(0.0, 1.66 * ymax)
        for i, mk in enumerate(t.markers):
            y_text = ymax * (1.10 + 0.24 * (i % 3))
            if mk.kind == "thick":
                region = pg.LinearRegionItem(
                    (mk.z0, mk.z1), movable=False,
                    brush=(160, 160, 210, 40), pen=pg.mkPen((160, 160, 210, 120)),
                )
                self.plot.addItem(region, ignoreBounds=True)
                z_text = (mk.z0 + mk.z1) / 2
            else:
                color = _LENS_COLOR if mk.kind == "lens" else _INTERFACE_COLOR
                self.plot.addItem(
                    pg.InfiniteLine(pos=mk.z0, angle=90,
                                    pen=pg.mkPen(color, style=Qt.DashLine, width=1)),
                    ignoreBounds=True,
                )
                z_text = mk.z0
            # anchor labels away from the view edges they would clip against
            L = t.total_length
            if z_text < 0.12 * L:
                anchor = (0, 1)
            elif z_text > 0.88 * L:
                anchor = (1, 1)
            else:
                anchor = (0.5, 1)
            text = pg.TextItem(text=mk.label, color=(210, 210, 210), anchor=anchor)
            self.plot.addItem(text, ignoreBounds=True)
            text.setPos(z_text, y_text)

    def refresh_results(self):
        f = self.trace.final
        self.res_w0.setText(f"Output waist w<sub>0</sub>′ = {format_length(f['w0'])}")
        self.res_loc.setText(f"Waist position (rel. last element) = {format_length(f['w0_loc'])}")
        self.res_zR.setText(f"Rayleigh range z<sub>R</sub> = {format_length(f['zR'])}")
        self.res_div.setText(f"Divergence half-angle θ = {f['theta'] / mrad:.4g} mrad")

    # ------------------------------------------------------------------
    # Ctrl+drag element sliding on the plot
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self.plot.viewport():
            et = event.type()
            if et == QEvent.MouseButtonPress:
                if (
                    event.button() == Qt.LeftButton
                    and event.modifiers() & Qt.ControlModifier
                    and self.trace is not None
                ):
                    spec = self._marker_at(event.position())
                    if spec is not None:
                        self._drag_spec = spec
                        self.plot.viewport().setCursor(Qt.SizeHorCursor)
                        return True  # consume: no ViewBox pan
            elif et == QEvent.MouseMove and self._drag_spec is not None:
                z = self._view_z(event.position())
                self._slide_element_to(self._drag_spec, z)
                self.refresh_plot()
                return True
            elif et == QEvent.MouseButtonRelease and self._drag_spec is not None:
                spec = self._drag_spec
                self._drag_spec = None
                self.plot.viewport().unsetCursor()
                self.row_tail.set_value(self.system.tail)
                row = (
                    self.system.elements.index(spec)
                    if spec in self.system.elements
                    else -1
                )
                self.refresh_all(select=row)
                return True
        return super().eventFilter(obj, event)

    def _view_z(self, pos):
        vb = self.plot.getViewBox()
        # toPoint(): PySide6's mapToScene(QPointF) overload is buggy
        return vb.mapSceneToView(self.plot.mapToScene(pos.toPoint())).x()

    def _marker_at(self, pos):
        """Non-FreeSpace element whose marker is within ~12 px of pos."""
        vb = self.plot.getViewBox()
        px_size = vb.viewPixelSize()[0]
        if px_size <= 0:
            return None
        z = self._view_z(pos)
        best, best_px = None, 12.0
        for mk in self.trace.markers:
            if mk.spec is None:
                continue
            zc = mk.z0 if mk.kind != "thick" else (mk.z0 + mk.z1) / 2
            dx = abs(z - zc) / px_size
            if dx < best_px:
                best, best_px = mk.spec, dx
        return best

    def _slide_element_to(self, spec, z_new):
        """Slide a non-FreeSpace element to z_new.

        Other elements keep their z positions; the free-space gaps around
        the dragged element merge and re-split, so crossing a neighbor
        naturally reorders the element list.  The draggable range extends
        over the trailing free space: sliding into it shortens the tail
        (the "auto" button restores the suggested length).
        """
        t_drag = spec.params.get("t", 0.0) if spec.type == "ThickLens" else 0.0
        others = []   # (spec, z0) of non-FreeSpace elements except the dragged one
        z_total = 0.0  # element path length (dragged element's own span excluded)
        for e in self.system.elements:
            if e is spec:
                continue
            if e.type == "FreeSpace":
                z_total += e.params["d"]
            elif e.type == "ThickLens":
                others.append((e, z_total))
                z_total += e.params["t"]
            else:
                others.append((e, z_total))
        tail = self.system.tail if self.system.tail is not None else 0.0
        z_max = z_total + tail - t_drag
        z_new = min(max(z_new, 0.0), max(z_max, 0.0))
        pos = sum(1 for _, z0 in others if z0 <= z_new)
        ordered = [e for e, _ in others]
        ordered.insert(pos, spec)
        positions = {id(e): z0 for e, z0 in others}
        positions[id(spec)] = z_new

        new_list, z = [], 0.0
        for e in ordered:
            gap = positions[id(e)] - z
            if gap > 0:
                new_list.append(ElementSpec("FreeSpace", {"d": gap}))
            new_list.append(e)
            z = positions[id(e)] + (
                e.params.get("t", 0.0) if e.type == "ThickLens" else 0.0
            )
        # Everything beyond the last element is the trailing free space:
        # sliding into it shortens it, sliding back out restores it.
        self.system.tail = max(z_total + tail - z, 0.0)
        self.system.elements = new_list

    # ------------------------------------------------------------------
    # Hover probe
    # ------------------------------------------------------------------

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        vb = self.plot.getViewBox()
        if self.trace is None:
            return
        if not vb.sceneBoundingRect().contains(pos):
            self._clear_probe()
            return
        z = vb.mapSceneToView(pos).x()
        res = self.trace.probe(z)
        if res is None:
            self._clear_probe()
            return
        w, R, zR, n = res
        self.probe_line.setVisible(True)
        self.probe_line.setPos(z)
        self.probe_pts.setData([z, z], [w, -w])
        self.lbl_z.setText(f"z = {format_length(z)}")
        self.lbl_w.setText(f"w(z) = {format_length(w)}")
        if np.isinf(R):
            r_text = "∞ (plane wavefront)"
        else:
            direction = "diverging" if R > 0 else "converging"
            r_text = f"{format_length(R)} ({direction})"
        self.lbl_R.setText(f"R(z) = {r_text}")
        self.lbl_zR.setText(f"z<sub>R</sub> = {format_length(zR)}")
        self.lbl_n.setText(f"n = {n:.4g}")

    def _clear_probe(self):
        if not hasattr(self, "probe_line"):
            return
        self.probe_line.setVisible(False)
        self.probe_pts.setData([], [])
        self.lbl_z.setText("z = —")
        self.lbl_w.setText("w(z) = —")
        self.lbl_R.setText("R(z) = —")
        self.lbl_zR.setText("z<sub>R</sub> = —")
        self.lbl_n.setText("n = —")

    # ------------------------------------------------------------------
    # Config save / load
    # ------------------------------------------------------------------

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save system configuration", "system.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.system.save(path)
        except OSError as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load system configuration", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.system = OpticalSystem.load(path)
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self.row_wl.set_value(self.system.wl)
        self.row_w0.set_value(self.system.w0)
        self.row_n.set_value(self.system.n0)
        self.row_tail.set_value(self.system.tail)
        self.refresh_all(select=0)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("GaussianBeam GUI")
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

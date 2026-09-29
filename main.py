import sys
import os
import threading
import pickle
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QComboBox, QPushButton, QDockWidget, QListWidget,
                             QMenu, QToolBar, QStatusBar, QFileDialog, QMessageBox, QDialog,
                             QDialogButtonBox, QTextEdit, QSlider, QColorDialog, QListWidgetItem,
                             QInputDialog, QTabWidget, QSplitter, QTableWidget, QTableWidgetItem,
                             QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QScrollArea, QAbstractItemView, QGridLayout, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMimeData
from PyQt6.QtGui import QAction, QIcon, QColor, QPalette, QFont, QKeySequence
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from core.data_engine import DataEngine
from core.viz_engine import VizEngine
from core.sheet_manager import SheetManager
from core.theme_manager import ThemeManager
from agent.graph_ds import agent_ds_app
from agent.graph_report import agent_report_app
from agent.voice_layer import VoiceCommandLayer
from services.experiment_registry import ExperimentRegistry
from services.lineage import DatasetLineage, dataset_fingerprint
from services.model_registry import ModelRegistry
from services.governance import (ScientificDataLeakageGate, DatasetCardBuilder, ModelCardBuilder,
                                 StatisticalUncertainty, ExperimentComparator, ModelPromotionRegistry,
                                 DataDiff, EvidenceDAG, HumanApprovalEvidence, AgentEvaluation)
from services.large_data import LargeDataEngine
from services.semantic_types import schema_table
from services.data_contracts import DataContractEngine
from services.split_wizard import SplitWizard
from services.error_analysis import ErrorAnalysisEngine
from services.statistical_analysis import StatisticalAnalysisEngine
from services.duckdb_workspace import DuckDBWorkspace
from services.notebook_export import NotebookExporter
from services.publication_package import PublicationPackageBuilder
from services.monitoring import ModelMonitoring
from services.sharing import SharingService
from services.macro_library import PYTHON_MACROS, SQL_MACROS
from services.data_quality import DataQualityEngine
from services.compute_backend import ComputeBackend
from services.agent_tools import default_registry
from services.analysis_recipe import AnalysisRecipe
from services.agent_console import AgentRunConsole
from services.analysis_state import AnalysisStateMachine, AnalysisState
from services.artifacts import ArtifactStore
from services.target_feature_selection import recommend_targets_and_features
from services.pdf_tables import PDFTableExtractor
from agent.graph_question_analysis import question_analysis_app
from services.project_io import build_manifest, save_manifest, load_manifest
from core.tableau_views import TableauSheet, TableauDashboard, TableauStory
from langchain_core.messages import HumanMessage
import config

class ColumnDragListWidget(QListWidget):
    """Data Management list that exports the actual field name as text/plain MIME."""
    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item is None:
            return
        mime = QMimeData()
        mime.setText(item.text())
        drag = __import__("PyQt6.QtGui", fromlist=["QDrag"]).QDrag(self)
        drag.setMimeData(mime)
        drag.exec(supportedActions)


class FieldDropComboBox(QComboBox):
    """Marks field selector with live column synchronization and real drag/drop."""
    fieldDropped = pyqtSignal(str)
    def __init__(self,parent=None):
        super().__init__(parent); self.setAcceptDrops(True); self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    def sync_fields(self,columns=None):
        if columns is None:
            w=self.window(); df=getattr(getattr(w,"data_engine",None),"df",None); columns=[] if df is None else [str(c) for c in df.columns]
        current=self.currentText().strip(); wanted=[""]+list(dict.fromkeys(columns))
        if [self.itemText(i) for i in range(self.count())] != wanted:
            self.blockSignals(True); self.clear(); self.addItems(wanted)
            if current in wanted: self.setCurrentIndex(wanted.index(current))
            self.blockSignals(False)
    def showPopup(self): self.sync_fields(); super().showPopup()
    def dragEnterEvent(self,event):
        event.acceptProposedAction() if event.mimeData().hasText() and event.mimeData().text().strip() else event.ignore()
    def dragMoveEvent(self,event): self.dragEnterEvent(event)
    def dropEvent(self,event):
        field=event.mimeData().text().strip() if event.mimeData().hasText() else ""
        w=self.window(); df=getattr(getattr(w,"data_engine",None),"df",None)
        if field and df is not None and field in df.columns:
            self.sync_fields([str(c) for c in df.columns]); self.setCurrentIndex(self.findText(field)); self.fieldDropped.emit(field); event.setDropAction(Qt.DropAction.CopyAction); event.accept()
        else: event.ignore()


class FieldDropListWidget(QListWidget):
    """Drop-only Filters target with explicit drag-move acceptance."""
    fieldDropped = pyqtSignal(str)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText() and event.mimeData().text().strip(): event.acceptProposedAction()
        else: event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText() and event.mimeData().text().strip(): event.acceptProposedAction()
        else: event.ignore()

    def dropEvent(self, event):
        field = event.mimeData().text().strip() if event.mimeData().hasText() else ""
        if field:
            self.fieldDropped.emit(field); event.acceptProposedAction()
        else: event.ignore()


class SimpleGraphWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    def __init__(self, app, state, config=None):
        super().__init__(); self.app=app; self.state=state; self.config=config
    def run(self):
        try:
            result=self.app.invoke(self.state, config=self.config) if self.config else self.app.invoke(self.state)
            self.finished.emit(result)
        except Exception as exc: self.error.emit(str(exc))


# ============================================================
# WORKER THREADS
# ============================================================

class AgentWorker(QThread):
    step_ready = pyqtSignal(str)
    finished = pyqtSignal(str, object)
    error = pyqtSignal(str)

    def __init__(self, agent_app, state, action='think'):
        super().__init__(); self.agent_app = agent_app; self.state = state; self.action = action

    def run(self):
        try:
            result = self.agent_app.invoke(self.state)
            if isinstance(result, dict):
                self.state.update(result)
            if self.state.get("needs_approval"):
                self.step_ready.emit(self.state.get("stage_summary") or self.state.get("current_step", "Awaiting approval..."))
            else:
                self.finished.emit(self.state.get("stage_summary", "Agent finished."), self.state)
        except Exception as e:
            self.error.emit(str(e))


class ReportWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, agent_app, state):
        super().__init__()
        self.agent_app = agent_app
        self.state = state

    def run(self):
        try:
            result = self.agent_app.invoke(self.state)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# DIALOGS
# ============================================================

class LocalLLMConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Local LLM Model")
        self.resize(500, 300)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select a local HuggingFace model from your cache:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(config.LOCAL_LLM_MODELS.keys()))
        layout.addWidget(self.model_combo)
        layout.addWidget(QLabel("Or enter a custom model path:"))
        self.custom_path = QLineEdit()
        layout.addWidget(self.custom_path)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selected_model(self):
        if self.custom_path.text().strip():
            return self.custom_path.text().strip()
        return self.model_combo.currentText()


class DataAnalysisDialog(QDialog):
    def __init__(self, df, analysis_type, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Data Analysis: {analysis_type}")
        self.resize(800, 600)
        self.df = df
        layout = QVBoxLayout(self)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont("Courier", 10))
        layout.addWidget(text)

        if analysis_type == "Descriptive Statistics":
            text.setText(df.describe(include='all').to_string())
        elif analysis_type == "Correlation Matrix":
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                text.setText("No numeric columns for correlation.")
            else:
                text.setText(numeric_df.corr().to_string())
        elif analysis_type == "Hypothesis Testing":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                from scipy import stats
                t_stat, p_val = stats.ttest_ind(df[numeric_cols[0]].dropna(), df[numeric_cols[1]].dropna())
                text.setText(f"T-test between {numeric_cols[0]} and {numeric_cols[1]}\nT-statistic: {t_stat:.4f}\nP-value: {p_val:.4f}")
            else:
                text.setText("Need at least 2 numeric columns.")
        elif analysis_type == "Time Series Analysis":
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if date_cols:
                text.setText(f"Time series analysis on {date_cols[0]}\nMean: {df[date_cols[0]].mean()}\nTrend: {df[date_cols[0]].diff().mean()}")
            else:
                text.setText("No datetime columns found.")
        elif analysis_type == "Regression Analysis":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                from sklearn.linear_model import LinearRegression
                X = df[[numeric_cols[0]]].dropna()
                y = df[numeric_cols[1]].dropna()
                common_idx = X.index.intersection(y.index)
                X, y = X.loc[common_idx], y.loc[common_idx]
                if len(X) > 1:
                    model = LinearRegression().fit(X, y)
                    text.setText(f"Linear Regression: {numeric_cols[1]} ~ {numeric_cols[0]}\n"
                                 f"Coefficient: {model.coef_[0]:.4f}\n"
                                 f"Intercept: {model.intercept_:.4f}\n"
                                 f"R²: {model.score(X, y):.4f}")
                else:
                    text.setText("Not enough data.")
            else:
                text.setText("Need at least 2 numeric columns.")


class MacroEditorDialog(QDialog):
    """Professional multi-run Python/SQL workspace; execution output stays on the same page."""
    run_requested = pyqtSignal(str)
    def __init__(self,macro_type,parent=None):
        super().__init__(parent); self.setWindowTitle(f"{macro_type} Macro Editor"); self.resize(1000,700); self.macro_type=macro_type
        root=QVBoxLayout(self); head=QHBoxLayout(); head.addWidget(QLabel(f"{macro_type} Macro Workspace — 100 professional starter commands"))
        self.library_combo=QComboBox(); self.library_combo.addItem("Choose a starter command…"); self._macros=PYTHON_MACROS if macro_type=="Python" else SQL_MACROS; self.library_combo.addItems([t for t,_ in self._macros]); self.library_combo.currentIndexChanged.connect(self._load_macro); head.addWidget(self.library_combo); root.addLayout(head)
        split=QSplitter(Qt.Orientation.Vertical); root.addWidget(split,1); self.editor=QTextEdit(); self.editor.setFont(QFont("Courier",10)); self.editor.setPlaceholderText("Python: use df, pd, np, plt. SQL: query table 'data'."); split.addWidget(self.editor)
        out=QWidget(); ol=QVBoxLayout(out); ol.addWidget(QLabel("Execution Output / History")); self.output=QTextEdit(); self.output.setReadOnly(True); self.output.setFont(QFont("Courier",10)); ol.addWidget(self.output); split.addWidget(out); split.setSizes([360,260])
        buttons=QHBoxLayout(); run=QPushButton("▶ Run Current"); run.clicked.connect(lambda:self.run_requested.emit(self.editor.toPlainText())); buttons.addWidget(run); clear=QPushButton("Clear Output"); clear.clicked.connect(self.output.clear); buttons.addWidget(clear); buttons.addStretch(); close=QPushButton("Close"); close.clicked.connect(self.reject); buttons.addWidget(close); root.addLayout(buttons)
    def _load_macro(self,index):
        if index>0:self.editor.setPlainText(self._macros[index-1][1])
    def append_output(self,text): self.output.append(str(text)); self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())


class PDFTableSelectionDialog(QDialog):
    def __init__(self,tables,parent=None):
        super().__init__(parent); self.setWindowTitle("PDF Tables — Select Tables to Load"); self.resize(900,650); self.tables=tables
        root=QVBoxLayout(self); root.addWidget(QLabel("Select the PDF tables you want to load as sheets. You can select all, clear all, or preview the selected table."))
        split=QSplitter(Qt.Orientation.Horizontal); root.addWidget(split,1)
        left=QWidget(); ll=QVBoxLayout(left); self.list=QListWidget(); self.list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection); ll.addWidget(self.list)
        buttons=QHBoxLayout(); allb=QPushButton("Select All"); noneb=QPushButton("Clear All"); buttons.addWidget(allb); buttons.addWidget(noneb); ll.addLayout(buttons); split.addWidget(left)
        self.preview=QTextEdit(); self.preview.setReadOnly(True); split.addWidget(self.preview); split.setSizes([360,540])
        for i,t in enumerate(tables):
            item=QListWidgetItem(f"{i+1}. {t['name']}  [{len(t['data']):,} × {len(t['data'].columns):,}]"); item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable); item.setCheckState(Qt.CheckState.Checked); item.setData(Qt.ItemDataRole.UserRole,i); self.list.addItem(item)
        self.list.itemClicked.connect(self._preview); allb.clicked.connect(lambda:self._set_all(True)); noneb.clicked.connect(lambda:self._set_all(False));
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel); bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); root.addWidget(bb)
        if self.list.count(): self.list.setCurrentRow(0); self._preview(self.list.item(0))
    def _set_all(self,state):
        for i in range(self.list.count()): self.list.item(i).setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)
    def _preview(self,item):
        if item is None:return
        idx=item.data(Qt.ItemDataRole.UserRole); df=self.tables[idx]['data']; self.preview.setPlainText(df.head(30).to_string(index=False))
    def selected_indices(self): return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count()) if self.list.item(i).checkState()==Qt.CheckState.Checked]


class GroupByDialog(QDialog):
    """V16.py Group By Dialog."""
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Group By")
        self.resize(600, 500)
        self.df = df
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select columns to group by:"))
        self.group_list = QListWidget()
        self.group_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for col in df.columns:
            self.group_list.addItem(col)
        layout.addWidget(self.group_list)

        layout.addWidget(QLabel("Aggregations (for numeric columns):"))
        self.agg_layout = QFormLayout()
        self.agg_combos = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            combo = QComboBox()
            combo.addItems(["Sum", "Average", "Count", "Min", "Max", "Count Distinct"])
            self.agg_combos[col] = combo
            self.agg_layout.addRow(col, combo)
        layout.addLayout(self.agg_layout)

        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        layout.addWidget(QLabel("Preview:"))
        layout.addWidget(self.preview)

        btn_layout = QHBoxLayout()
        self.replace_btn = QPushButton("Replace Current Data")
        self.new_sheet_btn = QPushButton("Create New Sheet")
        self.replace_btn.clicked.connect(self.accept)
        self.new_sheet_btn.clicked.connect(self.accept)
        self.new_sheet_requested = False
        self.new_sheet_btn.clicked.connect(lambda: setattr(self, "new_sheet_requested", True))
        btn_layout.addWidget(self.replace_btn)
        btn_layout.addWidget(self.new_sheet_btn)
        layout.addLayout(btn_layout)

        self.group_list.itemSelectionChanged.connect(self.update_preview)
        for combo in self.agg_combos.values():
            combo.currentTextChanged.connect(self.update_preview)

    def update_preview(self):
        selected = [item.text() for item in self.group_list.selectedItems()]
        if not selected:
            self.preview.setText("Select at least one column to group by.")
            return
        agg_dict = {}
        for col, combo in self.agg_combos.items():
            agg_func = combo.currentText()
            if agg_func == "Sum":
                agg_dict[col] = 'sum'
            elif agg_func == "Average":
                agg_dict[col] = 'mean'
            elif agg_func == "Count":
                agg_dict[col] = 'count'
            elif agg_func == "Min":
                agg_dict[col] = 'min'
            elif agg_func == "Max":
                agg_dict[col] = 'max'
            elif agg_func == "Count Distinct":
                agg_dict[col] = 'nunique'
        try:
            result = self.df.groupby(selected).agg(agg_dict).reset_index()
            self.preview.setText(result.head(15).to_string())
        except Exception as e:
            self.preview.setText(f"Error: {e}")


# ============================================================
# MAIN WINDOW
# ============================================================

class EvidenceObjectProxy:
    @staticmethod
    def from_dict(data, title):
        return {"evidence_id": f"gate-{abs(hash(title + str(data))) % 10**10}", "kind": "governance_gate",
                "title": title, "status": data.get("status", "complete"), "data": data, "parent_ids": []}



def load_manifest_from_text(text: str) -> dict:
    data = json.loads(text)
    if data.get("format") != "dssp-project-manifest":
        raise ValueError("This file is not a valid Data Science Studio Pro project manifest.")
    return data

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Science Studio Pro")
        self.resize(1600, 1000)

        # --- Core Components ---
        self.data_engine = DataEngine()
        self.viz_engine = VizEngine()
        self.sheet_manager = SheetManager()
        self.theme_manager = ThemeManager(self)

        # --- AI Agent State ---
        self.agent_ds_state = {
            "messages": [], "approved_steps": [], "rejected_steps": [],
            "user_approved": False, "dataframe": None,
            "api_key": config.DEFAULT_API_KEY,
            "model_name": config.ALL_LLM_MODELS[0], "max_steps": 20, "abort_requested": False,
            "human_approval_evidence": []
        }
        self.agent_report_state = {
            "analysis_results": "", "pdf_buffer": None,
            "api_key": config.DEFAULT_API_KEY,
            "model_name": config.ALL_LLM_MODELS[0]
        }
        self.agent_worker = None
        self.report_worker = None
        self.voice_layer = VoiceCommandLayer()
        self.experiment_registry = ExperimentRegistry()
        self.lineage = DatasetLineage()
        self.model_registry = ModelRegistry()
        self.promotion_registry = ModelPromotionRegistry()
        self.large_data_engine = LargeDataEngine()
        self.duckdb_workspace = DuckDBWorkspace()
        self.data_quality_engine = DataQualityEngine()
        self.leakage_gate = ScientificDataLeakageGate()
        self.dataset_card = {}
        self.data_contract = {}
        self.model_cards = []
        self.evidence_dag = EvidenceDAG()
        self.evidence_records = []
        self.experiment_records = []
        self.history = []
        self.redo_history = []
        self.current_extra_insight_col = None
        self.agent_abort_requested = False
        self.hierarchies = {}
        self.mark_aggregations = {"Color":"sum","Size":"sum","Text":"count","Detail":"count","Tooltip":"count"}
        self.table_agent_memory = []
        self.presentation_agent_memory = []
        self.last_share_package_path = ""
        self.last_presentation_path = ""
        self.compute_mode = "CPU"
        self.compute_info = ComputeBackend.detect()
        self.agent_tool_registry = default_registry()
        self.analysis_recipe = AnalysisRecipe()
        self.agent_console = AgentRunConsole()
        self.analysis_state = AnalysisStateMachine(AnalysisState.DATA_LOADED.value)
        self.artifact_store = ArtifactStore()
        self.question_agent_memory = []

        # --- UI Setup ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 1. Rows and Columns Shelves
        self.top_shelf_layout = QHBoxLayout()
        self.rows_input = QLineEdit()
        self.rows_input.setPlaceholderText("Drop Rows here")
        self.rows_input.setAcceptDrops(True)
        self.rows_input.dragEnterEvent = self.shelf_drag_enter
        self.rows_input.dropEvent = self.row_drop
        self.rows_input.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.rows_input.customContextMenuRequested.connect(lambda pos: self.shelf_context_menu(pos, "Rows"))
        self.cols_input = QLineEdit()
        self.cols_input.setPlaceholderText("Drop Columns here")
        self.cols_input.setAcceptDrops(True)
        self.cols_input.dragEnterEvent = self.shelf_drag_enter
        self.cols_input.dropEvent = self.col_drop
        self.cols_input.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.cols_input.customContextMenuRequested.connect(lambda pos: self.shelf_context_menu(pos, "Columns"))
        self.top_shelf_layout.addWidget(QLabel("Rows:"))
        self.top_shelf_layout.addWidget(self.rows_input)
        self.top_shelf_layout.addWidget(QLabel("Columns:"))
        self.top_shelf_layout.addWidget(self.cols_input)
        self.layout.addLayout(self.top_shelf_layout)

        # 2. Marks Panel (Using QComboBox)
        self.marks_layout = QHBoxLayout()
        self.marks_layout.addWidget(QLabel("Marks:"))
        self.marks_widgets = {}
        marks_config = [("Color", "🎨"), ("Size", "📏"), ("Text", "🔤"), ("Detail", "🔍"), ("Tooltip", "💬")]
        for mark_name, icon in marks_config:
            self.marks_layout.addWidget(QLabel(f"{icon} {mark_name}:"))
            combo = FieldDropComboBox()
            combo.setMinimumWidth(100)
            combo.setToolTip(f"Marks → {mark_name}: choose a field or drag a field here from Data Management.")
            combo.fieldDropped.connect(lambda field, mn=mark_name: self.assign_mark_field(mn, field))
            combo.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            combo.customContextMenuRequested.connect(lambda pos, mn=mark_name: self.show_mark_context_menu(pos, mn))
            self.marks_layout.addWidget(combo)
            self.marks_widgets[mark_name] = combo
            combo.currentTextChanged.connect(lambda _text, _mn=mark_name: self.update_plot())

        self.marks_layout.addWidget(QLabel("📊 Chart Type:"))
        self.chart_combo = QComboBox()
        self.chart_combo.addItems(["auto", "horizontal_bar", "vertical_bar", "stacked_bar", "line_discrete", "line_continuous", "area", "dual_axis", "scatter", "histogram", "box", "density", "symbol_map", "filled_map", "gantt", "bullet", "heatmap", "highlight_table", "waterfall", "pareto", "donut", "bump", "sankey", "treemap", "crosstab"])
        self.chart_combo.currentTextChanged.connect(lambda _text: self.update_plot())
        self.marks_layout.addWidget(self.chart_combo)
        self.layout.addLayout(self.marks_layout)

        # 3. Central Plot Canvas
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # 4. Right Side Panels
        self.setup_right_panels()

        # 5. Menus and Toolbar
        self.setup_menus()
        self.setup_toolbar()

        # 6. Status Bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

        # 7. Apply Default Theme
        self.theme_manager.apply_theme("#e3f2fd")

        # 8. Context Menu for Theme
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        # 9. Keyboard Shortcuts
        self.setup_shortcuts()

    # ============================================================
    # KEYBOARD SHORTCUTS
    # ============================================================
    def setup_shortcuts(self):
        QAction("Open", self, shortcut=QKeySequence.StandardKey.Open, triggered=self.open_file).setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        QAction("Save", self, shortcut=QKeySequence.StandardKey.Save, triggered=self.save_project).setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        QAction("Undo", self, shortcut=QKeySequence.StandardKey.Undo, triggered=self.undo).setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        QAction("Redo", self, shortcut=QKeySequence.StandardKey.Redo, triggered=self.redo).setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        QAction("Exit", self, shortcut=QKeySequence.StandardKey.Quit, triggered=self.close).setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.addAction(QAction("Open", self, shortcut=QKeySequence.StandardKey.Open, triggered=self.open_file))
        self.addAction(QAction("Save", self, shortcut=QKeySequence.StandardKey.Save, triggered=self.save_project))
        self.addAction(QAction("Undo", self, shortcut=QKeySequence.StandardKey.Undo, triggered=self.undo))
        self.addAction(QAction("Redo", self, shortcut=QKeySequence.StandardKey.Redo, triggered=self.redo))

    # ============================================================
    # RIGHT PANELS
    # ============================================================
    def setup_right_panels(self):
        # Panel 1: Loaded Sheets
        self.sheets_dock = QDockWidget("Loaded Sheets", self)
        self.sheets_list = QListWidget()
        self.sheets_list.itemClicked.connect(self.on_loaded_sheet_select)
        self.sheets_dock.setWidget(self.sheets_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.sheets_dock)

        # Panel 2: Data Management
        self.data_mgmt_dock = QDockWidget("Data Management", self)
        data_widget = QWidget()
        data_layout = QVBoxLayout()
        data_layout.addWidget(QLabel("Numerical Columns"))
        self.num_list = ColumnDragListWidget()
        self.num_list.setDragEnabled(True)
        self.num_list.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.num_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.num_list.customContextMenuRequested.connect(self.show_column_menu)
        data_layout.addWidget(self.num_list)
        data_layout.addWidget(QLabel("Categorical Columns"))
        self.cat_list = ColumnDragListWidget()
        self.cat_list.setDragEnabled(True)
        self.cat_list.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.cat_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.cat_list.customContextMenuRequested.connect(self.show_column_menu)
        data_layout.addWidget(self.cat_list)
        data_widget.setLayout(data_layout)
        self.data_mgmt_dock.setWidget(data_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.data_mgmt_dock)

        # Panel 3: Filters
        self.filters_dock = QDockWidget("Filters", self)
        self.filters_list = FieldDropListWidget()
        self.filters_list.setAcceptDrops(True)
        self.filters_list.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.filters_list.setDropIndicatorShown(True)
        self.filters_list.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.filters_list.fieldDropped.connect(self._add_filter_v16)
        self.filters_list.itemDoubleClicked.connect(self._edit_filter_from_list)
        self.filters_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.filters_list.customContextMenuRequested.connect(self._filter_context_menu)
        self.filters_dock.setWidget(self.filters_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.filters_dock)

        # Panel 4: Extra Insight
        self.insight_dock = QDockWidget("Extra Insight", self)
        self.insight_label = QLabel("Drop column for extra analysis")
        self.insight_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.insight_label.setAcceptDrops(True)
        self.insight_label.dragEnterEvent = self.shelf_drag_enter
        self.insight_label.dropEvent = self.insight_drop
        self.insight_dock.setWidget(self.insight_label)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.insight_dock)

    def show_mark_context_menu(self, pos, mark_name):
        combo = self.marks_widgets.get(mark_name)
        if combo is None:
            return
        menu = QMenu(self)
        menu.addAction("Clear", lambda: (combo.setCurrentIndex(0), self.update_plot()))
        if mark_name in {"Color", "Size", "Text", "Detail", "Tooltip"}:
            agg_menu = menu.addMenu("Aggregation")
            for label, value in [("Sum", "sum"), ("Average", "mean"), ("Count", "count"), ("Min", "min"), ("Max", "max")]:
                agg_menu.addAction(label, lambda v=value, mn=mark_name: self._set_mark_aggregation(mn, v))
        menu.exec(combo.mapToGlobal(pos))

    # ============================================================
    # DRAG AND DROP — Tableau-style shelves/marks/filters/insight
    # ============================================================
    def shelf_drag_enter(self, event):
        if event.mimeData().hasText() and event.mimeData().text().strip():
            event.acceptProposedAction()
        else:
            event.ignore()

    def _dropped_column(self, event):
        return event.mimeData().text().strip() if event.mimeData().hasText() else ""

    def _append_shelf(self, widget, col):
        if not col or self.data_engine.df is None or col not in self.data_engine.df.columns:
            return False
        fields = self.viz_engine.parse_shelf(widget.text())
        if col not in fields:
            fields.append(col)
            widget.setText(self.viz_engine.format_shelf(fields))
        return True

    def shelf_context_menu(self, pos, shelf_name):
        widget = self.rows_input if shelf_name == "Rows" else self.cols_input
        fields = self.viz_engine.parse_shelf(widget.text())
        menu = QMenu(self)
        if not fields:
            menu.addAction("No fields in shelf").setEnabled(False)
        else:
            remove_menu = menu.addMenu(f"Remove from {shelf_name}")
            for field in fields:
                remove_menu.addAction(field, lambda f=field, w=widget: self._remove_shelf_field(w, f))
            drill_menu=menu.addMenu("Drill Down")
            has_drill=False
            for field in fields:
                if any(field in h and h.index(field)<len(h)-1 for h in self.hierarchies.values()):
                    drill_menu.addAction(field, lambda f=field, w=widget: self.drill_down_field(w,f)); has_drill=True
            if not has_drill: drill_menu.setEnabled(False)
            menu.addSeparator()
            menu.addAction("Clear shelf", lambda w=widget: (w.clear(), self.update_plot()))
        menu.exec(widget.mapToGlobal(pos))

    def _remove_shelf_field(self, widget, field):
        widget.setText(self.viz_engine.remove_from_shelf(widget.text(), field))
        self._record_view_change()
        self.update_plot()

    def _record_view_change(self):
        self.history.append({"rows": self.rows_input.text(), "columns": self.cols_input.text(),
                             "chart": self.chart_combo.currentText(),
                             "marks": {k: v.currentText() for k,v in self.marks_widgets.items()}})
        self.redo_history.clear()

    def row_drop(self, event):
        col = self._dropped_column(event)
        if self._append_shelf(self.rows_input, col):
            self.update_plot()
        event.acceptProposedAction()

    def col_drop(self, event):
        col = self._dropped_column(event)
        if self._append_shelf(self.cols_input, col):
            self.update_plot()
        event.acceptProposedAction()

    def _set_mark_aggregation(self, mark_name, value):
        if not hasattr(self, "mark_aggregations"): self.mark_aggregations={}
        self.mark_aggregations[mark_name]=value
        self.update_plot()

    def assign_mark_field(self, mark_name, col):
        combo=self.marks_widgets.get(mark_name); df=self.data_engine.df
        if combo is None or df is None or col not in df.columns: return False
        combo.sync_fields([str(c) for c in df.columns]); idx=combo.findText(str(col))
        if idx < 0: combo.addItem(str(col)); idx=combo.findText(str(col))
        combo.blockSignals(True); combo.setCurrentIndex(idx); combo.blockSignals(False)
        self._record_view_change(); self.status.showMessage(f"{col} assigned to Marks → {mark_name}."); self.update_plot(); return True

    def mark_drop(self, event, mark_name):
        col = self._dropped_column(event)
        if col: self.assign_mark_field(mark_name, col)
        event.acceptProposedAction()

    def filter_drop(self, event):
        col = self._dropped_column(event)
        source_df = self.data_engine.original_df if self.data_engine.original_df is not None else self.data_engine.df
        if col and source_df is not None and col in source_df.columns:
            self._add_filter_v16(col)
            event.setDropAction(Qt.DropAction.CopyAction); event.accept()
        else:
            event.ignore()

    def insight_drop(self, event):
        col = self._dropped_column(event)
        if col and self.data_engine.df is not None and col in self.data_engine.df.columns:
            self.current_extra_insight_col = col
            self.show_extra_insight(col)
        event.acceptProposedAction()

    # ============================================================
    # FILTERS — persistent, source-based Tableau-style filters
    # ============================================================
    def _add_filter_v16(self, col, edit_existing=False):
        df = self.data_engine.original_df if self.data_engine.original_df is not None else self.data_engine.df
        if df is None or col not in df.columns:
            return
        existing = self.data_engine.filter_specs.get(col, {}) if edit_existing else {}
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if series.empty: return
                lo, hi = float(series.min()), float(series.max())
                dlg = QDialog(self); dlg.setWindowTitle(f"Filter: {col}"); dlg.resize(420, 170)
                form = QFormLayout(dlg)
                min_box = QDoubleSpinBox(); max_box = QDoubleSpinBox()
                min_box.setRange(-1e15, 1e15); max_box.setRange(-1e15, 1e15); min_box.setDecimals(6); max_box.setDecimals(6)
                min_box.setValue(float(existing.get("min", lo))); max_box.setValue(float(existing.get("max", hi)))
                form.addRow("Minimum", min_box); form.addRow("Maximum", max_box)
                buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject); form.addRow(buttons)
                if dlg.exec() != QDialog.DialogCode.Accepted: return
                if min_box.value() > max_box.value():
                    QMessageBox.warning(self, "Invalid Filter", "Minimum must not exceed maximum."); return
                spec = {"kind": "numeric", "min": min_box.value(), "max": max_box.value()}
            elif pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in str(df[col].dtype).lower():
                series = pd.to_datetime(df[col], errors="coerce").dropna()
                if series.empty: return
                start, end = series.min().date(), series.max().date()
                start_text, ok1 = QInputDialog.getText(self, f"Filter: {col}", "Start date (YYYY-MM-DD):", text=str(existing.get("start", start)))
                if not ok1: return
                end_text, ok2 = QInputDialog.getText(self, f"Filter: {col}", "End date (YYYY-MM-DD):", text=str(existing.get("end", end)))
                if not ok2: return
                pd.to_datetime(start_text); pd.to_datetime(end_text)
                spec = {"kind": "date", "start": start_text, "end": end_text}
            else:
                values = sorted(df[col].dropna().astype(str).unique().tolist())
                if not values: return
                dlg = QDialog(self); dlg.setWindowTitle(f"Filter: {col}"); dlg.resize(420, 500)
                layout = QVBoxLayout(dlg); layout.addWidget(QLabel("Select one or more values:"))
                checks = []
                old = set(map(str, existing.get("values", values)))
                for value in values:
                    cb = QCheckBox(value); cb.setChecked(value in old); checks.append(cb); layout.addWidget(cb)
                buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject); layout.addWidget(buttons)
                if dlg.exec() != QDialog.DialogCode.Accepted: return
                selected = [cb.text() for cb in checks if cb.isChecked()]
                if not selected:
                    QMessageBox.warning(self, "Invalid Filter", "Select at least one value."); return
                spec = {"kind": "categorical", "values": selected}
            self.data_engine.set_filter(col, spec)
            self._refresh_filter_list()
            self.update_plot()
            self.status.showMessage(f"Filter applied: {col}")
        except Exception as exc:
            QMessageBox.critical(self, "Filter Error", str(exc))

    def _refresh_filter_list(self):
        self.filters_list.clear()
        for col, spec in self.data_engine.filter_specs.items():
            if spec.get("kind") == "numeric":
                label = f"{col}: {spec['min']:.4g} – {spec['max']:.4g}"
            elif spec.get("kind") == "date":
                label = f"{col}: {spec.get('start','')} – {spec.get('end','')}"
            else:
                vals = spec.get("values", [])
                label = f"{col}: {len(vals)} selected"
            item = QListWidgetItem(label); item.setData(Qt.ItemDataRole.UserRole, col); self.filters_list.addItem(item)

    def _edit_filter_from_list(self, item):
        col = item.data(Qt.ItemDataRole.UserRole)
        if col: self._add_filter_v16(col, edit_existing=True)

    def _filter_context_menu(self, pos):
        item = self.filters_list.itemAt(pos)
        if not item: return
        col = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self); menu.addAction("Edit Filter", lambda: self._add_filter_v16(col, True))
        menu.addAction("Remove Filter", lambda: (self.data_engine.remove_filter(col), self._refresh_filter_list(), self.update_plot()))
        menu.exec(self.filters_list.mapToGlobal(pos))

    # ============================================================
    # EXTRA INSIGHT WINDOW (V16.py Style)
    # ============================================================
    def show_extra_insight(self, column):
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        dialog = QDialog(self); dialog.setWindowTitle(f"Extra Insight: {column}"); dialog.resize(900, 700)
        layout = QVBoxLayout(dialog); controls = QHBoxLayout()
        controls.addWidget(QLabel("Top N:")); top_n = QSpinBox(); top_n.setRange(5, 50); top_n.setValue(10); controls.addWidget(top_n)
        controls.addWidget(QLabel("Chart Type:")); chart_type = QComboBox(); chart_type.addItems(["bar", "line", "scatter", "pie"]); controls.addWidget(chart_type)
        controls.addWidget(QLabel("Font Size:")); font_size = QSpinBox(); font_size.setRange(8, 30); font_size.setValue(12); controls.addWidget(font_size)
        controls.addWidget(QLabel("Color:")); color_btn = QPushButton("Pick"); controls.addWidget(color_btn)
        controls.addWidget(QLabel("Hover Color:")); hover_btn = QPushButton("Pick"); controls.addWidget(hover_btn); layout.addLayout(controls)
        fig = Figure(figsize=(8, 5), dpi=100); ax = fig.add_subplot(111); canvas = FigureCanvas(fig); layout.addWidget(canvas)
        save_btn = QPushButton("Save Plot as PNG"); layout.addWidget(save_btn)
        current = {"color": "#1976d2", "hover": "#ff9900"}; state = {"artists": [], "annotation": None}

        def pick(target):
            c = QColorDialog.getColor();
            if c.isValid():
                current[target] = c.name(); (color_btn if target == "color" else hover_btn).setStyleSheet(f"background-color:{c.name()}; color:white;"); update_plot()

        color_btn.clicked.connect(lambda: pick("color")); hover_btn.clicked.connect(lambda: pick("hover"))

        def update_plot():
            ax.clear(); state["artists"] = []; state["annotation"] = None
            data = self.data_engine.get_extra_insight(column, top_n.value())
            if data is None or data.empty:
                ax.text(.5, .5, "No data available", ha="center", va="center"); canvas.draw(); return
            is_num = pd.api.types.is_numeric_dtype(self.data_engine.df[column])
            labels = data[column].astype(str).tolist()
            vals = data[column].tolist() if is_num else data["Count"].tolist()
            kind = chart_type.currentText()
            if kind == "pie":
                wedges, _, _ = ax.pie(vals, labels=labels, autopct="%1.1f%%", colors=[current["color"]] * len(vals)); state["artists"] = list(wedges)
            elif kind == "line":
                line, = ax.plot(range(len(vals)), vals, marker="o", linewidth=2, color=current["color"]); state["artists"] = [line]
                ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=45, ha="right")
            elif kind == "scatter":
                pts = ax.scatter(range(len(vals)), vals, s=90, color=current["color"]); state["artists"] = [pts]
                ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=45, ha="right")
            else:
                bars = ax.bar(range(len(vals)), vals, color=current["color"]); state["artists"] = list(bars)
                ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_title(f"Top {top_n.value()} of {column}"); ax.set_xlabel(column); ax.tick_params(labelsize=font_size.value()); fig.subplots_adjust(left=.10,right=.97,bottom=.18,top=.92); canvas.draw()

        def on_move(event):
            if event.inaxes != ax or not state["artists"]: return
            found = None
            for artist in state["artists"]:
                try:
                    hit, _ = artist.contains(event)
                    if hit: found = artist; break
                except Exception: pass
            # Reset all artists, then highlight only the hovered mark.
            for artist in state["artists"]:
                try: artist.set_color(current["color"])
                except Exception: pass
            if found is not None:
                try: found.set_color(current["hover"])
                except Exception: pass
            canvas.draw_idle()

        canvas.mpl_connect("motion_notify_event", on_move)
        save_btn.clicked.connect(lambda: self._save_figure(dialog, fig))
        top_n.valueChanged.connect(update_plot); chart_type.currentTextChanged.connect(update_plot); font_size.valueChanged.connect(update_plot)
        update_plot(); dialog.exec()

    def _save_figure(self, dialog, fig):
        filepath, _ = QFileDialog.getSaveFileName(dialog, "Save Plot", "", "PNG Files (*.png)")
        if filepath:
            fig.savefig(filepath, dpi=180, bbox_inches="tight"); QMessageBox.information(dialog, "Saved", f"Saved: {filepath}")

    # ============================================================
    # MENUS
    # ============================================================
    def setup_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open CSV/Excel", self.open_file)
        file_menu.addAction("Open PDF", self.open_pdf)
        file_menu.addSeparator()
        file_menu.addAction("Save Project", self.save_project)
        file_menu.addAction("Load Project", self.load_project)
        file_menu.addAction("Export Analysis Snapshot", self.export_analysis_snapshot)
        file_menu.addAction("Export Reproducible Notebook", self.export_notebook)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        info_menu = menubar.addMenu("Data Info")
        info_menu.addAction("Missing Values", self.show_missing_values)
        info_menu.addAction("Describe Data", self.show_describe_data)
        info_menu.addAction("Data Info", self.show_data_info)
        info_menu.addAction("Data Quality Center", self.show_data_quality)
        info_menu.addAction("Data Dictionary", self.show_data_dictionary)
        info_menu.addAction("Core Schema Types", self.show_core_schema_types)

        prep_menu = menubar.addMenu("Data Pre-processing")
        prep_menu.addAction("Data Cleaning", self.data_cleaning)
        prep_menu.addAction("Data Integration", self.data_integration)
        prep_menu.addAction("Data Transformation", self.data_transformation)
        prep_menu.addAction("Data Reduction", self.data_reduction)
        prep_menu.addAction("Feature Engineering", self.feature_engineering)
        prep_menu.addAction("Data Encoding", self.data_encoding)
        prep_menu.addAction("Label Encoding", self.label_encoding)
        prep_menu.addAction("Data Scaling", self.data_scaling)
        prep_menu.addAction("Data Splitting", self.data_splitting)

        analysis_menu = menubar.addMenu("Data Analysis")
        analysis_menu.addAction("Descriptive Statistics", lambda: self.show_analysis("Descriptive Statistics"))
        analysis_menu.addAction("Correlation Matrix", lambda: self.show_analysis("Correlation Matrix"))
        analysis_menu.addAction("Hypothesis Testing", lambda: self.show_analysis("Hypothesis Testing"))
        analysis_menu.addAction("Time Series Analysis", lambda: self.show_analysis("Time Series Analysis"))
        analysis_menu.addAction("Regression Analysis", lambda: self.show_analysis("Regression Analysis"))
        analysis_menu.addAction("DuckDB Analytical Workspace", self.show_duckdb_workspace)

        ai_menu = menubar.addMenu("AI Agents")
        ai_menu.addAction("Agent Data Scientist", self.run_agent_data_scientist)
        ai_menu.addAction("AI Agent Plot", self.run_ai_agent_plot)
        ai_menu.addAction("AI Agent Report", self.run_ai_report)
        ai_menu.addAction("AI Agent Table Creation", self.run_ai_agent_table_creation)
        ai_menu.addAction("AI Agent Presentation", self.run_ai_agent_presentation)
        ai_menu.addAction("AI Agent Question → Analysis", self.run_ai_agent_question_analysis)
        ai_menu.addAction("Agent Run Console", self.show_agent_run_console)
        ai_menu.addSeparator()
        ai_menu.addAction("Select Local LLM Model", self.select_local_llm)

        macro_menu = menubar.addMenu("Macro")
        macro_menu.addAction("Python", lambda: self.open_macro_editor("Python"))
        macro_menu.addAction("SQL", lambda: self.open_macro_editor("SQL"))

        gov_menu = menubar.addMenu("Governance")
        gov_menu.addAction("Scientific/Data Leakage Gate", self.show_leakage_gate)
        gov_menu.addAction("Dataset Card", self.show_dataset_card)
        gov_menu.addAction("Model Cards", self.show_model_cards)
        gov_menu.addAction("Experiment Comparison", self.show_experiment_comparison)
        gov_menu.addAction("Model Promotion", self.show_model_promotion)
        gov_menu.addAction("Data Diff", self.show_data_diff)
        gov_menu.addAction("Evidence DAG", self.show_evidence_dag)
        gov_menu.addAction("Agent Evaluation", self.show_agent_evaluation)
        gov_menu.addAction("Data Contract & Schema Drift", self.show_data_contract)
        gov_menu.addAction("Leakage-Aware Split Wizard", self.show_split_wizard)
        gov_menu.addAction("Error Analysis Workspace", self.show_error_analysis)
        gov_menu.addAction("Statistical Analysis Wizard", self.show_statistical_wizard)
        gov_menu.addAction("Model Monitoring / Drift", self.show_model_monitoring)
        gov_menu.addAction("Publication Package", self.build_publication_package)
        gov_menu.addAction("Typed Agent Tool Registry", self.show_agent_tool_registry)
        gov_menu.addAction("Analysis Recipe", self.show_analysis_recipe)
        gov_menu.addAction("Analysis State Machine", self.show_analysis_state)
        gov_menu.addAction("Artifact Registry / Lineage", self.show_artifact_registry)

        access_menu = menubar.addMenu("Accessibility")
        access_menu.addAction("Voice Command Help", self.show_voice_help)
        access_menu.addAction("Start Voice Command", self.start_voice_command)
        access_menu.addAction("Stop Agent Run", self.stop_agent_run)

        help_menu = menubar.addMenu("Help")
        help_menu.addAction("Using Rows, Columns and Marks", lambda: self.show_help_topic("shelves"))
        help_menu.addAction("Filters and Drag-and-Drop", lambda: self.show_help_topic("filters"))
        help_menu.addAction("Sheets, Dashboards and Stories", lambda: self.show_help_topic("views"))
        help_menu.addAction("AI Agents and Human Approval", lambda: self.show_help_topic("agents"))
        help_menu.addAction("Chart Types and Hierarchies", lambda: self.show_help_topic("charts"))
        help_menu.addAction("About Data Science Studio Pro", lambda: self.show_help_topic("about"))

        share_menu = menubar.addMenu("Sharing")
        share_menu.addAction("Export PNG", self.export_png)
        share_menu.addAction("Email Current Sheet", self.email_current_sheet)
        share_menu.addAction("Email Story", self.email_story)
        share_menu.addAction("Export to Streamlit", self.export_to_streamlit)
        share_menu.addSeparator()
        share_menu.addAction("Create Collaboration Package", self.create_sharing_package)
        share_menu.addAction("Copy Share Package Path", self.copy_share_package_path)

    # ============================================================
    # TOOLBAR
    # ============================================================
    def setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        self.undo_action = QAction(QIcon(str(Path(__file__).parent / "assets" / "undo.svg")), "Undo", self); self.undo_action.triggered.connect(self.undo); toolbar.addAction(self.undo_action)
        self.redo_action = QAction(QIcon(str(Path(__file__).parent / "assets" / "redo.svg")), "Redo", self); self.redo_action.triggered.connect(self.redo); toolbar.addAction(self.redo_action)
        self.sort_asc_action = QAction(QIcon(str(Path(__file__).parent / "assets" / "sort_asc.svg")), "Sort Y-Axis Asc", self); self.sort_asc_action.triggered.connect(self.sort_asc); toolbar.addAction(self.sort_asc_action)
        self.sort_desc_action = QAction(QIcon(str(Path(__file__).parent / "assets" / "sort_desc.svg")), "Sort Y-Axis Desc", self); self.sort_desc_action.triggered.connect(self.sort_desc); toolbar.addAction(self.sort_desc_action)
        toolbar.addAction("Story", self.create_story)
        toolbar.addAction("Sheets", self.manage_sheets)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Compute: "))
        self.compute_combo = QComboBox(); self.compute_combo.addItems(["CPU","CPU+GPU","GPU"]); self.compute_combo.setCurrentText(self.compute_mode); self.compute_combo.setToolTip("Choose execution policy. GPU requires a CUDA-capable PyTorch installation."); self.compute_combo.currentTextChanged.connect(self.on_compute_mode_changed); toolbar.addWidget(self.compute_combo)
        toolbar.addAction("Dark Mode", self.toggle_dark_mode)

    def toggle_dark_mode(self):
        if self.theme_manager.dark_mode:
            self.theme_manager.apply_theme("#e3f2fd", dark=False)
        else:
            self.theme_manager.apply_theme("#2b2b2b", dark=True)

    # ============================================================
    # THEME CONTEXT MENU
    # ============================================================
    def show_context_menu(self, pos):
        menu = QMenu(self)
        change_color = QAction("Change Theme Color", self)
        change_color.triggered.connect(self.open_color_picker)
        menu.addAction(change_color)
        menu.exec(self.mapToGlobal(pos))

    def open_color_picker(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.theme_manager.apply_theme(color.name())

    def on_compute_mode_changed(self, mode):
        info=ComputeBackend.resolve(mode); self.compute_mode=mode if info.gpu_available or mode=="CPU" else "CPU"
        if self.compute_mode != mode: self.compute_combo.blockSignals(True); self.compute_combo.setCurrentText("CPU"); self.compute_combo.blockSignals(False)
        self.status.showMessage(info.message)
        if mode in {"GPU","CPU+GPU"} and not info.gpu_available:
            QMessageBox.warning(self,"GPU Not Found","A CUDA-capable GPU was not detected. The application will use CPU mode safely.\n\nInstall a compatible PyTorch CUDA build if GPU execution is required.")
    def run_ai_agent_question_analysis(self):
        if self.data_engine.df is None: QMessageBox.warning(self,"Question → Analysis","Load data first."); return
        q,ok=QInputDialog.getMultiLineText(self,"AI Agent Question → Analysis","Describe the data-science question:")
        if not ok or not q.strip(): return
        state={"question":q.strip(),"dataframe":self.data_engine.df.copy(),"memory":self.question_agent_memory}
        try:
            result=question_analysis_app.invoke(state); self.question_agent_memory=result.get("memory",[])
            dlg=QDialog(self); dlg.setWindowTitle("Question → Analysis — Human Review"); dlg.resize(900,650); root=QVBoxLayout(dlg); root.addWidget(QLabel(result.get("summary","Review the proposed analytical plan.")))
            form=QFormLayout(); target=QComboBox(); target.addItems([x["column"] for x in result.get("target_candidates",[]) if x.get("column") in self.data_engine.df.columns]); target.setCurrentText(result.get("suggested_target","") or ""); form.addRow("Target",target); root.addLayout(form)
            features=QListWidget(); features.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
            for x in result.get("feature_candidates",[]):
                item=QListWidgetItem(f"{x['column']}  | score={x['score']} | {', '.join(x.get('reasons',[]))}"); item.setData(Qt.ItemDataRole.UserRole,x['column']); item.setSelected(x['column'] in result.get("suggested_features",[])); features.addItem(item)
            root.addWidget(QLabel("Candidate features — review before execution:")); root.addWidget(features,1)
            plan=QTextEdit(); plan.setReadOnly(True); plan.setPlainText("\n".join(f"• {x}" for x in result.get("plan",[]))); plan.setMaximumHeight(120); root.addWidget(plan)
            bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel); bb.button(QDialogButtonBox.StandardButton.Ok).setText("Approve Plan & Run Master Agent"); bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); root.addWidget(bb)
            self.agent_console.log("Question Agent","understand","approval_required",result.get("summary",""))
            if dlg.exec()!=QDialog.DialogCode.Accepted:return
            chosen=target.currentText(); self._question_selected_features=[features.item(i).data(Qt.ItemDataRole.UserRole) for i in range(features.count()) if features.item(i).isSelected()]
            self.run_agent_data_scientist(target_override=chosen, feature_override=self._question_selected_features)
        except Exception as exc: QMessageBox.critical(self,"Question Agent",str(exc))
    def show_agent_run_console(self):
        self._show_text_dialog("Agent Run Console",self.agent_console.to_text() or "No agent events recorded yet.")
    def show_agent_tool_registry(self):
        self._show_json_dialog("Typed Agent Tool Registry",self.agent_tool_registry.describe())
    def show_analysis_recipe(self):
        self._show_json_dialog("Analysis Recipe",self.analysis_recipe.to_dict())
    def show_analysis_state(self):
        self._show_json_dialog("Analysis State Machine",self.analysis_state.to_dict())
    def show_artifact_registry(self):
        self._show_json_dialog("Artifact Registry / Lineage",self.artifact_store.to_dict())
    # ============================================================
    # FILE OPERATIONS
    # ============================================================
    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xlsx *.xls)")
        if filepath:
            try:
                previous_contract = self.data_contract.copy() if self.data_contract else None
                self.data_engine.load_data(filepath)
                self.analysis_state=AnalysisStateMachine(AnalysisState.DATA_LOADED.value)
                self.artifact_store.register("dataset",Path(filepath).name,{"fingerprint":dataset_fingerprint(self.data_engine.df),"source":filepath,"rows":len(self.data_engine.df),"columns":len(self.data_engine.df.columns)})
                self.analysis_recipe.add("load_dataset",{"source":filepath,"rows":len(self.data_engine.df),"columns":len(self.data_engine.df.columns)})
                backend_plan = self.large_data_engine.plan(self.data_engine.df)
                self.status.showMessage(f"Data backend plan: {backend_plan.backend} — {backend_plan.reason}")
                self.lineage.snapshot(self.data_engine.df, operation="load", metadata={"source": filepath})
                self.experiment_registry.start(dataset_hash=dataset_fingerprint(self.data_engine.df))
                self.dataset_card = DatasetCardBuilder.build(self.data_engine.df, source=filepath, purpose="interactive analysis")
                self.data_contract = DataContractEngine.build(self.data_engine.df, Path(filepath).stem)
                if previous_contract:
                    drift = DataContractEngine.validate(self.data_engine.df, previous_contract)
                    self._add_evidence({"evidence_id":f"schema-drift-{len(self.evidence_records)+1}","kind":"schema_drift","title":"Schema Drift Check","data":drift,"parent_ids":[]})
                    if drift.get("status") == "drift": self.status.showMessage("Schema drift detected; review Governance → Data Contract & Schema Drift.")
                self._add_evidence({"evidence_id":f"dataset-{dataset_fingerprint(self.data_engine.df)[:12]}","kind":"dataset_card","title":"Dataset Card","data":self.dataset_card,"parent_ids":[]})
                sheet_name = Path(filepath).stem
                self.sheet_manager.sheets[sheet_name] = {"x_col": None, "y_col": None, "mark_type": "Bar", "color_col": None, "size_col": None, "agg_func": "sum", "data": self.data_engine.df.copy()}
                if self.sheets_list.findItems(sheet_name, Qt.MatchFlag.MatchExactly) == []:
                    self.sheets_list.addItem(sheet_name)
                self.refresh_data_management()
                self.populate_marks_combos()
                self.status.showMessage(f"Loaded {filepath}")
                self.update_plot()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def open_pdf(self):
        filepath,_=QFileDialog.getOpenFileName(self,"Open PDF Tables","","PDF Files (*.pdf)")
        if not filepath:return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            tables=PDFTableExtractor.extract(filepath)
        except Exception as exc:
            QMessageBox.critical(self,"PDF Table Import",str(exc)); return
        finally:
            QApplication.restoreOverrideCursor()
        if not tables:
            QMessageBox.information(self,"PDF Table Import","No tables were detected."); return
        dlg=PDFTableSelectionDialog(tables,self)
        if dlg.exec()!=QDialog.DialogCode.Accepted:return
        selected=dlg.selected_indices()
        if not selected:
            QMessageBox.warning(self,"PDF Table Import","Select at least one table."); return
        imported_names=[]; self.sheets_list.blockSignals(True)
        try:
            for idx in selected:
                t=tables[idx]; name=t["name"]; df=t["data"].copy(); base=name
                n=2
                while name in self.sheet_manager.sheets: name=f"{base} ({n})"; n+=1
                self.sheet_manager.sheets[name]={"x_col":None,"y_col":None,"mark_type":"Bar","color_col":None,"size_col":None,"agg_func":"sum","data":df,"source_pdf":filepath,"pdf_page":t["page"],"pdf_table_index":t["table_index"]}
                self.sheets_list.addItem(name); imported_names.append(name)
        finally:self.sheets_list.blockSignals(False)
        # Explicitly activate the first imported table; a programmatic selection does not emit itemClicked.
        first_item=self.sheets_list.findItems(imported_names[0],Qt.MatchFlag.MatchExactly)[0]
        self.sheets_list.setCurrentItem(first_item); self.on_loaded_sheet_select(first_item)
        self.status.showMessage(f"Loaded {len(selected)} PDF table(s). Select a sheet in Loaded Sheets to work on it.")

    def save_project(self):
        """Save a portable .dssp project package without unsafe pickle deserialization."""
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data before saving a project.")
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Project", "DataScienceStudioProject.dssp", "DSP Project (*.dssp)")
        if not filepath:
            return
        try:
            view_state = {
                "rows": self.rows_input.text(),
                "columns": self.cols_input.text(),
                "chart": self.chart_combo.currentText(),
                "marks": {k: v.currentText() for k, v in self.marks_widgets.items()},
                "sort_order": getattr(self, "_sort_order", []),
                "sort_desc": getattr(self, "_sort_desc", False),
            }
            manifest = build_manifest(
                dataset_fingerprint=dataset_fingerprint(self.data_engine.original_df if self.data_engine.original_df is not None else self.data_engine.df),
                source=self.data_engine.source_path,
                rows=len(self.data_engine.df), columns=len(self.data_engine.df.columns),
                view_state=view_state, filters=list(self.data_engine.filter_specs.items()),
                evidence=self.evidence_records,
            )
            sheets = {}
            for name, cfg in self.sheet_manager.sheets.items():
                sheets[name] = {k: v for k, v in cfg.items() if k != "data"}
            manifest["sheets"] = sheets
            manifest["active_sheet"] = self.sheet_manager.active_sheet
            manifest["dataset_card"] = self.dataset_card
            manifest["data_contract"] = self.data_contract
            manifest["mark_aggregations"] = getattr(self, "mark_aggregations", {})
            # JSON table format preserves column names, dtypes and index semantics better than CSV.
            data_json = self.data_engine.original_df.to_json(orient="table", date_format="iso")
            active_json = self.data_engine.df.to_json(orient="table", date_format="iso")
            with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
                z.writestr("data_original.json", data_json)
                z.writestr("data_active.json", active_json)
            self.status.showMessage(f"Portable project saved to {filepath}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Project Error", str(exc))

    def load_project(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "DSP Project (*.dssp)")
        if not filepath:
            return
        try:
            with zipfile.ZipFile(filepath, "r") as z:
                names = set(z.namelist())
                required = {"manifest.json", "data_original.json", "data_active.json"}
                missing = required - names
                if missing:
                    raise ValueError(f"Invalid project package; missing: {', '.join(sorted(missing))}")
                manifest = load_manifest_from_text(z.read("manifest.json").decode("utf-8"))
                original = pd.read_json(io.StringIO(z.read("data_original.json").decode("utf-8")), orient="table")
                active = pd.read_json(io.StringIO(z.read("data_active.json").decode("utf-8")), orient="table")
            self.data_engine.original_df = original.copy()
            self.data_engine.df = active.copy()
            self.data_engine.source_path = manifest.get("dataset", {}).get("source")
            self.data_engine.filter_specs = dict(manifest.get("filters", []))
            self.data_engine._refresh_schema()
            self.sheet_manager.sheets = manifest.get("sheets") or {"Sheet 1": {"x_col": None, "y_col": None, "mark_type": "Bar", "color_col": None, "size_col": None, "agg_func": "sum"}}
            self.sheet_manager.active_sheet = manifest.get("active_sheet", next(iter(self.sheet_manager.sheets)))
            self.dataset_card = manifest.get("dataset_card", {})
            view = manifest.get("view_state", {})
            self.rows_input.setText(view.get("rows", "")); self.cols_input.setText(view.get("columns", ""))
            if view.get("chart"):
                self.chart_combo.setCurrentText(view["chart"])
            for k, v in view.get("marks", {}).items():
                if k in self.marks_widgets:
                    self.marks_widgets[k].setCurrentText(v)
            self.evidence_records = manifest.get("evidence", []) or []
            self.evidence_dag = EvidenceDAG()
            for e in self.evidence_records:
                if isinstance(e, dict) and e.get("evidence_id"):
                    self.evidence_dag.add(e)
            self.refresh_data_management(); self.populate_marks_combos(); self.update_plot()
            self.status.showMessage(f"Project loaded from {filepath}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Project Error", str(exc))

    def export_analysis_snapshot(self):
        """Export a human-readable JSON snapshot for reproducibility and issue reporting."""
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Analysis Snapshot", "analysis_snapshot.json", "JSON Files (*.json)")
        if not path:
            return
        view_state = {"rows": self.rows_input.text(), "columns": self.cols_input.text(), "chart": self.chart_combo.currentText(), "marks": {k: v.currentText() for k, v in self.marks_widgets.items()}}
        snapshot = build_manifest(dataset_fingerprint=dataset_fingerprint(self.data_engine.df), source=self.data_engine.source_path, rows=len(self.data_engine.df), columns=len(self.data_engine.df.columns), view_state=view_state, filters=list(self.data_engine.filter_specs.items()), evidence=self.evidence_records)
        snapshot["quality"] = self.data_quality_engine.assess(self.data_engine.df)
        snapshot["schema"] = self.data_quality_engine.data_dictionary(self.data_engine.df).to_dict(orient="records")
        save_manifest(path, snapshot)
        self.status.showMessage(f"Analysis snapshot exported to {path}")
        QMessageBox.information(self, "Analysis Snapshot", f"Reproducibility snapshot saved to:\n{path}")

    def show_data_quality(self):
        if self.data_engine.df is None:
            QMessageBox.warning(self, "Data Quality", "Load data first.")
            return
        target = None
        if self.agent_ds_state.get("target") in self.data_engine.df.columns:
            target = self.agent_ds_state.get("target")
        result = self.data_quality_engine.assess(self.data_engine.df, target=target)
        lines = [f"Overall diagnostic score: {result['score']}/100", f"Status: {result['status']}", f"Rows: {result['rows']:,} | Columns: {result['columns']}", f"Duplicate rows: {result['duplicate_rows']:,}", ""]
        for issue in result.get("issues", []):
            lines.append(f"[{issue['severity'].upper()}] {issue['check']} — {issue.get('column') or 'dataset'}: {issue['message']}")
            if issue.get("recommendation"):
                lines.append(f"    Recommendation: {issue['recommendation']}")
        self._show_text_dialog("Data Quality Center", "\n".join(lines) if len(lines) > 4 else "No quality issues were detected by the built-in checks.")
        self._add_evidence({"evidence_id": f"quality-{len(self.evidence_records)+1}", "kind": "data_quality", "title": "Data Quality Assessment", "data": result, "parent_ids": []})

    def show_data_dictionary(self):
        if self.data_engine.df is None:
            QMessageBox.warning(self, "Data Dictionary", "Load data first.")
            return
        table = self.data_quality_engine.data_dictionary(self.data_engine.df)
        d = QDialog(self); d.setWindowTitle("Data Dictionary"); d.resize(1050, 650); layout = QVBoxLayout(d)
        widget = QTableWidget(len(table), len(table.columns)); widget.setHorizontalHeaderLabels(list(table.columns)); widget.setAlternatingRowColors(True)
        for i, row in table.iterrows():
            for j, col in enumerate(table.columns): widget.setItem(i, j, QTableWidgetItem(str(row[col])))
        widget.resizeColumnsToContents(); layout.addWidget(widget)
        buttons = QHBoxLayout(); export = QPushButton("Export CSV"); close = QPushButton("Close"); buttons.addWidget(export); buttons.addStretch(); buttons.addWidget(close); layout.addLayout(buttons)
        export.clicked.connect(lambda: self._export_dataframe(table, "data_dictionary.csv", "Data Dictionary CSV (*.csv)")); close.clicked.connect(d.accept); d.exec()

    def _export_dataframe(self, table, default_name, file_filter):
        path, _ = QFileDialog.getSaveFileName(self, "Export", default_name, file_filter)
        if path:
            table.to_csv(path, index=False)
            self.status.showMessage(f"Exported to {path}")

    # ============================================================
    # TABLE CREATION
    # ============================================================
    def web_scraping(self):
        QMessageBox.information(self, "Web Scraping", "Web scraping dialog not implemented yet.")

    def set_api_key(self):
        key, ok = QInputDialog.getText(self, "API Key", "Enter your OpenAI API Key:", QLineEdit.EchoMode.Password)
        if ok and key:
            self.agent_ds_state["api_key"] = key
            self.agent_report_state["api_key"] = key
            self.status.showMessage("API Key updated.")

    # ============================================================
    # DATA INFO
    # ============================================================
    def show_missing_values(self):
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        missing = self.data_engine.get_missing_values()
        msg = "\n".join([f"{k}: {v}" for k, v in missing.items() if v > 0])
        QMessageBox.information(self, "Missing Values", msg or "No missing values.")

    def show_describe_data(self):
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        desc = self.data_engine.describe_data()
        dialog = QDialog(self)
        dialog.setWindowTitle("Describe Data")
        dialog.resize(800, 600)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont("Courier", 10))
        text.setText(desc)
        layout.addWidget(text)
        dialog.exec()

    def show_data_info(self):
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        info = self.data_engine.get_data_info()
        QMessageBox.information(self, "Data Info", info)

    # ============================================================
    # DATA PRE-PROCESSING
    # ============================================================
    def data_cleaning(self):
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        msg = []
        msg.append(self.data_engine.remove_duplicates())
        msg.append(self.data_engine.fill_missing_mean())
        msg.append(self.data_engine.cap_outliers())
        QMessageBox.information(self, "Data Cleaning", "\n".join(msg))
        self.refresh_data_management()
        self.update_plot()

    def data_integration(self):
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first."); return
        path, _ = QFileDialog.getOpenFileName(self, "Select file to merge", "", "CSV/Excel (*.csv *.xlsx *.xls)")
        if not path: return
        try:
            other = DataEngine().read_file(path); common = [c for c in self.data_engine.df.columns if c in other.columns]
            if not common:
                QMessageBox.warning(self, "Merge", "No common columns were found."); return
            key, ok = QInputDialog.getItem(self, "Merge", "Select join key:", common, 0, False)
            if not ok: return
            how, ok = QInputDialog.getItem(self, "Merge", "Join type:", ["inner", "left", "right", "outer"], 0, False)
            if not ok: return
            result = self.data_engine.df.merge(other, on=key, how=how, suffixes=("", "_right"))
            self.data_engine.commit_dataframe(result, keep_as_source=True); self.refresh_data_management(); self.populate_marks_combos(); self.update_plot()
            QMessageBox.information(self, "Data Integration", f"Merged {len(result):,} rows using {how} join on '{key}'.")
        except Exception as exc: QMessageBox.critical(self, "Data Integration", str(exc))

    def data_transformation(self):
        if self.data_engine.df is None: QMessageBox.warning(self, "No Data", "Load data first."); return
        cols = self.data_engine.measures
        if not cols: QMessageBox.information(self, "Data Transformation", "No numeric columns."); return
        col, ok = QInputDialog.getItem(self, "Data Transformation", "Column:", cols, 0, False)
        if not ok: return
        op, ok = QInputDialog.getItem(self, "Data Transformation", "Transformation:", ["log1p", "sqrt", "square", "Box-Cox"], 0, False)
        if not ok: return
        work = self.data_engine.df.copy(); values = pd.to_numeric(work[col], errors="coerce")
        if op == "log1p": work[col] = np.log1p(values.clip(lower=0))
        elif op == "sqrt": work[col] = np.sqrt(values.clip(lower=0))
        elif op == "square": work[col] = values ** 2
        else:
            from scipy.stats import boxcox
            valid = values.dropna(); shift = 1 - valid.min() if valid.min() <= 0 else 0
            transformed, _ = boxcox(valid + shift); work.loc[valid.index, col] = transformed
        self.data_engine.commit_dataframe(work, keep_as_source=True); self.refresh_data_management(); self.update_plot()

    def data_reduction(self):
        if self.data_engine.df is None: QMessageBox.warning(self, "No Data", "Load data first."); return
        action, ok = QInputDialog.getItem(self, "Data Reduction", "Action:", ["Drop Column", "Random Sample"], 0, False)
        if not ok: return
        work = self.data_engine.df.copy()
        if action == "Drop Column":
            col, ok = QInputDialog.getItem(self, "Drop Column", "Column:", list(work.columns), 0, False)
            if not ok: return
            work = work.drop(columns=[col])
        else:
            n, ok = QInputDialog.getInt(self, "Random Sample", "Rows:", min(100, len(work)), 1, len(work), 1)
            if not ok: return
            work = work.sample(n=n, random_state=42).reset_index(drop=True)
        self.data_engine.commit_dataframe(work, keep_as_source=True); self.refresh_data_management(); self.populate_marks_combos(); self.update_plot()

    def feature_engineering(self):
        self.create_calculated_field()

    def data_encoding(self):
        if self.data_engine.df is None: QMessageBox.warning(self, "No Data", "Load data first."); return
        cats = self.data_engine.dimensions
        if not cats: QMessageBox.information(self, "Data Encoding", "No categorical columns."); return
        col, ok = QInputDialog.getItem(self, "Data Encoding", "Column:", cats, 0, False)
        if not ok: return
        mode, ok = QInputDialog.getItem(self, "Data Encoding", "Encoding:", ["Label", "One-Hot"], 0, False)
        if not ok: return
        work = self.data_engine.df.copy()
        if mode == "Label": work[col] = pd.factorize(work[col].astype(str))[0]
        else:
            dummies = pd.get_dummies(work[col], prefix=col, dtype=int); work = pd.concat([work.drop(columns=[col]), dummies], axis=1)
        self.data_engine.commit_dataframe(work, keep_as_source=True); self.refresh_data_management(); self.populate_marks_combos(); self.update_plot()

    def label_encoding(self):
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        cat_cols = self.data_engine.dimensions
        if not cat_cols:
            QMessageBox.information(self, "Label Encoding", "No categorical columns.")
            return
        col, ok = QInputDialog.getItem(self, "Label Encoding", "Select column:", cat_cols, 0, False)
        if ok and col:
            msg = self.data_engine.label_encode(col)
            QMessageBox.information(self, "Label Encoding", msg)
            self.refresh_data_management()
            self.update_plot()

    def data_scaling(self):
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        num_cols = self.data_engine.measures
        if not num_cols:
            QMessageBox.information(self, "Data Scaling", "No numerical columns.")
            return
        col, ok = QInputDialog.getItem(self, "Data Scaling", "Select column:", num_cols, 0, False)
        if ok and col:
            msg = self.data_engine.scale_data(col)
            QMessageBox.information(self, "Data Scaling", msg)
            self.refresh_data_management()
            self.update_plot()

    def data_splitting(self):
        if self.data_engine.df is None: QMessageBox.warning(self, "No Data", "Load data first."); return
        from sklearn.model_selection import train_test_split
        ratio, ok = QInputDialog.getDouble(self, "Data Splitting", "Test fraction:", .2, .05, .9, 2)
        if not ok: return
        train, test = train_test_split(self.data_engine.df, test_size=ratio, random_state=42)
        self.sheet_manager.sheets["Train"] = {"data": train.copy(), "x_col": None, "y_col": None, "mark_type": "Bar", "color_col": None, "size_col": None, "agg_func": "sum"}
        self.sheet_manager.sheets["Test"] = {"data": test.copy(), "x_col": None, "y_col": None, "mark_type": "Bar", "color_col": None, "size_col": None, "agg_func": "sum"}
        self.sheets_list.clear()
        for name in self.sheet_manager.sheets: self.sheets_list.addItem(name)
        QMessageBox.information(self, "Data Splitting", f"Created Train ({len(train):,}) and Test ({len(test):,}) sheets.")

    # ============================================================
    # DATA ANALYSIS
    # ============================================================
    def show_analysis(self, analysis_type):
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        dialog = DataAnalysisDialog(self.data_engine.df, analysis_type, self)
        dialog.exec()

    # ============================================================
    # AI AGENTS
    # ============================================================
    def run_agent_data_scientist(self, target_override=None, feature_override=None):
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first."); return
        target_analysis = recommend_targets_and_features(self.data_engine.df)
        target = target_override or target_analysis.get("suggested_target") or self.data_engine.df.columns[-1]
        if feature_override: target_analysis["approved_features"] = list(feature_override)
        gate = self.leakage_gate.evaluate(self.data_engine.df, target=target)
        self.agent_ds_state = {
            "messages": [HumanMessage(content="Master Agent: analyze this dataset and choose ML or DL.")],
            "approved_steps": [], "rejected_steps": [], "user_approved": False,
            "dataframe": self.data_engine.df.copy(), "api_key": config.DEFAULT_API_KEY,
            "model_name": self.agent_ds_state.get("model_name", config.ALL_LLM_MODELS[0]),
            "target": target, "target_analysis": target_analysis, "feature_candidates": target_analysis.get("suggested_features", []),
            "compute_mode": self.compute_mode, "leakage_gate": gate, "max_steps": 20, "abort_requested": False, "dataset_card": DatasetCardBuilder.build(self.data_engine.df),
            "memory": [],
            "human_approval_evidence": [], "abort_requested": False,
        }
        gate_evidence = EvidenceObjectProxy.from_dict(gate, "Scientific/Data Leakage Gate")
        self._add_evidence(gate_evidence)
        if gate.get("status") == "blocked":
            box = QMessageBox(self); box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Scientific/Data Leakage Gate")
            box.setText("Potential data leakage was detected. The Master Agent is blocked until you explicitly acknowledge the gate.")
            box.setDetailedText(json.dumps(gate, indent=2, default=str))
            override = box.addButton("Review and Override", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole); box.exec()
            if box.clickedButton() is not override:
                self.status.showMessage("Agent run cancelled by Scientific/Data Leakage Gate."); return
            approval = HumanApprovalEvidence.create("Scientific/Data Leakage Gate", "approved", "User explicitly acknowledged the gate and allowed the analysis to continue.")
            self._add_evidence(approval); self.agent_ds_state["human_approval_evidence"].append(approval)
        self.agent_console.clear(); self.agent_console.log("Master Agent","route","started",f"Target candidate: {target}; compute mode: {self.compute_mode}")
        self.analysis_recipe.add("agent_start",{"target":target,"compute_mode":self.compute_mode})
        self.agent_worker = AgentWorker(agent_ds_app, self.agent_ds_state)
        self.agent_worker.step_ready.connect(self.on_agent_step_ready)
        self.agent_worker.finished.connect(self.on_agent_finished)
        self.agent_worker.error.connect(self.on_agent_error); self.agent_worker.start()
        self.status.showMessage("Master Agent is thinking...")

    def on_agent_step_ready(self, step_text):
        summary = step_text or self.agent_ds_state.get("stage_summary", "Awaiting your approval.")
        self.agent_console.log("Master Agent", self.agent_ds_state.get("current_step","unknown"), "approval_required", summary)
        # Optional accessible voice channel: set DSP_VOICE_APPROVAL=1. Keyboard/mouse remains the default.
        voice_intent = None
        if os.environ.get("DSP_VOICE_APPROVAL", "0") == "1":
            heard = self.voice_layer.listen_once(summary)
            voice_intent = self.voice_layer.resolve(heard, ["approve", "reject", "pause", "abort"])
        if voice_intent in {"pause", "abort"}:
            self.agent_ds_state["user_approved"] = False
            self.status.showMessage("Voice command requested pause/abort; the current step was not approved.")
            return
        if voice_intent in {"approve", "reject"}:
            self.agent_ds_state["user_approved"] = voice_intent == "approve"
        else:
            msg_box = QMessageBox(self); msg_box.setWindowTitle("Human Approval — Agent Data Scientist")
            msg_box.setText(f"{summary}\n\nApprove this step to continue, Reject to revise/repeat it, or Abort to stop the run safely.")
            approve_btn = msg_box.addButton("Approve", QMessageBox.ButtonRole.AcceptRole)
            reject_btn = msg_box.addButton("Reject", QMessageBox.ButtonRole.RejectRole)
            abort_btn = msg_box.addButton("Abort Run", QMessageBox.ButtonRole.DestructiveRole)
            msg_box.exec()
            clicked = msg_box.clickedButton()
            if clicked is abort_btn:
                self.agent_abort_requested = True; self.agent_ds_state["abort_requested"] = True
                self.agent_ds_state["user_approved"] = False
                approval = HumanApprovalEvidence.create(self.agent_ds_state.get("current_step", "unknown"), "aborted", "User safely terminated the agent run.")
                self._add_evidence(approval); self.agent_ds_state.setdefault("human_approval_evidence", []).append(approval)
                self.status.showMessage("Agent run aborted safely; no further agent step will be started."); return
            self.agent_ds_state["user_approved"] = clicked is approve_btn
            self.agent_console.log("Human", self.agent_ds_state.get("current_step","unknown"), "approved" if clicked is approve_btn else "rejected", "Human decision recorded.")
            decision = "approved" if clicked is approve_btn else "rejected"
            approval = HumanApprovalEvidence.create(self.agent_ds_state.get("current_step", "unknown"), decision, summary)
            self._add_evidence(approval); self.agent_ds_state.setdefault("human_approval_evidence", []).append(approval)
        if not self.agent_ds_state["user_approved"]:
            self.agent_ds_state.setdefault("rejected_steps", []).append(self.agent_ds_state.get("current_step", "unknown"))
            self.agent_ds_state["messages"].append(HumanMessage(content="User rejected the step. Revise or repeat it."))
        if self.agent_abort_requested or self.agent_ds_state.get("abort_requested"):
            self.status.showMessage("Agent run stopped safely."); return
        self.agent_worker = AgentWorker(agent_ds_app, self.agent_ds_state)
        self.agent_worker.step_ready.connect(self.on_agent_step_ready); self.agent_worker.finished.connect(self.on_agent_finished); self.agent_worker.error.connect(self.on_agent_error); self.agent_worker.start()

    def on_agent_finished(self, message, final_state):
        self.agent_ds_state = final_state; self.status.showMessage(message)
        self.agent_console.log("Master Agent", final_state.get("current_step","complete"), "complete", message)
        try:
            if self.analysis_state.state == AnalysisState.DATA_LOADED.value: self.analysis_state.transition(AnalysisState.QUALITY_CHECKED.value); self.analysis_state.transition(AnalysisState.CONTRACT_VALIDATED.value); self.analysis_state.transition(AnalysisState.ANALYSIS_READY.value)
            if final_state.get("ml_results") or final_state.get("dl_results"): self.analysis_state.transition(AnalysisState.MODEL_READY.value)
        except Exception: pass
        if final_state.get("dataframe") is not None:
            self.data_engine.set_active_dataframe(final_state["dataframe"]); self.refresh_data_management(); self.populate_marks_combos(); self.update_plot()
        if final_state.get("ml_results") or final_state.get("dl_results"):
            for bundle in (final_state.get("ml_results"), final_state.get("dl_results")):
                if bundle and bundle.get("status") == "ok":
                    self.experiment_registry.log("agent_result", agent=bundle.get("agent"), metrics=bundle.get("best_metrics"))
                    self.agent_console.log(bundle.get("agent","Agent"),"model_analysis","complete",bundle.get("summary","Model result recorded."))
                    self.analysis_recipe.add("model_result",{"agent":bundle.get("agent"),"model":bundle.get("best_model"),"target":bundle.get("target")})
                    self.artifact_store.register("model",bundle.get("best_model",bundle.get("agent","model")),{"agent":bundle.get("agent"),"task":bundle.get("task"),"metrics":bundle.get("best_metrics",{})})
                    self.experiment_records.append({"experiment": f"Agent run {len(self.experiment_records)+1}", "agent": bundle.get("agent"), "model": bundle.get("best_model"), "task": bundle.get("task"), "metrics": bundle.get("best_metrics", {}), "dataset_fingerprint": self.agent_ds_state.get("dataset_card",{}).get("fingerprint"), "route": self.agent_ds_state.get("route"), "step_index": self.agent_ds_state.get("step_index")})
                    card = ModelCardBuilder.build(bundle, self.agent_ds_state.get("dataset_card"), self.agent_ds_state.get("leakage_gate"))
                    self.model_cards.append(card); self._add_evidence({"evidence_id": f"modelcard-{bundle.get('agent','model')}-{len(self.model_cards)}", "kind":"model_card", "title":f"{bundle.get('agent','Model')} Model Card", "data":card, "parent_ids":[]})
                    model = bundle.get("model_object")
                    if model is not None:
                        try:
                            self.model_registry.register(bundle.get("best_model", bundle.get("agent", "model")), model, bundle.get("best_metrics", {}), {"agent": bundle.get("agent"), "task": bundle.get("task")})
                        except Exception:
                            pass
            self.status.showMessage("Agent Data Scientist finished. ML/DL evidence is ready for Agent Plot and Agent Report.")
        QMessageBox.information(self, "Agent Data Scientist", message)

    def on_agent_error(self, error_msg):
        QMessageBox.critical(self, "Agent Error", error_msg); self.status.showMessage("Agent error.")

    def run_ai_agent_plot(self):
        from agent.graph_plot import run_plot_agent
        ml = self.agent_ds_state.get("ml_results"); dl = self.agent_ds_state.get("dl_results")
        if not ml and not dl:
            QMessageBox.warning(self, "No AI Results", "Run Agent Data Scientist first so ML/DL results are available."); return
        result = run_plot_agent(self.data_engine.df, ml, dl, self.agent_ds_state.get("target"), self.chart_combo.currentText())
        self.agent_ds_state["plot_results"] = result
        if result.get("status") != "ok":
            QMessageBox.warning(self, "Agent Plot", result.get("message", "Plot failed.")); return
        fig = result.get("figure")
        if fig is not None:
            try:
                html = fig.to_html(include_plotlyjs="cdn", full_html=True)
                path, _ = QFileDialog.getSaveFileName(self, "Save AI Agent Plot", "AI_Agent_Plot.html", "HTML Files (*.html)")
                if path:
                    Path(path).write_text(html, encoding="utf-8")
                    QMessageBox.information(self, "Agent Plot", f"Interactive plot saved to {path}")
            except Exception as exc:
                QMessageBox.information(self, "Agent Plot", str(result.get("summary", "Plot prepared.")) + f"\n\n{exc}")
        self.status.showMessage(result.get("summary", "Agent Plot completed."))

    def run_ai_report(self):
        if not (self.agent_ds_state.get("ml_results") or self.agent_ds_state.get("dl_results")):
            QMessageBox.warning(self, "No Analysis", "Run Agent Data Scientist first."); return
        def report_safe(bundle):
            if not isinstance(bundle, dict): return bundle
            return {k: v for k, v in bundle.items() if k not in {"model_object", "figure"}}
        self.agent_report_state["analysis_evidence"] = {
            "master_route": self.agent_ds_state.get("route"),
            "target": self.agent_ds_state.get("target"),
            "ML Agent": report_safe(self.agent_ds_state.get("ml_results")),
            "DL Agent": report_safe(self.agent_ds_state.get("dl_results")),
            "Plot Agent": report_safe(self.agent_ds_state.get("plot_results")),
            "approved_steps": self.agent_ds_state.get("approved_steps", []),
            "rejected_steps": self.agent_ds_state.get("rejected_steps", []),
            "dataset_card": self.agent_ds_state.get("dataset_card"),
            "leakage_gate": self.agent_ds_state.get("leakage_gate"),
            "human_approval_evidence": self.agent_ds_state.get("human_approval_evidence", []),
            "agent_evaluation": AgentEvaluation.evaluate(self.agent_ds_state),
            "evidence_dag": self.evidence_dag.to_dict(),
        }
        self.report_worker = ReportWorker(agent_report_app, self.agent_report_state)
        self.report_worker.finished.connect(self.on_report_finished); self.report_worker.error.connect(self.on_report_error); self.report_worker.start()
        self.status.showMessage("AI Agent Report is generating...")

    def on_report_finished(self, result):
        if isinstance(result, dict):
            pdf_buffer=result.get("pdf_buffer"); self.agent_report_state["report_text"]=result.get("report_text","")
        else:
            pdf_buffer=result
        if pdf_buffer:
            filepath, _ = QFileDialog.getSaveFileName(self, "Save AI Agent Report", "AI_Agent_Report.pdf", "PDF Files (*.pdf)")
            if filepath:
                with open(filepath, "wb") as f: f.write(pdf_buffer.getvalue())
                self.agent_report_state["last_report_path"] = filepath
                QMessageBox.information(self, "Report Saved", f"Report saved to {filepath}"); self.status.showMessage(f"Report saved to {filepath}")
        else: QMessageBox.warning(self, "Report Error", "Could not generate report.")

    def on_report_error(self, error_msg):
        QMessageBox.critical(self, "Report Error", error_msg)

    # ============================================================
    # SELECT LOCAL LLM MODEL
    # ============================================================
    def select_local_llm(self):
        dialog = LocalLLMConfigDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            model = dialog.get_selected_model()
            if model:
                self.agent_ds_state["model_name"] = model
                self.agent_report_state["model_name"] = model
                self.status.showMessage(f"Selected LLM: {model}")
                QMessageBox.information(self, "LLM Selected", f"Now using: {model}")

    # ============================================================
    # MACRO EDITOR
    # ============================================================
    def open_macro_editor(self, macro_type):
        dialog=MacroEditorDialog(macro_type,self)
        dialog.run_requested.connect(lambda code,d=dialog,t=macro_type:self._run_macro_in_dialog(t,code,d))
        dialog.exec()

    def _run_macro_in_dialog(self,macro_type,code,dialog):
        try:
            result=self.execute_python_macro(code) if macro_type=="Python" else self.execute_sql_macro(code)
            dialog.append_output(result)
        except Exception as exc: dialog.append_output(f"ERROR: {exc}")

    def execute_python_macro(self,code):
        if self.data_engine.df is None: raise ValueError("Load data first.")
        import ast,contextlib
        buffer=io.StringIO(); env={'df':self.data_engine.df.copy(),'pd':pd,'np':np,'plt':plt}
        tree=ast.parse(code,mode="exec")
        with contextlib.redirect_stdout(buffer):
            value=None
            if tree.body and isinstance(tree.body[-1],ast.Expr):
                last=tree.body.pop(); exec(compile(tree,"<Python Macro>","exec"),env); value=eval(compile(ast.Expression(last.value),"<Python Macro>","eval"),env)
            else:
                exec(compile(tree,"<Python Macro>","exec"),env); value=env.get("result")
        output=buffer.getvalue()
        if isinstance(value,pd.DataFrame):
            self.data_engine.set_active_dataframe(value); self.refresh_data_management(); self.populate_marks_combos(); self.update_plot(); output += "\n\nDataFrame result loaded into active analysis:\n"+value.head(30).to_string(index=False)
        elif value is not None: output += "\n"+str(value)
        return output.strip() or "Macro executed successfully (no printed/result output)."

    def execute_sql_macro(self,code):
        if self.data_engine.df is None: raise ValueError("Load data first.")
        result=self.duckdb_workspace.query(self.data_engine.df,code)
        if isinstance(result,pd.DataFrame): return f"SQL result — {len(result):,} rows × {len(result.columns):,} columns\n\n"+result.head(200).to_string(index=False)
        return f"SQL command completed successfully. Result: {result}"

    def run_python_macro(self,code):
        try: QMessageBox.information(self,"Python Macro Output",self.execute_python_macro(code))
        except Exception as exc: QMessageBox.critical(self,"Macro Error",str(exc))

    def run_sql_macro(self,code):
        try: self._show_text_dialog("SQL Macro Result",self.execute_sql_macro(code))
        except Exception as exc: QMessageBox.critical(self,"SQL Macro Error",str(exc))

    # ============================================================
    # PROFESSIONAL ANALYTICS / AGENTS
    # ============================================================
    def show_core_schema_types(self):
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load data first."); return
        self._show_text_dialog("Core Schema Types", schema_table(self.data_engine.df).to_string(index=False))

    def show_data_contract(self):
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load data first."); return
        contract=self.data_contract or DataContractEngine.build(self.data_engine.df, "Data Science Studio Pro Contract")
        dlg=QDialog(self); dlg.setWindowTitle("Data Contract & Schema Drift"); dlg.resize(900,650); l=QVBoxLayout(dlg)
        txt=QTextEdit(); txt.setReadOnly(True); txt.setPlainText(json.dumps(contract,indent=2,default=str)); l.addWidget(txt)
        row=QHBoxLayout(); save=QPushButton("Save Contract"); check=QPushButton("Check Current Dataset"); row.addWidget(save); row.addWidget(check); l.addLayout(row)
        def save_contract():
            path,_=QFileDialog.getSaveFileName(self,"Save Data Contract","data_contract.json","JSON (*.json)")
            if path: DataContractEngine.save(path,contract); self.status.showMessage(f"Contract saved to {path}")
        def check_contract():
            result=DataContractEngine.validate(self.data_engine.df,contract); txt.setPlainText(json.dumps(result,indent=2,default=str))
        save.clicked.connect(save_contract); check.clicked.connect(check_contract); dlg.exec()

    def show_split_wizard(self):
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load data first."); return
        cols=list(self.data_engine.df.columns); d=QDialog(self); d.setWindowTitle("Leakage-Aware Train / Validation / Test Wizard"); d.resize(760,560); l=QVBoxLayout(d)
        form=QFormLayout(); mode=QComboBox(); mode.addItems(SplitWizard.MODES); target=QComboBox(); target.addItems(cols); group=QComboBox(); group.addItem("None"); group.addItems(cols); timec=QComboBox(); timec.addItem("None"); timec.addItems(cols); test=QDoubleSpinBox(); test.setRange(.05,.8); test.setValue(.2); val=QDoubleSpinBox(); val.setRange(.05,.5); val.setValue(.1); form.addRow("Split strategy",mode); form.addRow("Target",target); form.addRow("Group column",group); form.addRow("Time column",timec); form.addRow("Test fraction",test); form.addRow("Validation fraction",val); l.addLayout(form)
        out=QTextEdit(); out.setReadOnly(True); l.addWidget(out); run=QPushButton("Validate & Create 3 Splits"); l.addWidget(run)
        def execute():
            g=None if group.currentText()=="None" else group.currentText(); tc=None if timec.currentText()=="None" else timec.currentText(); plan=SplitWizard.plan(self.data_engine.df,target.currentText(),mode.currentText(),g,tc,test.value(),val.value());
            try:
                tr,va,te=SplitWizard.execute(self.data_engine.df,target.currentText(),mode.currentText(),g,tc,test.value(),val.value());
                self.sheet_manager.sheets["Train"]={"data":tr,"x_col":None,"y_col":None,"mark_type":"Bar","color_col":None,"size_col":None,"agg_func":"sum"}; self.sheet_manager.sheets["Validation"]={"data":va,"x_col":None,"y_col":None,"mark_type":"Bar","color_col":None,"size_col":None,"agg_func":"sum"}; self.sheet_manager.sheets["Test"]={"data":te,"x_col":None,"y_col":None,"mark_type":"Bar","color_col":None,"size_col":None,"agg_func":"sum"};
                out.setPlainText(json.dumps({"plan":plan,"train":len(tr),"validation":len(va),"test":len(te)},indent=2)); self.status.showMessage("Leakage-aware Train/Validation/Test splits created.")
            except Exception as exc: out.setPlainText(json.dumps({"plan":plan,"error":str(exc)},indent=2))
        run.clicked.connect(execute); dlg.exec()

    def show_duckdb_workspace(self):
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load data first."); return
        engine=DuckDBWorkspace(); d=QDialog(self); d.setWindowTitle("DuckDB Analytical Workspace"); d.resize(1000,700); l=QVBoxLayout(d)
        l.addWidget(QLabel("Query the active dataset as the DuckDB table `data`. Results can be inspected or exported without changing the source dataset."))
        editor=QTextEdit(); editor.setPlainText("SELECT * FROM data LIMIT 100;"); l.addWidget(editor)
        run=QPushButton("Run SQL"); export=QPushButton("Export Result CSV"); l.addWidget(run); l.addWidget(export); out=QTableWidget(); l.addWidget(out)
        result_holder={"df":None}
        def execute():
            try:
                result=engine.query(self.data_engine.df,editor.toPlainText()); result_holder["df"]=result; out.setRowCount(len(result)); out.setColumnCount(len(result.columns)); out.setHorizontalHeaderLabels([str(c) for c in result.columns])
                for i,row in result.iterrows():
                    for j,c in enumerate(result.columns): out.setItem(i,j,QTableWidgetItem(str(row[c])))
                self.status.showMessage(f"DuckDB returned {len(result):,} rows.")
            except Exception as exc: QMessageBox.critical(d,"DuckDB Error",str(exc))
        def export_result():
            result=result_holder.get("df")
            if result is None:return
            path,_=QFileDialog.getSaveFileName(d,"Export Query Result","duckdb_result.csv","CSV (*.csv)")
            if path: result.to_csv(path,index=False)
        run.clicked.connect(execute); export.clicked.connect(export_result); d.exec()

    def show_statistical_wizard(self):
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load data first."); return
        df=self.data_engine.df; cols=list(df.columns); d=QDialog(self); d.setWindowTitle("Statistical Analysis Wizard"); d.resize(900,650); l=QVBoxLayout(d); form=QFormLayout(); typ=QComboBox(); typ.addItems(["Descriptive Statistics","Correlation","Hypothesis Test","Regression"]); c1=QComboBox(); c1.addItems(cols); c2=QComboBox(); c2.addItems(cols); form.addRow("Analysis",typ); form.addRow("Variable / Group",c1); form.addRow("Value / Outcome",c2); l.addLayout(form); out=QTextEdit(); out.setReadOnly(True); l.addWidget(out); run=QPushButton("Run Analysis"); l.addWidget(run)
        def execute():
            try:
                t=typ.currentText();
                if t=="Descriptive Statistics": result=StatisticalAnalysisEngine.describe(df)
                elif t=="Correlation": result=StatisticalAnalysisEngine.correlation(df).to_dict()
                elif t=="Hypothesis Test": result=StatisticalAnalysisEngine.hypothesis(df,c1.currentText(),c2.currentText())
                else: result=StatisticalAnalysisEngine.regression(df,c1.currentText(),c2.currentText())
                out.setPlainText(json.dumps(result,indent=2,default=str))
            except Exception as exc: out.setPlainText(f"Analysis error: {exc}")
        run.clicked.connect(execute); dlg.exec()

    def show_error_analysis(self):
        results=self.agent_ds_state.get("ml_results") or self.agent_ds_state.get("dl_results")
        if not results: QMessageBox.warning(self,"No Model Results","Run Agent Data Scientist first."); return
        models=results.get("models") or {}; diagnostics={name:bundle.get("error_analysis") for name,bundle in models.items()} if isinstance(models,dict) else {}
        self._show_json_dialog("Error Analysis Workspace", {"task":results.get("task"),"target":results.get("target"),"models":diagnostics,"guidance":"Use slice analysis on validation/test predictions before deployment; inspect class-specific errors or residual structure rather than relying only on aggregate metrics."})

    def show_model_monitoring(self):
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load current data first."); return
        path,_=QFileDialog.getOpenFileName(self,"Select Reference Dataset","","Data Files (*.csv *.xlsx *.parquet)")
        if not path:return
        try:
            reference=self.data_engine.read_file(path); result=ModelMonitoring.summary(reference,self.data_engine.df); self._show_json_dialog("Model Monitoring / Drift",result)
        except Exception as exc: QMessageBox.critical(self,"Monitoring Error",str(exc))

    def export_notebook(self):
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load data first."); return
        path,_=QFileDialog.getSaveFileName(self,"Export Reproducible Notebook","analysis.ipynb","Jupyter Notebook (*.ipynb)")
        if not path:return
        nb=NotebookExporter.build({"rows":self.rows_input.text(),"columns":self.cols_input.text()},self.data_engine.source_path or "data.csv",self.rows_input.text(),self.cols_input.text(),self.chart_combo.currentText(),{k:v.currentText() for k,v in self.marks_widgets.items()},self.data_engine.filter_specs,self.agent_report_state.get("analysis_evidence")); NotebookExporter.save(path,nb); self.status.showMessage(f"Notebook exported to {path}")

    def build_publication_package(self):
        path,_=QFileDialog.getSaveFileName(self,"Create Publication Package","DataScienceStudioPro_Publication.zip","ZIP (*.zip)")
        if not path:return
        artifacts={"README.md":self._publication_readme()}
        if self.agent_report_state.get("last_report_path") and Path(self.agent_report_state["last_report_path"]).exists(): artifacts["AI_Agent_Report.pdf"]=self.agent_report_state["last_report_path"]
        if getattr(self,"last_presentation_path",None) and Path(self.last_presentation_path).exists(): artifacts["AI_Agent_Presentation.py"]=self.last_presentation_path
        if self.data_engine.df is not None: artifacts["data_dictionary.csv"]=self._write_temp_dictionary()
        PublicationPackageBuilder.build(path,artifacts); self.status.showMessage(f"Publication package created: {path}")

    def _write_temp_dictionary(self):
        import tempfile
        p=Path(tempfile.gettempdir())/"dssp_data_dictionary.csv"; schema_table(self.data_engine.df).to_csv(p,index=False); return str(p)

    def _publication_readme(self):
        return "Data Science Studio Pro publication package\n\nIncludes the recorded analysis outputs available at export time. Verify all scientific conclusions against the underlying dataset and methods.\n"

    def run_ai_agent_table_creation(self):
        d=QDialog(self); d.setWindowTitle("AI Agent Table Creation"); d.resize(760,520); l=QVBoxLayout(d); form=QFormLayout(); mode=QComboBox(); mode.addItems(["Web Scraping","API Key"]); url=QLineEdit(); api=QLineEdit(); api.setEchoMode(QLineEdit.EchoMode.Password); header=QLineEdit("Authorization"); prompt=QLineEdit(); form.addRow("Source method",mode); form.addRow("URL",url); form.addRow("API key",api); form.addRow("API header",header); form.addRow("Agent instruction",prompt); l.addLayout(form); out=QTextEdit(); out.setReadOnly(True); l.addWidget(out); run=QPushButton("Run AI Agent Table Creation"); l.addWidget(run)
        def execute():
            from agent.graph_table_creation import agent_table_creation_app
            state={"mode":mode.currentText(),"url":url.text(),"api_key":api.text(),"api_header":header.text(),"prompt":prompt.text(),"memory":getattr(self,"table_agent_memory",[])}; self.table_agent_worker=SimpleGraphWorker(agent_table_creation_app,state,{"configurable":{"thread_id":"table-creation"}}); self.table_agent_worker.finished.connect(lambda r:self.on_table_agent_finished(r,d)); self.table_agent_worker.error.connect(lambda e:out.setPlainText(e)); self.table_agent_worker.start(); out.setPlainText("LangGraph Table Creation Agent is running…")
        run.clicked.connect(execute); d.exec()

    def on_table_agent_finished(self,result,dialog):
        self.table_agent_memory=result.get("memory",[]); df=result.get("dataframe");
        if isinstance(df,pd.DataFrame): self.data_engine.set_active_dataframe(df); self.refresh_data_management(); self.populate_marks_combos(); self.update_plot()
        if result.get("status")=="complete": self.status.showMessage(result.get("summary","Table created.")); dialog.accept()
        else: QMessageBox.critical(self,"Table Creation Agent",result.get("error","Agent failed."))

    def run_ai_agent_presentation(self):
        evidence=self.agent_report_state.get("analysis_evidence")
        if not evidence: QMessageBox.warning(self,"No Report","Run AI Agent Report first. The presentation agent consumes the report evidence."); return
        path,_=QFileDialog.getSaveFileName(self,"Save Streamlit Presentation","DataScienceStudioPro_Presentation.py","Python (*.py)")
        if not path:return
        from agent.graph_presentation import agent_presentation_app
        state={"report_evidence":evidence,"report_text":self.agent_report_state.get("report_text",""),"memory":getattr(self,"presentation_agent_memory",[]),"output_path":path,"title":"Data Science Studio Pro — Analysis Presentation"}
        self.presentation_agent_worker=SimpleGraphWorker(agent_presentation_app,state,{"configurable":{"thread_id":"presentation"}}); self.presentation_agent_worker.finished.connect(self.on_presentation_agent_finished); self.presentation_agent_worker.error.connect(lambda e:QMessageBox.critical(self,"Presentation Agent",e)); self.presentation_agent_worker.start(); self.status.showMessage("AI Agent Presentation is generating the Streamlit application…")

    def on_presentation_agent_finished(self,result):
        self.presentation_agent_memory=result.get("memory",[]); self.last_presentation_path=result.get("output_path");
        if result.get("status")=="complete": QMessageBox.information(self,"Presentation Created",result.get("summary")+"\n\nRun it with: streamlit run "+str(result.get("output_path")))
        else: QMessageBox.critical(self,"Presentation Agent",result.get("error","Presentation generation failed."))

    # ============================================================
    # SHARING
    # ============================================================
    def export_png(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG Files (*.png)")
        if filepath:
            self.fig.savefig(filepath, dpi=150)
            self.status.showMessage(f"Exported to {filepath}")

    def email_current_sheet(self):
        self.create_sharing_package()

    def email_story(self):
        self.create_sharing_package()

    def export_to_streamlit(self):
        self.run_ai_agent_presentation()

    def create_sharing_package(self):
        path,_=QFileDialog.getSaveFileName(self,"Create Collaboration Package","DataScienceStudioPro_Share.zip","ZIP (*.zip)")
        if not path:return
        artifacts={"analysis_snapshot.json":json.dumps(self._analysis_snapshot_dict(),indent=2,default=str).encode("utf-8"),"view_summary.txt":self._view_summary()}
        if self.agent_report_state.get("last_report_path") and Path(self.agent_report_state["last_report_path"]).exists(): artifacts["AI_Agent_Report.pdf"]=Path(self.agent_report_state["last_report_path"]).read_bytes()
        if getattr(self,"last_presentation_path",None) and Path(self.last_presentation_path).exists(): artifacts["AI_Agent_Presentation.py"]=Path(self.last_presentation_path).read_text(encoding="utf-8")
        SharingService.build_bundle(path,artifacts); self.last_share_package_path=path; self.status.showMessage(f"Collaboration package created: {path}")
        QMessageBox.information(self,"Collaboration Package", f"Package created successfully.\n\n{path}\n\nYou can place this ZIP on your approved team drive, send it to a colleague, or attach it to a project record.")

    def copy_share_package_path(self):
        path=getattr(self,"last_share_package_path","")
        if not path: QMessageBox.information(self,"Sharing","Create a Collaboration Package first."); return
        QApplication.clipboard().setText(path); self.status.showMessage("Share package path copied to clipboard.")

    def _view_summary(self):
        return json.dumps({"rows":self.rows_input.text(),"columns":self.cols_input.text(),"chart":self.chart_combo.currentText(),"marks":{k:v.currentText() for k,v in self.marks_widgets.items()},"filters":self.data_engine.filter_specs},indent=2,default=str)

    def _analysis_snapshot_dict(self):
        return {"dataset_fingerprint":self.data_engine.dataset_hash(),"source":self.data_engine.source_path,"view":{ "rows":self.rows_input.text(),"columns":self.cols_input.text(),"chart":self.chart_combo.currentText(),"marks":{k:v.currentText() for k,v in self.marks_widgets.items()}},"filters":self.data_engine.filter_specs,"evidence":self.evidence_dag.to_dict()}

    # ============================================================
    # TOOLBAR ACTIONS
    # ============================================================
    def undo(self):
        if not self.history: self.status.showMessage("Nothing to undo."); return
        current={"rows":self.rows_input.text(),"columns":self.cols_input.text(),"chart":self.chart_combo.currentText(),"marks":{k:v.currentText() for k,v in self.marks_widgets.items()}}
        self.redo_history.append(current); state=self.history.pop()
        self.rows_input.setText(state["rows"]); self.cols_input.setText(state["columns"]); self.chart_combo.setCurrentText(state["chart"])
        for k,v in state["marks"].items(): self.marks_widgets[k].setCurrentText(v)
        self.update_plot(); self.status.showMessage("View change undone.")

    def redo(self):
        if not self.redo_history: self.status.showMessage("Nothing to redo."); return
        current={"rows":self.rows_input.text(),"columns":self.cols_input.text(),"chart":self.chart_combo.currentText(),"marks":{k:v.currentText() for k,v in self.marks_widgets.items()}}
        self.history.append(current); state=self.redo_history.pop()
        self.rows_input.setText(state["rows"]); self.cols_input.setText(state["columns"]); self.chart_combo.setCurrentText(state["chart"])
        for k,v in state["marks"].items(): self.marks_widgets[k].setCurrentText(v)
        self.update_plot(); self.status.showMessage("View change redone.")

    def sort_asc(self): self._sort_plot_axis(False)
    def sort_desc(self): self._sort_plot_axis(True)
    def _sort_plot_axis(self, descending=False):
        if self.data_engine.df is None: return
        rows=self.viz_engine.parse_shelf(self.rows_input.text()); cols=self.viz_engine.parse_shelf(self.cols_input.text())
        dims=[c for c in rows+cols if c in self.data_engine.df.columns and not pd.api.types.is_numeric_dtype(self.data_engine.df[c])]
        nums=[c for c in rows+cols if c in self.data_engine.df.columns and pd.api.types.is_numeric_dtype(self.data_engine.df[c])]
        if not dims or not nums: self.status.showMessage("Add a dimension and measure to sort the view."); return
        g=self.data_engine.df.groupby(dims,dropna=False)[nums[0]].sum().sort_values(ascending=not descending)
        order=[str(x) for x in g.index] if len(dims)==1 else [" | ".join(map(str,x)) for x in g.index]
        self.status.showMessage(f"Sorted Y-axis {'descending' if descending else 'ascending'} by {nums[0]}.")
        # Keep the data unchanged; update_plot uses this ordering hint.
        self._sort_order=order; self._sort_desc=descending; self.update_plot()

    def create_story(self): self.manage_sheets(mode="story")
    def manage_sheets(self, mode="sheet"): self._open_tableau_workspace(mode)


    def _add_evidence(self, evidence):
        self.evidence_records.append(evidence); self.evidence_dag.add(evidence)

    def show_leakage_gate(self):
        if self.data_engine.df is None: QMessageBox.information(self,"Leakage Gate","Load data first."); return
        gate=self.leakage_gate.evaluate(self.data_engine.df,self.data_engine.df.columns[-1]); QMessageBox.information(self,"Scientific/Data Leakage Gate",json.dumps(gate,indent=2,default=str))
    def show_dataset_card(self):
        if self.data_engine.df is None: QMessageBox.information(self,"Dataset Card","Load data first."); return
        self.dataset_card=DatasetCardBuilder.build(self.data_engine.df); self._show_json_dialog("Dataset Card",self.dataset_card)
    def show_model_cards(self): self._show_json_dialog("Model Cards",self.model_cards or {"message":"No model cards yet."})
    def show_experiment_comparison(self):
        df=ExperimentComparator.compare(self.experiment_records)
        if df.empty: self._show_text_dialog("Experiment Comparison 2.0","No experiment comparison records yet."); return
        # Add reproducibility-oriented context without ranking political-style or arbitrary winners.
        self._show_text_dialog("Experiment Comparison 2.0", df.to_string(index=False) + "\n\nCompare dataset fingerprint, task, model, metrics, uncertainty and recorded preprocessing before drawing conclusions.")
    def show_data_diff(self):
        if self.data_engine.original_df is None or self.data_engine.df is None: QMessageBox.information(self,"Data Diff","No source/active dataframe pair is available."); return
        self._show_json_dialog("Data Diff",DataDiff.compare(self.data_engine.original_df,self.data_engine.df))
    def show_evidence_dag(self):
        d=QDialog(self); d.setWindowTitle("Visual Evidence DAG"); d.resize(1000,700); l=QVBoxLayout(d)
        scene=QGraphicsScene(d); view=QGraphicsView(scene); l.addWidget(view)
        nodes=list(self.evidence_dag.nodes.values())
        if not nodes: scene.addText("No evidence recorded yet.")
        else:
            pos={}
            for i,n in enumerate(nodes):
                x=(i%3)*300; y=(i//3)*120; rect=QGraphicsRectItem(x,y,260,80); rect.setBrush(QColor("#e3f2fd")); scene.addItem(rect); t=QGraphicsTextItem(str(n.get("title","Evidence"))[:60],rect); t.setPos(x+8,y+8); pos[n.get("evidence_id")]=(x+130,y+80)
            for a,b,_ in self.evidence_dag.edges:
                if a in pos and b in pos: scene.addLine(pos[a][0],pos[a][1],pos[b][0],pos[b][1])
        d.exec()
    def show_model_promotion(self):
        if not self.model_cards:
            QMessageBox.information(self,"Model Promotion","Run an ML/DL analysis first."); return
        ids=[]
        for c in self.model_cards:
            ids.append(str(c.get("best_model") or c.get("agent")))
        model,ok=QInputDialog.getItem(self,"Model Promotion","Model:",ids,0,False)
        if not ok: return
        stage,ok=QInputDialog.getItem(self,"Model Promotion","Promote to:",list(ModelPromotionRegistry.STAGES),1,False)
        if not ok: return
        approval=HumanApprovalEvidence.create("Model Promotion", "approved", f"User approved promotion of {model} to {stage}.")
        self._add_evidence(approval)
        try:
            self.promotion_registry.register(model, next((c.get("metrics",{}) for c in self.model_cards if str(c.get("best_model") or c.get("agent"))==model),{}), [approval["evidence_id"]])
            record=self.promotion_registry.promote(model,stage,approval); self._show_json_dialog("Model Promotion",record)
        except Exception as exc: QMessageBox.critical(self,"Model Promotion",str(exc))
    def show_agent_evaluation(self): self._show_json_dialog("Agent Evaluation",AgentEvaluation.evaluate(self.agent_ds_state))
    def show_voice_help(self):
        QMessageBox.information(self,"Voice Command Help","For hands-free operation, enable DSP_VOICE_APPROVAL=1 before starting the app, then use Accessibility → Start Voice Command. Supported commands include approve, reject, pause and abort. Ambiguous speech never approves a step.")
    def start_voice_command(self):
        if self.agent_worker and self.agent_worker.isRunning(): QMessageBox.information(self,"Voice Command","Voice approval is active for the current agent gate when DSP_VOICE_APPROVAL=1."); return
        os.environ["DSP_VOICE_APPROVAL"]="1"; QMessageBox.information(self,"Voice Command","Voice approval mode enabled for the next Agent Data Scientist run.")
    def stop_agent_run(self):
        self.agent_abort_requested=True
        if self.agent_worker and self.agent_worker.isRunning(): self.agent_ds_state["abort_requested"]=True
        self.status.showMessage("Stop requested. The current human gate will terminate the run safely.")
    def show_help_topic(self, topic):
        text={
          "shelves":"Drag fields from Data Management to Rows or Columns. Multiple fields are supported. Marks Color/Size/Text/Detail/Tooltip also accept real field drags; their dropdowns are functional secondary selectors. Measures aggregate; dimensions define groups. The shelf containing the measure determines horizontal/vertical orientation.",
          "filters":"Drag any numerical, categorical or datetime field from Data Management directly into Filters. The dedicated drop target accepts the field and opens the appropriate filter editor. The filter remains persistent until edited or removed.",
          "views":"Sheets are standalone analytical views. Story opens the combined Dashboard + Story workspace: arrange multiple sheets in a grid, apply a global filter, and navigate story steps with captions.",
          "agents":"Agent Data Scientist is the Master Agent with exactly two internal agents: ML and DL. LangGraph orchestrates state and transitions; human approval gates every executable milestone. AI Agent Table Creation handles Web Scraping/API-key table acquisition. AI Agent Report produces evidence-bound PDF output, and AI Agent Presentation consumes that report to create an editable Streamlit application. Abort Run safely terminates the Master workflow.",
          "charts":"Supported native-style views include bar, stacked bar, line, area, dual-axis, scatter, histogram, box plot, heatmap, highlight table, pie/donut, Pareto, waterfall, Gantt and bullet. Date fields can be drilled through Year → Quarter → Month → Week → Day; categorical hierarchies can be defined through the Sheets/Story workspace.",
          "about":"Data Science Studio Pro is a desktop, evidence-first data-science environment. Governance, reproducibility, model cards, dataset cards, lineage and human approvals are recorded alongside analytical results."}[topic]
        self._show_text_dialog("Data Science Studio Pro Help",text)
    def _show_text_dialog(self,title,text):
        d=QDialog(self); d.setWindowTitle(title); d.resize(900,650); l=QVBoxLayout(d); w=QTextEdit(); w.setReadOnly(True); w.setFont(QFont("Arial",10)); w.setPlainText(str(text)); l.addWidget(w); d.exec()
    def _show_json_dialog(self,title,obj): self._show_text_dialog(title,json.dumps(obj,indent=2,default=str))

    def _open_tableau_workspace(self, mode="sheet"):
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load data first."); return
        d=QDialog(self); d.setWindowTitle("Sheets / Dashboards / Stories"); d.resize(1150,800); root=QVBoxLayout(d)
        tabs=QTabWidget(); root.addWidget(tabs)
        # Tableau-like standalone Sheet manager.
        sheet_tab=QWidget(); sl=QVBoxLayout(sheet_tab); buttons=QHBoxLayout(); newb=QPushButton("New Sheet"); dupb=QPushButton("Duplicate"); delb=QPushButton("Delete"); buttons.addWidget(newb); buttons.addWidget(dupb); buttons.addWidget(delb); sl.addLayout(buttons)
        sheet_list=QListWidget(); sl.addWidget(sheet_list); preview=FigureCanvas(Figure(figsize=(9,4),dpi=90)); sl.addWidget(preview)
        for name in self.sheet_manager.sheets: sheet_list.addItem(name)
        def render_sheet(name):
            cfg=self.sheet_manager.sheets.get(name,{})
            df=cfg.get("data") if isinstance(cfg,dict) and isinstance(cfg.get("data"),pd.DataFrame) else self.data_engine.df
            fig=preview.figure; fig.clear(); ax=fig.add_subplot(111)
            rows=self.viz_engine.parse_shelf(self.rows_input.text()); cols=self.viz_engine.parse_shelf(self.cols_input.text());
            plan=self.viz_engine.plan_shelves(df,rows,cols); data=self.viz_engine.aggregate_for_shelves(df,rows,cols,cfg.get("agg_func","sum") if isinstance(cfg,dict) else "sum")
            if data.empty: ax.text(.5,.5,"Empty sheet",ha="center",va="center")
            else:
                nums=[c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]; dims=[c for c in data.columns if c not in nums]; y=nums[0] if nums else data.columns[-1]; x=dims[0] if dims else None
                if self.chart_combo.currentText() in {"line","area"} and x: ax.plot(data[x].astype(str),data[y],marker="o")
                elif self.chart_combo.currentText()=="scatter" and len(nums)>=2: ax.scatter(data[nums[0]],data[nums[1]])
                elif x: ax.bar(data[x].astype(str),data[y]); ax.tick_params(axis="x",rotation=45)
                else: ax.bar([y],[float(data[y].iloc[0])])
                ax.set_title(name)
            fig.subplots_adjust(left=.10,right=.97,bottom=.18,top=.90); preview.draw()
        def selected_name():
            i=sheet_list.currentItem(); return i.text() if i else None
        sheet_list.itemClicked.connect(lambda i: render_sheet(i.text()))
        def create_sheet():
            name=f"Sheet {len(self.sheet_manager.sheets)+1}"; self.sheet_manager.sheets[name]={"x_col":None,"y_col":None,"mark_type":"Bar","color_col":None,"size_col":None,"agg_func":"sum","data":self.data_engine.df.copy()}; sheet_list.addItem(name); sheet_list.setCurrentRow(sheet_list.count()-1); render_sheet(name)
        def duplicate_sheet():
            name=selected_name();
            if not name:return
            new=f"{name} Copy"; self.sheet_manager.sheets[new]=dict(self.sheet_manager.sheets[name]); sheet_list.addItem(new)
        def delete_sheet():
            name=selected_name();
            if not name:return
            if len(self.sheet_manager.sheets)<=1: QMessageBox.warning(d,"Sheets","At least one sheet must remain."); return
            self.sheet_manager.sheets.pop(name,None); sheet_list.takeItem(sheet_list.currentRow())
        newb.clicked.connect(create_sheet); dupb.clicked.connect(duplicate_sheet); delb.clicked.connect(delete_sheet)
        tabs.addTab(sheet_tab,"Sheets")

        # Combined Dashboard + Story workspace: grid of Sheet objects + global filters + step navigation.
        story_tab=QWidget(); gl=QVBoxLayout(story_tab); controls=QHBoxLayout(); add=QPushButton("Add Current Sheet"); prevb=QPushButton("Previous"); nextb=QPushButton("Next"); field_combo=QComboBox(); value_combo=QComboBox(); controls.addWidget(add); controls.addWidget(prevb); controls.addWidget(nextb); controls.addWidget(QLabel("Global Filter:")); controls.addWidget(field_combo); controls.addWidget(value_combo); gl.addLayout(controls)
        grid=QGridLayout(); gl.addLayout(grid); caption=QTextEdit(); caption.setMaximumHeight(95); gl.addWidget(caption)
        dashboard=TableauDashboard("Dashboard",[]); story=TableauStory("Story",[dashboard],[])
        categorical=[c for c in self.data_engine.df.columns if not pd.api.types.is_numeric_dtype(self.data_engine.df[c])]
        field_combo.addItems(categorical)
        def refresh_values():
            value_combo.clear(); value_combo.addItem("All"); f=field_combo.currentText()
            if f in self.data_engine.df.columns: value_combo.addItems([str(x) for x in self.data_engine.df[f].dropna().astype(str).unique()[:200]])
        field_combo.currentTextChanged.connect(refresh_values); refresh_values()
        def mini_canvas(sheet):
            cv=FigureCanvas(Figure(figsize=(4.4,2.7),dpi=80)); fig=cv.figure; ax=fig.add_subplot(111); result=sheet.render(); dat=result.get("data") if result else None
            if dat is not None and not dat.empty:
                nums=[c for c in dat.columns if pd.api.types.is_numeric_dtype(dat[c])]; dims=[c for c in dat.columns if c not in nums]; y=nums[0] if nums else None; x=dims[0] if dims else None
                if y and x: ax.bar(dat[x].astype(str),dat[y]); ax.tick_params(axis="x",rotation=45,labelsize=7)
                elif len(nums)>=2: ax.scatter(dat[nums[0]],dat[nums[1]],s=25)
            ax.set_title(sheet.name,fontsize=9); fig.subplots_adjust(left=.10,right=.97,bottom=.18,top=.90); return cv
        def rebuild():
            while grid.count(): item=grid.takeAt(0); w=item.widget(); w.deleteLater() if w else None
            for i,sheet in enumerate(dashboard.filtered_sheets()): grid.addWidget(mini_canvas(sheet),i//2,i%2)
            caption.setPlainText(story.captions[story.current_step] if story.captions and story.current_step<len(story.captions) else "Dashboard step: all sheets share the global filter.")
        def add_sheet():
            rows=self.viz_engine.parse_shelf(self.rows_input.text()); cols=self.viz_engine.parse_shelf(self.cols_input.text()); marks={"color_col":self._mark_value("Color"),"size_col":self._mark_value("Size"),"text_col":self._mark_value("Text"),"detail_col":self._mark_value("Detail"),"tooltip_col":self._mark_value("Tooltip")}
            sheet=TableauSheet(f"Sheet {len(dashboard.sheets)+1}",self.data_engine.df.copy(),rows,cols,self.chart_combo.currentText(),marks,self.sheet_manager.get_sheet_config(self.sheet_manager.active_sheet).get("agg_func","sum")); dashboard.sheets.append(sheet); story.captions.append(f"Story step {len(story.captions)+1}: review the current dashboard and the recorded analytical view."); rebuild()
        def global_filter():
            f=field_combo.currentText(); v=value_combo.currentText(); dashboard.global_filters={f:v} if f and v!="All" else {}; rebuild()
        add.clicked.connect(add_sheet); value_combo.currentTextChanged.connect(global_filter); nextb.clicked.connect(lambda:(story.next(),rebuild())); prevb.clicked.connect(lambda:(story.previous(),rebuild())); rebuild(); tabs.addTab(story_tab,"Dashboard + Story")
        d.exec()

    # ============================================================
    # DATA MANAGEMENT REFRESH
    # ============================================================
    def refresh_data_management(self):
        self.num_list.clear()
        self.cat_list.clear()
        if self.data_engine.df is None:
            return
        for col in self.data_engine.dimensions:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsDragEnabled)
            self.cat_list.addItem(item)
        for col in self.data_engine.measures:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsDragEnabled)
            self.num_list.addItem(item)

    def populate_marks_combos(self):
        cols=[] if self.data_engine.df is None else [str(c) for c in self.data_engine.df.columns]
        for combo in [self.marks_widgets.get("Color"),self.marks_widgets.get("Size"),self.marks_widgets.get("Text"),self.marks_widgets.get("Detail"),self.marks_widgets.get("Tooltip")]:
            if combo: combo.sync_fields(cols)

    def show_column_menu(self, pos):
        list_widget = self.sender()
        item = list_widget.itemAt(pos)
        if not item:
            return
        col = item.text()
        menu = QMenu(self)
        menu.addAction(f"Add '{col}' to Sheet", lambda: self.add_column_to_sheet(col))
        menu.addSeparator()
        menu.addAction("Duplicate", lambda: self.duplicate_column(col))
        menu.addAction("Rename", lambda: self.rename_column(col))
        menu.addAction("Convert to Measure", lambda: self.toggle_measure(col))
        menu.addSeparator()
        menu.addAction("Group By...", lambda: self.group_by_tableau_like())
        menu.addAction("Create Hierarchy...", lambda: self.create_hierarchy(col))
        menu.addAction("Describe...", lambda: self.describe_column(col))
        menu.addAction("Create Calculated Field", lambda: self.create_calculated_field(col))
        menu.exec(list_widget.mapToGlobal(pos))

    def create_hierarchy(self, root_col):
        if self.data_engine.df is None: return
        cols=list(self.data_engine.df.columns); ordered=[root_col]+[c for c in cols if c!=root_col]
        selected,ok=QInputDialog.getItem(self,"Create Hierarchy",f"Select next level after '{root_col}':",[c for c in ordered if c!=root_col],0,False)
        if not ok:return
        self.hierarchies[root_col]=[root_col,selected]
        QMessageBox.information(self,"Hierarchy Created",f"Hierarchy: {' → '.join(self.hierarchies[root_col])}\nRight-click the field on Rows/Columns to drill down.")
    def drill_down_field(self, widget, field):
        hierarchy=next((h for h in self.hierarchies.values() if field in h),None)
        if not hierarchy:return
        i=hierarchy.index(field)
        if i+1>=len(hierarchy): QMessageBox.information(self,"Hierarchy","Already at the lowest level."); return
        fields=self.viz_engine.parse_shelf(widget.text()); fields=[hierarchy[i+1] if x==field else x for x in fields]; widget.setText(self.viz_engine.format_shelf(fields)); self.update_plot()

    def add_column_to_sheet(self, col):
        self._append_shelf(self.rows_input, col)
        self._record_view_change()
        self.update_plot()

    def duplicate_column(self, col):
        new_name, ok = QInputDialog.getText(self, "Duplicate Column", f"New name for copy of '{col}':", text=f"{col}_copy")
        if ok and new_name.strip():
            work = self.data_engine.df.copy(); work[new_name] = work[col]
            self.data_engine.commit_dataframe(work, keep_as_source=True)
            self.refresh_data_management()
            self.update_plot()

    def rename_column(self, col):
        new_name, ok = QInputDialog.getText(self, "Rename Column", f"New name for '{col}':", text=col)
        if ok and new_name and new_name != col:
            work = self.data_engine.df.copy().rename(columns={col: new_name})
            self.data_engine.commit_dataframe(work, keep_as_source=True)
            self.refresh_data_management()
            self.update_plot()

    def toggle_measure(self, col):
        work = self.data_engine.df.copy()
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].astype(str)
        else:
            work[col] = pd.to_numeric(work[col], errors='coerce')
        self.data_engine.commit_dataframe(work, keep_as_source=True)
        self.refresh_data_management()
        self.populate_marks_combos()
        self.update_plot()

    def describe_column(self, col):
        if self.data_engine.df is None or col not in self.data_engine.df.columns:
            return
        desc = self.data_engine.df[col].describe(include='all').to_string()
        QMessageBox.information(self, f"Describe: {col}", desc)

    def create_calculated_field(self, col=None):
        """V16.py Calculated Field implementation."""
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        name, ok = QInputDialog.getText(self, "Calculated Field", "New field name:")
        if not ok or not name:
            return
        expr, ok = QInputDialog.getText(self, "Calculated Field", "Formula (use [column] for columns):")
        if not ok or not expr:
            return
        try:
            import re
            expr_py = re.sub(r'\[([^\]]+)\]', r"df['\1']", expr)
            result = eval(expr_py, {"__builtins__": {}}, {"df": self.data_engine.df, "np": np, "pd": pd})
            if isinstance(result, (int, float)):
                result = pd.Series([result] * len(self.data_engine.df), index=self.data_engine.df.index)
            work = self.data_engine.df.copy(); work[name] = result
            self.data_engine.commit_dataframe(work, keep_as_source=True)
            self.refresh_data_management(); self.populate_marks_combos()
            self.update_plot()
            QMessageBox.information(self, "Success", f"Created calculated field '{name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Formula error: {e}")

    def group_by_tableau_like(self):
        """V16.py Group By implementation."""
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        dialog = GroupByDialog(self.data_engine.df, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected = [item.text() for item in dialog.group_list.selectedItems()]
            if not selected:
                QMessageBox.warning(self, "Error", "Select at least one column to group by.")
                return
            agg_dict = {}
            for col, combo in dialog.agg_combos.items():
                agg_func = combo.currentText()
                if agg_func == "Sum":
                    agg_dict[col] = 'sum'
                elif agg_func == "Average":
                    agg_dict[col] = 'mean'
                elif agg_func == "Count":
                    agg_dict[col] = 'count'
                elif agg_func == "Min":
                    agg_dict[col] = 'min'
                elif agg_func == "Max":
                    agg_dict[col] = 'max'
                elif agg_func == "Count Distinct":
                    agg_dict[col] = 'nunique'
            try:
                result = self.data_engine.df.groupby(selected).agg(agg_dict).reset_index()
                if dialog.new_sheet_requested:
                    sheet_name = f"Group By {len(self.sheet_manager.sheets) + 1}"
                    self.sheet_manager.sheets[sheet_name] = {"x_col": None, "y_col": None, "mark_type": "Bar", "color_col": None, "size_col": None, "agg_func": "sum", "data": result.copy()}
                    QMessageBox.information(self, "Success", f"Created new sheet '{sheet_name}'.")
                else:
                    self.data_engine.commit_dataframe(result, keep_as_source=True)
                    self.refresh_data_management()
                    self.populate_marks_combos()
                    self.update_plot()
                    QMessageBox.information(self, "Success", "Group By applied successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    # ============================================================
    # FILTERS
    # ============================================================
    def apply_filter(self, col):
        if self.data_engine.df is None or col not in self.data_engine.df.columns:
            return
        if pd.api.types.is_numeric_dtype(self.data_engine.df[col]):
            min_val = float(self.data_engine.df[col].min())
            max_val = float(self.data_engine.df[col].max())
            val, ok = QInputDialog.getDouble(self, f"Filter {col}", f"Enter min value (max: {max_val}):", min_val, min_val, max_val)
            if ok:
                self.data_engine.df = self.data_engine.df[self.data_engine.df[col] >= val]
                self.update_plot()
        else:
            values = self.data_engine.df[col].dropna().unique().tolist()
            item, ok = QInputDialog.getItem(self, f"Filter {col}", "Select value:", values, 0, False)
            if ok and item:
                self.data_engine.df = self.data_engine.df[self.data_engine.df[col] == item]
                self.update_plot()

    # ============================================================
    # SHEET SELECTION
    # ============================================================
    def on_loaded_sheet_select(self, item):
        name = item.text(); cfg = self.sheet_manager.sheets.get(name, {})
        if isinstance(cfg, dict) and isinstance(cfg.get("data"), pd.DataFrame):
            self.data_engine.set_active_dataframe(cfg["data"])
            self.refresh_data_management(); self.populate_marks_combos(); self.update_plot()
        self.status.showMessage(f"Selected sheet: {name}")

    # ============================================================
    # PLOT UPDATING
    # ============================================================
    def _mark_value(self, name):
        combo = self.marks_widgets.get(name); value = combo.currentText().strip() if combo else ""
        return value if value and self.data_engine.df is not None and value in self.data_engine.df.columns else None

    def update_plot(self):
        self.ax.clear()
        df = self.data_engine.df
        if df is None:
            self.ax.text(.5,.5,"No data loaded",ha="center",va="center"); self.canvas.draw(); return
        rows=[x for x in self.viz_engine.parse_shelf(self.rows_input.text()) if x in df.columns]
        cols=[x for x in self.viz_engine.parse_shelf(self.cols_input.text()) if x in df.columns]
        chart=self.chart_combo.currentText()
        color_col=self._mark_value("Color"); size_col=self._mark_value("Size"); text_col=self._mark_value("Text"); detail_col=self._mark_value("Detail"); tooltip_col=self._mark_value("Tooltip")
        if not rows and not cols:
            self.ax.text(.5,.5,"Drag fields to Rows and Columns to begin",ha="center",va="center"); self.canvas.draw(); return
        try:
            cfg=self.sheet_manager.get_sheet_config(self.sheet_manager.active_sheet)
            agg=cfg.get("agg_func","sum") if isinstance(cfg,dict) else "sum"
            func=agg if agg in {"sum","mean","count","min","max"} else "sum"
            shelf_fields=rows+cols
            numeric_shelf=[c for c in shelf_fields if pd.api.types.is_numeric_dtype(df[c])]
            base_dims=[c for c in shelf_fields if c not in numeric_shelf]

            # Tableau semantics: Color/Detail may define the level of detail; numeric
            # mark fields become aggregated measures. Text/Tooltip are carried through
            # as representative values when they are not already grouped.
            group_dims=list(base_dims)
            for mark in ("Color","Detail"):
                field=self._mark_value(mark)
                if field and field not in group_dims:
                    group_dims.append(field)
            mark_numeric=[]
            for field in (color_col,size_col,text_col,tooltip_col):
                if field and field in df.columns and pd.api.types.is_numeric_dtype(df[field]):
                    mark_numeric.append(field)
            measures=list(dict.fromkeys(numeric_shelf + mark_numeric))

            if measures:
                agg_map={m:func for m in measures}
                mark_to_col={"Color":color_col,"Size":size_col,"Text":text_col,"Tooltip":tooltip_col}
                for mark_name, field in mark_to_col.items():
                    if field in agg_map and hasattr(self,"mark_aggregations"):
                        chosen=self.mark_aggregations.get(mark_name,func)
                        agg_map[field]=chosen if chosen in {"sum","mean","count","min","max"} else func
                if group_dims:
                    work=df.groupby(group_dims,dropna=False)[measures].agg(agg_map).reset_index()
                else:
                    work=pd.DataFrame([{m:getattr(df[m],agg_map[m])() for m in measures}])
            else:
                measures=["Number of Records"]
                work=df.groupby(group_dims,dropna=False).size().reset_index(name=measures[0]) if group_dims else pd.DataFrame({measures:[len(df)]})
            primary=numeric_shelf[0] if numeric_shelf else measures[0]

            # Carry categorical/nonnumeric Text and Tooltip fields into the aggregated view.
            carry_fields=[]
            for field in (text_col,tooltip_col):
                if field and field not in work.columns and field in df.columns:
                    carry_fields.append(field)
            if carry_fields:
                if group_dims:
                    extras=df.groupby(group_dims,dropna=False)[carry_fields].first().reset_index()
                    key=group_dims
                    work=work.merge(extras,on=key,how="left")
                else:
                    for field in carry_fields: work[field]=df[field].dropna().iloc[0] if df[field].notna().any() else ""

            if chart=="auto":
                chart="vertical_bar" if any(c in cols for c in numeric_shelf) else "horizontal_bar"
            orientation="horizontal" if any(c in cols for c in numeric_shelf) else "vertical"
            if hasattr(self,"_sort_order") and len(base_dims)==1 and base_dims[0] in work.columns:
                work[base_dims[0]]=pd.Categorical(work[base_dims[0]].astype(str),categories=self._sort_order,ordered=True); work=work.sort_values(base_dims[0])
            label_col=base_dims[0] if base_dims else None
            labels=work[label_col].astype(str) if label_col else pd.Series(["Value"]*len(work),index=work.index)
            if len(base_dims)>1:
                labels=work[base_dims].astype(str).agg(" | ".join,axis=1)

            # Mark Color: categorical values define grouping; numeric values map to a continuous colormap.
            color_values=work[color_col] if color_col and color_col in work.columns else None
            if chart in {"line_discrete","line_continuous","line","area"}:
                if color_col and not pd.api.types.is_numeric_dtype(df[color_col]) and color_col in work.columns:
                    for value, grp in work.groupby(color_col,dropna=False):
                        self.ax.plot(grp.index,grp[primary],marker="o",label=str(value))
                    self.ax.legend(title=color_col)
                else:
                    self.ax.plot(np.arange(len(work)),work[primary],marker="o",label=primary)
                    if chart=="area": self.ax.fill_between(np.arange(len(work)),work[primary].to_numpy(float),alpha=.25)
                self.ax.set_xticks(np.arange(len(work))); self.ax.set_xticklabels(labels,rotation=45,ha="right")
            elif chart=="scatter":
                xfield=numeric_shelf[0] if numeric_shelf else primary; yfield=numeric_shelf[1] if len(numeric_shelf)>1 else primary
                if xfield not in work.columns or yfield not in work.columns: raise ValueError("Scatter requires numeric fields on Rows/Columns.")
                sizes=self._sizes(work,size_col)
                if color_values is not None and pd.api.types.is_numeric_dtype(color_values):
                    self.ax.scatter(work[xfield],work[yfield],s=sizes,c=pd.to_numeric(color_values,errors="coerce"),cmap="viridis",alpha=.8)
                    self.fig.colorbar(self.ax.collections[-1],ax=self.ax,label=color_col)
                elif color_col and color_col in work.columns:
                    cats={v:i for i,v in enumerate(pd.unique(work[color_col]))}; vals=work[color_col].map(cats).to_numpy(float); self.ax.scatter(work[xfield],work[yfield],s=sizes,c=vals,cmap="tab10",alpha=.8)
                else: self.ax.scatter(work[xfield],work[yfield],s=sizes,alpha=.8)
                self.ax.set_xlabel(xfield); self.ax.set_ylabel(yfield)
            elif chart=="histogram": self.ax.hist(pd.to_numeric(df[primary],errors="coerce").dropna(),bins="auto",alpha=.8); self.ax.set_xlabel(primary)
            elif chart=="box":
                vals=[pd.to_numeric(df[c],errors="coerce").dropna() for c in numeric_shelf[:6] or [primary]]; self.ax.boxplot(vals,labels=numeric_shelf[:len(vals)] or [primary])
            elif chart in {"heatmap","highlight_table"} and len(base_dims)>=2:
                pivot=pd.pivot_table(df,index=base_dims[0],columns=base_dims[1],values=numeric_shelf[0] if numeric_shelf else None,aggfunc=func,fill_value=0); im=self.ax.imshow(pivot.to_numpy(),aspect="auto"); self.fig.colorbar(im,ax=self.ax); self.ax.set_xticks(range(len(pivot.columns))); self.ax.set_xticklabels([str(x) for x in pivot.columns],rotation=45,ha="right"); self.ax.set_yticks(range(len(pivot.index))); self.ax.set_yticklabels([str(x) for x in pivot.index])
            elif chart in {"donut","pie"}:
                self.ax.pie(work[primary],labels=labels,autopct="%1.1f%%");
                if chart=="donut": self.ax.add_artist(plt.Circle((0,0),.55,fc="white"))
            elif chart=="stacked_bar" and color_col and color_col in work.columns and not pd.api.types.is_numeric_dtype(work[color_col]):
                pivot=work.pivot_table(index=label_col or work.index,columns=color_col,values=primary,aggfunc="sum",fill_value=0); pivot.plot(kind="bar",stacked=True,ax=self.ax); self.ax.legend(title=color_col)
            elif chart=="dual_axis" and len(numeric_shelf)>=2:
                x=np.arange(len(work)); self.ax.bar(x,work[numeric_shelf[0]],alpha=.7); ax2=self.ax.twinx(); ax2.plot(x,work[numeric_shelf[1]],marker="o"); self.ax.set_xticks(x); self.ax.set_xticklabels(labels,rotation=45,ha="right"); ax2.set_ylabel(numeric_shelf[1])
            elif chart=="gantt" and label_col and len(numeric_shelf)>=2:
                starts=pd.to_numeric(work[numeric_shelf[0]],errors="coerce"); durations=pd.to_numeric(work[numeric_shelf[1]],errors="coerce"); self.ax.barh(labels,durations,left=starts); self.ax.set_xlabel(f"{numeric_shelf[0]} + {numeric_shelf[1]}")
            elif chart=="bullet": self.ax.barh(labels,work[primary],height=.55); self.ax.axvline(float(work[primary].mean()),ls="--")
            elif chart=="pareto":
                order=work[primary].sort_values(ascending=False).index; vals=work.loc[order,primary].to_numpy(float); labs=labels.loc[order]; self.ax.bar(labs,vals); ax2=self.ax.twinx(); total=vals.sum(); ax2.plot(np.arange(len(vals)),np.cumsum(vals)/total*100,marker="o"); ax2.set_ylabel("Cumulative %") if total else None
            elif chart=="waterfall":
                vals=work[primary].to_numpy(float); starts=np.r_[0,np.cumsum(vals)[:-1]]; self.ax.bar(labels,vals,bottom=np.minimum(starts,starts+vals)); self.ax.axhline(0,linewidth=1)
            elif chart=="bump" and label_col:
                ranks=work.groupby(label_col)[primary].rank(method="first",ascending=False); self.ax.plot(labels,ranks,marker="o"); self.ax.invert_yaxis(); self.ax.set_ylabel("Rank")
            elif chart=="density" and len(numeric_shelf)>=2: self.ax.hexbin(pd.to_numeric(df[numeric_shelf[0]],errors="coerce"),pd.to_numeric(df[numeric_shelf[1]],errors="coerce"),gridsize=30); self.ax.set_xlabel(numeric_shelf[0]); self.ax.set_ylabel(numeric_shelf[1])
            elif chart=="crosstab":
                tab=pd.pivot_table(df,index=base_dims[0] if base_dims else None,columns=base_dims[1] if len(base_dims)>1 else None,values=primary,aggfunc=func,fill_value=0); self.ax.axis("off"); self.ax.table(cellText=np.round(tab.to_numpy(),3),rowLabels=[str(x) for x in tab.index],colLabels=[str(x) for x in tab.columns],loc="center")
            elif chart in {"symbol_map","filled_map"} and {"Latitude","Longitude"}.issubset(df.columns): self.ax.scatter(pd.to_numeric(df["Longitude"],errors="coerce"),pd.to_numeric(df["Latitude"],errors="coerce"),s=self._sizes(df,size_col),alpha=.7); self.ax.set_xlabel("Longitude"); self.ax.set_ylabel("Latitude")
            elif chart in {"sankey","treemap"}: self.ax.bar(labels,work[primary]); self.ax.set_title(f"{chart.title()} fallback — export the sheet for interactive Plotly rendering")
            else:
                if orientation=="horizontal": bars=self.ax.barh(labels,work[primary],height=self._bar_sizes(work,size_col)); self.ax.set_xlabel(primary); self.ax.set_ylabel(label_col or "Rows")
                else: bars=self.ax.bar(labels,work[primary],width=self._bar_sizes(work,size_col)); self.ax.set_ylabel(primary); self.ax.set_xlabel(label_col or "Columns"); self.ax.tick_params(axis="x",rotation=45)
                self._apply_bar_colors(bars,work,color_col)

            # Text mark: annotate each displayed mark with the selected field.
            if text_col and text_col in work.columns:
                for i,(_,r) in enumerate(work.iterrows()):
                    try:
                        value=r.get(text_col,"")
                        self.ax.annotate(str(value),(i,r[primary]),fontsize=8)
                    except Exception: pass
            self.ax.set_title(f"{primary} by {', '.join(base_dims) or 'Records'}")
            self._install_hover_handler(work,color_col,tooltip_col,size_col,rows,cols)
        except Exception as exc:
            self.ax.text(.5,.5,f"Plot error: {exc}",ha="center",va="center")
        self.fig.subplots_adjust(left=.10,right=.97,bottom=.18,top=.90); self.canvas.draw()

    def _bar_sizes(self, plot_df, size_col):
        if not size_col or size_col not in plot_df.columns or not pd.api.types.is_numeric_dtype(plot_df[size_col]): return 0.8
        s=pd.to_numeric(plot_df[size_col],errors="coerce").fillna(0).to_numpy(float); lo,hi=float(np.nanmin(s)),float(np.nanmax(s))
        return 0.8 if hi==lo else 0.35+0.9*(s-np.nanmin(s))/(hi-lo)

    def _sizes(self, plot_df, size_col):
        if not size_col or size_col not in plot_df.columns or not pd.api.types.is_numeric_dtype(plot_df[size_col]): return 70
        s = pd.to_numeric(plot_df[size_col], errors="coerce").fillna(0).to_numpy(float);
        if len(s) == 0: return 70
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s)); return np.full(len(s), 70.) if hi == lo else 40 + 260 * (s-lo)/(hi-lo)

    def _apply_bar_colors(self, bars, plot_df, color_col):
        if not color_col or color_col not in plot_df.columns: return
        vals = plot_df[color_col]
        if pd.api.types.is_numeric_dtype(vals):
            cmap = plt.get_cmap("viridis"); arr = pd.to_numeric(vals, errors="coerce").fillna(vals.mean()).to_numpy(float); lo, hi = np.nanmin(arr), np.nanmax(arr); norm = np.ones(len(arr))*.5 if hi == lo else (arr-lo)/(hi-lo)
            for b, n in zip(bars, norm): b.set_color(cmap(float(n)))
        else:
            categories = {v:i for i,v in enumerate(pd.unique(vals))}; cmap=plt.get_cmap("tab10"); n=max(1,len(categories)-1)
            for b, v in zip(bars, vals): b.set_color(cmap(categories[v]/n if n else 0))

    def _install_hover_handler(self, plot_df, color_col, tooltip_col, size_col, rows=None, cols=None):
        if hasattr(self, "_hover_cid") and self._hover_cid is not None:
            try: self.canvas.mpl_disconnect(self._hover_cid)
            except Exception: pass
        artists = list(self.ax.patches) + list(self.ax.lines) + list(self.ax.collections)
        base = []
        for a in artists:
            try: base.append((a, a.get_facecolor() if hasattr(a, "get_facecolor") else a.get_color()))
            except Exception: base.append((a, None))
        annotation = {"obj": None}
        def move(event):
            if event.inaxes != self.ax: return
            hit = None; hit_index = None
            for a, _ in base:
                try:
                    ok, info = a.contains(event)
                    if ok:
                        hit = a
                        inds = info.get("ind", []) if isinstance(info, dict) else []
                        hit_index = int(inds[0]) if len(inds) else None
                        break
                except Exception: pass
            for a, c in base:
                try:
                    if c is not None: a.set_facecolor(c) if hasattr(a,"set_facecolor") else a.set_color(c)
                except Exception: pass
            if hit is not None:
                try: hit.set_facecolor("#ff9900") if hasattr(hit,"set_facecolor") else hit.set_color("#ff9900")
                except Exception: pass
                if hit_index is not None and hit_index < len(plot_df):
                    row = plot_df.iloc[hit_index]
                    text = "<br>".join(f"{c}: {row[c]}" for c in plot_df.columns if c in ({tooltip_col} if tooltip_col else set()) or c in (rows or []) or c in (cols or []))
                    if annotation["obj"] is None:
                        annotation["obj"] = self.ax.annotate(text, xy=(event.xdata, event.ydata), xytext=(12, 12),
                            textcoords="offset points", bbox=dict(boxstyle="round", fc="white", alpha=.9), arrowprops=dict(arrowstyle="->"))
                    else:
                        annotation["obj"].set_text(text); annotation["obj"].xy = (event.xdata, event.ydata); annotation["obj"].set_visible(True)
                else:
                    if annotation["obj"] is not None: annotation["obj"].set_visible(False)
            else:
                if annotation["obj"] is not None: annotation["obj"].set_visible(False)
            self.canvas.draw_idle()
        self._hover_cid = self.canvas.mpl_connect("motion_notify_event", move)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
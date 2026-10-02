"""Module duty: Main.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

import sys
import os
import threading
import pickle
import io
import json
import zipfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QComboBox, QPushButton, QDockWidget, QListWidget,
                             QMenu, QToolBar, QStatusBar, QFileDialog, QMessageBox, QDialog,
                             QDialogButtonBox, QTextEdit, QSlider, QColorDialog, QListWidgetItem,
                             QInputDialog, QTabWidget, QSplitter, QTableWidget, QTableWidgetItem,
                             QCheckBox, QSpinBox, QFontDialog, QDoubleSpinBox, QGroupBox, QFormLayout, QScrollArea, QAbstractItemView, QGridLayout, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMimeData, QPointF, QRectF
from PyQt6.QtGui import QAction, QIcon, QColor, QPalette, QFont, QKeySequence, QImage, QPainter
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
from agent.graph_cnn import run_cnn_image_step
from services.project_io import build_manifest, save_manifest, load_manifest
from services.llm_config import LLMConfig, preflight as llm_preflight, local_model_available, local_model_path, infer_api_provider
from services.serialization import json_safe
from services.supervised_analysis import ClassificationAnalysisEngine, RegressionAnalysisEngine
from services.clustering_analysis import ClusteringAnalysisEngine
from services.agent_provider_registry import AgentProviderRegistry, DEFAULT_AGENTS
from services.cnn_image_analysis import CNNConfig, CNNImageAnalysisEngine
from core.tableau_views import TableauSheet, TableauDashboard, TableauStory
from langchain_core.messages import HumanMessage
import config

class ColumnDragListWidget(QListWidget):
    """Data Management list that exports the actual field name as text/plain MIME."""
    def startDrag(self, supportedActions):
        """Perform the start drag operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        item = self.currentItem()
        if item is None:
            return
        mime = QMimeData()
        mime.setText(item.text())
        drag = __import__("PyQt6.QtGui", fromlist=["QDrag"]).QDrag(self)
        drag.setMimeData(mime)
        drag.exec(supportedActions)


class FieldDropComboBox(QComboBox):
    """Tableau-like Marks field target. Multi-field properties support repeated drops/selections; Size is single-field."""
    fieldDropped = pyqtSignal(str)
    fieldsChanged = pyqtSignal(list)
    def __init__(self,parent=None,multi=True):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        super().__init__(parent); self.multi=multi; self._selected_fields=[]
        self.setAcceptDrops(True); self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert); self.setEditable(True); self.lineEdit().setReadOnly(True)
        self.activated.connect(self._toggle_from_popup)
    def selected_fields(self):
        """Perform the selected fields operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return list(self._selected_fields)
    def clear_selection(self):
        """Perform the clear selection operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._selected_fields = []
        self._sync_display()
        self.fieldsChanged.emit([])

    def _display(self):
        """Perform the display operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return ", ".join(self._selected_fields)
    def _sync_display(self):
        """Perform the sync display operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.lineEdit().setText(self._display())
        self.setToolTip(self._display() or "No field assigned. Drag a field here or use the dropdown.")
    def sync_fields(self,columns=None):
        """Perform the sync fields operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if columns is None:
            w=self.window(); df=getattr(getattr(w,"data_engine",None),"df",None); columns=[] if df is None else [str(c) for c in df.columns]
        columns=list(dict.fromkeys(map(str,columns))); selected=[x for x in self._selected_fields if x in columns]
        if [self.itemText(i) for i in range(self.count())] != columns:
            self.blockSignals(True); self.clear(); self.addItems(columns); self.blockSignals(False)
        self._selected_fields=selected; self._sync_display()
    def set_selected_fields(self,fields):
        """Perform the set selected fields operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        cols=[] if self.window() is None or getattr(self.window(),"data_engine",None).df is None else [str(c) for c in self.window().data_engine.df.columns]
        self.sync_fields(cols); vals=[str(x) for x in fields if str(x) in cols]
        self._selected_fields=vals[-1:] if not self.multi else list(dict.fromkeys(vals)); self._sync_display()
    def setCurrentText(self,text):
        """Perform the set current text operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        vals=[x.strip() for x in str(text).split(",") if x.strip()]
        self.set_selected_fields(vals)
    def _toggle_from_popup(self,index):
        """Perform the toggle from popup operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if index < 0:return
        field=self.itemText(index)
        if not field:return
        if self.multi:
            if field in self._selected_fields: self._selected_fields.remove(field)
            else: self._selected_fields.append(field)
        else:
            self._selected_fields=[field]
        self._sync_display(); self.fieldsChanged.emit(self.selected_fields())
    def showPopup(self):
        """Perform the show popup operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.sync_fields(); super().showPopup()
    def dragEnterEvent(self,event):
        """Perform the drag enter event operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        event.acceptProposedAction() if event.mimeData().hasText() and event.mimeData().text().strip() else event.ignore()
    def dragMoveEvent(self,event):
        """Perform the drag move event operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.dragEnterEvent(event)
    def dropEvent(self,event):
        """Perform the drop event operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        field=event.mimeData().text().strip() if event.mimeData().hasText() else ""
        w=self.window(); df=getattr(getattr(w,"data_engine",None),"df",None)
        if field and df is not None and field in df.columns:
            self.sync_fields([str(c) for c in df.columns])
            add=bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
            if self.multi and (add or field not in self._selected_fields):
                if not add: self._selected_fields=[]
                if field not in self._selected_fields:self._selected_fields.append(field)
            else:
                self._selected_fields=[field]
            self._sync_display(); self.fieldDropped.emit(field); self.fieldsChanged.emit(self.selected_fields()); event.setDropAction(Qt.DropAction.CopyAction); event.accept()
        else: event.ignore()


class FieldDropListWidget(QListWidget):
    """Drop-only Filters target with explicit drag-move acceptance."""
    fieldDropped = pyqtSignal(str)

    def dragEnterEvent(self, event):
        """Perform the drag enter event operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if event.mimeData().hasText() and event.mimeData().text().strip(): event.acceptProposedAction()
        else: event.ignore()

    def dragMoveEvent(self, event):
        """Perform the drag move event operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if event.mimeData().hasText() and event.mimeData().text().strip(): event.acceptProposedAction()
        else: event.ignore()

    def dropEvent(self, event):
        """Perform the drop event operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        field = event.mimeData().text().strip() if event.mimeData().hasText() else ""
        if field:
            self.fieldDropped.emit(field); event.acceptProposedAction()
        else: event.ignore()


class SimpleGraphWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    def __init__(self, app, state, config=None):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        super().__init__(); self.app=app; self.state=state; self.config=config
    def run(self):
        """Perform the run operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        super().__init__(); self.agent_app = agent_app; self.state = state; self.action = action

    def run(self):
        """Perform the run operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        super().__init__()
        self.agent_app = agent_app
        self.state = state

    def run(self):
        """Perform the run operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        try:
            result = self.agent_app.invoke(self.state)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# DIALOGS
# ============================================================

class LocalLLMConfigDialog(QDialog):
    """Single explicit provider gate for all governed AI agents."""
    def __init__(self, parent=None, current=None):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        super().__init__(parent)
        self.setWindowTitle("Select Local LLM model or API key")
        self.resize(560, 380)
        current = current or {}
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select the AI provider for one governed agent. Each agent checks its own selection immediately before execution."))
        self.agent_combo = QComboBox(); self.agent_combo.addItems(list(DEFAULT_AGENTS)); layout.addWidget(QLabel("Agent:")); layout.addWidget(self.agent_combo)
        self.agent_combo.currentTextChanged.connect(self._load_agent_config)
        self.provider_combo = QComboBox(); self.provider_combo.addItems(["None", "Local LLM model", "API key / cloud model"])
        layout.addWidget(self.provider_combo)
        self.model_combo = QComboBox(); self.model_combo.addItems(list(config.LOCAL_LLM_MODELS.keys())); layout.addWidget(QLabel("Local model:")); layout.addWidget(self.model_combo)
        self.custom_path = QLineEdit(); self.custom_path.setPlaceholderText("Optional custom local model path"); layout.addWidget(self.custom_path)
        self.api_model = QComboBox(); self.api_model.setEditable(True); self.api_model.addItems(config.CLOUD_LLM_MODELS); self.api_model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert); layout.addWidget(QLabel("API model:")); layout.addWidget(self.api_model)
        self.api_key = QLineEdit(); self.api_key.setEchoMode(QLineEdit.EchoMode.Password); self.api_key.setPlaceholderText("API key"); layout.addWidget(self.api_key)
        self.status = QLabel(""); self.status.setWordWrap(True); layout.addWidget(self.status)
        self.provider_combo.currentIndexChanged.connect(self._refresh)
        self.model_combo.currentTextChanged.connect(self._refresh)
        self.custom_path.textChanged.connect(self._refresh)
        self.api_model.currentTextChanged.connect(self._refresh)
        self.provider_combo.setCurrentText({"none":"None","local":"Local LLM model","api":"API key / cloud model"}.get(current.get("provider"),"None"))
        if current.get("model_name") in config.LOCAL_LLM_MODELS: self.model_combo.setCurrentText(current["model_name"])
        if current.get("api_key"): self.api_key.setText(current["api_key"])
        if current.get("provider") == "api" and current.get("model_name"): self.api_model.setCurrentText(current["model_name"])
        if current.get("local_path"): self.custom_path.setText(current["local_path"])
        self._refresh()
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)

    def _load_agent_config(self, agent_name):
        """Load the currently stored provider for the selected agent."""
        parent=self.parent()
        cfg=(parent._agent_llm_config(agent_name) if parent is not None and hasattr(parent,"_agent_llm_config") else LLMConfig())
        self.provider_combo.blockSignals(True); self.model_combo.blockSignals(True); self.api_model.blockSignals(True); self.custom_path.blockSignals(True); self.api_key.blockSignals(True)
        self.provider_combo.setCurrentText({"none":"None","local":"Local LLM model","api":"API key / cloud model"}.get(cfg.provider,"None"))
        if cfg.model_name in config.LOCAL_LLM_MODELS: self.model_combo.setCurrentText(cfg.model_name)
        if cfg.provider=="api": self.api_model.setCurrentText(cfg.model_name)
        self.custom_path.setText(cfg.local_path or ""); self.api_key.setText(cfg.api_key or "")
        for w in (self.provider_combo,self.model_combo,self.api_model,self.custom_path,self.api_key): w.blockSignals(False)
        self._refresh()

    def selected_agent(self):
        """Return the agent selected in the provider dialog."""
        return self.agent_combo.currentText()

    def _refresh(self):
        """Perform the refresh operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        provider=self.provider_combo.currentText()
        local=provider=="Local LLM model"; api=provider=="API key / cloud model"
        self.model_combo.setEnabled(local); self.custom_path.setEnabled(local); self.api_model.setEnabled(api); self.api_key.setEnabled(api)
        if local:
            name=self.model_combo.currentText(); ok=local_model_available(name,self.custom_path.text())
            self.status.setText(("Ready: local model is available." if ok else "Model not found in the local cache/path. Choose another model or a custom path."))
        elif api:
            family=infer_api_provider(self.api_model.currentText()); self.status.setText(f"Ready when a non-empty API key is provided. Detected API family: {family}. The model field is editable so newly released provider/model IDs can be entered without an application update.")
        else: self.status.setText("No provider selected. Governed AI agents will remain blocked.")

    def get_config(self):
        """Perform the get config operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        provider=self.provider_combo.currentText()
        if provider=="Local LLM model":
            return {"provider":"local","model_name":self.model_combo.currentText(),"api_key":"","local_path":self.custom_path.text().strip(),"api_base":""}
        if provider=="API key / cloud model":
            return {"provider":"api","model_name":self.api_model.currentText().strip(),"api_key":self.api_key.text().strip(),"local_path":"","api_base":""}
        return {"provider":"none","model_name":"","api_key":"","local_path":"","api_base":""}


class DataAnalysisDialog(QDialog):
    def __init__(self, df, analysis_type, parent=None):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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


class ClusteringAnalysisDialog(QDialog):
    """Professional unsupervised clustering workspace for senior data analysts."""

    def __init__(self, df: pd.DataFrame, parent=None):
        """Build the clustering workspace without changing the main application layout."""
        super().__init__(parent)
        self.df = df.copy()
        self.result: dict[str, Any] | None = None
        self.setWindowTitle("Clustering Analysis — Professional Workspace")
        self.resize(1120, 780)
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        setup = QWidget(); sl = QVBoxLayout(setup)
        form = QFormLayout()
        self.features = QListWidget(); self.features.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for col in self.df.select_dtypes(include=[np.number]).columns:
            item = QListWidgetItem(str(col)); item.setSelected(True); self.features.addItem(item)
        self.features.setMaximumHeight(150)
        self.scaling = QComboBox(); self.scaling.addItems(["standard", "robust", "minmax", "none"])
        self.imputation = QComboBox(); self.imputation.addItems(["median", "mean", "most_frequent"])
        self.k_min = QSpinBox(); self.k_min.setRange(2, 12); self.k_min.setValue(2)
        self.k_max = QSpinBox(); self.k_max.setRange(2, 12); self.k_max.setValue(8)
        self.eps = QDoubleSpinBox(); self.eps.setRange(0.01, 20.0); self.eps.setSingleStep(0.05); self.eps.setValue(0.50)
        self.min_samples = QSpinBox(); self.min_samples.setRange(2, 100); self.min_samples.setValue(5)
        self.repeats = QSpinBox(); self.repeats.setRange(2, 20); self.repeats.setValue(5)
        self.alg_checks = {}
        alg_box = QHBoxLayout()
        for name in ("K-Means", "Agglomerative", "Gaussian Mixture", "DBSCAN"):
            cb = QCheckBox(name); cb.setChecked(True); self.alg_checks[name] = cb; alg_box.addWidget(cb)
        form.addRow("Numeric features", self.features)
        form.addRow("Imputation", self.imputation)
        form.addRow("Scaling", self.scaling)
        form.addRow("K minimum", self.k_min); form.addRow("K maximum", self.k_max)
        form.addRow("DBSCAN eps", self.eps); form.addRow("DBSCAN min samples", self.min_samples)
        form.addRow("Stability repeats", self.repeats)
        sl.addLayout(form); sl.addWidget(QLabel("Candidate algorithms")); sl.addLayout(alg_box)
        sl.addWidget(QLabel("The workspace compares clustering families using internal validity, cluster sizes, PCA visualization, and repeatability. Internal metrics do not establish domain meaning."))
        self.run_button = QPushButton("Run Professional Clustering Analysis")
        self.run_button.clicked.connect(self._run)
        sl.addWidget(self.run_button)
        self.tabs.addTab(setup, "1. Setup & Preprocessing")

        self.methods_text = QTextEdit(); self.methods_text.setReadOnly(True); self.tabs.addTab(self.methods_text, "2. Method Comparison")
        self.profile_text = QTextEdit(); self.profile_text.setReadOnly(True); self.tabs.addTab(self.profile_text, "3. Cluster Profiles")
        self.validation_text = QTextEdit(); self.validation_text.setReadOnly(True); self.tabs.addTab(self.validation_text, "4. Validation & Stability")
        viz = QWidget(); vl = QVBoxLayout(viz)
        self.figure = Figure(figsize=(7, 4)); self.canvas = FigureCanvas(self.figure); vl.addWidget(self.canvas)
        self.tabs.addTab(viz, "5. PCA Visualization")
        self.export_text = QTextEdit(); self.export_text.setReadOnly(True); self.tabs.addTab(self.export_text, "6. Reproducibility & Export")
        export_row = QHBoxLayout()
        export_json = QPushButton("Export Analysis JSON"); export_json.clicked.connect(self._export_json); export_row.addWidget(export_json)
        export_csv = QPushButton("Export Cluster Assignments CSV"); export_csv.clicked.connect(self._export_assignments); export_row.addWidget(export_csv)
        export_row.addStretch(); root.addLayout(export_row)

    def _selected_features(self) -> list[str]:
        """Return the user-approved numeric feature selection."""
        return [self.features.item(i).text() for i in range(self.features.count()) if self.features.item(i).isSelected()]

    def _run(self):
        """Execute the governed clustering comparison and refresh all workspace tabs."""
        try:
            features = self._selected_features()
            algorithms = [name for name, cb in self.alg_checks.items() if cb.isChecked()]
            if not algorithms:
                raise ValueError("Select at least one clustering algorithm.")
            if self.k_max.value() < self.k_min.value():
                raise ValueError("K maximum must be greater than or equal to K minimum.")
            self.result = ClusteringAnalysisEngine.run(
                self.df, features=features, scaling=self.scaling.currentText(), imputation=self.imputation.currentText(),
                algorithms=algorithms, k_min=self.k_min.value(), k_max=self.k_max.value(),
                dbscan_eps=self.eps.value(), dbscan_min_samples=self.min_samples.value(), seed=42,
            )
            candidates = self.result.get("candidates", [])
            rows = []
            for item in candidates:
                m = item.get("metrics", {})
                rows.append({"algorithm": item.get("algorithm"), "parameters": item.get("parameters"), "silhouette": m.get("silhouette"), "calinski_harabasz": m.get("calinski_harabasz"), "davies_bouldin": m.get("davies_bouldin"), "noise_fraction": m.get("noise_fraction"), "inertia": item.get("inertia"), "AIC": item.get("aic"), "BIC": item.get("bic")})
            self.methods_text.setPlainText(json.dumps(rows, indent=2, default=str))
            self.profile_text.setPlainText(json.dumps(self.result.get("cluster_profile", {}), indent=2, default=str))
            selected = self.result.get("selected_method") or {}
            stability = {}
            if selected:
                stability = ClusteringAnalysisEngine.stability_analysis(self.df, self.result.get("features", []), selected.get("algorithm"), selected.get("parameters", {}), self.repeats.value())
            self.validation_text.setPlainText(json.dumps({"selected_method": selected, "stability": stability, "interpretation_note": self.result.get("interpretation_note")}, indent=2, default=str))
            self._draw_pca()
            self.export_text.setPlainText(json.dumps({"preprocessing": self.result.get("preprocessing"), "reproducibility": self.result.get("reproducibility"), "selected_method": selected, "stability": stability}, indent=2, default=str))
            self.tabs.setCurrentIndex(1)
        except Exception as exc:
            QMessageBox.critical(self, "Clustering Analysis Error", str(exc))

    def _draw_pca(self):
        """Render a clean two-component PCA view of the selected cluster solution."""
        self.figure.clear(); ax = self.figure.add_subplot(111)
        if not self.result or not self.result.get("labels"):
            ax.text(.5, .5, "Run clustering analysis to display the PCA projection.", ha="center", va="center"); self.canvas.draw(); return
        projection = ClusteringAnalysisEngine.pca_projection(self.df, self.result["features"], self.result["labels"])
        z = np.asarray(projection["projection"], dtype=float); labels = np.asarray(projection["labels"])
        for cluster in np.unique(labels):
            mask = labels == cluster; ax.scatter(z[mask, 0], z[mask, 1], s=24, alpha=.75, label=f"Cluster {cluster}")
        ax.set_xlabel(f"PC1 ({projection['explained_variance_ratio'][0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({projection['explained_variance_ratio'][1]*100:.1f}% variance)")
        ax.set_title("PCA Projection of Selected Clustering Solution"); ax.legend(); self.figure.tight_layout(); self.canvas.draw()

    def _export_json(self):
        """Export the complete JSON-safe clustering evidence package."""
        if not self.result: return
        path, _ = QFileDialog.getSaveFileName(self, "Export Clustering Analysis", "clustering_analysis.json", "JSON (*.json)")
        if path: Path(path).write_text(json.dumps(self.result, indent=2, default=str), encoding="utf-8")

    def _export_assignments(self):
        """Export the active cluster labels alongside the source dataframe."""
        if not self.result or not self.result.get("labels"): return
        path, _ = QFileDialog.getSaveFileName(self, "Export Cluster Assignments", "cluster_assignments.csv", "CSV (*.csv)")
        if path:
            out = self.df.copy(); out["Cluster"] = self.result["labels"]; out.to_csv(path, index=False)


class UnsupervisedLearningDialog(QDialog):
    """Professional unsupervised-learning workspace covering discovery, reduction and anomaly analysis."""
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent); self.df=df.copy(); self.result=None; self.setWindowTitle("Unsupervised Learning — Professional Workspace"); self.resize(1120,780)
        root=QVBoxLayout(self); tabs=QTabWidget(); root.addWidget(tabs)
        setup=QWidget(); form=QFormLayout(setup); self.features=QListWidget(); self.features.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for c in self.df.select_dtypes(include=[np.number]).columns: it=QListWidgetItem(str(c)); it.setSelected(True); self.features.addItem(it)
        self.features.setMaximumHeight(150); self.imputation=QComboBox(); self.imputation.addItems(["median","mean","most_frequent"]); self.scaling=QComboBox(); self.scaling.addItems(["standard","robust","minmax","none"])
        form.addRow("Numeric features",self.features); form.addRow("Imputation",self.imputation); form.addRow("Scaling",self.scaling)
        setup.addLayout(form); setup.addWidget(QLabel("The workflow compares latent structure, dimensionality reduction, clustering families and anomaly detectors. Interpret unsupervised patterns with domain evidence."))
        run=QPushButton("Run Professional Unsupervised Analysis"); run.clicked.connect(self._run); setup.addWidget(run); tabs.addTab(setup,"1. Setup")
        self.structure=QTextEdit(); self.structure.setReadOnly(True); tabs.addTab(self.structure,"2. Structure Discovery")
        self.validation=QTextEdit(); self.validation.setReadOnly(True); tabs.addTab(self.validation,"3. Validation & Stability")
        self.figure=Figure(figsize=(7,4)); self.canvas=FigureCanvas(self.figure); tabs.addTab(self.canvas,"4. PCA / Latent Structure")
        self.export=QTextEdit(); self.export.setReadOnly(True); tabs.addTab(self.export,"5. Reproducibility")
    def _run(self):
        try:
            features=[self.features.item(i).text() for i in range(self.features.count()) if self.features.item(i).isSelected()]
            self.result=UnsupervisedAnalysisEngine.run(self.df,features,self.imputation.currentText(),self.scaling.currentText())
            self.structure.setPlainText(json.dumps({"features":features,"PCA":self.result["pca"],"cluster_methods":list(self.result["cluster_labels"]),"anomaly_methods":list(self.result["anomaly_labels"])},indent=2,default=str))
            self.validation.setPlainText(json.dumps({"stability_requirement":"repeat across seeds before interpreting clusters","interpretation_note":self.result["interpretation_note"]},indent=2))
            self.figure.clear(); ax=self.figure.add_subplot(111); z=np.asarray(self.result["pca"]["projection"]); ax.scatter(z[:,0],z[:,1],s=20,alpha=.65); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Latent Structure — PCA Projection"); self.figure.tight_layout(); self.canvas.draw()
            self.export.setPlainText(json.dumps(self.result,indent=2,default=str)); tabs=self.findChild(QTabWidget); tabs.setCurrentIndex(1)
        except Exception as exc: QMessageBox.critical(self,"Unsupervised Learning Error",str(exc))


class ReinforcementLearningDialog(QDialog):
    """Professional finite-state RL workspace for Q-learning and SARSA with reproducibility controls."""
    def __init__(self,parent=None):
        super().__init__(parent); self.setWindowTitle("Reinforcement Learning — Professional Workspace"); self.resize(1050,720); self.result={}; root=QVBoxLayout(self); tabs=QTabWidget(); root.addWidget(tabs)
        setup=QWidget(); form=QFormLayout(setup); self.states=QSpinBox(); self.states.setRange(4,500); self.states.setValue(16); self.actions=QSpinBox(); self.actions.setRange(2,20); self.actions.setValue(4); self.episodes=QSpinBox(); self.episodes.setRange(50,10000); self.episodes.setValue(500); self.alpha=QDoubleSpinBox(); self.alpha.setRange(.001,1); self.alpha.setValue(.1); self.gamma=QDoubleSpinBox(); self.gamma.setRange(0,1); self.gamma.setValue(.95); self.alg=QComboBox(); self.alg.addItems(["Q-learning","SARSA"])
        for label,w in (("States",self.states),("Actions",self.actions),("Episodes",self.episodes),("Learning rate α",self.alpha),("Discount γ",self.gamma),("Algorithm",self.alg)): form.addRow(label,w)
        setup.addLayout(form); setup.addWidget(QLabel("Define the state/action/reward semantics before applying RL to a real problem. The built-in finite environment is intentionally bounded for reproducibility and teaching.")); b=QPushButton("Run Reinforcement Learning Experiment"); b.clicked.connect(self._run); setup.addWidget(b); tabs.addTab(setup,"1. Environment & Policy")
        self.results=QTextEdit(); self.results.setReadOnly(True); tabs.addTab(self.results,"2. Learning Results"); self.fig=Figure(figsize=(7,4)); self.cv=FigureCanvas(self.fig); tabs.addTab(self.cv,"3. Learning Curve"); self.governance=QTextEdit(); self.governance.setReadOnly(True); tabs.addTab(self.governance,"4. Evaluation & Reproducibility")
    def _run(self):
        try:
            algo="q_learning" if self.alg.currentText().startswith("Q") else "sarsa"; r=TabularRLEngine.train(self.states.value(),self.actions.value(),self.episodes.value(),self.alpha.value(),self.gamma.value(),algorithm=algo); self.result=r
            self.results.setPlainText(json.dumps({"algorithm":r["algorithm"],"mean_last_50":r["mean_last_50"],"parameters":r["parameters"],"policy":r["policy"]},indent=2)); self.governance.setPlainText(r["interpretation_note"]+"\n\nFor professional RL, add environment-specific off-policy evaluation, safety constraints and multiple seeds.")
            self.fig.clear(); ax=self.fig.add_subplot(111); ax.plot(r["episode_rewards"]); ax.set_xlabel("Episode"); ax.set_ylabel("Return"); ax.set_title(f"{self.alg.currentText()} Learning Curve"); self.fig.tight_layout(); self.cv.draw(); self.findChild(QTabWidget).setCurrentIndex(1)
        except Exception as exc: QMessageBox.critical(self,"Reinforcement Learning Error",str(exc))


class CNNImageAnalysisDialog(QDialog):
    """Professional CNN image-classification workspace with transfer learning and fine-tuning controls."""
    def __init__(self, parent=None):
        """Build the CNN workspace while keeping the main application layout unchanged."""
        super().__init__(parent)
        self.setWindowTitle("CNN Image Analysis — Professional Workspace")
        self.resize(1160, 820)
        self.result = None
        self.image_dataset_path = ""
        root = QVBoxLayout(self)
        tabs = QTabWidget(); root.addWidget(tabs)

        data = QWidget(); form = QFormLayout(data)
        self.root_path = QLineEdit(); browse = QPushButton("Browse Image Dataset…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout(); row.addWidget(self.root_path); row.addWidget(browse); form.addRow("ImageFolder root", row)
        self.arch = QComboBox(); self.arch.addItems(["resnet18", "resnet50", "vgg16", "efficientnet_b0"]); form.addRow("CNN architecture", self.arch)
        self.pretrained = QCheckBox("Use pretrained weights (transfer learning)"); self.pretrained.setChecked(True); form.addRow("Transfer learning", self.pretrained)
        self.finetune = QComboBox(); self.finetune.addItems(["freeze_backbone", "last_block", "full_finetune"]); form.addRow("Fine-tuning policy", self.finetune)
        self.image_size = QSpinBox(); self.image_size.setRange(64, 512); self.image_size.setValue(224); form.addRow("Image size", self.image_size)
        self.batch = QSpinBox(); self.batch.setRange(1, 256); self.batch.setValue(16); form.addRow("Batch size", self.batch)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 500); self.epochs.setValue(10); form.addRow("Maximum epochs", self.epochs)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(6); self.lr.setRange(0.000001, 1.0); self.lr.setValue(0.001); self.lr.setSingleStep(0.0005); form.addRow("Learning rate", self.lr)
        self.weight_decay = QDoubleSpinBox(); self.weight_decay.setDecimals(6); self.weight_decay.setRange(0, 1); self.weight_decay.setValue(0.0001); form.addRow("Weight decay", self.weight_decay)
        self.optimizer = QComboBox(); self.optimizer.addItems(["AdamW", "Adam"]); form.addRow("Optimizer", self.optimizer)
        self.scheduler = QComboBox(); self.scheduler.addItems(["ReduceLROnPlateau", "None"]); form.addRow("LR scheduler", self.scheduler)
        self.val_fraction = QDoubleSpinBox(); self.val_fraction.setRange(.05,.4); self.val_fraction.setValue(.15); form.addRow("Validation fraction", self.val_fraction)
        self.test_fraction = QDoubleSpinBox(); self.test_fraction.setRange(.05,.4); self.test_fraction.setValue(.15); form.addRow("Test fraction", self.test_fraction)
        self.patience = QSpinBox(); self.patience.setRange(1,50); self.patience.setValue(3); form.addRow("Early stopping patience", self.patience)
        self.augment = QCheckBox("Enable training augmentation"); self.augment.setChecked(True); form.addRow("Augmentation", self.augment)
        self.class_weighting = QCheckBox("Use class-balanced sampling"); self.class_weighting.setChecked(True); form.addRow("Imbalance handling", self.class_weighting)
        self.device = QComboBox(); self.device.addItems(["auto", "cpu", "cuda"]); form.addRow("Compute device", self.device)
        run = QPushButton("Run Professional CNN Image Analysis"); run.clicked.connect(self._run); form.addRow(run)
        tabs.addTab(data, "1. Dataset & Training")

        self.dataset_info = QTextEdit(); self.dataset_info.setReadOnly(True); tabs.addTab(self.dataset_info, "2. Dataset & Split")
        self.training = QTextEdit(); self.training.setReadOnly(True); tabs.addTab(self.training, "3. Training & Fine-tuning")
        self.metrics = QTextEdit(); self.metrics.setReadOnly(True); tabs.addTab(self.metrics, "4. Evaluation & Diagnostics")
        explain_page = QWidget(); explain_layout = QVBoxLayout(explain_page)
        explain_controls = QHBoxLayout(); self.explain_index = QSpinBox(); self.explain_index.setRange(0, 0); explain_btn = QPushButton("Generate Grad-CAM for Test Image"); explain_btn.clicked.connect(self._grad_cam); explain_controls.addWidget(QLabel("Test image index:")); explain_controls.addWidget(self.explain_index); explain_controls.addWidget(explain_btn); explain_controls.addStretch(); explain_layout.addLayout(explain_controls)
        self.explain = QTextEdit(); self.explain.setReadOnly(True); self.explain.setMaximumHeight(130); self.explain.setPlainText("Grad-CAM is available after a successful CNN run. The explanation is linked to a specific evaluation image, predicted class, true class and model architecture."); explain_layout.addWidget(self.explain)
        self.explain_figure = Figure(figsize=(8,4)); self.explain_canvas = FigureCanvas(self.explain_figure); explain_layout.addWidget(self.explain_canvas,1); tabs.addTab(explain_page, "5. Explainability")
        self.repro = QTextEdit(); self.repro.setReadOnly(True); tabs.addTab(self.repro, "6. Reproducibility & Export")
        export = QPushButton("Export Experiment Evidence JSON"); export.clicked.connect(self._export); root.addWidget(export)

    def _browse(self):
        """Select an ImageFolder-compatible dataset directory."""
        path = QFileDialog.getExistingDirectory(self, "Select CNN Image Dataset")
        if path:
            self.root_path.setText(path)
            self.image_dataset_path = path
            if self.parent() is not None: setattr(self.parent(), "cnn_image_dataset_path", path)
            try: self.dataset_info.setPlainText(json.dumps(CNNImageAnalysisEngine.scan_image_folder(path), indent=2, default=str))
            except Exception as exc: self.dataset_info.setPlainText(str(exc))

    def _run(self):
        """Run the configured CNN experiment and populate professional evidence tabs."""
        try:
            path = self.root_path.text().strip()
            if not path:
                raise ValueError("Select an ImageFolder-compatible image dataset first.")
            cfg = CNNConfig(architecture=self.arch.currentText(), pretrained=self.pretrained.isChecked(), fine_tune_mode=self.finetune.currentText(), image_size=self.image_size.value(), batch_size=self.batch.value(), epochs=self.epochs.value(), learning_rate=self.lr.value(), weight_decay=self.weight_decay.value(), optimizer=self.optimizer.currentText(), scheduler=self.scheduler.currentText(), validation_fraction=self.val_fraction.value(), test_fraction=self.test_fraction.value(), early_stopping_patience=self.patience.value(), augment=self.augment.isChecked(), class_weighting=self.class_weighting.isChecked())
            self.result = CNNImageAnalysisEngine.run(path, cfg, self.device.currentText())
            self.dataset_info.setPlainText(json.dumps({"dataset":self.result["dataset"],"train_images":self.result["train_images"],"validation_images":self.result["validation_images"],"test_images":self.result["test_images"],"class_counts_train":self.result["class_counts_train"]},indent=2,default=str))
            self.training.setPlainText(json.dumps({"config":self.result["config"],"device":self.result["device"],"trainable_parameters":self.result["model_parameters_trainable"],"total_parameters":self.result["model_parameters_total"],"history":self.result["history"]},indent=2,default=str))
            self.metrics.setPlainText(json.dumps(self.result["metrics"],indent=2,default=str))
            self.repro.setPlainText(json.dumps({"task":self.result["task"],"dataset":self.result["dataset"],"config":self.result["config"],"device":self.result["device"],"seed":cfg.seed,"class_names":self.result["classes"]},indent=2,default=str))
            self.explain_index.setRange(0, max(0, int(self.result.get("test_images", 1))-1))
            tabs = self.findChild(QTabWidget); tabs.setCurrentIndex(3)
        except Exception as exc:
            QMessageBox.critical(self, "CNN Image Analysis Error", str(exc))

    def _grad_cam(self):
        """Generate and display a Grad-CAM explanation for a selected test image."""
        if not self.result or not self.result.get("model"):
            QMessageBox.warning(self, "Explainability", "Run a CNN experiment first."); return
        try:
            from PIL import Image
            result = CNNImageAnalysisEngine.grad_cam(self.result["model"], self.result["eval_dataset"], self.explain_index.value(), self.result["config"]["architecture"], self.result["device"])
            arr = np.asarray(result["input_tensor"]).transpose(1,2,0); arr = (arr - arr.min()) / (arr.max()-arr.min()+1e-8)
            self.explain_figure.clear(); ax=self.explain_figure.add_subplot(111); ax.imshow(arr); heat=Image.fromarray((result["heatmap"]*255).astype(np.uint8)).resize((arr.shape[1],arr.shape[0]))
            ax.imshow(np.asarray(heat)/255.0, cmap="jet", alpha=.42, extent=(0,arr.shape[1],arr.shape[0],0)); ax.set_title(f"Grad-CAM — predicted={self.result['classes'][result['predicted_class']]} | true={self.result['classes'][result['true_class']]} | confidence={result['score']:.3f}"); ax.axis("off"); self.explain_figure.tight_layout(); self.explain_canvas.draw()
            self.explain.setPlainText(json.dumps({"method":"Grad-CAM","test_image_index":self.explain_index.value(),"predicted_class":self.result["classes"][result["predicted_class"]],"true_class":self.result["classes"][result["true_class"]],"confidence":result["score"],"interpretation":"Highlighted regions identify image areas that contributed most strongly to the selected predicted class; they are explanatory diagnostics, not causal proof."},indent=2))
        except Exception as exc:
            QMessageBox.critical(self, "Grad-CAM Error", str(exc))

    def _export(self):
        """Export the current CNN experiment as evidence JSON."""
        if not self.result:
            QMessageBox.warning(self, "CNN Image Analysis", "Run an experiment first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Export CNN Experiment Evidence", "cnn_image_analysis.json", "JSON (*.json)")
        if path:
            CNNImageAnalysisEngine.export_evidence(self.result, path)
            QMessageBox.information(self, "Export Complete", f"Experiment evidence saved to {path}")


class MacroEditorDialog(QDialog):
    """Professional multi-run Python/SQL workspace; execution output stays on the same page."""
    run_requested = pyqtSignal(str)
    def __init__(self,macro_type,parent=None):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        super().__init__(parent); self.setWindowTitle(f"{macro_type} Macro Editor"); self.resize(1000,700); self.macro_type=macro_type
        root=QVBoxLayout(self); head=QHBoxLayout(); head.addWidget(QLabel(f"{macro_type} Macro Workspace — 100 professional starter commands"))
        self.library_combo=QComboBox(); self.library_combo.addItem("Choose a starter command…"); self._macros=PYTHON_MACROS if macro_type=="Python" else SQL_MACROS; self.library_combo.addItems([t for t,_ in self._macros]); self.library_combo.currentIndexChanged.connect(self._load_macro); head.addWidget(self.library_combo); root.addLayout(head)
        split=QSplitter(Qt.Orientation.Vertical); root.addWidget(split,1); self.editor=QTextEdit(); self.editor.setFont(QFont("Courier",10)); self.editor.setPlaceholderText("Python: use df, pd, np, plt. SQL: query table 'data'."); split.addWidget(self.editor)
        out=QWidget(); ol=QVBoxLayout(out); ol.addWidget(QLabel("Execution Output / History")); self.output=QTextEdit(); self.output.setReadOnly(True); self.output.setFont(QFont("Courier",10)); ol.addWidget(self.output); split.addWidget(out); split.setSizes([360,260])
        buttons=QHBoxLayout(); run=QPushButton("▶ Run All Commands"); run.clicked.connect(lambda:self.run_requested.emit(self.editor.toPlainText())); buttons.addWidget(run); run_sel=QPushButton("Run Selection"); run_sel.clicked.connect(lambda:self.run_requested.emit(self.editor.textCursor().selectedText())); buttons.addWidget(run_sel); clear=QPushButton("Clear Output"); clear.clicked.connect(self.output.clear); buttons.addWidget(clear); buttons.addStretch(); close=QPushButton("Close"); close.clicked.connect(self.reject); buttons.addWidget(close); root.addLayout(buttons)
    def _load_macro(self,index):
        """Perform the load macro operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if index>0:self.editor.setPlainText(self._macros[index-1][1])
    def append_output(self,text):
        """Perform the append output operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.output.append(str(text)); self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())


class AgentPlotEditorDialog(QDialog):
    """Professional per-plot preview/editor used by AI Agent Plot."""
    def __init__(self, plots, parent=None):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        super().__init__(parent); self.setWindowTitle("AI Agent Plot — Professional Visualization Review"); self.resize(1180,760); self.plots=plots or []; self.figures=[]
        root=QVBoxLayout(self); root.addWidget(QLabel("Each tab contains the evidence-linked plot selected by the Plot Agent. Edit presentation properties without changing the analytical fields."))
        tabs=QTabWidget(); root.addWidget(tabs,1); self.tabs=tabs
        for i,p in enumerate(self.plots):
            page=QWidget(); lay=QVBoxLayout(page); controls=QHBoxLayout(); title_size=QSpinBox(); title_size.setRange(8,40); title_size.setValue(18); axis_size=QSpinBox(); axis_size.setRange(7,28); axis_size.setValue(12); title_color=QPushButton("Title Color"); bg_color=QPushButton("Plot Background"); apply=QPushButton("Apply Format"); save_png=QPushButton("PNG"); save_jpg=QPushButton("JPG"); save_pdf=QPushButton("PDF")
            controls.addWidget(QLabel("Title size")); controls.addWidget(title_size); controls.addWidget(QLabel("Axis size")); controls.addWidget(axis_size); controls.addWidget(title_color); controls.addWidget(bg_color); controls.addWidget(apply); controls.addWidget(save_png); controls.addWidget(save_jpg); controls.addWidget(save_pdf); controls.addStretch(); lay.addLayout(controls)
            info=QTextEdit(); info.setReadOnly(True); info.setMaximumHeight(125); info.setPlainText(f"{p.get('title','Plot')}\nQuantity: {p.get('quantity','')}\nFields: {', '.join(map(str,p.get('fields',[])))}\nWhy: {p.get('rationale','')}\nFindings: {'; '.join(map(str,p.get('insights',[])))}"); lay.addWidget(info)
            image=QLabel("Rendering plot…"); image.setAlignment(Qt.AlignmentFlag.AlignCenter); image.setMinimumHeight(420); lay.addWidget(image,1); tabs.addTab(page,p.get("title",f"Plot {i+1}"))
            self.figures.append({"plot":p,"image":image,"title_size":title_size,"axis_size":axis_size,"title_color":QColor("#000000"),"bg_color":QColor("#ffffff")})
            title_color.clicked.connect(lambda _=False,idx=i:self._pick_color(idx,"title_color")); bg_color.clicked.connect(lambda _=False,idx=i:self._pick_color(idx,"bg_color")); apply.clicked.connect(lambda _=False,idx=i:self._render(idx)); save_png.clicked.connect(lambda _=False,idx=i:self._save(idx,"png")); save_jpg.clicked.connect(lambda _=False,idx=i:self._save(idx,"jpg")); save_pdf.clicked.connect(lambda _=False,idx=i:self._save(idx,"pdf"))
        buttons=QHBoxLayout(); save_all=QPushButton("Save All Interactive Plots"); close=QPushButton("Close"); buttons.addWidget(save_all); buttons.addStretch(); buttons.addWidget(close); root.addLayout(buttons); save_all.clicked.connect(self._save_all_html); close.clicked.connect(self.accept)
        for i in range(len(self.figures)): self._render(i)

    def _pick_color(self,idx,key):
        """Perform the pick color operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        c=QColorDialog.getColor(self.figures[idx][key],self);
        if c.isValid(): self.figures[idx][key]=c; self._render(idx)

    def _make_figure(self,idx):
        """Perform the make figure operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        import plotly.io as pio
        fig=pio.from_json(self.figures[idx]["plot"].get("figure_json"))
        item=self.figures[idx]; fig.update_layout(font=dict(size=item["axis_size"].value()),title=dict(font=dict(size=item["title_size"].value(),color=item["title_color"].name())),plot_bgcolor=item["bg_color"].name(),paper_bgcolor="#ffffff")
        return fig

    def _render(self,idx):
        """Perform the render operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        item=self.figures[idx]
        try:
            import plotly.io as pio
            fig=self._make_figure(idx); item["current_figure"]=fig; raw=pio.to_image(fig,format="png",width=1200,height=650,scale=1); from PyQt6.QtGui import QPixmap; pix=QPixmap(); pix.loadFromData(raw,"PNG"); item["image"].setPixmap(pix.scaled(item["image"].size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)); item["plot"]["figure_json"]=fig.to_json()
        except Exception as exc: item["image"].setText("Static preview unavailable. Install/verify kaleido for plot image rendering.\n\n"+str(exc))

    def resizeEvent(self,event):
        """Perform the resize event operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        super().resizeEvent(event)
        for i in range(len(self.figures)): self._render(i)

    def _save(self,idx,fmt):
        """Perform the save operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        try:
            fig=self._make_figure(idx); path,_=QFileDialog.getSaveFileName(self,"Save Plot",f"{self.figures[idx]['plot'].get('title','plot')}.{fmt}",f"{fmt.upper()} Files (*.{fmt})")
            if path: fig.write_image(path,format=fmt,width=1400,height=800,scale=1)
        except Exception as exc: QMessageBox.critical(self,"Plot Export",str(exc))

    def _save_all_html(self):
        """Perform the save all html operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        path,_=QFileDialog.getSaveFileName(self,"Save AI Agent Plots","AI_Agent_Plots.html","HTML (*.html)")
        if not path:return
        try:
            import plotly.io as pio
            parts=["<html><head><title>AI Agent Plot</title></head><body>"]
            for item in self.figures:
                fig=self._make_figure(self.figures.index(item)); parts.append(f"<h2>{item['plot'].get('title','Plot')}</h2>"); parts.append(pio.to_html(fig,include_plotlyjs="cdn",full_html=False))
            parts.append("</body></html>"); Path(path).write_text("\n".join(parts),encoding="utf-8")
        except Exception as exc: QMessageBox.critical(self,"Plot Export",str(exc))


class PDFTableSelectionDialog(QDialog):
    def __init__(self,tables,parent=None):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the set all operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        for i in range(self.list.count()): self.list.item(i).setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)
    def _preview(self,item):
        """Perform the preview operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if item is None:return
        idx=item.data(Qt.ItemDataRole.UserRole); df=self.tables[idx]['data']; self.preview.setPlainText(df.head(30).to_string(index=False))
    def selected_indices(self):
        """Perform the selected indices operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count()) if self.list.item(i).checkState()==Qt.CheckState.Checked]


class GroupByDialog(QDialog):
    """V16.py Group By Dialog."""
    def __init__(self, df, parent=None):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the update preview operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the from dict operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return {"evidence_id": f"gate-{abs(hash(title + str(data))) % 10**10}", "kind": "governance_gate",
                "title": title, "status": data.get("status", "complete"), "data": data, "parent_ids": []}



def load_manifest_from_text(text: str) -> dict:
    """Perform the load manifest from text operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    data = json.loads(text)
    if data.get("format") != "dssp-project-manifest":
        raise ValueError("This file is not a valid Data Science Studio Pro project manifest.")
    return data

class StoryCardItem(QGraphicsRectItem):
    """Movable/resizable story sheet card for the Tableau-like Story workspace."""
    def __init__(self, rect, pixmap, title, parent=None):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        super().__init__(rect,parent); self.sheet_name=title; self._resizing=False; self._press=QPointF(); self.setFlags(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable|QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable); self.setAcceptHoverEvents(True)
        from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsTextItem
        self.image_item=QGraphicsPixmapItem(pixmap,self); self.image_item.setPos(8,28); self.image_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.text_item=QGraphicsTextItem(title,self); self.text_item.setDefaultTextColor(QColor("#17324d")); self.text_item.setPos(8,5); self.text_item.setFont(QFont("Arial",11,QFont.Weight.Bold))
        self._sync_image()
    def _sync_image(self):
        """Perform the sync image operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        r=self.rect(); available_w=max(40,r.width()-16); available_h=max(40,r.height()-42); pm=self.image_item.pixmap();
        if not pm.isNull(): self.image_item.setPixmap(pm.scaled(int(available_w),int(available_h),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
    def mousePressEvent(self,event):
        """Perform the mouse press event operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        r=self.rect(); p=event.pos(); self._resizing=(p.x()>r.width()-18 and p.y()>r.height()-18); self._press=p; super().mousePressEvent(event)
    def mouseMoveEvent(self,event):
        """Perform the mouse move event operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self._resizing:
            p=event.pos(); w=max(180,p.x()); h=max(130,p.y()); self.setRect(self.rect().x(),self.rect().y(),w,h); self._sync_image(); event.accept(); return
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self,event):
        """Perform the mouse release event operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._resizing=False; super().mouseReleaseEvent(event)
    def paint(self,painter,option,widget=None):
        """Perform the paint operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        painter.setBrush(QColor("#ffffff")); painter.setPen(QColor("#8aa4b8")); painter.drawRoundedRect(self.rect(),8,8); super().paint(painter,option,widget)


class StoryCanvasView(QGraphicsView):
    def __init__(self,parent=None):
        # QGraphicsView must be constructed before creating a QGraphicsScene
        # whose parent is this view. Creating the scene first leaves the Qt
        # C++ object uninitialised and causes:
        # "super-class __init__() of type StoryCanvasView was never called".
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        super().__init__(parent)
        scene=QGraphicsScene(self)
        self.setScene(scene)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setBackgroundBrush(QColor("#f3f6f8"))
        self.setMinimumHeight(430)
    def clear_cards(self):
        """Perform the clear cards operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        for item in list(self.scene().items()):
            if isinstance(item,StoryCardItem) or item.data(0)=="story_title": self.scene().removeItem(item)
    def add_title(self,title):
        """Perform the add title operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        from PyQt6.QtWidgets import QGraphicsTextItem
        t=QGraphicsTextItem(title); t.setData(0,"story_title"); t.setDefaultTextColor(QColor("#0d47a1")); t.setFont(QFont("Arial",20,QFont.Weight.Bold)); t.setPos(20,15); self.scene().addItem(t)
    def set_background(self,color):
        """Perform the set background operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.setBackgroundBrush(color)
    def export_png(self,path):
        """Perform the export png operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        rect=self.scene().itemsBoundingRect().adjusted(-20,-20,20,20); image=QImage(int(rect.width()),int(rect.height()),QImage.Format.Format_ARGB32); image.fill(self.backgroundBrush().color()); from PyQt6.QtGui import QPainter; painter=QPainter(image); self.scene().render(painter,QRectF(0,0,rect.width(),rect.height()),rect); painter.end(); image.save(path)


class MainWindow(QMainWindow):
    def __init__(self):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        super().__init__()
        self.setWindowTitle("Data Science Studio Pro")
        self.resize(1600, 1000)

        # --- Core Components ---
        self.data_engine = DataEngine()
        self.viz_engine = VizEngine()
        self.sheet_manager = SheetManager()
        # Loaded Sheets are dataset sources; analytical workbook Sheets remain in SheetManager.
        self.loaded_datasets: dict[str, dict[str, Any]] = {}
        self.active_dataset_name: str | None = None
        self.theme_manager = ThemeManager(self)

        # --- AI Agent State ---
        # One canonical provider configuration is kept outside agent execution state.
        # This prevents LangGraph's evolving state from accidentally erasing the
        # provider selected in the AI Provider dialog.
        self.llm_config = LLMConfig()
        self.agent_provider_registry = AgentProviderRegistry()
        self.cnn_image_dataset_path = ""
        self.agent_ds_state = {
            "messages": [], "approved_steps": [], "rejected_steps": [],
            "user_approved": False, "dataframe": None,
            "api_key": "", "model_name": "", "llm_provider": "none", "local_path": "", "api_base": "",
            "max_steps": 20, "abort_requested": False, "human_approval_evidence": []
        }
        self.agent_report_state = {
            "analysis_results": "", "pdf_buffer": None,
            "api_key": "", "model_name": "", "llm_provider": "none", "local_path": "", "api_base": ""
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
        self.shelf_aggregations = {"Rows": {}, "Columns": {}}
        self.story_workspace_state = {"sheets": [], "stories": []}
        self.table_agent_memory = []
        self.presentation_agent_memory = []
        self.plot_agent_memory = []
        self.last_share_package_path = ""
        self.last_presentation_path = ""
        self.compute_mode = "CPU"
        self.compute_info = ComputeBackend.detect()
        self._recorded_agent_evidence_ids = set()
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
            combo = FieldDropComboBox(multi=(mark_name != "Size"))
            combo.setMinimumWidth(100)
            combo.setToolTip(f"Marks → {mark_name}: drag a field here. Multiple fields are supported for Color/Text/Detail/Tooltip; Size accepts one field. Hold Shift while dragging to add to Color instead of replacing it.")
            combo.fieldDropped.connect(lambda field, mn=mark_name: self.assign_mark_field(mn, field))
            combo.fieldsChanged.connect(lambda _fields, _mn=mark_name: self._mark_fields_changed(_mn))
            combo.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            combo.customContextMenuRequested.connect(lambda pos, mn=mark_name: self.show_mark_context_menu(pos, mn))
            self.marks_layout.addWidget(combo)
            self.marks_widgets[mark_name] = combo

        self.marks_layout.addWidget(QLabel("📊 Chart Type:"))
        self.chart_combo = QComboBox()
        self.chart_combo.addItems(["auto", "horizontal_bar", "vertical_bar", "stacked_bar", "line_discrete", "line_continuous", "area", "dual_axis", "scatter", "histogram", "box", "density", "symbol_map", "filled_map", "gantt", "bullet", "heatmap", "highlight_table", "waterfall", "pareto", "pie", "donut", "bump", "sankey", "treemap", "crosstab"])
        self.chart_combo.currentTextChanged.connect(lambda _text: self.update_plot())
        self.marks_layout.addWidget(self.chart_combo)
        self.layout.addLayout(self.marks_layout)

        # 3. Central Plot Canvas
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.show_plot_context_menu)
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
        """Perform the setup shortcuts operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the setup right panels operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.sheets_dock = QDockWidget("Loaded Sheets", self)
        self.sheets_list = QListWidget()
        self.sheets_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.sheets_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.sheets_list.itemClicked.connect(self.on_loaded_sheet_select)
        self.sheets_list.customContextMenuRequested.connect(self.loaded_sheets_context_menu)
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
        """Perform the show mark context menu operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        combo = self.marks_widgets.get(mark_name)
        if combo is None:
            return
        menu = QMenu(self)
        menu.addAction("Clear", lambda: (combo.clear_selection(), self._record_view_change(), self.update_plot()))
        if mark_name in {"Color", "Size", "Text", "Detail", "Tooltip"}:
            agg_menu = menu.addMenu("Aggregation")
            for label, value in [("Sum", "sum"), ("Average", "mean"), ("Count", "count"), ("Min", "min"), ("Max", "max")]:
                agg_menu.addAction(label, lambda v=value, mn=mark_name: self._set_mark_aggregation(mn, v))
        menu.exec(combo.mapToGlobal(pos))

    # ============================================================
    # DRAG AND DROP — Tableau-style shelves/marks/filters/insight
    # ============================================================
    def shelf_drag_enter(self, event):
        """Perform the shelf drag enter operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if event.mimeData().hasText() and event.mimeData().text().strip():
            event.acceptProposedAction()
        else:
            event.ignore()

    def _dropped_column(self, event):
        """Perform the dropped column operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return event.mimeData().text().strip() if event.mimeData().hasText() else ""

    def _append_shelf(self, widget, col):
        """Perform the append shelf operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if not col or self.data_engine.df is None or col not in self.data_engine.df.columns:
            return False
        fields = self.viz_engine.parse_shelf(widget.text())
        if col not in fields:
            fields.append(col)
            widget.setText(self.viz_engine.format_shelf(fields))
        return True

    def shelf_context_menu(self, pos, shelf_name):
        """Perform the shelf context menu operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        widget = self.rows_input if shelf_name == "Rows" else self.cols_input
        fields = self.viz_engine.parse_shelf(widget.text())
        menu = QMenu(self)
        if not fields:
            menu.addAction("No fields in shelf").setEnabled(False)
        else:
            for field in fields:
                field_menu = menu.addMenu(field)
                agg_menu = field_menu.addMenu("Aggregation")
                options = [("Automatic", None), ("Sum", "sum"), ("Average", "mean"), ("Count", "count"), ("Minimum", "min"), ("Maximum", "max"), ("Median", "median"), ("Count Distinct", "nunique")]
                for label, value in options:
                    agg_menu.addAction(label, lambda v=value, sn=shelf_name, f=field: self._set_shelf_aggregation(sn, f, v))
                field_menu.addAction("Remove", lambda f=field, w=widget: self._remove_shelf_field(w, f))
                if pd.api.types.is_numeric_dtype(self.data_engine.df[field]):
                    current = self.shelf_aggregations.get(shelf_name, {}).get(field) or "Automatic"
                    field_menu.setToolTip(f"Current aggregation: {current}")
            menu.addSeparator()
            menu.addAction("Clear shelf", lambda w=widget: self._clear_shelf(w, shelf_name))
        menu.exec(widget.mapToGlobal(pos))

    def _set_shelf_aggregation(self, shelf_name, field, value):
        """Perform the set shelf aggregation operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.shelf_aggregations.setdefault(shelf_name, {})[field] = value
        self._record_view_change(); self.update_plot()
        self.status.showMessage(f"{shelf_name} → {field}: {value or 'Automatic'}")

    def _clear_shelf(self, widget, shelf_name):
        """Perform the clear shelf operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        widget.clear(); self.shelf_aggregations[shelf_name] = {}; self._record_view_change(); self.update_plot()

    def _remove_shelf_field(self, widget, field):
        """Perform the remove shelf field operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        widget.setText(self.viz_engine.remove_from_shelf(widget.text(), field))
        self._record_view_change()
        self.update_plot()

    def _record_view_change(self):
        """Perform the record view change operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.history.append({"rows": self.rows_input.text(), "columns": self.cols_input.text(),
                             "chart": self.chart_combo.currentText(),
                             "marks": {k: v.currentText() for k,v in self.marks_widgets.items()}})
        self.redo_history.clear()

    def row_drop(self, event):
        """Perform the row drop operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        col = self._dropped_column(event)
        if self._append_shelf(self.rows_input, col):
            self.update_plot()
        event.acceptProposedAction()

    def col_drop(self, event):
        """Perform the col drop operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        col = self._dropped_column(event)
        if self._append_shelf(self.cols_input, col):
            self.update_plot()
        event.acceptProposedAction()

    def _set_mark_aggregation(self, mark_name, value):
        """Perform the set mark aggregation operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if not hasattr(self, "mark_aggregations"): self.mark_aggregations={}
        self.mark_aggregations[mark_name]=value
        self.update_plot()

    def _mark_fields_changed(self, mark_name):
        """Perform the mark fields changed operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._record_view_change(); self.status.showMessage(f"Marks → {mark_name}: {', '.join(self.marks_widgets[mark_name].selected_fields()) or 'none'}"); self.update_plot()

    def assign_mark_field(self, mark_name, col):
        """Perform the assign mark field operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        combo=self.marks_widgets.get(mark_name); df=self.data_engine.df
        if combo is None or df is None or col not in df.columns: return False
        current=combo.selected_fields(); add=False
        # A Shift-modified drag is handled inside the widget. A normal drag replaces
        # the field for Size and replaces the Color/Text/Detail/Tooltip set unless the
        # target widget already contains that field. This mirrors Tableau's default
        # replace behavior while permitting explicit multi-field addition.
        if combo.multi and col in current: return True
        combo.set_selected_fields(current+[col] if combo.multi and current and col in current else [col])
        self._record_view_change(); self.status.showMessage(f"{col} assigned to Marks → {mark_name}."); self.update_plot(); return True

    def mark_drop(self, event, mark_name):
        """Perform the mark drop operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        col = self._dropped_column(event)
        if col: self.assign_mark_field(mark_name, col)
        event.acceptProposedAction()

    def filter_drop(self, event):
        """Perform the filter drop operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        col = self._dropped_column(event)
        source_df = self.data_engine.original_df if self.data_engine.original_df is not None else self.data_engine.df
        if col and source_df is not None and col in source_df.columns:
            self._add_filter_v16(col)
            event.setDropAction(Qt.DropAction.CopyAction); event.accept()
        else:
            event.ignore()

    def insight_drop(self, event):
        """Perform the insight drop operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        col = self._dropped_column(event)
        if col and self.data_engine.df is not None and col in self.data_engine.df.columns:
            self.current_extra_insight_col = col
            self.show_extra_insight(col)
        event.acceptProposedAction()

    # ============================================================
    # FILTERS — persistent, source-based Tableau-style filters
    # ============================================================
    def _add_filter_v16(self, col, edit_existing=False):
        """Perform the add filter v16 operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the refresh filter list operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the edit filter from list operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        col = item.data(Qt.ItemDataRole.UserRole)
        if col: self._add_filter_v16(col, edit_existing=True)

    def _filter_context_menu(self, pos):
        """Perform the filter context menu operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the show extra insight operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
            """Perform the pick operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            c = QColorDialog.getColor();
            if c.isValid():
                current[target] = c.name(); (color_btn if target == "color" else hover_btn).setStyleSheet(f"background-color:{c.name()}; color:white;"); update_plot()

        color_btn.clicked.connect(lambda: pick("color")); hover_btn.clicked.connect(lambda: pick("hover"))

        def update_plot():
            """Perform the update plot operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
            """Perform the on move operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the save figure operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        filepath, _ = QFileDialog.getSaveFileName(dialog, "Save Plot", "", "PNG Files (*.png)")
        if filepath:
            fig.savefig(filepath, dpi=180, bbox_inches="tight"); QMessageBox.information(dialog, "Saved", f"Saved: {filepath}")

    # ============================================================
    # MENUS
    # ============================================================
    def setup_menus(self):
        """Perform the setup menus operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        prep_menu.addAction("Data Scaling", self.data_scaling)
        prep_menu.addAction("Data Splitting", self.data_splitting)

        analysis_menu = menubar.addMenu("Data Analysis")
        analysis_menu.addAction("Descriptive Statistics", lambda: self.show_analysis("Descriptive Statistics"))
        analysis_menu.addAction("Correlation Matrix", lambda: self.show_analysis("Correlation Matrix"))
        analysis_menu.addAction("Time Series Analysis", lambda: self.show_analysis("Time Series Analysis"))
        analysis_menu.addAction("Classification Analysis", self.show_classification_analysis)
        analysis_menu.addAction("Regression Analysis", self.show_regression_analysis)
        analysis_menu.addAction("Clustering Analysis", self.show_clustering_analysis)
        analysis_menu.addAction("Unsupervised Learning", self.show_unsupervised_learning)
        analysis_menu.addAction("Reinforcement Learning", self.show_reinforcement_learning)
        analysis_menu.addAction("CNN Image Analysis", self.show_cnn_image_analysis)

        ai_menu = menubar.addMenu("AI Agents")
        ai_menu.addAction("Agent Data Scientist", self.run_agent_data_scientist)
        ai_menu.addAction("AI Agent Plot", self.run_ai_agent_plot)
        ai_menu.addAction("AI Agent Report", self.run_ai_report)
        ai_menu.addAction("AI Agent Table Creation", self.run_ai_agent_table_creation)
        ai_menu.addAction("AI Agent Presentation", self.run_ai_agent_presentation)
        ai_menu.addAction("AI Agent Question → Analysis", self.run_ai_agent_question_analysis)
        ai_menu.addAction("Agent Run Console", self.show_agent_run_console)
        ai_menu.addSeparator()
        ai_menu.addAction("Select Local LLM model or API key", self.select_local_llm)

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
        help_menu.addAction("Help Center", self.show_help_center)
        help_menu.addAction("Quick Start Guide", lambda: self.show_help_topic("quickstart"))
        help_menu.addAction("Using Rows, Columns and Marks", lambda: self.show_help_topic("shelves"))
        help_menu.addAction("Filters and Drag-and-Drop", lambda: self.show_help_topic("filters"))
        help_menu.addAction("Classification Analysis", lambda: self.show_help_topic("classification"))
        help_menu.addAction("Regression Analysis", lambda: self.show_help_topic("regression"))
        help_menu.addAction("Clustering Analysis", lambda: self.show_help_topic("clustering"))
        help_menu.addAction("Sheets, Dashboards and Stories", lambda: self.show_help_topic("views"))
        help_menu.addAction("AI Agents and Human Approval", lambda: self.show_help_topic("agents"))
        help_menu.addAction("AI Provider Setup (LLM / API Key)", lambda: self.show_help_topic("provider"))
        help_menu.addAction("Macro Python / SQL Workspace", lambda: self.show_help_topic("macros"))
        help_menu.addAction("Report and PowerPoint Presentation", lambda: self.show_help_topic("report_presentation"))
        help_menu.addAction("Voice and Accessibility", self.show_voice_help)
        help_menu.addAction("Compute / CPU / GPU Troubleshooting", lambda: self.show_help_topic("compute"))
        help_menu.addAction("Chart Types and Hierarchies", lambda: self.show_help_topic("charts"))
        help_menu.addAction("Troubleshooting", lambda: self.show_help_topic("troubleshooting"))
        help_menu.addSeparator()
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
        """Perform the setup toolbar operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        self.undo_action = QAction(QIcon(str(Path(__file__).parent / "assets" / "undo.svg")), "Undo", self); self.undo_action.triggered.connect(self.undo); toolbar.addAction(self.undo_action)
        self.redo_action = QAction(QIcon(str(Path(__file__).parent / "assets" / "redo.svg")), "Redo", self); self.redo_action.triggered.connect(self.redo); toolbar.addAction(self.redo_action)
        self.sort_asc_action = QAction(QIcon(str(Path(__file__).parent / "assets" / "sort_asc.svg")), "Sort Y-Axis Asc", self); self.sort_asc_action.triggered.connect(self.sort_asc); toolbar.addAction(self.sort_asc_action)
        self.sort_desc_action = QAction(QIcon(str(Path(__file__).parent / "assets" / "sort_desc.svg")), "Sort Y-Axis Desc", self); self.sort_desc_action.triggered.connect(self.sort_desc); toolbar.addAction(self.sort_desc_action)
        toolbar.addAction("Story", self.create_story)
        toolbar.addAction("Sheets", self.manage_sheets)
        toolbar.addWidget(QLabel(" Sheet: "))
        self.sheet_nav_combo = QComboBox(); self.sheet_nav_combo.setMinimumWidth(150); self.sheet_nav_combo.currentTextChanged.connect(self._navigate_sheet_from_toolbar); toolbar.addWidget(self.sheet_nav_combo)
        toolbar.addWidget(QLabel(" Story: "))
        self.story_nav_combo = QComboBox(); self.story_nav_combo.setMinimumWidth(150); self.story_nav_combo.currentTextChanged.connect(self._navigate_story_from_toolbar); toolbar.addWidget(self.story_nav_combo)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Compute: "))
        self.compute_combo = QComboBox(); self.compute_combo.addItems(["CPU","CPU+GPU","GPU"]); self.compute_combo.setCurrentText(self.compute_mode); self.compute_combo.setToolTip("Choose execution policy. GPU requires a CUDA-capable PyTorch installation."); self.compute_combo.currentTextChanged.connect(self.on_compute_mode_changed); toolbar.addWidget(self.compute_combo)
        toolbar.addAction("Dark Mode", self.toggle_dark_mode)
        self._refresh_navigation_combos()

    def toggle_dark_mode(self):
        """Perform the toggle dark mode operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.theme_manager.dark_mode:
            self.theme_manager.apply_theme("#e3f2fd", dark=False)
        else:
            self.theme_manager.apply_theme("#2b2b2b", dark=True)

    # ============================================================
    # THEME CONTEXT MENU
    # ============================================================
    def show_context_menu(self, pos):
        """Perform the show context menu operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        menu = QMenu(self)
        change_color = QAction("Change Theme Color", self)
        change_color.triggered.connect(self.open_color_picker)
        menu.addAction(change_color)
        menu.exec(self.mapToGlobal(pos))

    def open_color_picker(self):
        """Perform the open color picker operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.theme_manager.apply_theme(color.name())

    def on_compute_mode_changed(self, mode):
        """Perform the on compute mode changed operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        info = ComputeBackend.resolve(mode, runtime_validate=True)
        # CPU+GPU is an opportunistic policy: keep the user's requested policy even
        # when the current runtime must execute on CPU. GPU is a strict request and
        # therefore returns to CPU when CUDA cannot be safely validated.
        if mode == "CPU+GPU":
            self.compute_mode = mode
        elif mode == "GPU" and not info.gpu_available:
            self.compute_mode = "CPU"
            self.compute_combo.blockSignals(True); self.compute_combo.setCurrentText("CPU"); self.compute_combo.blockSignals(False)
        else:
            self.compute_mode = mode
        self.compute_info = info
        self.status.showMessage(info.message)
        if mode in {"GPU", "CPU+GPU"} and not info.gpu_available:
            hardware = "NVIDIA hardware was detected." if info.hardware_gpu_detected else "No NVIDIA GPU was detected through nvidia-smi."
            details = "\n".join(info.diagnostics[-8:]) if info.diagnostics else "No additional diagnostics were returned."
            vram = f"{info.vram_total_gb:.2f} GB" if isinstance(info.vram_total_gb,(int,float)) else "unknown"
            QMessageBox.warning(self, "GPU Execution Unavailable",
                f"{hardware}\n\n{info.message}\n\n"
                f"PyTorch: {info.torch_version or 'not available'} | CUDA build: {info.cuda_version or 'CPU-only/unknown'}\n"
                f"GPU: {info.gpu_name or 'unknown'} | VRAM: {vram}\n\nDiagnostics:\n{details}\n\n"
                "The application will not force unsafe CUDA execution. CPU+GPU remains opportunistic and falls back to CPU safely.")
    def run_ai_agent_question_analysis(self):
        """Perform the run ai agent question analysis operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the show agent run console operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._show_text_dialog("Agent Run Console",self.agent_console.to_text() or "No agent events recorded yet.")
    def show_agent_tool_registry(self):
        """Perform the show agent tool registry operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._show_json_dialog("Typed Agent Tool Registry",self.agent_tool_registry.describe())
    def show_analysis_recipe(self):
        """Perform the show analysis recipe operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._show_json_dialog("Analysis Recipe",self.analysis_recipe.to_dict())
    def show_analysis_state(self):
        """Perform the show analysis state operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._show_json_dialog("Analysis State Machine",self.analysis_state.to_dict())
    def show_artifact_registry(self):
        """Perform the show artifact registry operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._show_json_dialog("Artifact Registry / Lineage",self.artifact_store.to_dict())
    # ============================================================
    # FILE OPERATIONS
    # ============================================================
    def _unique_loaded_dataset_name(self, base_name: str) -> str:
        """Return a unique user-facing name for a dataset in Loaded Sheets."""
        base = str(base_name).strip() or "Dataset"
        if base not in self.loaded_datasets:
            return base
        index = 2
        while f"{base} ({index})" in self.loaded_datasets:
            index += 1
        return f"{base} ({index})"

    def _register_loaded_dataset(self, name: str, dataframe: pd.DataFrame, metadata: dict[str, Any] | None = None, activate: bool = True) -> str:
        """Register a dataframe as a managed Loaded Sheets dataset and optionally activate it."""
        unique = self._unique_loaded_dataset_name(name)
        self.loaded_datasets[unique] = {
            "data": dataframe.copy(),
            "source": (metadata or {}).get("source", ""),
            "metadata": dict(metadata or {}),
        }
        item = QListWidgetItem(unique)
        item.setData(Qt.ItemDataRole.UserRole, unique)
        self.sheets_list.addItem(item)
        if activate:
            self.activate_loaded_dataset(unique)
        return unique

    def activate_loaded_dataset(self, name: str) -> None:
        """Make a Loaded Sheets dataset the canonical active dataframe for all application features."""
        if name not in self.loaded_datasets:
            raise KeyError(f"Unknown loaded dataset: {name}")
        cfg = self.loaded_datasets[name]
        df = cfg.get("data")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Loaded dataset '{name}' does not contain a DataFrame.")
        self.active_dataset_name = name
        self.data_engine.set_active_dataframe(df, source_path=cfg.get("source") or None)
        self.refresh_data_management()
        self.populate_marks_combos()
        self._sync_all_mark_fields()
        self.update_plot()
        for row in range(self.sheets_list.count()):
            item = self.sheets_list.item(row)
            item.setSelected(item.text() == name)
        current = self.sheets_list.findItems(name, Qt.MatchFlag.MatchExactly)
        if current:
            self.sheets_list.setCurrentItem(current[0])
        self.status.showMessage(f"Active dataset: {name} — {len(df):,} rows × {len(df.columns):,} columns")

    def _sync_active_dataset(self) -> None:
        """Persist current active dataframe changes back into its Loaded Sheets dataset."""
        if self.active_dataset_name and self.active_dataset_name in self.loaded_datasets and self.data_engine.df is not None:
            self.loaded_datasets[self.active_dataset_name]["data"] = self.data_engine.df.copy()

    def _selected_loaded_dataset_names(self) -> list[str]:
        """Return the Loaded Sheets dataset names selected by the user."""
        return [item.text() for item in self.sheets_list.selectedItems() if item.text() in self.loaded_datasets]

    def loaded_sheets_context_menu(self, pos) -> None:
        """Show professional dataset management actions for the Loaded Sheets pane."""
        item = self.sheets_list.itemAt(pos)
        if item and item.text() not in [x.text() for x in self.sheets_list.selectedItems()]:
            self.sheets_list.clearSelection()
            item.setSelected(True)
        names = self._selected_loaded_dataset_names()
        menu = QMenu(self)
        if names:
            menu.addAction("Set as Active Dataset", lambda: self.activate_loaded_dataset(names[0]))
            menu.addAction("Dataset Information", lambda: self.show_loaded_dataset_info(names[0]))
            menu.addSeparator()
            merge = menu.addAction("Merge Selected Datasets…")
            merge.setEnabled(len(names) >= 2)
            merge.triggered.connect(self.merge_loaded_datasets)
            union = menu.addAction("Create New Dataset — Union / Append Selected")
            union.setEnabled(len(names) >= 2)
            union.triggered.connect(self.union_loaded_datasets)
            duplicate = menu.addAction("Duplicate as New Dataset")
            duplicate.setEnabled(len(names) == 1)
            duplicate.triggered.connect(lambda: self.duplicate_loaded_dataset(names[0]))
            menu.addSeparator()
            rename = menu.addAction("Rename Dataset…")
            rename.setEnabled(len(names) == 1)
            rename.triggered.connect(lambda: self.rename_loaded_dataset(names[0]))
            remove = menu.addAction("Remove from Loaded Sheets")
            remove.triggered.connect(self.remove_loaded_datasets)
        else:
            action = menu.addAction("No dataset selected")
            action.setEnabled(False)
        menu.exec(self.sheets_list.viewport().mapToGlobal(pos))

    def show_loaded_dataset_info(self, name: str) -> None:
        """Display schema and provenance information for a selected loaded dataset."""
        cfg = self.loaded_datasets.get(name, {})
        df = cfg.get("data")
        if not isinstance(df, pd.DataFrame):
            return
        info = {
            "name": name,
            "source": cfg.get("source", ""),
            "rows": len(df),
            "columns": len(df.columns),
            "active": name == self.active_dataset_name,
            "columns_and_dtypes": {str(c): str(df[c].dtype) for c in df.columns},
            "missing_values": {str(c): int(df[c].isna().sum()) for c in df.columns if int(df[c].isna().sum()) > 0},
        }
        self._show_json_dialog(f"Loaded Dataset — {name}", info)

    def duplicate_loaded_dataset(self, name: str) -> None:
        """Create an independent managed copy of one Loaded Sheets dataset."""
        cfg = self.loaded_datasets.get(name)
        if not cfg:
            return
        new_name = self._register_loaded_dataset(f"{name} Copy", cfg["data"], {**cfg.get("metadata", {}), "source": cfg.get("source", "" )}, activate=True)
        self.status.showMessage(f"Created dataset: {new_name}")

    def rename_loaded_dataset(self, name: str) -> None:
        """Rename a Loaded Sheets dataset without changing its underlying dataframe."""
        if name not in self.loaded_datasets:
            return
        new_name, ok = QInputDialog.getText(self, "Rename Dataset", "New dataset name:", text=name)
        new_name = new_name.strip()
        if not ok or not new_name or new_name == name:
            return
        if new_name in self.loaded_datasets:
            QMessageBox.warning(self, "Rename Dataset", f"A dataset named '{new_name}' already exists.")
            return
        cfg = self.loaded_datasets.pop(name)
        self.loaded_datasets[new_name] = cfg
        for row in range(self.sheets_list.count()):
            item = self.sheets_list.item(row)
            if item.text() == name:
                item.setText(new_name)
                item.setData(Qt.ItemDataRole.UserRole, new_name)
                break
        if self.active_dataset_name == name:
            self.active_dataset_name = new_name
        self.status.showMessage(f"Renamed dataset to: {new_name}")

    def remove_loaded_datasets(self) -> None:
        """Remove selected datasets from Loaded Sheets while preserving unrelated datasets."""
        names = self._selected_loaded_dataset_names()
        if not names:
            return
        if self.active_dataset_name in names and len(self.loaded_datasets) <= len(names):
            QMessageBox.warning(self, "Loaded Sheets", "At least one loaded dataset must remain active.")
            return
        answer = QMessageBox.question(self, "Remove Dataset", f"Remove {len(names)} selected dataset(s) from Loaded Sheets? The original files are not deleted.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        for name in names:
            self.loaded_datasets.pop(name, None)
            for row in range(self.sheets_list.count() - 1, -1, -1):
                if self.sheets_list.item(row).text() == name:
                    self.sheets_list.takeItem(row)
        if self.active_dataset_name in names:
            remaining = next(iter(self.loaded_datasets), None)
            self.active_dataset_name = None
            if remaining:
                self.activate_loaded_dataset(remaining)
        self.status.showMessage(f"Removed {len(names)} dataset(s) from Loaded Sheets.")

    def merge_loaded_datasets(self) -> None:
        """Merge two or more selected datasets using explicit key and join-type controls."""
        names = self._selected_loaded_dataset_names()
        if len(names) < 2:
            QMessageBox.warning(self, "Merge Datasets", "Select at least two datasets in Loaded Sheets.")
            return
        frames = [self.loaded_datasets[n]["data"] for n in names]
        common = set(frames[0].columns)
        for frame in frames[1:]:
            common &= set(frame.columns)
        common = sorted(map(str, common))
        if not common:
            QMessageBox.warning(self, "Merge Datasets", "The selected datasets have no common columns that can be used as merge keys.")
            return
        dialog = QDialog(self); dialog.setWindowTitle("Merge Selected Datasets"); dialog.resize(560, 260)
        layout = QVBoxLayout(dialog); form = QFormLayout(); key = QComboBox(); key.addItems(common); join = QComboBox(); join.addItems(["inner", "left", "right", "outer"])
        form.addRow("Merge key", key); form.addRow("Join type", join); layout.addLayout(form)
        note = QLabel("Datasets are merged sequentially in the order shown in Loaded Sheets. Duplicate non-key columns receive pandas suffixes."); note.setWordWrap(True); layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); layout.addWidget(buttons); buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        result = frames[0].copy()
        merge_key = key.currentText(); how = join.currentText()
        try:
            for idx, frame in enumerate(frames[1:], 2):
                suffix = ("", f"_{idx}")
                result = pd.merge(result, frame.copy(), on=merge_key, how=how, suffixes=suffix)
            name = self._register_loaded_dataset("Merged Dataset", result, {"operation": "merge", "parents": names, "merge_key": merge_key, "join_type": how}, activate=True)
            QMessageBox.information(self, "Merge Datasets", f"Created '{name}' with {len(result):,} rows and {len(result.columns):,} columns.")
        except Exception as exc:
            QMessageBox.critical(self, "Merge Datasets", str(exc))

    def union_loaded_datasets(self) -> None:
        """Create a new dataset by appending the selected datasets with schema alignment."""
        names = self._selected_loaded_dataset_names()
        if len(names) < 2:
            QMessageBox.warning(self, "Union Datasets", "Select at least two datasets in Loaded Sheets.")
            return
        try:
            result = pd.concat([self.loaded_datasets[n]["data"] for n in names], axis=0, ignore_index=True, sort=False)
            name = self._register_loaded_dataset("Union Dataset", result, {"operation": "union", "parents": names}, activate=True)
            QMessageBox.information(self, "Union Datasets", f"Created '{name}' with {len(result):,} rows and {len(result.columns):,} columns.")
        except Exception as exc:
            QMessageBox.critical(self, "Union Datasets", str(exc))

    def open_file(self):
        """Perform the open file operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
                # Loaded Sheets is a dataset source manager; analytical workbook Sheets are separate.
                self._register_loaded_dataset(Path(filepath).stem, self.data_engine.df.copy(), {"source": filepath, "operation": "load"}, activate=True)
                self.status.showMessage(f"Loaded {filepath} as an active dataset")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def open_pdf(self):
        """Perform the open pdf operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
                unique = self._register_loaded_dataset(name, df, {"source": filepath, "operation": "pdf_table", "pdf_page": t["page"], "pdf_table_index": t["table_index"]}, activate=False)
                imported_names.append(unique)
        finally:self.sheets_list.blockSignals(False)
        # Explicitly activate the first imported table; a programmatic selection does not emit itemClicked.
        self.activate_loaded_dataset(imported_names[0])
        self.status.showMessage(f"Loaded {len(selected)} PDF table(s). Select a dataset in Loaded Sheets to work on it.")

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
            manifest["loaded_datasets"] = {
                name: {"source": cfg.get("source", ""), "metadata": cfg.get("metadata", {})}
                for name, cfg in self.loaded_datasets.items()
            }
            manifest["active_dataset_name"] = self.active_dataset_name
            manifest["dataset_card"] = self.dataset_card
            manifest["data_contract"] = self.data_contract
            manifest["mark_aggregations"] = getattr(self, "mark_aggregations", {})
            # JSON table format preserves column names, dtypes and index semantics better than CSV.
            data_json = self.data_engine.original_df.to_json(orient="table", date_format="iso")
            active_json = self.data_engine.df.to_json(orient="table", date_format="iso")
            with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr("data_original.json", data_json)
                z.writestr("data_active.json", active_json)
                for index, (name, cfg) in enumerate(self.loaded_datasets.items()):
                    frame = cfg.get("data")
                    if isinstance(frame, pd.DataFrame):
                        z.writestr(f"loaded_datasets/{index}.json", frame.to_json(orient="table", date_format="iso"))
                        manifest.setdefault("loaded_dataset_files", {})[name] = f"loaded_datasets/{index}.json"
                # Rewrite manifest after adding dataset file mappings.
                z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
            self.status.showMessage(f"Portable project saved to {filepath}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Project Error", str(exc))

    def load_project(self):
        """Perform the load project operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
                loaded_frames = {}
                for name, rel in (manifest.get("loaded_dataset_files") or {}).items():
                    if rel in names:
                        loaded_frames[name] = pd.read_json(io.StringIO(z.read(rel).decode("utf-8")), orient="table")
            self.data_engine.original_df = original.copy()
            self.data_engine.df = active.copy()
            self.data_engine.source_path = manifest.get("dataset", {}).get("source")
            self.data_engine.filter_specs = dict(manifest.get("filters", []))
            self.data_engine._refresh_schema()
            self.sheet_manager.sheets = manifest.get("sheets") or {"Sheet 1": {"x_col": None, "y_col": None, "mark_type": "Bar", "color_col": None, "size_col": None, "agg_func": "sum"}}
            self.sheet_manager.active_sheet = manifest.get("active_sheet", next(iter(self.sheet_manager.sheets)))
            self.loaded_datasets = {}
            self.sheets_list.clear()
            for name, meta in (manifest.get("loaded_datasets") or {}).items():
                rel = (manifest.get("loaded_dataset_files") or {}).get(name)
                frame = loaded_frames.get(name)
                if frame is not None:
                    self.loaded_datasets[name] = {"data": frame, "source": meta.get("source", ""), "metadata": meta.get("metadata", {})}
                    self.sheets_list.addItem(name)
            if not self.loaded_datasets:
                fallback_name = Path(self.data_engine.source_path).stem if self.data_engine.source_path else "Loaded Dataset"
                self.loaded_datasets[fallback_name] = {"data": self.data_engine.df.copy(), "source": self.data_engine.source_path or "", "metadata": {"operation": "project_load"}}
                self.sheets_list.addItem(fallback_name)
            self.active_dataset_name = manifest.get("active_dataset_name") if manifest.get("active_dataset_name") in self.loaded_datasets else next(iter(self.loaded_datasets))
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
            self.activate_loaded_dataset(self.active_dataset_name)
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
        """Perform the show data quality operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the show data dictionary operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the export dataframe operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        path, _ = QFileDialog.getSaveFileName(self, "Export", default_name, file_filter)
        if path:
            table.to_csv(path, index=False)
            self.status.showMessage(f"Exported to {path}")

    # ============================================================
    # TABLE CREATION
    # ============================================================
    def web_scraping(self):
        """Perform the web scraping operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        QMessageBox.information(self, "Web Scraping", "Web scraping dialog not implemented yet.")

    def set_api_key(self):
        """Perform the set api key operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.select_local_llm()

    # ============================================================
    # DATA INFO
    # ============================================================
    def show_missing_values(self):
        """Perform the show missing values operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        missing = self.data_engine.get_missing_values()
        msg = "\n".join([f"{k}: {v}" for k, v in missing.items() if v > 0])
        QMessageBox.information(self, "Missing Values", msg or "No missing values.")

    def show_describe_data(self):
        """Perform the show describe data operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the show data info operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        info = self.data_engine.get_data_info()
        QMessageBox.information(self, "Data Info", info)

    # ============================================================
    # DATA PRE-PROCESSING
    # ============================================================
    def data_cleaning(self):
        """Perform the data cleaning operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the data integration operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the data transformation operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the data reduction operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the feature engineering operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.create_calculated_field()

    def data_encoding(self):
        """Perform the data encoding operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the label encoding operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the data scaling operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the data splitting operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None: QMessageBox.warning(self, "No Data", "Load data first."); return
        from sklearn.model_selection import train_test_split
        ratio, ok = QInputDialog.getDouble(self, "Data Splitting", "Test fraction:", .2, .05, .9, 2)
        if not ok: return
        train, test = train_test_split(self.data_engine.df, test_size=ratio, random_state=42)
        train_name = self._register_loaded_dataset("Train", train.copy(), {"operation": "train_test_split", "parent": self.active_dataset_name, "role": "train"}, activate=False)
        test_name = self._register_loaded_dataset("Test", test.copy(), {"operation": "train_test_split", "parent": self.active_dataset_name, "role": "test"}, activate=False)
        QMessageBox.information(self, "Data Splitting", f"Created Loaded Sheets datasets: {train_name} ({len(train):,}) and {test_name} ({len(test):,}).")

    # ============================================================
    # DATA ANALYSIS
    # ============================================================
    def show_unsupervised_learning(self):
        """Open the professional unsupervised-learning workspace."""
        if self.data_engine.df is None: QMessageBox.warning(self, "No Data", "Load a dataset first."); return
        UnsupervisedLearningDialog(self.data_engine.df, self).exec()

    def show_reinforcement_learning(self):
        """Open the professional reinforcement-learning workspace."""
        ReinforcementLearningDialog(self).exec()

    def show_cnn_image_analysis(self):
        """Open the professional CNN image-analysis workspace without changing the main GUI layout."""
        CNNImageAnalysisDialog(self).exec()

    def _refresh_navigation_combos(self):
        """Refresh Sheet and Story navigation controls while preserving the existing toolbar geometry."""
        if hasattr(self,"sheet_nav_combo"):
            current=self.sheet_manager.active_sheet
            self.sheet_nav_combo.blockSignals(True); self.sheet_nav_combo.clear(); self.sheet_nav_combo.addItems(list(self.sheet_manager.sheets.keys())); self.sheet_nav_combo.setCurrentText(current); self.sheet_nav_combo.blockSignals(False)
        if hasattr(self,"story_nav_combo"):
            names=[x.get("name") for x in self.story_workspace_state.get("stories",[]) if x.get("name")]
            self.story_nav_combo.blockSignals(True); self.story_nav_combo.clear(); self.story_nav_combo.addItems(names or ["Story 1"]); self.story_nav_combo.setCurrentText(getattr(self,"active_story_name",names[0] if names else "Story 1")); self.story_nav_combo.blockSignals(False)

    def _navigate_sheet_from_toolbar(self, name):
        """Activate the selected Sheet without altering the main layout."""
        if not name or not hasattr(self, "sheet_manager"): return
        try:
            self.sheet_manager.activate(name)
            self.status.showMessage(f"Active Sheet: {name}")
        except Exception:
            pass

    def _navigate_story_from_toolbar(self, name):
        """Select the named Story in the story workspace state."""
        if not name: return
        self.active_story_name=name
        self.status.showMessage(f"Active Story: {name}")

    def show_analysis(self, analysis_type):
        """Perform the show analysis operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        dialog = DataAnalysisDialog(self.data_engine.df, analysis_type, self)
        dialog.exec()

    # ============================================================
    # AI AGENTS
    # ============================================================
    def run_agent_data_scientist(self, target_override=None, feature_override=None):
        """Perform the run agent data scientist operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first."); return
        cfg,check=self._agent_llm_preflight("Agent Data Scientist")
        if check.get("status")!="ready":
            QMessageBox.warning(self,"AI Provider Required","Select a local LLM model or configure an API key first using AI Agents → Select Local LLM model or API key.\n\n"+check.get("reason","No provider configured.")); return
        target_analysis = recommend_targets_and_features(self.data_engine.df)
        suggested = target_analysis.get("suggested_target") or (self.data_engine.df.columns[-1] if len(self.data_engine.df.columns) else None)
        target = target_override or suggested

        # Professional automatic target validation gate: the agent proposes, explains,
        # flags ambiguity, and asks the human to confirm before any model is trained.
        if target_override is None:
            d=QDialog(self); d.setWindowTitle("Automatic Target Validation & Feature Review"); d.resize(1000,700); root=QVBoxLayout(d)
            root.addWidget(QLabel("The Master Agent has inspected the dataset. Review the proposed target and flagged feature relationships before model training."))
            form=QFormLayout(); target_box=QComboBox(); target_box.addItem("No target — unsupervised / exploratory")
            target_box.addItems([x["column"] for x in target_analysis.get("target_candidates",[])])
            if suggested: target_box.setCurrentText(str(suggested))
            objective_box=QTextEdit(); objective_box.setPlainText("Perform a professional, evidence-bound analysis of the active dataset. Identify the appropriate analytical task and methods; consider supervised learning, clustering, anomaly detection, statistics, and time-series analysis where justified.")
            objective_box.setMaximumHeight(90)
            form.addRow("Proposed target",target_box); form.addRow("Analytical objective",objective_box); root.addLayout(form)
            cand=QTableWidget(); candidates=target_analysis.get("target_candidates",[])[:12]; cand.setRowCount(len(candidates)); cand.setColumnCount(5); cand.setHorizontalHeaderLabels(["Candidate","Score","dtype","Missing %","Reasons"])
            for i,x in enumerate(candidates):
                vals=[x.get("column"),x.get("score"),x.get("dtype"),x.get("missing_pct"),"; ".join(x.get("reasons",[]))]
                for j,v in enumerate(vals): cand.setItem(i,j,QTableWidgetItem(str(v)))
            cand.resizeColumnsToContents(); root.addWidget(QLabel("Target candidates")); root.addWidget(cand,1)
            warnings=target_analysis.get("warnings",[])
            red=target_analysis.get("redundancy_pairs",[])
            msg=QTextEdit(); msg.setReadOnly(True); msg.setPlainText(("Warnings / ambiguity:\n"+"\n".join("• "+w for w in warnings)) if warnings else "No high-severity target/feature ambiguity was detected by the automatic screening.")
            if red:
                msg.append("\n\nPotentially redundant or mathematically related numeric fields:\n"+"\n".join(f"• {x['columns']}: {x['type']} (r={x['correlation']}) — {x['recommendation']}" for x in red[:20]))
            root.addWidget(msg,1)
            note=QLabel("Important: correlation or a mathematical relationship cannot determine causal/temporal direction. A human must resolve whether a related field is a legitimate predictor or future/derived information.")
            note.setWordWrap(True); root.addWidget(note)
            bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel); bb.button(QDialogButtonBox.StandardButton.Ok).setText("Approve Target & Continue"); root.addWidget(bb)
            bb.accepted.connect(d.accept); bb.rejected.connect(d.reject)
            if d.exec()!=QDialog.DialogCode.Accepted:return
            selected_target=target_box.currentText()
            target = None if selected_target.startswith("No target") else selected_target
            objective = objective_box.toPlainText().strip()
        else:
            objective = "Perform a professional, evidence-bound analysis of the active dataset using the most appropriate analytical methods and specialist agents."
        if feature_override: target_analysis["approved_features"] = list(feature_override)
        gate = self.leakage_gate.evaluate(self.data_engine.df, target=target) if target else {"status":"not_applicable","reason":"No supervised target was selected; Master Agent may route to unsupervised, statistical, anomaly, or time-series analysis."}
        self.agent_ds_state = {
            "messages": [HumanMessage(content="Master Agent: analyze the active dataset and choose the appropriate analytical route and specialist sub-agents." )],
            "approved_steps": [], "rejected_steps": [], "user_approved": False,
            "dataframe": self.data_engine.df.copy(), "api_key": cfg.api_key,
            "model_name": cfg.model_name, "llm_provider": cfg.provider, "local_path": cfg.local_path, "api_base": cfg.api_base,
            "objective": objective,
            "target": target, "target_analysis": target_analysis, "feature_candidates": target_analysis.get("suggested_features", []), "image_dataset_path": getattr(self, "cnn_image_dataset_path", ""),
            "compute_mode": self.compute_mode, "leakage_gate": gate, "max_steps": 20, "abort_requested": False, "dataset_card": DatasetCardBuilder.build(self.data_engine.df),
            "memory": [], "human_approval_evidence": [], "evidence_ids": [],
        }
        gate_evidence = EvidenceObjectProxy.from_dict(gate, "Scientific/Data Leakage Gate")
        self._add_evidence(gate_evidence)
        self.agent_ds_state["evidence_ids"] = [gate_evidence.get("evidence_id")]
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
        self.agent_console.clear(); self.agent_console.log("Master Agent","target_validation","approved",f"Target: {target}; compute mode: {self.compute_mode}")
        self.analysis_recipe.add("agent_start",{"target":target,"compute_mode_requested":self.compute_mode,"compute_info":self.compute_info.to_dict() if hasattr(self.compute_info, "to_dict") else getattr(self.compute_info, "__dict__", {}),"target_validation":target_analysis})
        self.agent_worker = AgentWorker(agent_ds_app, self.agent_ds_state)
        self.agent_worker.step_ready.connect(self.on_agent_step_ready)
        self.agent_worker.finished.connect(self.on_agent_finished)
        self.agent_worker.error.connect(self.on_agent_error); self.agent_worker.start()
        self.status.showMessage("Master Agent is thinking...")

    def _approval_result_points(self):
        """Return very brief, evidence-bound results for the current approval gate."""
        st=self.agent_ds_state; stage=str(st.get("analysis_stage","")).upper(); points=list(st.get("approval_result_points") or [])
        if points: return points[:5]
        ta=st.get("target_analysis") or {}
        if stage=="PLAN":
            decision=st.get("master_decision") or {}
            selection=decision.get("method_selection") or {}
            selected=(selection.get("selected_method") or {}).get("method_id","not selected")
            points=[
                f"Dataset: {len(st.get('dataframe')):,} rows × {len(st.get('dataframe').columns):,} columns" if st.get('dataframe') is not None else "Dataset: not loaded",
                f"Identified task: {st.get('route','not specified')}",
                f"Selected method: {selected}",
                f"Selection status: {decision.get('selection_status','not recorded')}",
                f"Specialists: {', '.join(decision.get('sub_agents',[]) or []) or 'none'}",
            ]
        elif stage=="VALIDATION":
            ml=st.get("ml_results") or {}; dl=st.get("dl_results") or {}; points=["Validation protocol is designed before model fitting.","Final test partition remains locked.",f"Leakage gate: {(st.get('leakage_gate') or {}).get('status','not recorded')}"]
            if ml or dl: points.append(f"Existing model evidence: {', '.join([x for x,b in [('ML',ml),('DL',dl)] if b])}")
        elif stage=="PREPROCESSING":
            points=["Imputation/encoding/scaling are intended to be learned inside training folds.",f"Approved feature count: {len(ta.get('approved_features',[]) or st.get('feature_candidates',[]) or [])}","Potential leakage variables are reviewed before fitting."]
        elif stage=="MODEL_SELECTION":
            decision=st.get("master_decision") or {}
            selection=decision.get("method_selection") or {}
            selected=(selection.get("selected_method") or {}).get("method_id","not selected")
            candidates=decision.get("method_candidates") or []
            points=[
                f"Identified task: {st.get('route','not specified')}",
                f"Selected method: {selected}",
                f"Candidate methods compared: {len(candidates)}",
                f"Selection status: {decision.get('selection_status','not recorded')}",
                f"Next specialist: {st.get('current_subagent','not specified')}",
            ]
            gaps=decision.get("evidence_gaps") or []
            if gaps: points[-1]="Evidence gap: "+str(gaps[0])
        elif stage=="MODEL_EXECUTION":
            for label,key in (("ML","ml_results"),("DL","dl_results"),("Clustering","clustering_results"),("Unsupervised","unsupervised_results"),("Reinforcement Learning","reinforcement_results"),("Anomaly","anomaly_results"),("Statistics","statistics_results"),("Data Quality","data_quality_results"),("Time-Series","time_series_results")):
                b=st.get(key) or {}
                if b: points.append(f"{label} result status: {b.get('status','unknown')}; agent={b.get('agent',label)}")
            points.append(f"Next specialist: {st.get('current_subagent',st.get('route','not specified'))}")
        elif stage=="EVALUATION":
            for label,key in (("ML","ml_results"),("DL","dl_results")):
                b=st.get(key) or {}; m=b.get("best_metrics",{}) if isinstance(b,dict) else {}
                if m: points.append(f"{label} test metrics: "+", ".join(f"{k}={v:.4g}" for k,v in list(m.items())[:4] if isinstance(v,(int,float))))
            points.append("Prediction uncertainty, baseline and validation evidence are included where available.")
        elif stage=="ROBUSTNESS":
            for label,key in (("ML","ml_results"),("DL","dl_results")):
                b=st.get(key) or {}; r=(b.get("evaluation") or {}).get("robustness_perturbation") if isinstance(b,dict) else None
                if r: points.append(f"{label} perturbation status: {r.get('status','not recorded')}")
            points += ["Feature stability and temporal availability are reviewed.","Counterfactual leakage tests are diagnostic, not causal proof."]
        elif stage=="DIAGNOSIS":
            for label,key in (("ML","ml_results"),("DL","dl_results")):
                b=st.get(key) or {}; d=b.get("model_diagnosis") or {}; points.append(f"{label} diagnostic hypotheses: {len(d.get('hypotheses',[]) or [])}")
            points.append("Review subgroup/slice and generalisation-gap evidence before verification.")
        elif stage=="VERIFY":
            v=st.get("verification") or {}; points=[f"Evidence verification: {v.get('status','not recorded')}",f"Evidence IDs recorded: {len(st.get('evidence_ids',[]) or [])}","Verification checks internal consistency and evidence completeness; it does not certify scientific truth."]
        elif stage=="STOP":
            e=st.get("evidence_validation") or {}; points=[f"Evidence validation: {e.get('status','not recorded')}",f"Approved stages: {len(st.get('approved_steps',[]) or [])}","Completion is allowed only when required evidence and unresolved high-severity flags have been reviewed."]
        return points[:5] or ["No additional numeric result is available yet; this gate concerns the proposed analytical action."]

    def on_agent_step_ready(self, step_text):
        """Perform the on agent step ready operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        summary = step_text or self.agent_ds_state.get("stage_summary", "Awaiting your approval.")
        # Agent Why and Agent Self-Check are first-class evidence objects. Record
        # them before the human gate so approval is attached to the exact proposal.
        why = self.agent_ds_state.get("why_evidence")
        if isinstance(why, dict) and why.get("evidence_id") not in self._recorded_agent_evidence_ids:
            self._add_evidence(why); self._recorded_agent_evidence_ids.add(why.get("evidence_id"))
            self.agent_console.log("Agent", "why", "evidence", json.dumps(why.get("data", {}), default=str), evidence_id=why.get("evidence_id"))
        self_check = self.agent_ds_state.get("self_check")
        if isinstance(self_check, dict):
            eid = f"selfcheck-{abs(hash(json.dumps(self_check, sort_keys=True, default=str))) % 10**12}"
            if eid not in self._recorded_agent_evidence_ids:
                evidence = {"evidence_id": eid, "kind": "agent_self_check", "title": f"Agent Self-Check: {self.agent_ds_state.get('current_step','step')}", "status": self_check.get("status", "review"), "data": self_check, "parent_ids": [why.get("evidence_id")] if isinstance(why, dict) and why.get("evidence_id") else []}
                self._add_evidence(evidence); self._recorded_agent_evidence_ids.add(eid)
                self.agent_console.log("Agent", "self_check", self_check.get("status", "review"), "Pre-approval self-check recorded.", evidence_id=eid)
        # Verification gates are also first-class evidence and must be visible before approval.
        for key, title in (("verification_why", "Agent Verification Why"), ("verification_self_check", "Agent Verification Self-Check")):
            obj = self.agent_ds_state.get(key)
            if isinstance(obj, dict):
                if key == "verification_why" and obj.get("evidence_id") not in self._recorded_agent_evidence_ids:
                    self._add_evidence(obj); self._recorded_agent_evidence_ids.add(obj.get("evidence_id"))
                    self.agent_console.log("Agent", "verification_why", "evidence", json.dumps(obj.get("data", {}), default=str), evidence_id=obj.get("evidence_id"))
                elif key == "verification_self_check":
                    eid = f"verify-selfcheck-{abs(hash(json.dumps(obj, sort_keys=True, default=str))) % 10**12}"
                    if eid not in self._recorded_agent_evidence_ids:
                        evidence={"evidence_id":eid,"kind":"agent_self_check","title":title,"status":obj.get("status","review"),"data":obj,"parent_ids":[self.agent_ds_state.get("verification_why",{}).get("evidence_id")] if isinstance(self.agent_ds_state.get("verification_why"),dict) else []}
                        self._add_evidence(evidence); self._recorded_agent_evidence_ids.add(eid)
                        self.agent_console.log("Agent", "verification_self_check", obj.get("status","review"), "Verification self-check recorded.", evidence_id=eid)
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
            points=self._approval_result_points()
            result_block="\n".join("• "+str(x) for x in points)
            msg_box.setText(f"{summary}\n\nBrief results / evidence available at this gate:\n{result_block}\n\nApprove this step to continue, Reject to revise/repeat it, or Abort to stop the run safely.")
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
        # LangGraph may return a reduced state. Never allow that transient state
        # to erase the canonical provider selection.
        """Perform the on agent finished operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        provider_snapshot=self.llm_config.redacted()
        final_state.update({"api_key":self.llm_config.api_key,"model_name":self.llm_config.model_name,
                            "llm_provider":self.llm_config.provider,"local_path":self.llm_config.local_path,"api_base":self.llm_config.api_base})
        self.agent_ds_state = final_state; self.status.showMessage(message)
        self.agent_console.log("Master Agent", final_state.get("current_step","complete"), "complete", message)
        try:
            if self.analysis_state.state == AnalysisState.DATA_LOADED.value: self.analysis_state.transition(AnalysisState.QUALITY_CHECKED.value); self.analysis_state.transition(AnalysisState.CONTRACT_VALIDATED.value); self.analysis_state.transition(AnalysisState.ANALYSIS_READY.value)
            if final_state.get("ml_results") or final_state.get("dl_results"): self.analysis_state.transition(AnalysisState.MODEL_READY.value)
        except Exception: pass
        if final_state.get("dataframe") is not None:
            self.data_engine.set_active_dataframe(final_state["dataframe"]); self._sync_active_dataset(); self.refresh_data_management(); self.populate_marks_combos(); self.update_plot()
        specialist_keys=("ml_results","dl_results","cnn_results","clustering_results","unsupervised_results","reinforcement_results","anomaly_results","statistics_results","data_quality_results","time_series_results")
        if any(final_state.get(k) for k in specialist_keys):
            for bundle in (final_state.get("ml_results"), final_state.get("dl_results")):
                if bundle and bundle.get("status") == "ok":
                    self.experiment_registry.log("agent_result", agent=bundle.get("agent"), metrics=bundle.get("best_metrics"))
                    self.agent_console.log(bundle.get("agent","Agent"),"model_analysis","complete",bundle.get("summary","Model result recorded."))
                    self.analysis_recipe.add("model_result",{"agent":bundle.get("agent"),"model":bundle.get("best_model"),"target":bundle.get("target")})
                    self.artifact_store.register("model",bundle.get("best_model",bundle.get("agent","model")),{"agent":bundle.get("agent"),"task":bundle.get("task"),"metrics":bundle.get("best_metrics",{})})
                    self.experiment_records.append({"experiment": f"Agent run {len(self.experiment_records)+1}", "agent": bundle.get("agent"), "model": bundle.get("best_model"), "task": bundle.get("task"), "metrics": bundle.get("best_metrics", {}), "dataset_fingerprint": self.agent_ds_state.get("dataset_card",{}).get("fingerprint"), "route": self.agent_ds_state.get("route"), "step_index": self.agent_ds_state.get("step_index")})
                    card = ModelCardBuilder.build(bundle, self.agent_ds_state.get("dataset_card"), self.agent_ds_state.get("leakage_gate"))
                    self.model_cards.append(card); self._add_evidence({"evidence_id": f"modelcard-{bundle.get('agent','model')}-{len(self.model_cards)}", "kind":"model_card", "title":f"{bundle.get('agent','Model')} Model Card", "data":card, "parent_ids":[]})
                    # Register the professional evaluation stack as separate evidence nodes
                    # so Report/Presentation/Publication can distinguish raw metrics from diagnostics.
                    for kind, title, payload in (("professional_evaluation", "Professional Model Evaluation", bundle.get("evaluation")), ("validation_protocol", "Validation Protocol", bundle.get("validation_protocol")), ("validation_audit", "Validation Integrity Audit", bundle.get("validation_audit")), ("feature_stability", "Feature Stability Analysis", bundle.get("feature_stability")), ("feature_set_challenge", "Feature Set Challenge", bundle.get("feature_set_challenge")), ("temporal_availability", "Temporal Availability Matrix", bundle.get("temporal_availability")), ("counterfactual_leakage", "Counterfactual Leakage Test", bundle.get("counterfactual_leakage")), ("model_diagnosis", "Automatic Model Diagnosis", bundle.get("model_diagnosis")), ("feature_provenance", "Feature Provenance", bundle.get("feature_provenance")), ("resource_policy", "Resource Policy", bundle.get("resource_policy"))):
                        if payload:
                            eid = f"{kind}-{bundle.get('agent','model')}-{len(self.evidence_records)+1}"
                            self._add_evidence({"evidence_id": eid, "kind": kind, "title": f"{title}: {bundle.get('agent','Model')}", "data": payload, "parent_ids": []})
                    model = bundle.get("model_object")
                    if model is not None:
                        try:
                            self.model_registry.register(bundle.get("best_model", bundle.get("agent", "model")), model, bundle.get("best_metrics", {}), {"agent": bundle.get("agent"), "task": bundle.get("task")})
                        except Exception:
                            pass
            for label, key in (("CNN Image Analysis", "cnn_results"), ("Clustering", "clustering_results"), ("Unsupervised Learning", "unsupervised_results"), ("Reinforcement Learning", "reinforcement_results"), ("Anomaly Detection", "anomaly_results"), ("Statistical Insight", "statistics_results"), ("Data Quality", "data_quality_results"), ("Time-Series", "time_series_results")):
                payload = final_state.get(key)
                if payload:
                    self._add_evidence({"evidence_id": f"{key}-{len(self.evidence_records)+1}", "kind": key, "title": f"{label} Agent Evidence", "data": json_safe(payload), "parent_ids": []})
            try:
                from services.evidence_validation import EvidenceValidator
                required = []
                if final_state.get("ml_results") or final_state.get("dl_results"):
                    required = ["professional_evaluation", "model_diagnosis", "model_card"]
                validation = EvidenceValidator.validate(self.evidence_records, required_kinds=required)
                self.agent_ds_state["evidence_validation"] = validation
                self._add_evidence({"evidence_id": f"evidence-validation-{len(self.evidence_records)+1}", "kind":"evidence_validation", "title":"Evidence Integrity Validation", "data":validation, "parent_ids":[]})
                self.analysis_recipe.add("evidence_validation", validation)
            except Exception as exc:
                self.agent_console.log("Governance", "evidence_validation", "warning", str(exc))
            self.status.showMessage("Agent Data Scientist finished. Specialist evidence is ready for AI Agent Plot and AI Agent Report.")
        QMessageBox.information(self, "Agent Data Scientist", message)

    def on_agent_error(self, error_msg):
        """Perform the on agent error operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        QMessageBox.critical(self, "Agent Error", error_msg); self.status.showMessage("Agent error.")

    def run_ai_agent_plot(self):
        """Perform the run ai agent plot operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        from agent.graph_plot import agent_plot_app
        cfg,check=self._agent_llm_preflight("AI Agent Plot")
        if check.get("status")!="ready": QMessageBox.warning(self,"AI Provider Required",check.get("reason","Configure the provider for AI Agent Plot first.")); return
        ml = self.agent_ds_state.get("ml_results"); dl = self.agent_ds_state.get("dl_results")
        specialist_results={k:self.agent_ds_state.get(k) for k in ("clustering_results","unsupervised_results","reinforcement_results","anomaly_results","statistics_results","data_quality_results","time_series_results") if self.agent_ds_state.get(k)}
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load a dataset before running AI Agent Plot."); return
        if not ml and not dl and not specialist_results:
            QMessageBox.warning(self, "No AI Results", "Run Agent Data Scientist first. Agent Plot is evidence-linked to its target and specialist analytical evidence."); return
        state={"dataframe":self.data_engine.df.copy(),"ml_results":ml,"dl_results":dl,"specialist_results":specialist_results,"master_decision":self.agent_ds_state.get("master_decision"),"target":self.agent_ds_state.get("target"),"memory":getattr(self,"plot_agent_memory",[]),"llm_provider":cfg.provider,"model_name":cfg.model_name,"api_key":cfg.api_key,"local_path":cfg.local_path,"api_base":cfg.api_base}
        try:
            result=agent_plot_app.invoke(state,config={"configurable":{"thread_id":"agent-plot"}})
        except Exception as exc:
            QMessageBox.critical(self,"Agent Plot",f"Plot agent failed: {exc}"); return
        self.plot_agent_memory=result.get("memory",[])
        self.agent_ds_state["plot_results"]=result
        if result.get("status") != "ok":
            QMessageBox.warning(self,"Agent Plot",result.get("message",result.get("summary","Plot planning failed."))); return
        plots=result.get("plots",[])
        dlg=AgentPlotEditorDialog(plots,self); dlg.exec()
        self.status.showMessage(result.get("summary","Agent Plot completed."))

    def run_ai_report(self):
        """Perform the run ai report operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if not any(self.agent_ds_state.get(k) for k in ("ml_results","dl_results","cnn_results","clustering_results","unsupervised_results","reinforcement_results","anomaly_results","statistics_results","data_quality_results","time_series_results")):
            QMessageBox.warning(self, "No Analysis", "Run Agent Data Scientist first."); return
        if not self.agent_ds_state.get("plot_results"):
            QMessageBox.warning(self, "No Plot Evidence", "Run AI Agent Plot first. The professional report uses its evidence-linked visualizations."); return
        cfg,check=self._agent_llm_preflight("AI Agent Report")
        if check.get("status")!="ready": QMessageBox.warning(self,"AI Provider Required","Configure the local LLM model or API key first."); return
        def report_safe(bundle):
            """Perform the report safe operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            return json_safe(bundle)
        self.agent_report_state["analysis_evidence"] = {
            "master_route": self.agent_ds_state.get("route"),
            "target": self.agent_ds_state.get("target"),
            "ML Agent": report_safe(self.agent_ds_state.get("ml_results")),
            "DL Agent": report_safe(self.agent_ds_state.get("dl_results")),
            "CNN Image Analysis Agent": report_safe(self.agent_ds_state.get("cnn_results")),
            "Clustering Agent": report_safe(self.agent_ds_state.get("clustering_results")),
            "Anomaly Detection Agent": report_safe(self.agent_ds_state.get("anomaly_results")),
            "Statistical Insight Agent": report_safe(self.agent_ds_state.get("statistics_results")),
            "Data Quality Agent": report_safe(self.agent_ds_state.get("data_quality_results")),
            "Time-Series Agent": report_safe(self.agent_ds_state.get("time_series_results")),
            "Unsupervised Learning Agent": report_safe(self.agent_ds_state.get("unsupervised_results")),
            "Reinforcement Learning Agent": report_safe(self.agent_ds_state.get("reinforcement_results")),
            "validation_strategy": report_safe(self.agent_ds_state.get("validation_strategy")),
            "Master Decision": report_safe(self.agent_ds_state.get("master_decision")),
            "Plot Agent": report_safe(self.agent_ds_state.get("plot_results")),
            "data_visualizations": report_safe(self.agent_ds_state.get("plot_results")),
            "target_feature_analysis": self.agent_ds_state.get("target_analysis", {}),
            "approved_steps": self.agent_ds_state.get("approved_steps", []),
            "rejected_steps": self.agent_ds_state.get("rejected_steps", []),
            "dataset_card": self.agent_ds_state.get("dataset_card"),
            "leakage_gate": self.agent_ds_state.get("leakage_gate"),
            "human_approval_evidence": self.agent_ds_state.get("human_approval_evidence", []),
            "agent_why": [e for e in self.evidence_records if e.get("kind") == "agent_why"],
            "agent_self_checks": [e for e in self.evidence_records if e.get("kind") == "agent_self_check"],
            "professional_evaluation": {"ML": (self.agent_ds_state.get("ml_results") or {}).get("evaluation"), "DL": (self.agent_ds_state.get("dl_results") or {}).get("evaluation")},
            "advanced_diagnostics": {
                "ML": {k: (self.agent_ds_state.get("ml_results") or {}).get(k) for k in ("feature_stability", "temporal_availability", "counterfactual_leakage", "model_diagnosis")},
                "DL": {k: (self.agent_ds_state.get("dl_results") or {}).get(k) for k in ("temporal_availability", "model_diagnosis")},
            },
            "compute_info": self.compute_info.to_dict() if hasattr(self.compute_info, "to_dict") else getattr(self.compute_info, "__dict__", {}),
            "agent_evaluation": AgentEvaluation.evaluate(self.agent_ds_state),
            "evidence_dag": self.evidence_dag.to_dict(),
            "llm_provider": cfg.provider, "llm_model": cfg.model_name,
        }
        self.agent_report_state.update({"llm_provider":cfg.provider,"model_name":cfg.model_name,"api_key":cfg.api_key,"local_path":cfg.local_path})
        self.report_worker = ReportWorker(agent_report_app, self.agent_report_state)
        self.report_worker.finished.connect(self.on_report_finished); self.report_worker.error.connect(self.on_report_error); self.report_worker.start()
        self.status.showMessage("AI Agent Report is generating...")

    def on_report_finished(self, result):
        """Perform the on report finished operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the on report error operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        QMessageBox.critical(self, "Report Error", error_msg)

    # ============================================================
    # AI PROVIDER CONFIGURATION
    # ============================================================
    def _current_llm_config(self):
        """Return the legacy/global provider as a fallback only."""
        return self.llm_config

    def _agent_llm_config(self, agent_name: str) -> LLMConfig:
        """Read the selected provider for the named agent immediately before execution."""
        return self.agent_provider_registry.get(agent_name, self.llm_config)

    def _agent_llm_preflight(self, agent_name: str) -> tuple[LLMConfig, dict[str, Any]]:
        """Perform execution-time provider validation for one governed AI agent."""
        cfg = self._agent_llm_config(agent_name)
        return cfg, self.agent_provider_registry.preflight(agent_name, self.llm_config)

    def _sync_provider_to_agent_states(self):
        """Mirror selected providers into execution snapshots without making them authoritative."""
        for agent_name, state in (("Agent Data Scientist", self.agent_ds_state), ("AI Agent Report", self.agent_report_state)):
            cfg=self._agent_llm_config(agent_name)
            state.update({"api_key":cfg.api_key,"model_name":cfg.model_name,"llm_provider":cfg.provider,"local_path":cfg.local_path,"api_base":cfg.api_base})

    def select_local_llm(self):
        """Configure an independent provider for one of the four governed AI agents."""
        current = self._agent_llm_config("Agent Data Scientist")
        dialog=LocalLLMConfigDialog(self, current={"provider":current.provider,"model_name":current.model_name,"api_key":current.api_key,"local_path":current.local_path,"api_base":current.api_base})
        if dialog.exec()!=QDialog.DialogCode.Accepted: return
        cfg_dict=dialog.get_config(); agent_name=dialog.selected_agent(); cfg=LLMConfig(**cfg_dict)
        check=llm_preflight(cfg)
        if check.get("status")!="ready":
            QMessageBox.warning(self,"LLM Configuration",check.get("reason","Provider configuration is not ready.")); return
        self.agent_provider_registry.set(agent_name,cfg)
        if agent_name=="Agent Data Scientist": self.llm_config=cfg
        self._sync_provider_to_agent_states()
        family=check.get("api_provider", "local") if cfg.provider=="api" else "local"
        self.status.showMessage(f"AI provider configured for {agent_name}: {cfg.provider} / {cfg.model_name}")
        QMessageBox.information(self,"AI Provider Ready",f"Agent: {agent_name}\nProvider: {cfg.provider}\nModel: {cfg.model_name}\nRoute: {family}")

    # ============================================================
    # MACRO EDITOR
    # ============================================================
    def open_macro_editor(self, macro_type):
        """Perform the open macro editor operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        dialog=MacroEditorDialog(macro_type,self)
        dialog.run_requested.connect(lambda code,d=dialog,t=macro_type:self._run_macro_in_dialog(t,code,d))
        dialog.exec()

    def _run_macro_in_dialog(self,macro_type,code,dialog):
        """Perform the run macro in dialog operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        try:
            result=self.execute_python_macro(code) if macro_type=="Python" else self.execute_sql_macro(code)
            dialog.append_output(result)
        except Exception as exc: dialog.append_output(f"ERROR: {exc}")

    def execute_python_macro(self,code):
        """Perform the execute python macro operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None: raise ValueError("Load data first.")
        import ast,contextlib
        if not code.strip(): return "No Python command was provided."
        buffer=io.StringIO(); env={'df':self.data_engine.df.copy(),'pd':pd,'np':np,'plt':plt}; tree=ast.parse(code,mode="exec")
        with contextlib.redirect_stdout(buffer):
            for idx,node in enumerate(tree.body,1):
                is_expr=isinstance(node,ast.Expr)
                compiled=compile(ast.Module(body=[node],type_ignores=[]),f"<Python Macro {idx}>","exec")
                if is_expr:
                    value=eval(compile(ast.Expression(node.value),f"<Python Macro {idx}>","eval"),env)
                    if value is not None: print(f"[Command {idx}]\n{value}")
                else:
                    exec(compiled,env)
        output=buffer.getvalue()
        value=env.get("result")
        if isinstance(value,pd.DataFrame):
            self.data_engine.set_active_dataframe(value); self.refresh_data_management(); self.populate_marks_combos(); self._sync_all_mark_fields(); self.update_plot(); output += "\nDataFrame result loaded into active analysis:\n"+value.head(30).to_string(index=False)
        return output.strip() or "All Python commands executed successfully."

    def execute_sql_macro(self,code):
        """Perform the execute sql macro operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None: raise ValueError("Load data first.")
        if not code.strip(): return "No SQL command was provided."
        # Execute multiple statements sequentially. DuckDB returns the last SELECT result.
        try:
            import sqlparse
            statements=[x.strip() for x in sqlparse.split(code) if x.strip()]
        except Exception:
            statements=[x.strip() for x in code.split(";") if x.strip()]
        outputs=[]
        if not self.duckdb_workspace.available: raise RuntimeError("DuckDB is not installed. Install duckdb to use the SQL Macro workspace.")
        import duckdb
        con=duckdb.connect(database=":memory:")
        try:
            con.register("data",self.data_engine.df)
            for idx,stmt in enumerate(statements,1):
                result=con.execute(stmt)
                try:
                    frame=result.df(); outputs.append(f"[Command {idx}] {len(frame):,} rows × {len(frame.columns):,} columns\n{frame.head(200).to_string(index=False)}")
                except Exception:
                    outputs.append(f"[Command {idx}] SQL command completed successfully.")
        finally: con.close()
        return "\n\n".join(outputs)

    def run_python_macro(self,code):
        """Perform the run python macro operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        try: QMessageBox.information(self,"Python Macro Output",self.execute_python_macro(code))
        except Exception as exc: QMessageBox.critical(self,"Macro Error",str(exc))

    def run_sql_macro(self,code):
        """Perform the run sql macro operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        try: self._show_text_dialog("SQL Macro Result",self.execute_sql_macro(code))
        except Exception as exc: QMessageBox.critical(self,"SQL Macro Error",str(exc))

    # ============================================================
    # PROFESSIONAL ANALYTICS / AGENTS
    # ============================================================
    def show_core_schema_types(self):
        """Perform the show core schema types operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load data first."); return
        self._show_text_dialog("Core Schema Types", schema_table(self.data_engine.df).to_string(index=False))

    def show_data_contract(self):
        """Perform the show data contract operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load data first."); return
        contract=self.data_contract or DataContractEngine.build(self.data_engine.df, "Data Science Studio Pro Contract")
        dlg=QDialog(self); dlg.setWindowTitle("Data Contract & Schema Drift"); dlg.resize(900,650); l=QVBoxLayout(dlg)
        txt=QTextEdit(); txt.setReadOnly(True); txt.setPlainText(json.dumps(contract,indent=2,default=str)); l.addWidget(txt)
        row=QHBoxLayout(); save=QPushButton("Save Contract"); check=QPushButton("Check Current Dataset"); row.addWidget(save); row.addWidget(check); l.addLayout(row)
        def save_contract():
            """Perform the save contract operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            path,_=QFileDialog.getSaveFileName(self,"Save Data Contract","data_contract.json","JSON (*.json)")
            if path: DataContractEngine.save(path,contract); self.status.showMessage(f"Contract saved to {path}")
        def check_contract():
            """Perform the check contract operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            result=DataContractEngine.validate(self.data_engine.df,contract); txt.setPlainText(json.dumps(result,indent=2,default=str))
        save.clicked.connect(save_contract); check.clicked.connect(check_contract); dlg.exec()

    def show_split_wizard(self):
        """Perform the show split wizard operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load data first."); return
        cols=list(self.data_engine.df.columns); d=QDialog(self); d.setWindowTitle("Leakage-Aware Train / Validation / Test Wizard"); d.resize(760,560); l=QVBoxLayout(d)
        form=QFormLayout(); mode=QComboBox(); mode.addItems(SplitWizard.MODES); target=QComboBox(); target.addItems(cols); group=QComboBox(); group.addItem("None"); group.addItems(cols); timec=QComboBox(); timec.addItem("None"); timec.addItems(cols); test=QDoubleSpinBox(); test.setRange(.05,.8); test.setValue(.2); val=QDoubleSpinBox(); val.setRange(.05,.5); val.setValue(.1); form.addRow("Split strategy",mode); form.addRow("Target",target); form.addRow("Group column",group); form.addRow("Time column",timec); form.addRow("Test fraction",test); form.addRow("Validation fraction",val); l.addLayout(form)
        out=QTextEdit(); out.setReadOnly(True); l.addWidget(out); run=QPushButton("Validate & Create 3 Splits"); l.addWidget(run)
        def execute():
            """Perform the execute operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            g=None if group.currentText()=="None" else group.currentText(); tc=None if timec.currentText()=="None" else timec.currentText(); plan=SplitWizard.plan(self.data_engine.df,target.currentText(),mode.currentText(),g,tc,test.value(),val.value());
            try:
                tr,va,te=SplitWizard.execute(self.data_engine.df,target.currentText(),mode.currentText(),g,tc,test.value(),val.value());
                self._register_loaded_dataset("Train", tr, {"operation":"leakage_aware_split", "parent":self.active_dataset_name, "role":"train"}, activate=False)
                self._register_loaded_dataset("Validation", va, {"operation":"leakage_aware_split", "parent":self.active_dataset_name, "role":"validation"}, activate=False)
                self._register_loaded_dataset("Test", te, {"operation":"leakage_aware_split", "parent":self.active_dataset_name, "role":"test"}, activate=False)
                out.setPlainText(json.dumps({"plan":plan,"train":len(tr),"validation":len(va),"test":len(te)},indent=2)); self.status.showMessage("Leakage-aware Train/Validation/Test splits created.")
            except Exception as exc: out.setPlainText(json.dumps({"plan":plan,"error":str(exc)},indent=2))
        run.clicked.connect(execute); dlg.exec()

    def _show_supervised_workspace(self, kind):
        """Perform the show supervised workspace operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first."); return
        df=self.data_engine.df; cols=list(df.columns)
        d=QDialog(self); d.setWindowTitle(f"{kind} Analysis — Professional Workspace"); d.resize(1100,760); root=QVBoxLayout(d)
        tabs=QTabWidget(); root.addWidget(tabs)
        setup=QWidget(); sl=QVBoxLayout(setup); form=QFormLayout(); target=QComboBox(); target.addItems(cols); folds=QSpinBox(); folds.setRange(2,10); folds.setValue(5); test=QDoubleSpinBox(); test.setRange(.1,.5); test.setSingleStep(.05); test.setValue(.2); scaling=QComboBox(); scaling.addItems(["standard","minmax","none"]); pca=QCheckBox("Optional PCA (fit inside training folds)"); form.addRow("Target",target); form.addRow("Test fraction",test); form.addRow("CV folds",folds); form.addRow("Scaling",scaling); form.addRow("Dimensionality reduction",pca); sl.addLayout(form)
        desc=QTextEdit(); desc.setReadOnly(True); desc.setPlainText(("Classification pipeline: missing-value imputation → categorical one-hot encoding → training-fold-only scaling → model/CV/hyperparameter search → locked test evaluation → calibration and class-imbalance metrics → deployment/monitoring evidence." if kind=="Classification" else "Regression pipeline: missing-value imputation → categorical one-hot encoding → training-fold-only scaling → model/CV/hyperparameter search → locked test evaluation → residual/error metrics → uncertainty and deployment/monitoring evidence.")); sl.addWidget(desc)
        run=QPushButton(f"Run Professional {kind} Analysis"); sl.addWidget(run); tabs.addTab(setup,"1. Setup & Preprocessing")
        methods_text=QTextEdit(); methods_text.setReadOnly(True); tabs.addTab(methods_text,"2. Methods & Model Selection")
        eval_text=QTextEdit(); eval_text.setReadOnly(True); tabs.addTab(eval_text,"3. Evaluation & Diagnostics")
        deploy_text=QTextEdit(); deploy_text.setReadOnly(True); tabs.addTab(deploy_text,"4. Deployment & Monitoring")
        holder={"result":None}
        def execute():
            """Perform the execute operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            try:
                if kind=="Classification": result=ClassificationAnalysisEngine.run(df,target.currentText(),test_size=test.value(),folds=folds.value(),scaling=scaling.currentText(),pca=pca.isChecked())
                else: result=RegressionAnalysisEngine.run(df,target.currentText(),test_size=test.value(),folds=folds.value(),scaling=scaling.currentText(),pca=pca.isChecked())
                holder["result"]=result
                methods_text.setPlainText(json.dumps({"methods":result.get("methods"),"best_model":result.get("best_model"),"selection_metric":result.get("selection_metric"),"validation":result.get("validation")},indent=2,default=str))
                eval_text.setPlainText(json.dumps({"best_test_metrics":result.get("best_test_metrics"),"preprocessing":result.get("preprocessing"),"class_distribution":result.get("class_distribution"),"feature_screening":result.get("feature_screening"),"deployment":result.get("deployment")},indent=2,default=str))
                deploy_text.setPlainText(json.dumps(result.get("deployment",{}),indent=2,default=str)+"\n\nFinal test partition is locked and must not be used for tuning or feature selection.")
                tabs.setCurrentIndex(2); self.status.showMessage(f"Professional {kind} Analysis completed.")
            except Exception as exc: QMessageBox.critical(d,f"{kind} Analysis Error",str(exc))
        run.clicked.connect(execute)
        export=QPushButton("Export Analysis JSON"); root.addWidget(export)
        export.clicked.connect(lambda: self._export_supervised_result(holder.get("result"),kind))
        d.exec()

    def _export_supervised_result(self,result,kind):
        """Perform the export supervised result operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if not result: return
        path,_=QFileDialog.getSaveFileName(self,f"Export {kind} Analysis",f"{kind.lower().replace(' ','_')}_analysis.json","JSON (*.json)")
        if path: Path(path).write_text(json.dumps(result,indent=2,default=str),encoding="utf-8"); self.status.showMessage(f"Exported {kind} analysis to {path}")

    def show_classification_analysis(self):
        """Perform the show classification analysis operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._show_supervised_workspace("Classification")
    def show_regression_analysis(self):
        """Perform the show regression analysis operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._show_supervised_workspace("Regression")

    def show_clustering_analysis(self):
        """Open the professional clustering-analysis workspace for the active dataset."""
        if self.data_engine.df is None:
            QMessageBox.warning(self, "No Data", "Load data first."); return
        ClusteringAnalysisDialog(self.data_engine.df, self).exec()

    def show_duckdb_workspace(self):
        """Perform the show duckdb workspace operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load data first."); return
        engine=DuckDBWorkspace(); d=QDialog(self); d.setWindowTitle("DuckDB Analytical Workspace"); d.resize(1000,700); l=QVBoxLayout(d)
        l.addWidget(QLabel("Query the active dataset as the DuckDB table `data`. Results can be inspected or exported without changing the source dataset."))
        editor=QTextEdit(); editor.setPlainText("SELECT * FROM data LIMIT 100;"); l.addWidget(editor)
        run=QPushButton("Run SQL"); export=QPushButton("Export Result CSV"); l.addWidget(run); l.addWidget(export); out=QTableWidget(); l.addWidget(out)
        result_holder={"df":None}
        def execute():
            """Perform the execute operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            try:
                result=engine.query(self.data_engine.df,editor.toPlainText()); result_holder["df"]=result; out.setRowCount(len(result)); out.setColumnCount(len(result.columns)); out.setHorizontalHeaderLabels([str(c) for c in result.columns])
                for i,row in result.iterrows():
                    for j,c in enumerate(result.columns): out.setItem(i,j,QTableWidgetItem(str(row[c])))
                self.status.showMessage(f"DuckDB returned {len(result):,} rows.")
            except Exception as exc: QMessageBox.critical(d,"DuckDB Error",str(exc))
        def export_result():
            """Perform the export result operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            result=result_holder.get("df")
            if result is None:return
            path,_=QFileDialog.getSaveFileName(d,"Export Query Result","duckdb_result.csv","CSV (*.csv)")
            if path: result.to_csv(path,index=False)
        run.clicked.connect(execute); export.clicked.connect(export_result); d.exec()

    def show_statistical_wizard(self):
        """Perform the show statistical wizard operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load data first."); return
        df=self.data_engine.df; cols=list(df.columns); d=QDialog(self); d.setWindowTitle("Statistical Analysis Wizard"); d.resize(900,650); l=QVBoxLayout(d); form=QFormLayout(); typ=QComboBox(); typ.addItems(["Descriptive Statistics","Correlation","Hypothesis Test","Regression"]); c1=QComboBox(); c1.addItems(cols); c2=QComboBox(); c2.addItems(cols); form.addRow("Analysis",typ); form.addRow("Variable / Group",c1); form.addRow("Value / Outcome",c2); l.addLayout(form); out=QTextEdit(); out.setReadOnly(True); l.addWidget(out); run=QPushButton("Run Analysis"); l.addWidget(run)
        def execute():
            """Perform the execute operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the show error analysis operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        results=self.agent_ds_state.get("ml_results") or self.agent_ds_state.get("dl_results")
        if not results: QMessageBox.warning(self,"No Model Results","Run Agent Data Scientist first."); return
        models=results.get("models") or {}; diagnostics={name:bundle.get("error_analysis") for name,bundle in models.items()} if isinstance(models,dict) else {}
        self._show_json_dialog("Error Analysis Workspace", {"task":results.get("task"),"target":results.get("target"),"models":diagnostics,"guidance":"Use slice analysis on validation/test predictions before deployment; inspect class-specific errors or residual structure rather than relying only on aggregate metrics."})

    def show_model_monitoring(self):
        """Perform the show model monitoring operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load current data first."); return
        path,_=QFileDialog.getOpenFileName(self,"Select Reference Dataset","","Data Files (*.csv *.xlsx *.parquet)")
        if not path:return
        try:
            reference=self.data_engine.read_file(path); result=ModelMonitoring.summary(reference,self.data_engine.df); self._show_json_dialog("Model Monitoring / Drift",result)
        except Exception as exc: QMessageBox.critical(self,"Monitoring Error",str(exc))

    def export_notebook(self):
        """Perform the export notebook operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None: QMessageBox.warning(self,"No Data","Load data first."); return
        path,_=QFileDialog.getSaveFileName(self,"Export Reproducible Notebook","analysis.ipynb","Jupyter Notebook (*.ipynb)")
        if not path:return
        nb=NotebookExporter.build({"rows":self.rows_input.text(),"columns":self.cols_input.text()},self.data_engine.source_path or "data.csv",self.rows_input.text(),self.cols_input.text(),self.chart_combo.currentText(),{k:v.currentText() for k,v in self.marks_widgets.items()},self.data_engine.filter_specs,self.agent_report_state.get("analysis_evidence")); NotebookExporter.save(path,nb); self.status.showMessage(f"Notebook exported to {path}")

    def build_publication_package(self):
        """Perform the build publication package operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        path,_=QFileDialog.getSaveFileName(self,"Create Publication Package","DataScienceStudioPro_Publication.zip","ZIP (*.zip)")
        if not path:return
        artifacts={"README.md":self._publication_readme()}
        if self.agent_report_state.get("last_report_path") and Path(self.agent_report_state["last_report_path"]).exists(): artifacts["AI_Agent_Report.pdf"]=self.agent_report_state["last_report_path"]
        if getattr(self,"last_presentation_path",None) and Path(self.last_presentation_path).exists(): artifacts["AI_Agent_Presentation.py"]=self.last_presentation_path
        if self.data_engine.df is not None: artifacts["data_dictionary.csv"]=self._write_temp_dictionary()
        PublicationPackageBuilder.build(path,artifacts); self.status.showMessage(f"Publication package created: {path}")

    def _write_temp_dictionary(self):
        """Perform the write temp dictionary operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        import tempfile
        p=Path(tempfile.gettempdir())/"dssp_data_dictionary.csv"; schema_table(self.data_engine.df).to_csv(p,index=False); return str(p)

    def _publication_readme(self):
        """Perform the publication readme operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return "Data Science Studio Pro publication package\n\nIncludes the recorded analysis outputs available at export time. Verify all scientific conclusions against the underlying dataset and methods.\n"

    def run_ai_agent_table_creation(self):
        """Perform the run ai agent table creation operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        d=QDialog(self); d.setWindowTitle("AI Agent Table Creation"); d.resize(760,520); l=QVBoxLayout(d); form=QFormLayout(); mode=QComboBox(); mode.addItems(["Web Scraping","API Key"]); url=QLineEdit(); api=QLineEdit(); api.setEchoMode(QLineEdit.EchoMode.Password); header=QLineEdit("Authorization"); prompt=QLineEdit(); form.addRow("Source method",mode); form.addRow("URL",url); form.addRow("API key",api); form.addRow("API header",header); form.addRow("Agent instruction",prompt); l.addLayout(form); out=QTextEdit(); out.setReadOnly(True); l.addWidget(out); run=QPushButton("Run AI Agent Table Creation"); l.addWidget(run)
        def execute():
            """Perform the execute operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            from agent.graph_table_creation import agent_table_creation_app
            state={"mode":mode.currentText(),"url":url.text(),"api_key":api.text(),"api_header":header.text(),"prompt":prompt.text(),"memory":getattr(self,"table_agent_memory",[])}; self.table_agent_worker=SimpleGraphWorker(agent_table_creation_app,state,{"configurable":{"thread_id":"table-creation"}}); self.table_agent_worker.finished.connect(lambda r:self.on_table_agent_finished(r,d)); self.table_agent_worker.error.connect(lambda e:out.setPlainText(e)); self.table_agent_worker.start(); out.setPlainText("LangGraph Table Creation Agent is running…")
        run.clicked.connect(execute); d.exec()

    def on_table_agent_finished(self,result,dialog):
        """Perform the on table agent finished operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.table_agent_memory=result.get("memory",[]); df=result.get("dataframe");
        if isinstance(df,pd.DataFrame): self.data_engine.set_active_dataframe(df); self.refresh_data_management(); self.populate_marks_combos(); self._sync_all_mark_fields(); self.update_plot()
        if result.get("status")=="complete": self.status.showMessage(result.get("summary","Table created.")); dialog.accept()
        else: QMessageBox.critical(self,"Table Creation Agent",result.get("error","Agent failed."))

    def run_ai_agent_presentation(self):
        """Perform the run ai agent presentation operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        evidence=self.agent_report_state.get("analysis_evidence")
        if not evidence: QMessageBox.warning(self,"No Report","Run AI Agent Report first. The presentation agent consumes the report evidence."); return
        cfg,check=self._agent_llm_preflight("AI Agent Presentation")
        if check.get("status")!="ready": QMessageBox.warning(self,"AI Provider Required","Configure the local LLM model or API key first."); return
        presenter,ok=QInputDialog.getText(self,"Presentation Presenter","Presenter name:",text="Data Science Studio Pro")
        if not ok:return
        path,_=QFileDialog.getSaveFileName(self,"Save PowerPoint Presentation","DataScienceStudioPro_Analysis_Presentation.pptx","PowerPoint (*.pptx)")
        if not path:return
        from agent.graph_presentation import agent_presentation_app
        state={"report_evidence":json_safe(evidence),"report_text":self.agent_report_state.get("report_text",""),"plot_results":json_safe(self.agent_ds_state.get("plot_results",{})),"memory":getattr(self,"presentation_agent_memory",[]),"output_path":path,"title":"Data Science Studio Pro — Analysis Presentation","presenter":presenter.strip() or "Data Science Studio Pro","llm_provider":cfg.provider,"model_name":cfg.model_name,"api_key":cfg.api_key,"local_path":cfg.local_path,"api_base":cfg.api_base}
        self.presentation_agent_worker=SimpleGraphWorker(agent_presentation_app,state,{"configurable":{"thread_id":"presentation"}}); self.presentation_agent_worker.finished.connect(self.on_presentation_agent_finished); self.presentation_agent_worker.error.connect(lambda e:QMessageBox.critical(self,"Presentation Agent",e)); self.presentation_agent_worker.start(); self.status.showMessage("AI Agent Presentation is generating the PowerPoint presentation…")

    def on_presentation_agent_finished(self,result):
        """Perform the on presentation agent finished operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.presentation_agent_memory=result.get("memory",[]); self.last_presentation_path=result.get("output_path");
        if result.get("status")=="complete": QMessageBox.information(self,"Presentation Created",result.get("summary")+"\n\nPowerPoint file: "+str(result.get("output_path")))
        else: QMessageBox.critical(self,"Presentation Agent",result.get("error","Presentation generation failed."))

    # ============================================================
    # SHARING
    # ============================================================
    def show_plot_context_menu(self, pos):
        """Perform the show plot context menu operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        menu=QMenu(self)
        menu.addAction("Format Plot…", self.format_plot_dialog)
        menu.addAction("Format Legend…", self.format_legend_dialog)
        menu.addSeparator()
        menu.addAction("Save PNG", lambda:self.export_plot_format("png"))
        menu.addAction("Save JPG", lambda:self.export_plot_format("jpg"))
        menu.addAction("Save PDF", lambda:self.export_plot_format("pdf"))
        menu.exec(self.canvas.mapToGlobal(pos))

    def format_plot_dialog(self):
        """Perform the format plot dialog operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        d=QDialog(self); d.setWindowTitle("Plot Formatting and Export"); d.resize(520,460); l=QVBoxLayout(d); form=QFormLayout()
        title=QLineEdit(self.ax.get_title()); xlabel=QLineEdit(self.ax.get_xlabel()); ylabel=QLineEdit(self.ax.get_ylabel()); size=QSpinBox(); size.setRange(6,32); size.setValue(11)
        font_btn=QPushButton("Choose font…"); title_color=QPushButton("Choose…"); axis_color=QPushButton("Choose…"); grid=QCheckBox("Show grid"); grid.setChecked(any(line.get_visible() for line in self.ax.get_xgridlines()))
        form.addRow("Plot title",title); form.addRow("X-axis title",xlabel); form.addRow("Y-axis title",ylabel); form.addRow("Font",font_btn); form.addRow("Font size",size); form.addRow("Title / text color",title_color); form.addRow("Axis color",axis_color); form.addRow("Grid",grid); l.addLayout(form)
        colors={"title":None,"axis":None}; chosen_font={"font":self.ax.title.get_fontproperties().get_name()}
        def choose_font():
            """Perform the choose font operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            f,ok=QFontDialog.getFont(QFont(chosen_font["font"],size.value()),d,"Plot Font")
            if ok: chosen_font["font"]=f.family(); size.setValue(f.pointSize())
        font_btn.clicked.connect(choose_font)
        title_color.clicked.connect(lambda: colors.__setitem__("title",QColorDialog.getColor()))
        axis_color.clicked.connect(lambda: colors.__setitem__("axis",QColorDialog.getColor()))
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Apply|QDialogButtonBox.StandardButton.Cancel); l.addWidget(bb)
        def apply():
            """Perform the apply operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            title_kwargs={"fontsize":size.value(),"fontname":chosen_font["font"]}
            if colors["title"] is not None and colors["title"].isValid(): title_kwargs["color"]=colors["title"].name()
            self.ax.set_title(title.text(),**title_kwargs)
            self.ax.set_xlabel(xlabel.text(),fontsize=size.value(),fontname=chosen_font["font"]); self.ax.set_ylabel(ylabel.text(),fontsize=size.value(),fontname=chosen_font["font"])
            for label in self.ax.get_xticklabels()+self.ax.get_yticklabels(): label.set_fontname(chosen_font["font"]); label.set_fontsize(size.value())
            if colors["axis"] and colors["axis"].isValid():
                c=colors["axis"].name(); self.ax.tick_params(axis="both",colors=c); [sp.set_color(c) for sp in self.ax.spines.values()]
            self.ax.grid(grid.isChecked(),alpha=.25); self.canvas.draw_idle(); self.status.showMessage("Plot formatting applied.")
        bb.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(apply); bb.rejected.connect(d.reject); d.exec()

    def export_plot_format(self, fmt):
        """Perform the export plot format operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        filters={"png":"PNG Files (*.png)","jpg":"JPEG Files (*.jpg *.jpeg)","pdf":"PDF Files (*.pdf)"}; path,_=QFileDialog.getSaveFileName(self,f"Save Plot as {fmt.upper()}",f"plot.{fmt}",filters[fmt])
        if not path:return
        try:
            self.fig.savefig(path,dpi=300,bbox_inches="tight",format=fmt)
            self.status.showMessage(f"Plot exported to {path}")
        except Exception as exc: QMessageBox.critical(self,"Plot Export",str(exc))

    def export_png(self):
        """Perform the export png operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        filepath, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG Files (*.png)")
        if filepath:
            self.fig.savefig(filepath, dpi=150)
            self.status.showMessage(f"Exported to {filepath}")

    def email_current_sheet(self):
        """Perform the email current sheet operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.create_sharing_package()

    def email_story(self):
        """Perform the email story operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.create_sharing_package()

    def export_to_streamlit(self):
        """Perform the export to streamlit operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.run_ai_agent_presentation()

    def create_sharing_package(self):
        """Perform the create sharing package operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        path,_=QFileDialog.getSaveFileName(self,"Create Collaboration Package","DataScienceStudioPro_Share.zip","ZIP (*.zip)")
        if not path:return
        artifacts={"analysis_snapshot.json":json.dumps(self._analysis_snapshot_dict(),indent=2,default=str).encode("utf-8"),"view_summary.txt":self._view_summary()}
        if self.agent_report_state.get("last_report_path") and Path(self.agent_report_state["last_report_path"]).exists(): artifacts["AI_Agent_Report.pdf"]=Path(self.agent_report_state["last_report_path"]).read_bytes()
        if getattr(self,"last_presentation_path",None) and Path(self.last_presentation_path).exists(): artifacts["AI_Agent_Presentation.py"]=Path(self.last_presentation_path).read_text(encoding="utf-8")
        SharingService.build_bundle(path,artifacts); self.last_share_package_path=path; self.status.showMessage(f"Collaboration package created: {path}")
        QMessageBox.information(self,"Collaboration Package", f"Package created successfully.\n\n{path}\n\nYou can place this ZIP on your approved team drive, send it to a colleague, or attach it to a project record.")

    def copy_share_package_path(self):
        """Perform the copy share package path operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        path=getattr(self,"last_share_package_path","")
        if not path: QMessageBox.information(self,"Sharing","Create a Collaboration Package first."); return
        QApplication.clipboard().setText(path); self.status.showMessage("Share package path copied to clipboard.")

    def _view_summary(self):
        """Perform the view summary operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return json.dumps({"rows":self.rows_input.text(),"columns":self.cols_input.text(),"chart":self.chart_combo.currentText(),"marks":{k:v.currentText() for k,v in self.marks_widgets.items()},"filters":self.data_engine.filter_specs},indent=2,default=str)

    def _analysis_snapshot_dict(self):
        """Perform the analysis snapshot dict operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return {"dataset_fingerprint":self.data_engine.dataset_hash(),"source":self.data_engine.source_path,"view":{ "rows":self.rows_input.text(),"columns":self.cols_input.text(),"chart":self.chart_combo.currentText(),"marks":{k:v.currentText() for k,v in self.marks_widgets.items()}},"filters":self.data_engine.filter_specs,"evidence":self.evidence_dag.to_dict()}

    # ============================================================
    # TOOLBAR ACTIONS
    # ============================================================
    def undo(self):
        """Perform the undo operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if not self.history: self.status.showMessage("Nothing to undo."); return
        current={"rows":self.rows_input.text(),"columns":self.cols_input.text(),"chart":self.chart_combo.currentText(),"marks":{k:v.currentText() for k,v in self.marks_widgets.items()}}
        self.redo_history.append(current); state=self.history.pop()
        self.rows_input.setText(state["rows"]); self.cols_input.setText(state["columns"]); self.chart_combo.setCurrentText(state["chart"])
        for k,v in state["marks"].items(): self.marks_widgets[k].setCurrentText(v)
        self.update_plot(); self.status.showMessage("View change undone.")

    def redo(self):
        """Perform the redo operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if not self.redo_history: self.status.showMessage("Nothing to redo."); return
        current={"rows":self.rows_input.text(),"columns":self.cols_input.text(),"chart":self.chart_combo.currentText(),"marks":{k:v.currentText() for k,v in self.marks_widgets.items()}}
        self.history.append(current); state=self.redo_history.pop()
        self.rows_input.setText(state["rows"]); self.cols_input.setText(state["columns"]); self.chart_combo.setCurrentText(state["chart"])
        for k,v in state["marks"].items(): self.marks_widgets[k].setCurrentText(v)
        self.update_plot(); self.status.showMessage("View change redone.")

    def sort_asc(self):
        """Perform the sort asc operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._sort_plot_axis(False)
    def sort_desc(self):
        """Perform the sort desc operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._sort_plot_axis(True)
    def _sort_plot_axis(self, descending=False):
        """Perform the sort plot axis operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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

    def create_story(self):
        """Perform the create story operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.manage_sheets(mode="story")
    def manage_sheets(self, mode="sheet"):
        """Perform the manage sheets operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._open_tableau_workspace(mode)


    def _add_evidence(self, evidence):
        """Perform the add evidence operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.evidence_records.append(evidence); self.evidence_dag.add(evidence)

    def show_leakage_gate(self):
        """Perform the show leakage gate operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None: QMessageBox.information(self,"Leakage Gate","Load data first."); return
        gate=self.leakage_gate.evaluate(self.data_engine.df,self.data_engine.df.columns[-1]); QMessageBox.information(self,"Scientific/Data Leakage Gate",json.dumps(gate,indent=2,default=str))
    def show_dataset_card(self):
        """Perform the show dataset card operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None: QMessageBox.information(self,"Dataset Card","Load data first."); return
        self.dataset_card=DatasetCardBuilder.build(self.data_engine.df); self._show_json_dialog("Dataset Card",self.dataset_card)
    def show_model_cards(self):
        """Perform the show model cards operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._show_json_dialog("Model Cards",self.model_cards or {"message":"No model cards yet."})
    def show_experiment_comparison(self):
        """Perform the show experiment comparison operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        df=ExperimentComparator.compare(self.experiment_records)
        if df.empty: self._show_text_dialog("Experiment Comparison 2.0","No experiment comparison records yet."); return
        # Add reproducibility-oriented context without ranking political-style or arbitrary winners.
        self._show_text_dialog("Experiment Comparison 2.0", df.to_string(index=False) + "\n\nCompare dataset fingerprint, task, model, metrics, uncertainty and recorded preprocessing before drawing conclusions.")
    def show_data_diff(self):
        """Perform the show data diff operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.original_df is None or self.data_engine.df is None: QMessageBox.information(self,"Data Diff","No source/active dataframe pair is available."); return
        self._show_json_dialog("Data Diff",DataDiff.compare(self.data_engine.original_df,self.data_engine.df))
    def show_evidence_dag(self):
        """Perform the show evidence dag operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the show model promotion operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
    def show_agent_evaluation(self):
        """Perform the show agent evaluation operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._show_json_dialog("Agent Evaluation",AgentEvaluation.evaluate(self.agent_ds_state))
    def show_voice_help(self):
        """Perform the show voice help operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        d=QDialog(self); d.setWindowTitle("Voice Command & Accessibility"); d.resize(620,420); l=QVBoxLayout(d)
        l.addWidget(QLabel("Professional voice control is optional and never bypasses human approval. Only unambiguous commands are accepted."))
        status=QLabel("Status: disabled"); l.addWidget(status)
        mode=QComboBox(); mode.addItems(["Approval commands only","Approval + navigation commands"]); l.addWidget(mode)
        test=QPushButton("Test Microphone / Speech Recognition"); toggle=QPushButton("Enable Voice Approval"); close=QPushButton("Close"); row=QHBoxLayout(); row.addWidget(test); row.addWidget(toggle); row.addWidget(close); l.addLayout(row)
        def refresh():
            """Perform the refresh operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            enabled=os.environ.get("DSP_VOICE_APPROVAL","0")=="1"; status.setText("Status: ENABLED" if enabled else "Status: disabled"); toggle.setText("Disable Voice Approval" if enabled else "Enable Voice Approval")
        def toggle_fn():
            """Perform the toggle fn operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            os.environ["DSP_VOICE_APPROVAL"]="0" if os.environ.get("DSP_VOICE_APPROVAL","0")=="1" else "1"; refresh()
        def test_fn():
            """Perform the test fn operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            heard=self.voice_layer.listen_once("Please say a short command such as help or approve."); QMessageBox.information(d,"Voice Test",f"Recognized text:\n{heard or '[No unambiguous speech recognized]'}")
        test.clicked.connect(test_fn); toggle.clicked.connect(toggle_fn); close.clicked.connect(d.accept); refresh(); d.exec()
    def start_voice_command(self):
        """Perform the start voice command operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.show_voice_help()
    def stop_agent_run(self):
        """Perform the stop agent run operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.agent_abort_requested=True
        if self.agent_worker and self.agent_worker.isRunning(): self.agent_ds_state["abort_requested"]=True
        self.status.showMessage("Stop requested. The current human gate will terminate the run safely.")
    def show_help_center(self, initial_topic=None):
        """Perform the show help center operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        d=QDialog(self); d.setWindowTitle("Data Science Studio Pro — Professional Help Center"); d.resize(1100,720); root=QVBoxLayout(d)
        top=QHBoxLayout(); search=QLineEdit(); search.setPlaceholderText("Search Help: shelves, classification, regression, agents, GPU, stories, voice..."); top.addWidget(search); root.addLayout(top)
        split=QSplitter(); root.addWidget(split,1); topics=QListWidget(); content=QTextEdit(); content.setReadOnly(True); split.addWidget(topics); split.addWidget(content); split.setSizes([280,800])
        docs={
        "Quick Start":"1) File → Open CSV/Excel/PDF. 2) Review Data Info, Data Quality and Core Schema Types. 3) Drag fields into Rows/Columns and configure Marks. 4) Use Sheets to preserve alternative views. 5) Configure an AI provider before any governed AI agent. 6) Run Agent Data Scientist and review every approval gate. 7) Run AI Agent Plot. 8) Run AI Agent Report. 9) Create the PowerPoint presentation. 10) Export the analysis package when the review is complete.",
        "Rows / Columns / Aggregation":"Rows and Columns are analytical shelves. Drag one or more fields into either shelf. Right-click an individual field to choose Automatic, Sum, Average, Count, Minimum, Maximum, Median or Count Distinct, or remove the field. Aggregation is stored per shelf field and is part of the sheet view state.",
        "Filters":"Drag a field from Data Management to Filters. Numerical filters can use ranges; categorical filters can use values. Filters are part of the active analytical view and must be considered when interpreting visual results.",
        "Marks":"Color, Size, Text, Detail and Tooltip are Tableau-like encodings. Normal drag assigns a field; multi-field properties support explicit addition. Right-click → Clear clears the actual selection. Categorical Color creates a legend. Numeric Color is treated as a continuous encoding. Size accepts one field.",
        "Classification Analysis":"The classification workspace isolates a locked test partition, fits imputation/OneHotEncoder/scaling inside training folds, supports optional PCA, performs k-fold CV and hyperparameter search, and reports accuracy, balanced accuracy, precision, recall, F1, confusion matrix, ROC-AUC, PR-AUC, log loss and Brier/calibration evidence where available. Candidate families include Logistic Regression, SVM, Random Forest, Decision Tree, KNN, Naive Bayes and Neural Network, with optional XGBoost/LightGBM when installed. Use class balance and error cost—not accuracy alone—to interpret results.",
        "Regression Analysis":"The regression workspace uses the same leakage-safe pipeline and supports OLS, Ridge, Lasso, Elastic Net, Polynomial Regression, SVR, Decision Tree, Random Forest, Gradient Boosting and Neural Network, plus optional XGBoost/LightGBM. Review MAE, RMSE, R², median absolute error, bias and MAPE where meaningful. Hyperparameter tuning is training-only and the final test partition is locked.",
        "Clustering Analysis":"Professional clustering analysis with approved numeric feature selection, imputation/scaling, K-Means, Agglomerative, Gaussian Mixture and DBSCAN comparison, silhouette/Calinski-Harabasz/Davies-Bouldin metrics, cluster profiles, PCA visualization, repeatability checks and export.",
        "Agent Data Scientist":"The Master Agent uses LangGraph, layered analytical memory, evidence-bound method selection, validation-strategy planning, challenger methods, and specialist agents for regression, classification, clustering, unsupervised learning, anomaly detection, time series, reinforcement learning and deep learning. LangGraph governs one stage at a time. The normal path is Plan → Critic → Self-Check → Approval → Validation → Approval → Preprocessing → Approval → Model Selection → Approval → Execution → Approval → Evaluation → Approval → Robustness → Approval → Diagnosis → Approval → Verification → Approval → Stop. Each approval dialog now shows very brief evidence/result points for the current stage. Reject revises the stage; Abort stops safely.",
        "AI Provider Setup":"AI Agents are provider-gated. Use AI Agents → Select Local LLM model or API key. Choose a cached local HuggingFace model or a cloud model plus API key. The application preflights local model availability and resource risk. A 2 GB MX250 is not used for large local LLM weights; low-VRAM workloads are kept on CPU or bounded CUDA paths. Agent Data Scientist cannot start until the provider gate is ready.",
        "AI Agent Plot":"Agent Plot consumes the Data Scientist target, validation and model evidence. LangGraph combines deterministic analytical quantities with LLM-assisted candidate selection. Typical quantities include correlations, target distributions, category/measure comparisons, temporal trends and model metrics. Each tab provides the evidence rationale and a professional plot editor for title/font/color/background plus PNG/JPG/PDF/HTML export.",
        "AI Agent Report":"The Report Agent consumes Data Scientist evidence, Agent Plot evidence and the configured LLM. The report follows a professional structure inspired by the supplied scientific-report example: title page, table of contents, executive summary, introduction, data/governance, methods, results, evaluation/uncertainty/robustness, visual results, discussion, conclusion, recommendations, human oversight and evidence appendix. It never invents missing metrics.",
        "AI Agent Presentation":"The Presentation Agent consumes the Report, Agent Plot and Data Scientist evidence through LangGraph and the configured LLM. It produces a real PowerPoint .pptx with title/date/presenter, introduction, methods, governance, model results, analytical visualizations, interpretation, conclusion, acknowledgment and evidence appendix. The agent state is JSON/msgpack-safe; DataFrames are summarized before checkpointing.",
        "Macro Python / SQL Workspace":"Macro → Python and Macro → SQL are multi-command workspaces. You may edit starter commands, combine several commands, run all commands or run a selected block, and inspect output/history on the same page. Python exposes df/pandas/numpy; SQL exposes the active dataset as DuckDB table `data`. Multiple SQL statements are executed sequentially and each result is reported separately.",
        "Sheets and Stories":"Sheets are first-class analytical views with dataset, configuration, evidence and provenance context. Stories are presentation canvases that organize Sheets into an evidence-linked narrative. Sheets preserve independent rows/columns/Marks/chart configurations. Use Sheets to create, duplicate, rename, delete and activate views. In Stories, select multiple plotted sheets, add them as story points, reorder them, edit captions, set a story title/background, drag cards to position them, and resize cards using the lower-right resize handle. Export the composed story as PNG or PDF.",
        "Voice and Accessibility":"Voice is opt-in. Accessibility → Start Voice Command opens the control panel. At approval gates only unambiguous approve/reject/pause/abort commands are accepted. Ambiguous speech never grants approval. Test the microphone before enabling voice approval.",
        "CPU / GPU":"CPU is always the safe baseline. CPU+GPU is opportunistic: if NVIDIA hardware, a CUDA-enabled PyTorch build or runtime validation is missing, the workload falls back to CPU rather than crashing. GPU is strict. Hardware detection uses nvidia-smi plus PyTorch inspection and a separate CUDA smoke test. Low-VRAM devices such as an MX250 2 GB are treated as bounded accelerators, not large-model devices.",
        "Troubleshooting":"For GPU issues, distinguish hardware detection, driver/nvidia-smi, PyTorch CUDA build, compute capability and runtime smoke-test status. For local LLM memory failures, use a smaller model or an API provider. For convergence warnings, the application raises safe iteration limits and records convergence diagnostics instead of flooding the terminal. For presentation failures, ensure python-pptx is installed and verify that the selected provider is configured. For plot rendering/export, verify kaleido.",
        "Reproducibility":"Analysis Recipes, dataset fingerprints, model cards, evidence IDs, seeds, approval decisions, validation protocols and diagnostic artifacts are recorded where applicable. Recipe replay never bypasses current validation or human approval.",
        "About":"Data Science Studio Pro is an evidence-first professional desktop data-analysis environment. Its design principle is data → quality → leakage/split controls → governed analysis → diagnostics → visual evidence → report → PowerPoint → reproducible sharing and monitoring, with explicit human control over consequential agent stages."}
        for k in docs: topics.addItem(k)
        def show(item):
            """Perform the show operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            content.setPlainText(docs.get(item.text(),""))
        topics.currentItemChanged.connect(lambda cur,prev: show(cur) if cur else None)
        if initial_topic and initial_topic in docs: topics.setCurrentRow(list(docs.keys()).index(initial_topic))
        else: topics.setCurrentRow(0)
        search.textChanged.connect(lambda q: [topics.item(i).setHidden(bool(q.strip()) and q.lower() not in topics.item(i).text().lower() and q.lower() not in docs[topics.item(i).text()].lower()) for i in range(topics.count())])
        d.exec()

    def show_help_topic(self, topic):
        """Perform the show help topic operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        mapping={"quickstart":"Quick Start","shelves":"Rows / Columns / Aggregation","filters":"Filters","classification":"Classification Analysis","regression":"Regression Analysis","clustering":"Clustering Analysis","views":"Sheets and Stories","agents":"Agent Data Scientist","provider":"AI Provider Setup","macros":"Macro Python / SQL Workspace","report_presentation":"AI Agent Presentation","compute":"CPU / GPU","charts":"Marks","troubleshooting":"Troubleshooting","about":"About"}
        if topic in mapping:
            # Reuse the Help Center so every topic has searchable professional guidance.
            self.show_help_center(mapping[topic]); return
        self._show_text_dialog("Data Science Studio Pro Help", "Use Help Center for searchable documentation.")

    def _show_text_dialog(self,title,text):
        """Perform the show text dialog operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        d=QDialog(self); d.setWindowTitle(title); d.resize(900,650); l=QVBoxLayout(d); w=QTextEdit(); w.setReadOnly(True); w.setFont(QFont("Arial",10)); w.setPlainText(str(text)); l.addWidget(w); d.exec()
    def _show_json_dialog(self,title,obj):
        """Perform the show json dialog operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._show_text_dialog(title,json.dumps(obj,indent=2,default=str))

    def _open_tableau_workspace(self, mode="sheet"):
        """Professional Sheet/Story workspace. Main application geometry is unchanged."""
        if self.data_engine.df is None:
            QMessageBox.warning(self,"No Data","Load data first."); return
        d=QDialog(self); d.setWindowTitle("Sheets / Dashboards / Stories — Professional Workspace"); d.resize(1200,820); root=QVBoxLayout(d)
        tabs=QTabWidget(); root.addWidget(tabs)

        def snapshot(name):
            """Perform the snapshot operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            return {
                "name":name,
                "rows":self.viz_engine.parse_shelf(self.rows_input.text()),
                "columns":self.viz_engine.parse_shelf(self.cols_input.text()),
                "chart":self.chart_combo.currentText(),
                "marks":{k:self._mark_values(k) for k in self.marks_widgets},
                "shelf_aggregations":json.loads(json.dumps(self.shelf_aggregations,default=str)),
                "agg_func":self.sheet_manager.get_sheet_config(self.sheet_manager.active_sheet).get("agg_func","sum"),
                "data":self.data_engine.df.copy(),
            }

        def render_snapshot(snap, canvas=None):
            """Perform the render snapshot operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            fig=(canvas.figure if canvas is not None else Figure(figsize=(9,5),dpi=100)); fig.clear(); ax=fig.add_subplot(111)
            df=snap.get("data") if isinstance(snap.get("data"),pd.DataFrame) else self.data_engine.df
            rows=[c for c in snap.get("rows",[]) if c in df.columns]; cols=[c for c in snap.get("columns",[]) if c in df.columns]
            try:
                agg=(snap.get("agg_func") or "sum")
                data=self.viz_engine.aggregate_for_shelves(df,rows,cols,agg)
                nums=[c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]; dims=[c for c in data.columns if c not in nums]
                chart=snap.get("chart","auto"); y=nums[0] if nums else (data.columns[-1] if len(data.columns) else None); x=dims[0] if dims else None
                if data.empty or y is None: ax.text(.5,.5,"Empty analytical view",ha="center",va="center")
                elif chart in {"line","line_discrete","line_continuous","area"} and x: ax.plot(data[x].astype(str),data[y],marker="o"); ax.tick_params(axis="x",rotation=45)
                elif chart=="scatter" and len(nums)>=2: ax.scatter(data[nums[0]],data[nums[1]],s=40); ax.set_xlabel(nums[0]); ax.set_ylabel(nums[1])
                elif chart in {"pie","donut"} and x: ax.pie(data[y],labels=data[x].astype(str),autopct="%1.1f%%");
                elif x: ax.bar(data[x].astype(str),data[y]); ax.tick_params(axis="x",rotation=45); ax.set_ylabel(y)
                else: ax.bar([y],[float(data[y].iloc[0])]); ax.set_ylabel(y)
                ax.set_title(snap.get("name","Sheet"),fontsize=12); fig.subplots_adjust(left=.08,right=.97,bottom=.22,top=.88)
            except Exception as exc: ax.text(.5,.5,f"View error: {exc}",ha="center",va="center")
            if canvas is not None: canvas.draw()
            return fig

        # -------- Sheets tab --------
        sheet_tab=QWidget(); sl=QVBoxLayout(sheet_tab); toolbar=QHBoxLayout()
        newb=QPushButton("New Sheet"); dupb=QPushButton("Duplicate"); renameb=QPushButton("Rename"); delb=QPushButton("Delete"); applyb=QPushButton("Use Selected Sheet")
        for b in (newb,dupb,renameb,delb,applyb): toolbar.addWidget(b)
        toolbar.addStretch(); sl.addLayout(toolbar)
        split=QSplitter(); sl.addWidget(split,1); sheet_list=QListWidget(); preview=FigureCanvas(Figure(figsize=(9,5),dpi=100)); split.addWidget(sheet_list); split.addWidget(preview); split.setSizes([260,900])
        for name in self.sheet_manager.sheets: sheet_list.addItem(name)

        def selected():
            """Perform the selected operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            i=sheet_list.currentItem(); return i.text() if i else None
        def get_snap(name):
            """Perform the get snap operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            cfg=self.sheet_manager.sheets.get(name,{})
            if "view_snapshot" in cfg:return cfg["view_snapshot"]
            return {"name":name,"rows":self.viz_engine.parse_shelf(self.rows_input.text()),"columns":self.viz_engine.parse_shelf(self.cols_input.text()),"chart":cfg.get("mark_type","Bar").lower(),"marks":{},"shelf_aggregations":{},"agg_func":cfg.get("agg_func","sum"),"data":cfg.get("data",self.data_engine.df.copy())}
        def refresh_preview():
            """Perform the refresh preview operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            name=selected()
            if name: render_snapshot(get_snap(name),preview)
        def create_sheet():
            """Perform the create sheet operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            name=f"Sheet {len(self.sheet_manager.sheets)+1}"
            snap=snapshot(name); self.sheet_manager.sheets[name]={"data":self.data_engine.df.copy(),"agg_func":"sum","view_snapshot":snap,"x_col":None,"y_col":None,"mark_type":self.chart_combo.currentText(),"color_col":self._mark_value("Color"),"size_col":self._mark_value("Size")}
            sheet_list.addItem(name); sheet_list.setCurrentRow(sheet_list.count()-1); refresh_preview()
        def duplicate_sheet():
            """Perform the duplicate sheet operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            name=selected()
            if not name:return
            new=f"{name} Copy"; cfg=dict(self.sheet_manager.sheets[name]); cfg["view_snapshot"]=dict(cfg.get("view_snapshot",get_snap(name))); cfg["view_snapshot"]["name"]=new; self.sheet_manager.sheets[new]=cfg; sheet_list.addItem(new); sheet_list.setCurrentRow(sheet_list.count()-1); refresh_preview()
        def rename_sheet():
            """Perform the rename sheet operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            name=selected()
            if not name:return
            new,ok=QInputDialog.getText(d,"Rename Sheet","New sheet name:",text=name)
            if ok and new.strip() and new.strip()!=name:
                cfg=self.sheet_manager.sheets.pop(name); cfg["view_snapshot"]=dict(cfg.get("view_snapshot",{})); cfg["view_snapshot"]["name"]=new.strip(); self.sheet_manager.sheets[new.strip()]=cfg; sheet_list.currentItem().setText(new.strip()); refresh_preview()
        def delete_sheet():
            """Perform the delete sheet operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            name=selected()
            if not name:return
            if len(self.sheet_manager.sheets)<=1: QMessageBox.warning(d,"Sheets","At least one sheet must remain."); return
            self.sheet_manager.sheets.pop(name,None); sheet_list.takeItem(sheet_list.currentRow()); refresh_preview()
        def use_sheet():
            """Perform the use sheet operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            name=selected()
            if not name:return
            snap=get_snap(name); self.rows_input.setText(self.viz_engine.format_shelf(snap.get("rows",[]))); self.cols_input.setText(self.viz_engine.format_shelf(snap.get("columns",[]))); self.chart_combo.setCurrentText(snap.get("chart","auto"))
            for k,vals in snap.get("marks",{}).items():
                if k in self.marks_widgets: self.marks_widgets[k].set_selected_fields(vals if isinstance(vals,list) else [vals])
            self.shelf_aggregations=snap.get("shelf_aggregations",{"Rows":{},"Columns":{}}); self.update_plot(); self.status.showMessage(f"Activated sheet: {name}")
        sheet_list.currentItemChanged.connect(lambda cur,prev: refresh_preview()); newb.clicked.connect(create_sheet); dupb.clicked.connect(duplicate_sheet); renameb.clicked.connect(rename_sheet); delb.clicked.connect(delete_sheet); applyb.clicked.connect(use_sheet)
        tabs.addTab(sheet_tab,"Sheets")

        # -------- Story tab --------
        story_tab=QWidget(); gl=QVBoxLayout(story_tab)
        controls=QHBoxLayout(); story_name=QLineEdit("Story 1"); new_story=QPushButton("New Story"); add=QPushButton("Add Selected Sheets"); remove=QPushButton("Remove"); up=QPushButton("Move Up"); down=QPushButton("Move Down"); bg=QPushButton("Background"); controls.addWidget(QLabel("Story title:")); controls.addWidget(story_name); controls.addWidget(new_story); controls.addWidget(add); controls.addWidget(remove); controls.addWidget(up); controls.addWidget(down); controls.addWidget(bg); controls.addStretch(); gl.addLayout(controls)
        body=QSplitter(); gl.addWidget(body,1); available=QListWidget(); available.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection); ordered=QListWidget(); ordered.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove); body.addWidget(available); body.addWidget(ordered); body.setSizes([260,260])
        editor=QVBoxLayout(); editor_w=QWidget(); editor_w.setLayout(editor); body.addWidget(editor_w); caption=QTextEdit(); caption.setPlaceholderText("Story Point Text / analytical explanation…"); caption.setMaximumHeight(120); editor.addWidget(QLabel("Story Point Text")); editor.addWidget(caption)
        story_view=StoryCanvasView(); editor.addWidget(story_view,1); exportrow=QHBoxLayout(); pngb=QPushButton("Export Story PNG"); pdfb=QPushButton("Export Story PDF"); exportrow.addWidget(pngb); exportrow.addWidget(pdfb); editor.addLayout(exportrow)
        for name in self.sheet_manager.sheets: available.addItem(name)
        story_data={"name":"Story 1","items":[],"captions":{},"background":"#f3f6f8"}
        def get_ordered():
            """Perform the get ordered operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            return [ordered.item(i).text() for i in range(ordered.count())]
        def add_story_cards():
            """Perform the add story cards operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            story_view.clear_cards(); story_view.set_background(QColor(story_data.get("background","#f3f6f8"))); story_view.add_title(story_data.get("name","Story 1"))
            x,y=25,65
            for i,name in enumerate(get_ordered()):
                snap=get_snap(name); fig=render_snapshot(snap); buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=100,bbox_inches="tight"); buf.seek(0); pm=__import__("PyQt6.QtGui",fromlist=["QPixmap"]).QPixmap(); pm.loadFromData(buf.getvalue(),"PNG"); card=StoryCardItem(QRectF(0,0,380,250),pm,name); card.setPos(x,y); story_view.scene().addItem(card); x+=400
                if x+380>1150: x=25; y+=275
            story_view.scene().setSceneRect(0,0,max(1200,x+400),max(600,y+300))
        def add_sheet_story():
            """Perform the add sheet story operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            for item in available.selectedItems():
                name=item.text()
                if name not in story_data["items"]: story_data["items"].append(name); ordered.addItem(name); story_data["captions"][name]=""
            add_story_cards()
        def remove_story():
            """Perform the remove story operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            i=ordered.currentRow()
            if i>=0: story_data["items"].pop(i) if i<len(story_data["items"]) else None; ordered.takeItem(i); add_story_cards()
        def move(delta):
            """Perform the move operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            i=ordered.currentRow(); j=i+delta
            if i<0 or j<0 or j>=ordered.count():return
            item=ordered.takeItem(i); ordered.insertItem(j,item); ordered.setCurrentRow(j); story_data["items"]=get_ordered(); add_story_cards()
        def caption_changed():
            """Perform the caption changed operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            item=ordered.currentItem()
            if item: story_data["captions"][item.text()]=caption.toPlainText()
        def select_story_item(cur,prev):
            """Perform the select story item operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            if cur: caption.setPlainText(story_data["captions"].get(cur.text(),""))
        def new_story_fn():
            """Perform the new story fn operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            story_data["name"]=story_name.text().strip() or f"Story {len(self.story_workspace_state.get('stories',[]))+1}"; story_data["items"]=[]; story_data["captions"]={}; ordered.clear(); caption.clear(); self.active_story_name=story_data["name"]; self.story_workspace_state.setdefault("stories",[]); existing=[x for x in self.story_workspace_state["stories"] if x.get("name")!=story_data["name"]]; existing.append({"name":story_data["name"],"items":[]}); self.story_workspace_state["stories"]=existing; self._refresh_navigation_combos(); add_story_cards()
        def choose_bg():
            """Perform the choose bg operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            c=QColorDialog.getColor(QColor(story_data.get("background","#f3f6f8")),d,"Story Background")
            if c.isValid(): story_data["background"]=c.name(); add_story_cards()
        def export_story(fmt):
            """Perform the export story operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            if not story_data["items"]: QMessageBox.warning(d,"Story","Add at least one plotted sheet to the story."); return
            path,_=QFileDialog.getSaveFileName(d,f"Export Story as {fmt.upper()}",f"{story_data['name']}.{fmt}",f"{fmt.upper()} Files (*.{fmt})")
            if not path:return
            try:
                if fmt=="png": story_view.export_png(path)
                else:
                    from matplotlib.backends.backend_pdf import PdfPages
                    from matplotlib.figure import Figure
                    with PdfPages(path) as pdf:
                        fig=Figure(figsize=(13.333,7.5),dpi=120); canvas=FigureCanvas(fig); render_snapshot(get_snap(story_data["items"][0]),canvas); pdf.savefig(fig,bbox_inches="tight")
                        for name in story_data["items"][1:]:
                            fig=Figure(figsize=(13.333,7.5),dpi=120); canvas=FigureCanvas(fig); render_snapshot(get_snap(name),canvas); pdf.savefig(fig,bbox_inches="tight")
                self.status.showMessage(f"Story exported to {path}")
            except Exception as exc: QMessageBox.critical(d,"Story Export",str(exc))
        add.clicked.connect(add_sheet_story); remove.clicked.connect(remove_story); up.clicked.connect(lambda:move(-1)); down.clicked.connect(lambda:move(1)); ordered.currentItemChanged.connect(select_story_item); caption.textChanged.connect(caption_changed); new_story.clicked.connect(new_story_fn); bg.clicked.connect(choose_bg); pngb.clicked.connect(lambda:export_story("png")); pdfb.clicked.connect(lambda:export_story("pdf")); add_story_cards(); tabs.addTab(story_tab,"Stories")
        d.exec()
        self._refresh_navigation_combos()

    # ============================================================
    # DATA MANAGEMENT REFRESH
    # ============================================================
    def refresh_data_management(self):
        """Perform the refresh data management operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the populate marks combos operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        cols=[] if self.data_engine.df is None else [str(c) for c in self.data_engine.df.columns]
        for combo in [self.marks_widgets.get("Color"),self.marks_widgets.get("Size"),self.marks_widgets.get("Text"),self.marks_widgets.get("Detail"),self.marks_widgets.get("Tooltip")]:
            if combo: combo.sync_fields(cols)

    def show_column_menu(self, pos):
        """Perform the show column menu operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the create hierarchy operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if self.data_engine.df is None: return
        cols=list(self.data_engine.df.columns); ordered=[root_col]+[c for c in cols if c!=root_col]
        selected,ok=QInputDialog.getItem(self,"Create Hierarchy",f"Select next level after '{root_col}':",[c for c in ordered if c!=root_col],0,False)
        if not ok:return
        self.hierarchies[root_col]=[root_col,selected]
        QMessageBox.information(self,"Hierarchy Created",f"Hierarchy: {' → '.join(self.hierarchies[root_col])}\nRight-click the field on Rows/Columns to drill down.")
    def drill_down_field(self, widget, field):
        """Perform the drill down field operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        hierarchy=next((h for h in self.hierarchies.values() if field in h),None)
        if not hierarchy:return
        i=hierarchy.index(field)
        if i+1>=len(hierarchy): QMessageBox.information(self,"Hierarchy","Already at the lowest level."); return
        fields=self.viz_engine.parse_shelf(widget.text()); fields=[hierarchy[i+1] if x==field else x for x in fields]; widget.setText(self.viz_engine.format_shelf(fields)); self.update_plot()

    def add_column_to_sheet(self, col):
        """Perform the add column to sheet operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self._append_shelf(self.rows_input, col)
        self._record_view_change()
        self.update_plot()

    def duplicate_column(self, col):
        """Perform the duplicate column operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        new_name, ok = QInputDialog.getText(self, "Duplicate Column", f"New name for copy of '{col}':", text=f"{col}_copy")
        if ok and new_name.strip():
            work = self.data_engine.df.copy(); work[new_name] = work[col]
            self.data_engine.commit_dataframe(work, keep_as_source=True)
            self.refresh_data_management()
            self.update_plot()

    def rename_column(self, col):
        """Perform the rename column operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        new_name, ok = QInputDialog.getText(self, "Rename Column", f"New name for '{col}':", text=col)
        if ok and new_name and new_name != col:
            work = self.data_engine.df.copy().rename(columns={col: new_name})
            self.data_engine.commit_dataframe(work, keep_as_source=True)
            self.refresh_data_management()
            self.update_plot()

    def toggle_measure(self, col):
        """Perform the toggle measure operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Perform the describe column operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
                    dataset_name = self._register_loaded_dataset("Group By Dataset", result.copy(), {"operation": "group_by", "parent": self.active_dataset_name, "group_columns": selected}, activate=True)
                    QMessageBox.information(self, "Success", f"Created new dataset '{dataset_name}'.")
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
        """Perform the apply filter operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
        """Activate the dataset represented by a Loaded Sheets item."""
        name = item.text()
        if name in self.loaded_datasets:
            try:
                self.activate_loaded_dataset(name)
            except Exception as exc:
                QMessageBox.critical(self, "Activate Dataset", str(exc))
        else:
            self.status.showMessage(f"Dataset '{name}' is not available in Loaded Sheets.")

    # ============================================================
    # PLOT UPDATING
    # ============================================================
    def _mark_values(self, name):
        """Perform the mark values operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        combo=self.marks_widgets.get(name)
        if combo is None or self.data_engine.df is None:return []
        combo.sync_fields([str(c) for c in self.data_engine.df.columns])
        return [x for x in combo.selected_fields() if x in self.data_engine.df.columns]
    def _mark_value(self,name):
        """Perform the mark value operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        vals=self._mark_values(name); return vals[0] if vals else None

    def _sync_all_mark_fields(self):
        """Perform the sync all mark fields operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        cols=[] if self.data_engine.df is None else [str(c) for c in self.data_engine.df.columns]
        for combo in getattr(self,"marks_widgets",{}).values(): combo.sync_fields(cols)

    def update_plot(self):
        """Rebuild the plot pane from current shelves and remove all stale auxiliary axes/artists."""
        # Matplotlib creates additional axes for twin-axis charts and colorbars.
        # ``ax.clear()`` only clears the primary axes, so those auxiliary axes could
        # survive after Rows/Columns were emptied. Rebuild the figure-level axes
        # before every render to guarantee a clean Tableau-like view.
        for extra_ax in list(self.fig.axes):
            if extra_ax is not self.ax:
                try:
                    self.fig.delaxes(extra_ax)
                except Exception:
                    pass
        self.ax.clear()
        self.ax.set_title("")
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        df = self.data_engine.df
        if df is None:
            self.ax.text(.5,.5,"No data loaded",ha="center",va="center"); self.canvas.draw(); return
        self._sync_active_dataset()
        rows=[x for x in self.viz_engine.parse_shelf(self.rows_input.text()) if x in df.columns]
        cols=[x for x in self.viz_engine.parse_shelf(self.cols_input.text()) if x in df.columns]
        chart=self.chart_combo.currentText()
        color_cols=self._mark_values("Color"); size_cols=self._mark_values("Size"); text_cols=self._mark_values("Text"); detail_cols=self._mark_values("Detail"); tooltip_cols=self._mark_values("Tooltip")
        color_col=color_cols[0] if color_cols else None; size_col=size_cols[0] if size_cols else None; text_col=text_cols[0] if text_cols else None; detail_col=detail_cols[0] if detail_cols else None; tooltip_col=tooltip_cols[0] if tooltip_cols else None
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
            # Tableau semantics: Detail always increases granularity; Color only
            # increases granularity for discrete/categorical fields. Numeric Color
            # remains a continuous encoding and must not also be grouped.
            for field in detail_cols:
                if field not in group_dims: group_dims.append(field)
            for field in color_cols:
                if field not in group_dims and not pd.api.types.is_numeric_dtype(df[field]): group_dims.append(field)
            mark_numeric=[]
            for field in (color_cols + size_cols + text_cols + tooltip_cols):
                if field and field in df.columns and pd.api.types.is_numeric_dtype(df[field]) and field not in group_dims:
                    mark_numeric.append(field)
            measures=list(dict.fromkeys(numeric_shelf + mark_numeric))

            # Two quantitative fields on opposite shelves are kept row-wise for a
            # Tableau-style paired-measure view. Collapsing them to one aggregate row
            # would produce the previous single filled circle.
            raw_measure_pair = (
                len(rows) == 1 and len(cols) == 1
                and pd.api.types.is_numeric_dtype(df[rows[0]])
                and pd.api.types.is_numeric_dtype(df[cols[0]])
                and not group_dims
            )
            pair_x = cols[0] if raw_measure_pair else None
            pair_y = rows[0] if raw_measure_pair else None

            if raw_measure_pair:
                measures=[pair_x, pair_y]
                work=df[measures].replace([np.inf, -np.inf], np.nan).dropna().copy()
                primary=pair_y
            elif measures:
                agg_map={m:func for m in measures}
                for shelf_name, shelf_fields in (("Rows", rows), ("Columns", cols)):
                    for f in shelf_fields:
                        if f in agg_map:
                            chosen = self.shelf_aggregations.get(shelf_name, {}).get(f)
                            if chosen in {"sum","mean","count","min","max","median","nunique"}: agg_map[f] = chosen
                mark_to_col={"Color":color_col,"Size":size_col,"Text":text_col,"Tooltip":tooltip_col}
                for mark_name, field in mark_to_col.items():
                    if field in agg_map and hasattr(self,"mark_aggregations"):
                        chosen=self.mark_aggregations.get(mark_name,func)
                        agg_map[field]=chosen if chosen in {"sum","mean","count","min","max"} else func
                if group_dims:
                    work=df.groupby(group_dims,dropna=False)[measures].agg(agg_map).reset_index()
                else:
                    work=pd.DataFrame([{m:getattr(df[m],agg_map[m])() for m in measures}])
                primary=numeric_shelf[0] if numeric_shelf else measures[0]
            else:
                measures=["Number of Records"]
                work=df.groupby(group_dims,dropna=False).size().reset_index(name=measures[0]) if group_dims else pd.DataFrame({measures:[len(df)]})
                primary=measures[0]

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

            # Multiple categorical Color fields are represented as a combined
            # Tableau-style color dimension; this avoids asking matplotlib for
            # several incompatible color encodings at once.
            categorical_color_fields=[f for f in color_cols if f in work.columns and not pd.api.types.is_numeric_dtype(df[f])]
            if len(categorical_color_fields)>1:
                work["__dsp_color__"]=work[categorical_color_fields].astype(str).agg(" | ".join,axis=1); color_col="__dsp_color__"
            elif categorical_color_fields:
                color_col=categorical_color_fields[0]

            if chart=="auto":
                chart="scatter" if len(numeric_shelf)>=2 else ("vertical_bar" if any(c in cols for c in numeric_shelf) else "horizontal_bar")
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
                if raw_measure_pair:
                    ordered=work.sort_values(pair_x)
                    self.ax.plot(ordered[pair_x],ordered[pair_y],marker="o",label=f"{pair_y} vs {pair_x}")
                    self.ax.set_xlabel(pair_x); self.ax.set_ylabel(pair_y)
                    if chart=="area": self.ax.fill_between(ordered[pair_x].to_numpy(float),ordered[pair_y].to_numpy(float),alpha=.25)
                elif color_col and not pd.api.types.is_numeric_dtype(df[color_col]) and color_col in work.columns:
                    for value, grp in work.groupby(color_col,dropna=False):
                        self.ax.plot(grp.index,grp[primary],marker="o",label=str(value))
                    self._style_legend(self.ax.legend(title=color_col), color_col)
                    self.ax.set_xticks(np.arange(len(work))); self.ax.set_xticklabels(labels,rotation=45,ha="right")
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
                    cats={v:i for i,v in enumerate(pd.unique(work[color_col]))}; vals=work[color_col].map(cats).to_numpy(float); self.ax.scatter(work[xfield],work[yfield],s=sizes,c=vals,cmap="tab10",alpha=.8); self._add_categorical_legend(work,color_col)
                else: self.ax.scatter(work[xfield],work[yfield],s=sizes,alpha=.8)
                self.ax.set_xlabel(xfield); self.ax.set_ylabel(yfield)
            elif chart=="histogram": self.ax.hist(pd.to_numeric(df[primary],errors="coerce").dropna(),bins="auto",alpha=.8); self.ax.set_xlabel(primary)
            elif chart=="box":
                vals=[pd.to_numeric(df[c],errors="coerce").dropna() for c in numeric_shelf[:6] or [primary]]; self.ax.boxplot(vals,labels=numeric_shelf[:len(vals)] or [primary])
            elif chart in {"heatmap","highlight_table"} and len(base_dims)>=2:
                pivot=pd.pivot_table(df,index=base_dims[0],columns=base_dims[1],values=numeric_shelf[0] if numeric_shelf else None,aggfunc=func,fill_value=0); im=self.ax.imshow(pivot.to_numpy(),aspect="auto"); self.fig.colorbar(im,ax=self.ax); self.ax.set_xticks(range(len(pivot.columns))); self.ax.set_xticklabels([str(x) for x in pivot.columns],rotation=45,ha="right"); self.ax.set_yticks(range(len(pivot.index))); self.ax.set_yticklabels([str(x) for x in pivot.index])
            elif chart in {"donut","pie"}:
                if raw_measure_pair:
                    values=np.array([float(work[pair_y].mean()), float(work[pair_x].mean())])
                    values=np.abs(values); labels_p=[pair_y, pair_x]
                    if values.sum() > 0:
                        self.ax.pie(values,labels=labels_p,autopct="%1.1f%%")
                else:
                    self.ax.pie(work[primary],labels=labels,autopct="%1.1f%%")
                if chart=="donut": self.ax.add_artist(plt.Circle((0,0),.55,fc="white"))
            elif chart=="stacked_bar" and color_col and color_col in work.columns and not pd.api.types.is_numeric_dtype(work[color_col]):
                pivot=work.pivot_table(index=label_col or work.index,columns=color_col,values=primary,aggfunc="sum",fill_value=0); pivot.plot(kind="bar",stacked=True,ax=self.ax); self._style_legend(self.ax.legend(title=color_col), color_col)
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
            elif raw_measure_pair and chart in {"horizontal_bar","vertical_bar"}:
                values=[float(work[pair_x].mean()), float(work[pair_y].mean())]
                labels_pair=[pair_x, pair_y]
                if chart=="horizontal_bar":
                    self.ax.barh(labels_pair,values); self.ax.set_xlabel("Mean value")
                else:
                    self.ax.bar(labels_pair,values); self.ax.set_ylabel("Mean value"); self.ax.tick_params(axis="x",rotation=30)
            else:
                if orientation=="horizontal": bars=self.ax.barh(labels,work[primary],height=self._bar_sizes(work,size_col)); self.ax.set_xlabel(primary); self.ax.set_ylabel(label_col or "Rows")
                else: bars=self.ax.bar(labels,work[primary],width=self._bar_sizes(work,size_col)); self.ax.set_ylabel(primary); self.ax.set_xlabel(label_col or "Columns"); self.ax.tick_params(axis="x",rotation=45)
                self._apply_bar_colors(bars,work,color_col)
                if color_col and color_col in work.columns and not pd.api.types.is_numeric_dtype(work[color_col]):
                    self._add_categorical_legend(work, color_col)

            # Text/Label mark: multiple fields are concatenated for each mark.
            active_text=[f for f in text_cols if f in work.columns]
            if active_text:
                for i,(_,r) in enumerate(work.iterrows()):
                    try:
                        value=" | ".join(f"{f}: {r.get(f,'')}" for f in active_text)
                        yv=r[primary] if primary in r else 0
                        self.ax.annotate(str(value),(i,yv),fontsize=8)
                    except Exception: pass
            self.ax.set_title(f"{pair_y} vs {pair_x}" if raw_measure_pair else f"{primary} by {', '.join(base_dims) or 'Records'}")
            self._install_hover_handler(work,color_col,tooltip_cols,size_col,rows,cols)
        except Exception as exc:
            self.ax.text(.5,.5,f"Plot error: {exc}",ha="center",va="center")
        self.fig.subplots_adjust(left=.10,right=.97,bottom=.18,top=.90); self.canvas.draw()

    def _style_legend(self, legend, title=None):
        """Professional legend defaults: outside the axes, readable, draggable."""
        if legend is None: return None
        try:
            if title is not None: legend.set_title(str(title))
            legend.set_draggable(True, use_blit=False)
            legend.get_frame().set_alpha(0.92)
            legend.get_frame().set_linewidth(0.8)
            # Keep the initial legend outside the data region so it cannot
            # obscure marks. Users can subsequently drag it interactively.
            legend.set_bbox_to_anchor((1.02, 1.0), transform=self.ax.transAxes)
            legend._loc = 2  # upper-left relative to the external anchor
            for txt in legend.get_texts():
                txt.set_fontsize(9)
            if legend.get_title():
                legend.get_title().set_fontsize(10)
            self.fig.subplots_adjust(right=0.78)
        except Exception:
            pass
        return legend

    def _add_categorical_legend(self, plot_df, color_col):
        """Perform the add categorical legend operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        try:
            from matplotlib.patches import Patch
            vals = list(pd.unique(plot_df[color_col]))
            if not vals: return
            cmap = plt.get_cmap("tab10"); n=max(1,len(vals)-1)
            handles=[Patch(facecolor=cmap(i/n if n else 0), label=str(v)) for i,v in enumerate(vals)]
            self._style_legend(self.ax.legend(handles=handles, title=str(color_col), loc="upper left", bbox_to_anchor=(1.02,1.0), frameon=True), color_col)
        except Exception: pass

    def format_legend_dialog(self):
        """Perform the format legend dialog operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        legend=self.ax.get_legend()
        if legend is None:
            QMessageBox.information(self,"Legend Formatting","The current plot has no legend. Assign a categorical field to Color, or use a chart that produces a legend.")
            return
        d=QDialog(self); d.setWindowTitle("Professional Legend Formatting"); d.resize(560,430); l=QVBoxLayout(d); form=QFormLayout()
        title=QLineEdit(legend.get_title().get_text() if legend.get_title() else "")
        font_size=QSpinBox(); font_size.setRange(6,32); font_size.setValue(int(legend.get_texts()[0].get_fontsize()) if legend.get_texts() else 9)
        title_size=QSpinBox(); title_size.setRange(7,36); title_size.setValue(int(legend.get_title().get_fontsize()) if legend.get_title() else 10)
        text_color=QPushButton("Choose…"); title_color=QPushButton("Choose…"); frame_color=QPushButton("Choose…")
        alpha=QDoubleSpinBox(); alpha.setRange(0.1,1.0); alpha.setSingleStep(.05); alpha.setValue(float(legend.get_frame().get_alpha() or 1.0))
        loc=QComboBox(); loc.addItems(["Upper right","Upper left","Lower right","Lower left","Center right","Center left","Best"]); loc.setCurrentText("Upper left")
        form.addRow("Legend title",title); form.addRow("Text font size",font_size); form.addRow("Title font size",title_size); form.addRow("Text color",text_color); form.addRow("Title color",title_color); form.addRow("Frame color",frame_color); form.addRow("Frame opacity",alpha); form.addRow("Initial position",loc); l.addLayout(form)
        note=QLabel("The legend is draggable directly on the plot. It starts outside the data area to prevent obscuring marks."); note.setWordWrap(True); l.addWidget(note)
        colors={"text":None,"title":None,"frame":None}
        text_color.clicked.connect(lambda: colors.__setitem__("text",QColorDialog.getColor()))
        title_color.clicked.connect(lambda: colors.__setitem__("title",QColorDialog.getColor()))
        frame_color.clicked.connect(lambda: colors.__setitem__("frame",QColorDialog.getColor()))
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Apply|QDialogButtonBox.StandardButton.Cancel); l.addWidget(bb)
        def apply():
            """Perform the apply operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
            legend.set_title(title.text())
            for t in legend.get_texts():
                t.set_fontsize(font_size.value())
                if colors["text"] is not None and colors["text"].isValid(): t.set_color(colors["text"].name())
            if legend.get_title():
                legend.get_title().set_fontsize(title_size.value())
                if colors["title"] is not None and colors["title"].isValid(): legend.get_title().set_color(colors["title"].name())
            frame=legend.get_frame(); frame.set_alpha(alpha.value())
            if colors["frame"] is not None and colors["frame"].isValid(): frame.set_edgecolor(colors["frame"].name())
            anchors={"Upper right":(1.02,1.0),"Upper left":(0.02,1.0),"Lower right":(1.02,0.0),"Lower left":(0.02,0.0),"Center right":(1.02,.5),"Center left":(0.02,.5),"Best":(0,0)}
            if loc.currentText()=="Best": legend._loc="best"
            else:
                legend.set_bbox_to_anchor(anchors[loc.currentText()], transform=self.ax.transAxes); legend._loc={"Upper right":1,"Upper left":2,"Lower left":3,"Lower right":4,"Center left":6,"Center right":7}[loc.currentText()]
            legend.set_draggable(True,use_blit=False)
            self.fig.subplots_adjust(right=0.78 if loc.currentText() in {"Upper right","Center right","Lower right"} else .97)
            self.canvas.draw_idle(); self.status.showMessage("Legend formatting applied; drag the legend to reposition it.")
        bb.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(apply); bb.rejected.connect(d.reject); d.exec()

    def _bar_sizes(self, plot_df, size_col):
        """Perform the bar sizes operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if not size_col or size_col not in plot_df.columns or not pd.api.types.is_numeric_dtype(plot_df[size_col]): return 0.8
        s=pd.to_numeric(plot_df[size_col],errors="coerce").fillna(0).to_numpy(float); lo,hi=float(np.nanmin(s)),float(np.nanmax(s))
        return 0.8 if hi==lo else 0.35+0.9*(s-np.nanmin(s))/(hi-lo)

    def _sizes(self, plot_df, size_col):
        """Perform the sizes operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if not size_col or size_col not in plot_df.columns or not pd.api.types.is_numeric_dtype(plot_df[size_col]): return 70
        s = pd.to_numeric(plot_df[size_col], errors="coerce").fillna(0).to_numpy(float);
        if len(s) == 0: return 70
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s)); return np.full(len(s), 70.) if hi == lo else 40 + 260 * (s-lo)/(hi-lo)

    def _apply_bar_colors(self, bars, plot_df, color_col):
        """Perform the apply bar colors operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if not color_col or color_col not in plot_df.columns: return
        vals = plot_df[color_col]
        if pd.api.types.is_numeric_dtype(vals):
            cmap = plt.get_cmap("viridis"); arr = pd.to_numeric(vals, errors="coerce").fillna(vals.mean()).to_numpy(float); lo, hi = np.nanmin(arr), np.nanmax(arr); norm = np.ones(len(arr))*.5 if hi == lo else (arr-lo)/(hi-lo)
            for b, n in zip(bars, norm): b.set_color(cmap(float(n)))
        else:
            categories = {v:i for i,v in enumerate(pd.unique(vals))}; cmap=plt.get_cmap("tab10"); n=max(1,len(categories)-1)
            for b, v in zip(bars, vals): b.set_color(cmap(categories[v]/n if n else 0))

    def _install_hover_handler(self, plot_df, color_col, tooltip_cols, size_col, rows=None, cols=None):
        """Perform the install hover handler operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
            """Perform the move operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
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
                    tipset=set(tooltip_cols or [])
                    text = "<br>".join(f"{c}: {row[c]}" for c in plot_df.columns if c in tipset or c in (rows or []) or c in (cols or []))
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
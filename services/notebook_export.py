"""Module duty: Notebook export.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

class NotebookExporter:
    @staticmethod
    def build(view_state, dataset_source, rows, columns, chart, marks, filters, analysis_evidence=None):
        """Perform the build operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        cells = [
            {"cell_type":"markdown","metadata":{},"source":["# Data Science Studio Pro — Reproducible Analysis\n"]},
            {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["import pandas as pd\n", "df = pd.read_csv(" + repr(dataset_source) + ")\n"]},
            {"cell_type":"markdown","metadata":{},"source":[f"## View\nRows: `{rows}`  \nColumns: `{columns}`  \nChart: `{chart}`  \nMarks: `{json.dumps(marks)}`\n"]},
            {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["# Recreate the active view\n", f"# Filters: {json.dumps(filters, default=str)}\n", "df.head()\n"]},
        ]
        if analysis_evidence:
            cells.append({"cell_type":"markdown","metadata":{},"source":["## Recorded evidence\n", json.dumps(analysis_evidence, indent=2, default=str)]})
        return {"cells": cells, "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}}, "nbformat":4,"nbformat_minor":5}

    @staticmethod
    def save(path, notebook):
        """Perform the save operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        Path(path).write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")

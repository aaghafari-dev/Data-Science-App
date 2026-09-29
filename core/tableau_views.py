from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
from core.viz_engine import VizEngine

@dataclass
class TableauSheet:
    name: str
    dataframe: pd.DataFrame
    rows: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    chart_type: str = "auto"
    marks: dict[str, str | None] = field(default_factory=lambda: {"Color":None,"Size":None,"Text":None,"Detail":None,"Tooltip":None})
    agg_func: str = "sum"
    def render(self):
        return VizEngine.render_chart(self.dataframe, self.rows, self.columns, self.chart_type, **self.marks, agg_func=self.agg_func)

@dataclass
class TableauDashboard:
    name: str
    sheets: list[TableauSheet] = field(default_factory=list)
    global_filters: dict[str, Any] = field(default_factory=dict)
    columns: int = 2
    def apply_filter(self, field: str, value: Any):
        self.global_filters[field] = value
    def filtered_sheets(self):
        out=[]
        for s in self.sheets:
            df=s.dataframe
            for f,v in self.global_filters.items():
                if f in df.columns and v not in (None, "", "All"):
                    df=df[df[f].astype(str)==str(v)]
            out.append(TableauSheet(s.name, df, s.rows, s.columns, s.chart_type, dict(s.marks), s.agg_func))
        return out

@dataclass
class TableauStory:
    name: str
    dashboards: list[TableauDashboard] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    current_step: int = 0
    def next(self): self.current_step=min(self.current_step+1, max(0,len(self.dashboards)-1)); return self.current_step
    def previous(self): self.current_step=max(self.current_step-1,0); return self.current_step
    def current(self): return self.dashboards[self.current_step] if self.dashboards else None

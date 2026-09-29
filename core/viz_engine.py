from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ShelfPlan:
    rows: list[str]
    columns: list[str]
    dimensions: list[str]
    measures: list[str]
    x_field: str | None
    y_field: str | None
    orientation: str


class VizEngine:
    """Tableau-like shelf planner and matplotlib/Qt-friendly rendering helpers."""

    AGGREGATIONS = {"sum": "sum", "mean": "mean", "count": "count", "min": "min", "max": "max"}

    @staticmethod
    def parse_shelf(value: str | list[str] | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        return [x.strip() for x in str(value).split(",") if x.strip()]

    @staticmethod
    def format_shelf(fields: list[str]) -> str:
        return ", ".join(dict.fromkeys(fields))

    @staticmethod
    def add_to_shelf(value: str | None, field: str) -> str:
        fields = VizEngine.parse_shelf(value)
        if field not in fields:
            fields.append(field)
        return VizEngine.format_shelf(fields)

    @staticmethod
    def remove_from_shelf(value: str | None, field: str) -> str:
        return VizEngine.format_shelf([x for x in VizEngine.parse_shelf(value) if x != field])

    @staticmethod
    def plan_shelves(df: pd.DataFrame, rows_shelf: list[str], columns_shelf: list[str]) -> ShelfPlan:
        fields = [x for x in rows_shelf + columns_shelf if x in df.columns]
        dimensions = [x for x in fields if not pd.api.types.is_numeric_dtype(df[x])]
        measures = [x for x in fields if pd.api.types.is_numeric_dtype(df[x])]

        row_measures = [x for x in rows_shelf if x in measures]
        col_measures = [x for x in columns_shelf if x in measures]
        row_dims = [x for x in rows_shelf if x in dimensions]
        col_dims = [x for x in columns_shelf if x in dimensions]

        # Tableau's common orientation rule: a quantitative field on Columns gives
        # a horizontal quantitative axis; on Rows it gives a vertical axis.
        if col_measures:
            y_field = row_dims[0] if row_dims else None
            x_field = col_measures[0]
            orientation = "horizontal"
        elif row_measures:
            x_field = col_dims[0] if col_dims else None
            y_field = row_measures[0]
            orientation = "vertical"
        elif row_dims and col_dims:
            x_field, y_field, orientation = col_dims[0], row_dims[0], "crosstab"
        elif row_dims:
            x_field, y_field, orientation = row_dims[0], None, "categorical"
        elif col_dims:
            x_field, y_field, orientation = col_dims[0], None, "categorical"
        else:
            x_field = y_field = None
            orientation = "empty"
        return ShelfPlan(rows_shelf, columns_shelf, dimensions, measures, x_field, y_field, orientation)

    @staticmethod
    def aggregate_for_shelves(df: pd.DataFrame, rows_shelf: list[str], columns_shelf: list[str], agg_func: str = "sum") -> pd.DataFrame:
        """Return a long, plotting-ready table for multiple Tableau-like shelves."""
        rows = [x for x in rows_shelf if x in df.columns]
        cols = [x for x in columns_shelf if x in df.columns]
        shelf_fields = rows + cols
        numeric_fields = [x for x in shelf_fields if pd.api.types.is_numeric_dtype(df[x])]
        dimension_fields = [x for x in shelf_fields if not pd.api.types.is_numeric_dtype(df[x])]
        if not shelf_fields:
            return df.copy()

        if numeric_fields:
            func = VizEngine.AGGREGATIONS.get(agg_func, "sum")
            if dimension_fields:
                grouped = df.groupby(dimension_fields, dropna=False, as_index=False)[numeric_fields].agg(func)
            else:
                values = {m: getattr(df[m], func)() for m in numeric_fields}
                grouped = pd.DataFrame([values])
            return grouped

        # Dimensions only: Tableau can still show a count of records.
        grouped = df.groupby(dimension_fields, dropna=False, as_index=False).size()
        grouped = grouped.rename(columns={"size": "Number of Records"})
        return grouped

    @staticmethod
    def build_cross_tab(df: pd.DataFrame, rows: list[str], cols: list[str], agg_func: str = "sum") -> pd.DataFrame:
        rows = [x for x in rows if x in df.columns]
        cols = [x for x in cols if x in df.columns]
        if not rows or not cols:
            return pd.DataFrame()
        row_dims = [x for x in rows if not pd.api.types.is_numeric_dtype(df[x])]
        col_dims = [x for x in cols if not pd.api.types.is_numeric_dtype(df[x])]
        measures = [x for x in rows + cols if pd.api.types.is_numeric_dtype(df[x])]
        if not measures:
            return pd.crosstab([df[x] for x in row_dims], [df[x] for x in col_dims])
        value = measures[0]
        return pd.pivot_table(df, index=row_dims or rows, columns=col_dims or cols,
                              values=value, aggfunc=VizEngine.AGGREGATIONS.get(agg_func, "sum"), fill_value=0)

    @staticmethod
    def render_chart(df: pd.DataFrame, rows_shelf: list[str], columns_shelf: list[str], chart_type: str = "auto",
                     color_col: str | None = None, size_col: str | None = None,
                     text_col: str | None = None, detail_col: str | None = None,
                     tooltip_col: str | None = None, agg_func: str = "sum") -> dict[str, Any]:
        """Create a Plotly figure for exports/tests; GUI uses the same planning rules."""
        import plotly.express as px

        plan = VizEngine.plan_shelves(df, rows_shelf, columns_shelf)
        plot_df = VizEngine.aggregate_for_shelves(df, rows_shelf, columns_shelf, agg_func)
        if plot_df.empty:
            return {"figure": None, "plan": plan, "data": plot_df}

        numeric = [c for c in plot_df.columns if pd.api.types.is_numeric_dtype(plot_df[c])]
        dims = [c for c in plot_df.columns if c not in numeric]
        x = plan.x_field if plan.x_field in plot_df.columns else (dims[0] if dims else None)
        y = plan.y_field if plan.y_field in plot_df.columns else (numeric[0] if numeric else "Number of Records")
        if plan.orientation == "horizontal" and x in plot_df.columns and y not in plot_df.columns:
            x, y = y, x
        if chart_type == "auto":
            chart_type = "bar"
        kwargs = {}
        if color_col and color_col in df.columns:
            kwargs["color"] = color_col
        if size_col and size_col in df.columns and pd.api.types.is_numeric_dtype(df[size_col]):
            kwargs["size"] = size_col
        if text_col and text_col in plot_df.columns:
            kwargs["text"] = text_col
        if chart_type == "line":
            fig = px.line(plot_df, x=x, y=y, **kwargs)
        elif chart_type == "area":
            fig = px.area(plot_df, x=x, y=y, **kwargs)
        elif chart_type == "scatter":
            fig = px.scatter(plot_df, x=x, y=y, **kwargs)
        elif chart_type in {"histogram"}:
            fig = px.histogram(df, x=y if y in df.columns else x, **kwargs)
        elif chart_type in {"box"}:
            fig = px.box(df, x=x, y=y, **kwargs)
        elif chart_type in {"heatmap", "highlight_table"} and len(dims) >= 2:
            pivot = plot_df.pivot_table(index=dims[0], columns=dims[1], values=y, aggfunc=agg_func, fill_value=0)
            fig = px.imshow(pivot, text_auto=(chart_type == "highlight_table"), aspect="auto", color_continuous_scale="Blues")
        elif chart_type in {"pie", "donut"}:
            fig = px.pie(plot_df, names=x, values=y, hole=0.5 if chart_type == "donut" else 0, **({"color": color_col} if color_col in plot_df.columns else {}))
        elif chart_type == "stacked_bar":
            fig = px.bar(plot_df, x=x, y=y, color=color_col if color_col in plot_df.columns else None, barmode="stack", **kwargs)
        elif chart_type == "dual_axis" and len(numeric) >= 2:
            fig = px.line(plot_df, x=x, y=numeric[:2])
        elif chart_type == "gantt" and x and y:
            fig = px.timeline(plot_df, x_start=x, x_end=y, y=dims[0] if dims else None)
        else:
            fig = px.bar(plot_df, x=x, y=y, **kwargs)
        fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=50, b=40))
        return {"figure": fig, "plan": plan, "data": plot_df}

    @staticmethod
    def render_extra_insight_plot(df: pd.DataFrame, column: str, top_n: int = 10,
                                   font_size: int = 12, font_family: str = "Arial",
                                   color: str = "#1976d2", hover_color: str = "#ff9900",
                                   chart_type: str = "bar"):
        """Plotly version used by external integrations. Hover styling is explicit."""
        import plotly.express as px
        if df is None or column not in df.columns:
            return None
        if pd.api.types.is_numeric_dtype(df[column]):
            data = df.nlargest(top_n, column)[[column]].copy()
            data["Count"] = 1
        else:
            data = df[column].value_counts().head(top_n).reset_index()
            data.columns = [column, "Count"]
        if chart_type == "pie":
            fig = px.pie(data, names=column, values=column if "Count" not in data else "Count")
        elif chart_type == "line":
            fig = px.line(data, x=column, y="Count", markers=True)
        elif chart_type == "scatter":
            fig = px.scatter(data, x=column, y="Count")
        else:
            fig = px.bar(data, x=column, y="Count")
        fig.update_traces(marker_color=color, hovertemplate="<b>%{x}</b><br>Count=%{y}<extra></extra>")
        fig.update_layout(font=dict(family=font_family, size=font_size), hoverlabel=dict(bgcolor=hover_color), height=500)
        return fig

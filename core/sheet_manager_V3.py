"""Module duty: Sheet manager V3.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

import streamlit as st

class SheetManager:
    def __init__(self):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if 'sheets' not in st.session_state:
            st.session_state.sheets = {}
        if 'active_sheet' not in st.session_state:
            st.session_state.active_sheet = "Sheet 1"

    def create_sheet(self, name):
        """Perform the create sheet operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        st.session_state.sheets[name] = {
            'x_col': None,
            'y_col': None,
            'mark_type': 'Bar',
            'color_col': None,
            'size_col': None,
            'agg_func': 'sum'
        }

    def get_sheet_config(self, name):
        """Perform the get sheet config operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return st.session_state.sheets.get(name, {})

    def update_sheet_config(self, name, key, value):
        """Perform the update sheet config operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if name in st.session_state.sheets:
            st.session_state.sheets[name][key] = value

    def get_all_sheets(self):
        """Perform the get all sheets operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return list(st.session_state.sheets.keys())

    def render_dashboard(self, df, viz_engine):
        """Duty: Dashboard Architecture - Aggregate sheets into a grid."""
        st.subheader("Dashboard View")
        
        # Example: Arrange Sheet 1 and Sheet 2 side-by-side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Sheet 1: Sales Trend")
            config1 = self.get_sheet_config("Sheet 1")
            fig1 = viz_engine.render_chart(df, config1['x_col'], config1['y_col'], 
                                           config1['mark_type'], config1['color_col'], 
                                           config1['size_col'], config1['agg_func'])
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("Configure Sheet 1 to see the chart.")

        with col2:
            st.markdown("### Sheet 2: Category Breakdown")
            config2 = self.get_sheet_config("Sheet 2")
            fig2 = viz_engine.render_chart(df, config2['x_col'], config2['y_col'], 
                                           config2['mark_type'], config2['color_col'], 
                                           config2['size_col'], config2['agg_func'])
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Configure Sheet 2 to see the chart.")

        st.markdown("---")
        st.markdown("### Sheet 3: Key Metrics")
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        
        with kpi_col1:
            if df is not None:
                total_rows = len(df)
                st.plotly_chart(viz_engine.render_kpi_card(total_rows, "Total Rows"), use_container_width=True)
        with kpi_col2:
            if df is not None and len(df.select_dtypes(include=['number']).columns) > 0:
                num_col = df.select_dtypes(include=['number']).columns[0]
                avg_val = df[num_col].mean()
                st.plotly_chart(viz_engine.render_kpi_card(avg_val, f"Avg {num_col}"), use_container_width=True)
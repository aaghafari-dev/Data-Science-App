"""Module duty: Sheet manager.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

class SheetManager:
    """Workbook sheet state. A new workbook opens on Sheet 1 only."""
    def __init__(self):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.sheets = {
            "Sheet 1": {
                'x_col': None, 'y_col': None, 'mark_type': 'Bar',
                'color_col': None, 'size_col': None, 'agg_func': 'sum'
            }
        }
        self.active_sheet = "Sheet 1"

    def update_sheet_config(self, name, key, value):
        """Perform the update sheet config operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if name in self.sheets:
            self.sheets[name][key] = value

    def get_sheet_config(self, name):
        """Perform the get sheet config operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return self.sheets.get(name, {})

    def create_sheet(self, name, snapshot=None, dataframe=None):
        """Perform the create sheet operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.sheets[name] = {
            'x_col': None, 'y_col': None, 'mark_type': 'Bar',
            'color_col': None, 'size_col': None, 'agg_func': 'sum',
            'data': dataframe,
            'view_snapshot': snapshot,
        }
        self.active_sheet = name
        return name

    def activate(self, name):
        """Perform the activate operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        if name not in self.sheets:
            raise KeyError(name)
        self.active_sheet = name
        return self.sheets[name]

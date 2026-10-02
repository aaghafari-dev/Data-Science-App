"""Module duty: Theme manager.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

class ThemeManager:
    def __init__(self, app):
        """Perform the init operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.app = app
        self.theme_color = "#e3f2fd"
        self.dark_mode = False

    def apply_theme(self, color_hex, dark=False):
        """Perform the apply theme operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        self.theme_color = color_hex
        self.dark_mode = dark
        text_color = "white" if dark else "black"
        self.app.setStyleSheet(f"""
            QMainWindow {{ background-color: {color_hex}; color: {text_color}; }}
            QDockWidget {{ background-color: {color_hex}; color: {text_color}; }}
            QLineEdit {{ background-color: white; border: 1px solid #ccc; color: black; }}
            QComboBox {{ background-color: white; border: 1px solid #ccc; color: black; }}
            QLabel {{ color: {text_color}; }}
            QListWidget {{ background-color: white; color: black; }}
            QMenuBar {{ background-color: {color_hex}; color: {text_color}; }}
            QMenu {{ background-color: {color_hex}; color: {text_color}; }}
            QToolBar {{ background-color: {color_hex}; color: {text_color}; }}
        """)
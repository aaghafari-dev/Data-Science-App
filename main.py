import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, 
                             QDockWidget, QListWidget, QMenu, QToolBar, QStatusBar)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Science Studio Pro")
        self.resize(1400, 900)
        
        # 1. Setup Central Widget (The Canvas Area)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # 2. Top Section: Rows and Columns (Tableau Style)
        self.top_shelf_layout = QHBoxLayout()
        self.rows_input = QLineEdit()
        self.rows_input.setPlaceholderText("Drop Rows here")
        self.cols_input = QLineEdit()
        self.cols_input.setPlaceholderText("Drop Columns here")
        
        self.top_shelf_layout.addWidget(QLabel("Rows:"))
        self.top_shelf_layout.addWidget(self.rows_input)
        self.top_shelf_layout.addWidget(QLabel("Columns:"))
        self.top_shelf_layout.addWidget(self.cols_input)
        self.layout.addLayout(self.top_shelf_layout)

        # 3. Marks Panel
        self.marks_layout = QHBoxLayout()
        self.marks_layout.addWidget(QLabel("Marks:"))
        self.marks_layout.addWidget(QLabel("Color:"))
        self.marks_layout.addWidget(QComboBox())
        self.marks_layout.addWidget(QLabel("Size:"))
        self.marks_layout.addWidget(QComboBox())
        self.marks_layout.addWidget(QLabel("Text:"))
        self.marks_layout.addWidget(QComboBox())
        self.marks_layout.addWidget(QLabel("Detail:"))
        self.marks_layout.addWidget(QComboBox())
        self.marks_layout.addWidget(QLabel("Tooltip:"))
        self.marks_layout.addWidget(QComboBox())
        self.marks_layout.addWidget(QLabel("Chart Type:"))
        self.chart_combo = QComboBox()
        self.chart_combo.addItems(["auto", "bar", "line", "scatter", "pie"])
        self.marks_layout.addWidget(self.chart_combo)
        self.layout.addLayout(self.marks_layout)

        # Placeholder for the actual chart
        self.canvas_placeholder = QLabel("No data loaded")
        self.canvas_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_placeholder.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.layout.addWidget(self.canvas_placeholder)

        # 4. Setup Right Side Panels (Dock Widgets)
        self.setup_right_panels()
        
        # 5. Setup Menus and Toolbar
        self.setup_menus()
        self.setup_toolbar()
        
        # 6. Apply Default Light Blue Theme
        self.apply_theme("#e3f2fd") # Light Blue

    def setup_right_panels(self):
        # Panel 1: Loaded Sheets
        self.sheets_dock = QDockWidget("Loaded Sheets", self)
        self.sheets_list = QListWidget()
        self.sheets_dock.setWidget(self.sheets_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.sheets_dock)

        # Panel 2: Data Management
        self.data_mgmt_dock = QDockWidget("Data Management", self)
        data_widget = QWidget()
        data_layout = QVBoxLayout()
        data_layout.addWidget(QLabel("Numerical Columns"))
        data_layout.addWidget(QListWidget())
        data_layout.addWidget(QLabel("Categorical Columns"))
        data_layout.addWidget(QListWidget())
        data_widget.setLayout(data_layout)
        self.data_mgmt_dock.setWidget(data_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.data_mgmt_dock)

        # Panel 3: Filters
        self.filters_dock = QDockWidget("Filters", self)
        self.filters_dock.setWidget(QListWidget())
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.filters_dock)

        # Panel 4: Extra Insight
        self.insight_dock = QDockWidget("Extra Insight", self)
        self.insight_dock.setWidget(QLabel("Drop column for extra analysis"))
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.insight_dock)

    def setup_menus(self):
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open CSV/Excel")
        file_menu.addAction("Open PDF")
        file_menu.addAction("Save Project")
        file_menu.addAction("Load Project")
        file_menu.addAction("Exit")

        # Table Creation
        table_menu = menubar.addMenu("Table Creation")
        table_menu.addAction("Web Scraping")
        table_menu.addAction("API Key")

        # Data Info
        info_menu = menubar.addMenu("Data Info")
        info_menu.addAction("Missing Values")
        info_menu.addAction("Describe Data")
        info_menu.addAction("Data Info")

        # Data Pre-processing
        prep_menu = menubar.addMenu("Data Pre-processing")
        prep_menu.addAction("Data Cleaning")
        prep_menu.addAction("Data Integration")
        prep_menu.addAction("Data Transformation")
        prep_menu.addAction("Data Reduction")
        prep_menu.addAction("Feature Engineering")
        prep_menu.addAction("Data Encoding")
        prep_menu.addAction("Label Encoding")
        prep_menu.addAction("Data Scaling")
        prep_menu.addAction("Data Splitting")

        # Data Analysis (Suggested Submenus)
        analysis_menu = menubar.addMenu("Data Analysis")
        analysis_menu.addAction("Descriptive Statistics")
        analysis_menu.addAction("Correlation Matrix")
        analysis_menu.addAction("Hypothesis Testing")
        analysis_menu.addAction("Time Series Analysis")
        analysis_menu.addAction("Regression Analysis")

        # AI Agent
        ai_menu = menubar.addMenu("AI Agent")
        ai_menu.addAction("Agent Data Science")
        ai_menu.addAction("Report")

        # Macro
        macro_menu = menubar.addMenu("Macro")
        macro_menu.addAction("Python")
        macro_menu.addAction("SQL")

        # Sharing
        share_menu = menubar.addMenu("Sharing")
        share_menu.addAction("Export PNG")
        share_menu.addAction("Email Current Sheet")
        share_menu.addAction("Email Story")
        share_menu.addAction("Export to Streamlit")

    def setup_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        toolbar.addAction("Undo")
        toolbar.addAction("Redo")
        toolbar.addAction("Sort Y-Axis Asc")
        toolbar.addAction("Sort Y-Axis Desc")
        toolbar.addAction("Story")
        toolbar.addAction("Sheets")

    def apply_theme(self, color_hex):
        # This function allows changing theme colors
        # In a real app, this would use a CSS stylesheet file
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {color_hex}; }}
            QDockWidget {{ background-color: {color_hex}; }}
            QLineEdit {{ background-color: white; border: 1px solid #ccc; }}
        """)

    # Override context menu event to allow right-click theme changes
    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        change_color_action = QAction("Change Theme Color", self)
        change_color_action.triggered.connect(self.open_color_picker)
        context_menu.addAction(change_color_action)
        context_menu.exec(event.globalPos())

    def open_color_picker(self):
        # Placeholder for color picker logic
        print("Color picker would open here. Select a professional color scale.")
        # Example: self.apply_theme("#ff0000")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
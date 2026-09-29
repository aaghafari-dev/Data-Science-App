class SheetManager:
    def __init__(self):
        self.sheets = {
            "Sheet 1": {'x_col': None, 'y_col': None, 'mark_type': 'Bar',
                        'color_col': None, 'size_col': None, 'agg_func': 'sum'},
            "Sheet 2": {'x_col': None, 'y_col': None, 'mark_type': 'Bar',
                        'color_col': None, 'size_col': None, 'agg_func': 'sum'}
        }
        self.active_sheet = "Sheet 1"

    def update_sheet_config(self, name, key, value):
        if name in self.sheets:
            self.sheets[name][key] = value

    def get_sheet_config(self, name):
        return self.sheets.get(name, {})
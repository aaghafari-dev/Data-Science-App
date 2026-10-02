"""Module duty: Tools.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

def check_missing_values(df):
    """Perform the check missing values operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    return df.isnull().sum().to_dict()

def fill_missing_values(df, strategy='mean'):
    """Perform the fill missing values operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    if strategy == 'mean':
        num_cols = df.select_dtypes(include=['number']).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif strategy == 'drop':
        df.dropna(inplace=True)
    return df, f"Missing values handled with strategy: {strategy}"

def encode_categorical(df):
    """Perform the encode categorical operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df, f"Encoded categorical columns: {list(cat_cols)}"

def scale_data(df):
    """Perform the scale data operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, f"Scaled numerical columns: {list(num_cols)}"

def train_model(df, target_col, task_type='classification'):
    """Perform the train model operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
    features = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    if not features:
        return None, "No numeric features."
    X = df[features].fillna(0)
    y = df[target_col]
    if task_type == 'classification':
        y = y.astype('category').cat.codes
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        return model, f"Random Forest Classifier accuracy: {acc:.3f}"
    else:
        y = pd.to_numeric(y, errors='coerce').fillna(0)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        r2 = r2_score(y, model.predict(X))
        return model, f"Random Forest Regressor R²: {r2:.3f}"

TOOL_MAP = {
    "check_missing_values": check_missing_values,
    "fill_missing_values": fill_missing_values,
    "encode_categorical": encode_categorical,
    "scale_data": scale_data,
    "train_model": train_model
}
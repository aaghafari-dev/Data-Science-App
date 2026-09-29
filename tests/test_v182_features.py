import numpy as np
import pandas as pd

from services.semantic_types import schema_table
from services.data_contracts import DataContractEngine
from services.split_wizard import SplitWizard
from services.error_analysis import ErrorAnalysisEngine
from services.statistical_analysis import StatisticalAnalysisEngine
from services.monitoring import ModelMonitoring
from services.macro_library import ALL_MACROS
from agent.graph_ml import run_ml_step


def test_core_types_and_schema_drift():
    df = pd.DataFrame({
        "i": pd.Series([1, 2, 3], dtype="int64"),
        "f": pd.Series([1.0, 2.0, 3.0], dtype="float64"),
        "o": ["a", "b", "c"],
        "dt": pd.date_range("2025-01-01", periods=3),
    })
    table = schema_table(df)
    assert set(table["core_type"]) == {"int64", "float64", "object", "datetime"}
    contract = DataContractEngine.build(df)
    changed = df.rename(columns={"o": "text"})
    assert DataContractEngine.validate(changed, contract)["status"] == "drift"


def test_leakage_aware_time_split_is_ordered():
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=100), "x": np.arange(100), "y": np.arange(100)})
    tr, va, te = SplitWizard.execute(df, "y", "Time-aware", time_col="date")
    assert tr["date"].max() < va["date"].min() < te["date"].min()


def test_error_analysis_and_monitoring():
    y = np.array([0, 0, 1, 1]); p = np.array([0, 1, 1, 0])
    result = ErrorAnalysisEngine.classification(y, p)
    assert result["errors"] == 2
    ref = pd.DataFrame({"x": np.arange(10), "c": list("AABBCCDDEE")})
    cur = pd.DataFrame({"x": np.arange(10) + 10, "c": list("AAAAAAAAAA")})
    monitored = ModelMonitoring.summary(ref, cur)
    assert monitored["review_count"] >= 1


def test_macro_library_has_exactly_100_entries():
    assert len(ALL_MACROS) == 100


def test_datetime_and_string_model_features_do_not_crash():
    rng = np.random.default_rng(7)
    n = 80
    df = pd.DataFrame({
        "when": pd.date_range("2025-01-01", periods=n, freq="D"),
        "category": rng.choice(["A", "B", "C"], n),
        "x": rng.normal(size=n),
        "target": rng.normal(size=n),
    })
    result = run_ml_step(df, "target")
    assert result["status"] == "ok"

import pandas as pd
from services.data_quality import DataQualityEngine
from services.project_io import build_manifest, save_manifest, load_manifest


def test_data_quality_detects_common_issues(tmp_path):
    df = pd.DataFrame({
        "id": [1, 2, 2, 4],
        "value": [1.0, None, None, 2.0],
        "constant": [7, 7, 7, 7],
        "label": [0, 0, 0, 0],
    })
    result = DataQualityEngine().assess(df, target="label")
    checks = {x["check"] for x in result["issues"]}
    assert "duplicate_rows" in checks
    assert "missing_values" in checks
    assert "constant_column" in checks
    assert "invalid_target" in checks


def test_data_dictionary_is_exportable():
    df = pd.DataFrame({"A": [1, 2], "B": ["x", None]})
    table = DataQualityEngine.data_dictionary(df)
    assert list(table.columns) == ["column", "dtype", "rows", "non_null", "missing", "missing_pct", "unique", "example"]
    assert int(table.loc[table["column"] == "B", "missing"].iloc[0]) == 1


def test_project_manifest_round_trip(tmp_path):
    manifest = build_manifest(
        dataset_fingerprint="abc123", source="demo.csv", rows=3, columns=2,
        view_state={"rows": "region", "columns": "sales", "chart": "bar"},
        filters=[("region", {"kind": "categorical", "values": ["EU"]})],
        evidence=[]
    )
    path = tmp_path / "snapshot.json"
    save_manifest(path, manifest)
    loaded = load_manifest(path)
    assert loaded["format"] == "dssp-project-manifest"
    assert loaded["dataset"]["fingerprint"] == "abc123"
    assert loaded["view_state"]["rows"] == "region"

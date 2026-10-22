import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_stage1_counts_real():
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    s1 = m["stage1_rule_counts_on_raw_file"]
    assert s1["rows_total"] == 541909
    assert s1["missing_customer_id"] > 100000
    assert s1["rows_failing_any_rule"] > 0


def test_stage2_labelled():
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    s2 = m["stage2_labelled_test"]
    assert s2["injected_anomalies"] == 400
    c = s2["rules_plus_forest"]
    assert 0 <= c["precision"] <= 1 and 0 <= c["recall"] <= 1
    # combined must catch at least as much as either stage alone
    assert c["recall"] >= s2["isolation_forest_only"]["recall"] - 1e-9
    assert c["recall"] >= s2["rules_only"]["recall"] - 1e-9

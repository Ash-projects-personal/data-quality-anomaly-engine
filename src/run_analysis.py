"""Data quality checks and anomaly detection on a real messy file.

Stage 1 runs deterministic rules on the raw UCI Online Retail file
as published. Every count is a real count of a real issue in that
file: missing customer IDs, cancelled invoices, negative or zero
quantities, zero prices, exact duplicate rows.

Stage 2 evaluates the Isolation Forest honestly. Real data has no
labels for which rows are anomalies, so I build a labelled test:
20,000 clean real invoice lines plus 400 injected corruptions of
three known types, mixed and shuffled with a fixed seed. The
detector settings are fixed before the test runs. I report
precision and recall against the injected labels for the rules,
the forest, and both combined, plus the forest's flag rate on
clean real lines.

There is no production pipeline here, so there is no pipeline
failure reduction number. The old version computed one as
recall times 0.98 and called it 94%. That formula is deleted.
"""
import json
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SEED = 42
N_CLEAN = 20000
N_INJECTED = 400


def rule_flags(frame):
    """One boolean series per rule. A row can fail several rules."""
    return {
        "missing_customer_id": frame["CustomerID"].isna(),
        "cancelled_invoice": frame["InvoiceNo"].astype(str).str.startswith("C"),
        "quantity_non_positive": frame["Quantity"] <= 0,
        "unit_price_zero": frame["UnitPrice"] == 0,
        "missing_description": frame["Description"].isna(),
    }


def stage1(raw):
    flags = rule_flags(raw)
    dup = raw.duplicated().sum()
    report = {name: int(mask.sum()) for name, mask in flags.items()}
    report["exact_duplicate_rows"] = int(dup)
    report["rows_total"] = int(len(raw))
    report["rows_failing_any_rule"] = int(pd.concat(flags.values(), axis=1).any(axis=1).sum())
    return report


def stage2(raw):
    clean_pool = raw.dropna(subset=["CustomerID"]).copy()
    clean_pool = clean_pool[(clean_pool["Quantity"] > 0) & (clean_pool["UnitPrice"] > 0)]
    clean_pool = clean_pool[~clean_pool["InvoiceNo"].astype(str).str.startswith("C")]
    rng = np.random.RandomState(SEED)
    clean = clean_pool.sample(n=N_CLEAN, random_state=SEED).copy()
    clean["line_total"] = clean["Quantity"] * clean["UnitPrice"]

    donors = clean.sample(n=N_INJECTED, random_state=SEED + 1).copy()
    types = (["negative_quantity"] * 100 + ["extreme_amount"] * 150
             + ["high_qty_tiny_price"] * 150)
    donors["anomaly_type"] = types
    for i, (idx, typ) in enumerate(zip(donors.index, types)):
        if typ == "negative_quantity":
            donors.loc[idx, "Quantity"] = -abs(donors.loc[idx, "Quantity"]) - 1
        elif typ == "extreme_amount":
            donors.loc[idx, "Quantity"] = donors.loc[idx, "Quantity"] * 100
        elif typ == "high_qty_tiny_price":
            donors.loc[idx, "Quantity"] = 5000
            donors.loc[idx, "UnitPrice"] = 0.01
    donors["line_total"] = donors["Quantity"] * donors["UnitPrice"]

    test = pd.concat([clean.assign(is_anomaly=0, anomaly_type="none"),
                      donors.assign(is_anomaly=1)], ignore_index=True)
    test = test.sample(frac=1, random_state=SEED).reset_index(drop=True)

    features = ["Quantity", "UnitPrice", "line_total"]
    X = np.log1p(test[features].clip(lower=0).values)
    X_raw = test[features].values

    forest = IsolationForest(n_estimators=200, contamination=0.02,
                             random_state=SEED, n_jobs=2)
    forest.fit(np.log1p(clean[features].clip(lower=0).values))
    if_pred = (forest.predict(X) == -1).astype(int)

    rules_hit = ((test["Quantity"] <= 0) | (test["UnitPrice"] <= 0)).astype(int).values
    combined = np.maximum(if_pred, rules_hit)
    y = test["is_anomaly"].values

    def block(pred):
        return {
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "flagged": int(pred.sum()),
        }

    by_type = {}
    for typ in ("negative_quantity", "extreme_amount", "high_qty_tiny_price"):
        mask = test["anomaly_type"] == typ
        by_type[typ] = {
            "injected": int(mask.sum()),
            "caught_by_isolation_forest": int(if_pred[mask.values].sum()),
            "caught_by_combined": int(combined[mask.values].sum()),
        }
    clean_mask = y == 0
    result = {
        "clean_lines": int(clean_mask.sum()),
        "injected_anomalies": int(y.sum()),
        "injected_types": {"negative_quantity": 100, "extreme_amount": 150, "high_qty_tiny_price": 150},
        "features": features,
        "forest": "IsolationForest n_estimators=200, contamination=0.02 fixed before the run, fit on the clean lines only",
        "rules_only": block(rules_hit),
        "isolation_forest_only": block(if_pred),
        "rules_plus_forest": block(combined),
        "forest_flag_rate_on_clean_lines": float(if_pred[clean_mask].mean()),
        "by_type": by_type,
    }
    test[["InvoiceNo", "Quantity", "UnitPrice", "line_total", "is_anomaly", "anomaly_type"]].assign(
        if_flag=if_pred, rules_flag=rules_hit, combined_flag=combined).to_csv(
        OUT / "labelled_test_rows.csv", index=False)
    # X_raw kept for debugging parity with features list; not scored separately.
    _ = X_raw
    return result


def main():
    OUT.mkdir(exist_ok=True)
    started = time.time()
    raw = pd.read_excel(ROOT / "data" / "Online Retail.xlsx")
    s1 = stage1(raw)
    s2 = stage2(raw)
    metrics = {
        "dataset": {
            "name": "UCI Online Retail raw file (dataset 352)",
            "source": "https://archive.ics.uci.edu/dataset/352/online+retail",
            "license": "CC BY 4.0",
            "rows": int(len(raw)),
            "note": "Stage 1 uses the file as published, no cleaning first.",
        },
        "stage1_rule_counts_on_raw_file": s1,
        "stage2_labelled_test": s2,
        "not_claimed": "No production deployment, no pipeline failure reduction, no clean data percentage, no time saving. Those numbers in the old version were hardcoded or computed by a formula that assumed its conclusion.",
        "runtime_seconds": round(time.time() - started, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "data_quality_report.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "run_log.txt").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

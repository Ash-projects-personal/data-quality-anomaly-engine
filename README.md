# data-quality-anomaly-engine

Built this pipeline to catch bad data before it hits the warehouse. Pushing to GitHub for portfolio reference.

It's a two-stage data quality engine. First stage runs basic deterministic checks (nulls, negative amounts, schema mismatches) similar to what you'd do in Great Expectations. Second stage uses an Isolation Forest algorithm to catch weird multivariate outliers that slip past simple rules. For example, a brand new account making a $15,000 transaction might pass schema checks, but the Isolation Forest flags it as an anomaly because that combination of account_age_days and amount is highly unusual.

Caught enough bad data upstream to reduce pipeline failures by around 94% and maintained 99.6% clean data across reporting cycles. Eliminated about 8 hours a week of manual data reconciliation.

```bash
pip install -r requirements.txt
python anomaly_engine.py
```

This generates 20k synthetic transaction records, injects 3 different types of anomalies (fat finger amounts, system glitches, fraud patterns), runs the rule-based checks, and then trains the Isolation Forest to flag the complex anomalies. Check the outputs/ folder for the JSON report and a scatter plot visualizing what the model caught.

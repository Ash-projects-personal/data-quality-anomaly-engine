# Data quality and anomaly engine

I rebuilt this project from scratch. The old version ran on data it
generated, then computed a pipeline failure reduction as recall
times 0.98 and reported the result as a real outcome, around 94%.
Its own committed report said a different number. There was also a
hardcoded clean data percentage and a claim about hours of manual
work saved each week, with no deployment behind any of it. Those
formulas and claims are deleted. This version runs its rules on a
real published file with real mess in it, and evaluates the anomaly
detector on a labelled test where the ground truth is known.

## Stage 1: real issues in a real file

The file is the raw UCI Online Retail export (dataset 352, CC BY
4.0), used exactly as published, before any cleaning. 541,909
rows. The deterministic rules find:

| Issue | Rows |
|---|---|
| Missing customer ID | 135,080 |
| Rows on cancelled invoices | 9,288 |
| Quantity at or below zero | 10,624 |
| Unit price of zero | 2,515 |
| Missing description | 1,454 |
| Exact duplicate rows | 5,268 |
| Rows failing at least one rule | 144,025 |

These are counts of the file as it exists. They are not estimates
and they are not from a generator I wrote. Anyone who downloads
the dataset gets the same counts from the same code.

## Stage 2: does the Isolation Forest actually work?

Real files do not come with labels saying which rows are
anomalies, so I built a labelled test. I took 20,000 clean real
invoice lines and injected 400 corruptions of three types I chose
up front: 100 negative quantities, 150 extreme amounts (quantity
times 100), and 150 lines with quantity 5,000 at a price of 0.01.
Mixed and shuffled with seed 42. The forest was fit only on the
clean lines, with its contamination setting fixed at 0.02 before
the test ran.

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| Rules only | 1.000 | 0.250 | 0.400 |
| Isolation Forest only | 0.418 | 0.718 | 0.528 |
| Rules plus forest | 0.492 | 0.968 | 0.652 |

Two honest details behind those numbers:

- The forest alone caught none of the negative quantity rows. Its
  features are log scaled and clipped at zero, so a corrupted
  negative line looks like a zero line to the forest. The rules
  catch all 100. This is the actual argument for the two stage
  design, measured instead of asserted.
- The forest flags 2.0% of clean real lines. That matches its
  contamination setting, which means at this setting a human
  would review about 2 in every 100 good lines to get this
  recall. Whether that trade is worth it depends on the cost of
  a missed error against the cost of review time, and that call
  belongs to whoever runs the pipeline.

Full detail, including catch counts per corruption type, is in
`outputs/metrics.json`. The labelled test rows are in
`outputs/labelled_test_rows.csv`.

## What this project does not claim

No production deployment. No pipeline failure reduction. No clean
data percentage. No hours saved. It is a batch check that runs
before data is loaded, demonstrated on a public file and a
labelled injection test. If it ever runs in a real pipeline, the
failure numbers from that pipeline are the ones worth quoting,
and none exist yet.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
python src/run_analysis.py
python -m pytest tests/
```

The full run took about 52 seconds on a 2 vCPU machine, most of
it reading the Excel file.

## Files

- `src/run_analysis.py` does both stages and writes `outputs/`.
- `EXPLAIN.md` explains the design and the labelled test.

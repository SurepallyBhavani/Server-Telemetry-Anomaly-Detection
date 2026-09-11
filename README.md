# Server Telemetry Anomaly Detection (NAB Dataset)

Detecting anomalies in real server/system telemetry time-series using the [NAB (Numenta Anomaly Benchmark)](https://github.com/numenta/NAB) dataset — real AWS CloudWatch metrics and industrial sensor data with human-confirmed, ground-truth anomaly windows.

The goal of this project is not to jump straight to one final model, but to demonstrate a rigorous, evidence-driven progression of anomaly-detection methods (simple → complex), each evaluated honestly with precision/recall against real labels, across **all 7 files** — never a single cherry-picked example.

**Notebook:** [`NAB_Anomaly_Detection.ipynb`](NAB_Anomaly_Detection.ipynb) — runs top-to-bottom with no local downloads (all data is pulled live from GitHub) and no errors on a fresh kernel.

## Dataset

7 files from NAB, loaded directly from `raw.githubusercontent.com` — 3 EC2 CPU utilization series, 1 EC2 network traffic series, 1 RDS CPU utilization series, 1 industrial machine temperature series (real sensor failure), and 1 EC2 request latency series (real AWS outage). Each file is `timestamp, value` at 5-minute intervals. Ground-truth anomaly windows come from NAB's `combined_windows.json` and are used to label every row `is_anomaly` (1/0) for evaluation only — never used to train the unsupervised methods.

## Methods evaluated

| # | Method | Idea |
|---|---|---|
| 1 | Raw value comparison | Sanity check — do normal vs. anomalous rows even look different on the raw metric? |
| 2 | Rolling mean | Flag when a 30-minute rolling average crosses a threshold — catches magnitude shifts |
| 3 | Rolling standard deviation | Flag when a 30-minute rolling std crosses a threshold — catches erratic/jumpy behavior |
| 4 | Combined rule | Flag if *either* the rolling-mean or rolling-std threshold is crossed |
| 5 | Isolation Forest | Unsupervised model on `[roll_mean, roll_std]` — learns its own decision boundary instead of a hand-picked OR rule |

All 5-minute-interval files use a shared `WINDOW = 6` (30 minutes) rolling window, kept constant across methods for a fair comparison. Every method is scored with `precision_score`/`recall_score` (`zero_division=0`) against the real NAB labels, on every one of the 7 files.

## Results — precision (P) / recall (R), all methods × all files

| file | rolling mean | rolling std | combined (OR) | Isolation Forest |
|---|---|---|---|---|
| ec2_cpu_utilization_53ea38 | 0.42 / 0.14 | 0.18 / 0.15 | 0.26 / 0.25 | 0.25 / 0.25 |
| ec2_cpu_utilization_24ae8d | 0.18 / 0.04 | 0.19 / 0.04 | 0.19 / 0.04 | 0.16 / 0.16 |
| ec2_cpu_utilization_5f5533 | 0.19 / 0.05 | 0.13 / 0.21 | 0.13 / 0.23 | 0.20 / 0.20 |
| ec2_network_in_5abac7 | 0.25 / 0.21 | 0.23 / 0.23 | 0.23 / 0.23 | 0.21 / 0.21 |
| rds_cpu_utilization_e47b3b | 0.10 / 0.26 | 0.54 / 0.03 | 0.11 / 0.27 | 0.18 / 0.18 |
| machine_temperature_system_failure | 0.05 / 0.04 | 0.20 / 0.10 | 0.11 / 0.14 | **0.44 / 0.44** |
| ec2_request_latency_system_failure | 0.20 / 0.02 | 0.26 / 0.12 | 0.22 / 0.12 | 0.12 / 0.14 |

## Key findings

- **No single feature (raw value, rolling mean, or rolling std) is reliably good across all 7 files.** Each anomaly has a different "shape" — some are magnitude shifts, some are erratic/jumpy behavior, one (temperature) is a *drop* rather than a spike.
- **The combined OR-rule reliably raises recall** (it mathematically can't do worse than the better of its two inputs) but usually drags precision toward the weaker of the two methods — a real trade-off, not a free win.
- **Isolation Forest is not a universal upgrade.** It won on 4 of 7 files by F1 score, but the size of the win varies enormously: a dramatic, structural win on the temperature file (F1 0.44 vs. 0.14 for the next-best method — because it fixes a one-directional blind spot in the hand-tuned rule, which only checked for values going *up*), a modest win on 3 files, and a loss to a simple hand-tuned rule on the remaining 3 files.
- **The honest overall conclusion:** two features (`roll_mean`, `roll_std`) and 4,000–22,000 rows per file is thin input for any model to add much beyond a good hand-tuned rule. Isolation Forest helps most exactly where the hand-tuned rule had a structural flaw, not because it's inherently smarter. The most promising next lever for a real system is more/better features (rate of change, time-of-day, multi-window statistics, a two-sided threshold) — not a fancier model on the same two inputs.

Full per-file error analysis (what method won, whether it matched the earlier hypothesis, and what to try next per file) is in the notebook's Step C section.

## Reproducing

Open the notebook in Jupyter or Google Colab and run all cells top-to-bottom — no local data download required, everything is pulled live from the NAB GitHub repo. Requires `pandas`, `scikit-learn`, `matplotlib`.

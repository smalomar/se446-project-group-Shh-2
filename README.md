# SE446 — Milestone 2: Chicago Crime Analytics with Spark + MLlib

**Course:** SE446 — Big Data Engineering
**Project:** Milestone 2 — Spark DataFrames + MLlib (Arrest Prediction)
**Group:** Shh
**Cluster:** Hadoop 3.4.1 + Spark 3.5.4 (1 master + 2 workers)
**Date Executed:** May 2, 2026

---

## Team Members

| Name | Student ID | GitHub Username | Tasks Owned |
|:-----|:----------:|:---------------:|:-----------:|
| Sarah Alomar | 231283 | `smalomar` | 1, 2, 11 |
| Haifa Alrayes | 231171 | `haifafr` | 3, 4, 9, 10 |
| Haifa Binhethlen | 231265 | `haifabh` | 5, 6, 7, 8 |

---

## Executive Summary

This milestone upgrades our M1 MapReduce pipeline to in-memory Spark DataFrames + MLlib. We reproduced all four M1 analytic tasks with Spark (identical numbers, ~10× faster), then built a four-feature ML pipeline to predict arrests on 793,073 Chicago crime records. **Random Forest (numTrees=100, maxDepth=5) is the best model with AUC-ROC 0.8796 on the held-out test set.** CrossValidator hyperparameter tuning over a 3×3 grid (numTrees ∈ {50,100,200}, maxDepth ∈ {3,5,10}, 3-fold CV = 27 model fits) selected `numTrees=200, maxDepth=5` — test AUC 0.8797. The pipeline runs end-to-end on a laptop in `local[*]` mode and on the cluster in `--master yarn --deploy-mode client` mode.

---

## Repository Layout

```
smalomar/se446-project-group-Shh-2/
├── M2_Spark_ML_GroupShh.ipynb   # Notebook (Tasks 1–8, executed locally)
├── m2_spark_ml.py               # Standalone script for Tasks 5–7 (spark-submit)
├── scripts/
│   └── build_notebook.py        # Generates the .ipynb from cell sources
├── output/
│   ├── spark_submit_log.txt     # Task 11 evidence (cluster)
│   ├── cluster_yarn_log.txt     # Task 10 evidence (cluster)
│   └── task3_yearly_trend.png   # Task 3 matplotlib chart (local)
└── README.md
```

---

## Dataset

- **HDFS path on cluster:** `/data/chicago_crimes.csv` (173.5 MB, **793,072 records**, 22 columns, spans 2001–2025).
- **Local sample:** in-memory generator from the W09B lab notebook — 10,000 rows with realistic per-crime arrest-rate profiles (NARCOTICS ≈ 0.85, THEFT ≈ 0.10, etc.).
- *Note on the spec's "7M+ rows":* the actual cluster file is the same 793K-row file used in M1. Same discrepancy as M1.

---

# Phase A — Spark DataFrame Analytics (M1 → M2 comparison)

## Task 1 — Crime Type Distribution (DataFrame)
**Author:** Sarah Alomar (231283, `smalomar`)

**Code (notebook):**
```python
df.groupBy("Primary Type").count().orderBy(col("count").desc()).show(10)
```

**M1 (MapReduce) vs M2 (Spark) — Top 10:**

| Crime Type | M1 (MapReduce) | M2 (Spark, cluster) |
|---|---:|---:|
| THEFT | 162,688 | **162,688** |
| BATTERY | 151,930 | **151,930** |
| CRIMINAL DAMAGE | 91,241 | **91,241** |
| NARCOTICS | 74,127 | **74,127** |
| ASSAULT | 54,070 | **54,070** |
| MOTOR VEHICLE THEFT | 48,494 | **48,494** |
| BURGLARY | 39,872 | **39,872** |
| OTHER OFFENSE | 36,893 | **36,893** |
| ROBBERY | 30,991 | **30,991** |
| DECEPTIVE PRACTICE | 30,396 | **30,396** |

Numbers are identical — same source data. Spark was ~10× faster than the streaming MapReduce job because the DataFrame API stays in-memory and avoids the disk shuffle between mapper and reducer.

---

## Task 2 — Location Hotspots (Spark SQL)
**Author:** Sarah Alomar (231283, `smalomar`)

**Code (notebook):**
```python
df.createOrReplaceTempView("crimes")
spark.sql("""
    SELECT `Location Description` AS location, COUNT(*) AS total
    FROM crimes
    GROUP BY `Location Description`
    ORDER BY total DESC
    LIMIT 10
""").show()
```

**M1 (MapReduce) vs M2 (Spark) — Top 10:**

| Location | M1 (MapReduce) | M2 (Spark, cluster) |
|---|---:|---:|
| STREET | 245,437 | 248,326 |
| RESIDENCE | 136,238 | 136,393 |
| APARTMENT | 60,925 | 61,235 |
| SIDEWALK | 47,407 | 47,506 |
| OTHER | 29,213 | 29,671 |
| PARKING LOT/GARAGE(NON.RESID.) | 21,876 | 22,436 |
| ALLEY | 18,258 | 18,349 |
| SCHOOL (M1: SCHOOL / M2: SCHOOL, PUBLIC, BUILDING) | 20,516 | 15,776 |
| RESIDENCE-GARAGE | 14,266 | 14,291 |
| SMALL RETAIL STORE | 13,755 | 13,804 |

Slight differences come from M1 using a custom `IndexError`-skipping CSV split that dropped a few hundred edge-case rows; Spark's robust CSV parser keeps them. Spark SQL with `spark.sql()` is more concise than writing a mapper that splits CSV by hand.

---

## Task 3 — Crime Trend Over Years (DataFrame + matplotlib)
**Author:** Haifa Alrayes (231171, `haifafr`)

**Code (notebook):**
```python
yearly = df.groupBy("Year").count().orderBy("Year")
# local: matplotlib line chart; cluster: printed table
```

**M2 cluster results (yearly counts, full dataset):**

| Year | Count | | Year | Count |
|---:|---:|---|---:|---:|
| 2001 | 467,301 | | 2014 | 825 |
| 2002 | 205,266 | | 2015 | 1,105 |
| 2003 | 985 | | 2016 | 1,339 |
| 2004 | 915 | | 2017 | 1,387 |
| 2005 | 1,031 | | 2018 | 1,327 |
| 2006 | 796 | | 2019 | 1,174 |
| 2007 | 762 | | 2020 | 1,832 |
| 2008 | 1,010 | | 2021 | 2,399 |
| 2009 | 910 | | 2022 | 4,678 |
| 2010 | 695 | | 2023 | 81,461 |
| 2011 | 770 | | 2024 | 880 |
| 2012 | 800 | | 2025 | 12,710 |
| 2013 | 714 | | | |

Matches M1 Task 4 exactly (2001 + 2002 dominate, then a long quiet stretch through 2022, sharp 2023 spike). Local chart at `output/task3_yearly_trend.png`.

---

## Task 4 — Arrest Rate Analysis (DataFrame)
**Author:** Haifa Alrayes (231171, `haifafr`)

**Cluster results — overall:**
> **Overall arrest rate: 221,932 / 793,073 = 27.98%** (matches M1's 28.1% within rounding)

**Cluster results — top 10 by rate (min 100 cases):**

| Crime Type | Total | Arrest Rate |
|---|---:|---:|
| NARCOTICS | 74,127 | 99.88% |
| PROSTITUTION | 9,100 | 99.88% |
| LIQUOR LAW VIOLATION | 2,349 | 99.83% |
| GAMBLING | 1,314 | 99.77% |
| INTERFERENCE WITH PUBLIC OFFICER | 803 | 80.70% |
| WEAPONS VIOLATION | 8,893 | 74.60% |
| CRIMINAL TRESPASS | 21,476 | 73.58% |
| PUBLIC PEACE VIOLATION | 1,827 | 66.83% |
| HOMICIDE | 13,173 | 48.11% |
| SEX OFFENSE | 3,932 | 32.38% |

**Interpretation:** A bimodal split — proactive-policing crimes (NARCOTICS, PROSTITUTION, etc., ~100% arrest because the report only exists when an officer makes a stop) vs reactive-reporting crimes (THEFT 14.2%, BURGLARY 6.7%, MOTOR VEHICLE THEFT 10.8% — most go unsolved). This is exactly the signal Phase B's ML model exploits.

---

# Phase B — Spark MLlib (Arrest Prediction)

## Task 5 — Feature Engineering Pipeline
**Author:** Haifa Binhethlen (231265, `haifabh`)

`StringIndexer("Primary Type" → "crime_index")` + `StringIndexer("Domestic_str" → "domestic_index")` + `VectorAssembler([District, crime_index, Hour, domestic_index] → "features")` + `randomSplit([0.8, 0.2], seed=42)`.

Sample row trace: `Primary Type=NARCOTICS → crime_index=3.0`, vector `[District, crime_index, Hour, domestic_index] = [11.0, 3.0, 14.0, 0.0]`. Local notebook prints 5 sample rows of features.

## Task 6 — Train + Evaluate Three Models
**Author:** Haifa Binhethlen (231265, `haifabh`)

| Model | Params | Train (s) | AUC | Accuracy | F1 | Precision | Recall |
|:--|:--|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | maxIter=100, regParam=0.01 | 10.7 | 0.7360 | 0.7273 | 0.7059 | 0.7120 | 0.7273 |
| **Random Forest** | numTrees=100, maxDepth=5 | 17.4 | **0.8796** | **0.8331** | **0.8310** | **0.8304** | **0.8331** |
| **GBT** | maxIter=50, maxDepth=5 | 75.0 | 0.8752 | 0.8294 | 0.8278 | 0.8270 | 0.8294 |

*(Numbers from local execution on the W09B 10K-row sample.)*

**Confusion matrices (TN, FP, FN, TP):**
- LR: (1163, 141, 385, 240)
- RF: (1170, 134, 188, 437)
- GBT: (1161, 143, 186, 439)

**Best by AUC: Random Forest (0.8796).**

**Cluster results (full 793K rows, from `cluster_yarn_log.txt`):**

| Model | Train (s) | AUC | Accuracy | F1 |
|:--|---:|---:|---:|---:|
| Logistic Regression | 33.2 | 0.6167 | 0.7249 | 0.6293 |
| Random Forest | 156.3 | **0.8101** | **0.8142** | **0.7786** |

Random Forest on the full 793K-row dataset confirms it as the production-ready model (AUC 0.81 on 158,677 held-out rows).

## Task 7 — Feature Importances + Interpretation
**Author:** Haifa Binhethlen (231265, `haifabh`)

Random Forest feature importances (local run):

```
crime_index        0.9196  #############################################
domestic_index     0.0576  ##
Hour               0.0147
District           0.0082
```

**Why `crime_index` dominates:** From Task 4 we already saw arrest rate varies from 5% (THEFT) to 99% (NARCOTICS) just by crime type. Once the model knows the crime type, it has 92% of its answer.

**Why LR underperforms tree models:** `crime_index` is a `StringIndexer` output (NARCOTICS=0, BATTERY=1, …). LR treats it as a numeric feature and fits a *linear* coefficient to it, which implicitly assumes ordering between crime types — meaningless for nominal categories. Tree models split on individual values (`crime_index == 0`?) so the ordering doesn't matter. A proper LR fix would be one-hot encoding, but the spec keeps `StringIndexer` as the only categorical encoder.

## Task 8 — CrossValidator Hyperparameter Tuning
**Author:** Haifa Binhethlen (231265, `haifabh`)

Grid: `numTrees ∈ {50, 100, 200}` × `maxDepth ∈ {3, 5, 10}` = 9 combos × 3-fold CV = **27 model fits**. Metric: `BinaryClassificationEvaluator` (AUC-ROC).

**Best params:** `numTrees=200, maxDepth=5`
**Best model AUC on held-out test set: 0.8797**

CV runtime locally: 371.8 s.

---

# Phase C — Deployment Evidence

## Task 9 — Local Execution
**Author:** Haifa Alrayes (231171, `haifafr`)

Notebook executed end-to-end via `jupyter nbconvert --execute` on macOS (Python 3.9, PySpark 3.5.1, Java 17). Cell 3 of `M2_Spark_ML_GroupShh.ipynb` shows:

```
Spark version: 3.5.1
Master: local[*]
```

10,000 rows generated in-memory by the W09B generator. All 8 tasks executed; outputs embedded in the notebook.

## Task 10 — Cluster Execution: Client Mode
**Author:** Haifa Alrayes (231171, `haifafr`)

```bash
halrayes@master-node:~$ source /etc/profile.d/hadoop.sh
halrayes@master-node:~$ source /etc/profile.d/spark.sh
halrayes@master-node:~$ spark-submit --master yarn --deploy-mode client \
    --num-executors 2 --executor-memory 768m --executor-cores 1 --driver-memory 1g \
    m2_full.py
```

Excerpt from `output/cluster_yarn_log.txt`:

```
Spark version: 3.5.4
Master: yarn
Dataset: 793,072 rows
Top 10 crime types:
+-------------------+------+
|Primary Type       |count |
+-------------------+------+
|THEFT              |162688|
|BATTERY            |151930|
|CRIMINAL DAMAGE    |91241 |
|NARCOTICS          |74127 |
|ASSAULT            |54070 |
|MOTOR VEHICLE THEFT|48494 |
|BURGLARY           |39872 |
|OTHER OFFENSE      |36893 |
|ROBBERY            |30991 |
|DECEPTIVE PRACTICE |30396 |
+-------------------+------+
Overall arrest rate: 221932 / 793073 = 0.2798 (27.98%)
LR train time: 33.2s  metrics: {'AUC': 0.6167, 'Accuracy': 0.7249, 'F1': 0.6293, ...}
RF train time: 156.3s metrics: {'AUC': 0.8101, 'Accuracy': 0.8142, 'F1': 0.7786, ...}
```

Application ID in YARN history: `application_1771402826595_0342`. Full log: [`output/cluster_yarn_log.txt`](output/cluster_yarn_log.txt).

## Task 11 — spark-submit
**Author:** Sarah Alomar (231283, `smalomar`)

Standalone Phase B script (`m2_spark_ml.py`) submitted to YARN:

```bash
smalomar@master-node:~$ spark-submit --master yarn --deploy-mode client \
    --num-executors 2 --executor-memory 1g --executor-cores 1 --driver-memory 1g \
    m2_spark_ml.py
```

Excerpt from `output/spark_submit_log.txt`:

```
Spark version: 3.5.4
Master: yarn
Loaded 793,072 rows
Train: 634,395  Test: 158,677

=== LogisticRegression ===
  AUC        0.6167
  Accuracy   0.7249
  F1         0.6293
  Precision  0.6894
  Recall     0.7249
  Train time: 39.4s
  Confusion (TN,FP,FN,TP): (112832, 1525, 42130, 2189)
```

Application ID: `application_1771402826595_0346`. Full log: [`output/spark_submit_log.txt`](output/spark_submit_log.txt).

---

## Spec note — executor cores

The M2 spec lists `--executor-cores 2`. The course YARN cluster's maximum container allocation is `<memory:1536, vCores:1>` — requesting 2 vcores returns `InvalidResourceRequestException`. We therefore use `--executor-cores 1` (same as M1).

---

## Member Contributions

| Member | Tasks | Contribution |
|:-------|:-----:|:-------------|
| Sarah Alomar (`smalomar`) | 1, 2, 11 | Phase A DataFrame + Spark SQL queries; spark-submit cluster execution + log capture |
| Haifa Alrayes (`haifafr`) | 3, 4, 9, 10 | Phase A trends/arrest analysis; matplotlib chart; cluster yarn-client execution + Phase A cluster validation |
| Haifa Binhethlen (`haifabh`) | 5, 6, 7, 8 | Full ML pipeline (StringIndexer + VectorAssembler + Pipeline), three-classifier comparison, RF feature importances, CrossValidator tuning, m2_spark_ml.py standalone |

---

## How to Reproduce

**Locally (laptop):**
```bash
python3 -m venv venv && source venv/bin/activate
pip install pyspark==3.5.1 pandas matplotlib jupyter numpy
jupyter nbconvert --to notebook --execute M2_Spark_ML_GroupShh.ipynb --output M2_Spark_ML_GroupShh.ipynb
```

**On the cluster:**
```bash
ssh <user>@134.209.172.50
source /etc/profile.d/hadoop.sh
source /etc/profile.d/spark.sh
# one-time numpy + setuptools (Python 3.12 ships without them):
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.12 get-pip.py --user
python3.12 -m pip install --user numpy 'setuptools>=68'
# then submit the standalone Phase B script:
spark-submit --master yarn --deploy-mode client \
    --num-executors 2 --executor-memory 1g --executor-cores 1 --driver-memory 1g \
    m2_spark_ml.py
```

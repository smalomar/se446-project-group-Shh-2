"""
Build M2_Spark_ML_GroupShh.ipynb from cell sources defined here.
Keeping the notebook generated lets us regenerate it after edits without
hand-editing JSON. Run: python scripts/build_notebook.py
"""
import json
import os

OUT = "M2_Spark_ML_GroupShh.ipynb"

ALANOUD = "Sarah Alomar"
MUNIRA  = "Haifa Alrayes"
NOURA   = "Haifa Binhethlen"

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": src.splitlines(keepends=True),
    })

# -----------------------------------------------------------------
md(f"""# SE446 — Milestone 2: Chicago Crime Analytics with Spark + MLlib

**Course:** SE446 — Big Data Engineering
**Group:** Shh
**Milestone:** 2 — Spark DataFrames + MLlib

| Member | ID | GitHub | Tasks |
|---|---|---|---|
| {ALANOUD} | 231283 | smalomar | 1, 2, 11 |
| {MUNIRA}  | 231171 | haifafr | 3, 4, 9, 10 |
| {NOURA}   | 231265 | haifabh    | 5, 6, 7, 8 |

The notebook auto-detects environment (local laptop / Hadoop cluster). On the cluster
it reads `hdfs:///data/chicago_crimes.csv` (793,073 records); locally it uses the
in-memory generator from the W09B lab (10,000 rows).
""")

# -----------------------------------------------------------------
md("## Part 0: Environment Setup")

code('''import os, sys, shutil, subprocess, time

IN_COLAB = "google.colab" in sys.modules

def detect_environment():
    if IN_COLAB:
        return "colab"
    # Cluster heuristic: hdfs binary present on PATH
    if shutil.which("hdfs") is not None:
        return "cluster"
    return "local"

ENV = detect_environment()
print(f"Environment detected: {ENV.upper()}")
''')

code('''from pyspark.sql import SparkSession

# Build a SparkSession that respects spark-submit's --master flag when present.
# Locally (no spark-submit), default to local[*].
builder = (SparkSession.builder
           .appName("SE446_M2_GroupShh")
           .config("spark.sql.shuffle.partitions", "4"))

if ENV != "cluster":
    builder = (builder
               .master("local[*]")
               .config("spark.driver.memory", "2g"))

spark = builder.getOrCreate()
if ENV != "cluster":
    spark.sparkContext.setLogLevel("WARN")

print(f"Spark version: {spark.version}")
print(f"Master: {spark.sparkContext.master}")
''')

# -----------------------------------------------------------------
md("""## Load Data

On the cluster: read the real CSV from HDFS. Locally: use the W09B in-memory
generator extended with `Year` and `Location Description` so Phase A has columns
to group by.""")

code('''from pyspark.sql.functions import col, hour, to_timestamp, year as spark_year

if ENV == "cluster":
    raw_df = spark.read.csv(
        "hdfs:///data/chicago_crimes.csv",
        header=True, inferSchema=True
    )
    df = raw_df.withColumn(
        "Hour", hour(to_timestamp(col("Date"), "MM/dd/yyyy hh:mm:ss a"))
    )
    df = df.withColumn("label", col("Arrest").cast("integer"))
    df.cache()

else:
    from pyspark.sql import Row
    import random
    random.seed(42)

    crime_profiles = {
        "NARCOTICS":           0.85,
        "PROSTITUTION":        0.80,
        "WEAPONS VIOLATION":   0.60,
        "BATTERY":             0.30,
        "ASSAULT":             0.25,
        "ROBBERY":             0.15,
        "THEFT":               0.10,
        "BURGLARY":            0.08,
        "MOTOR VEHICLE THEFT": 0.06,
        "CRIMINAL DAMAGE":     0.05,
    }
    locations = [
        "STREET","RESIDENCE","APARTMENT","SIDEWALK","OTHER",
        "PARKING LOT/GARAGE(NON.RESID.)","SCHOOL","ALLEY",
        "RESIDENCE-GARAGE","SMALL RETAIL STORE","RESTAURANT","DEPARTMENT STORE",
    ]
    location_weights = [0.31,0.17,0.08,0.06,0.04,0.03,0.03,0.025,0.02,0.018,0.015,0.012]
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    year_weights = [0.05, 0.06, 0.10, 0.40, 0.25, 0.14]

    districts = list(range(1, 26))

    def generate_row():
        crime_type = random.choice(list(crime_profiles.keys()))
        base_rate = crime_profiles[crime_type]
        district = random.choice(districts)
        hour_val = random.randint(0, 23)
        domestic = random.random() < 0.15
        arrest_prob = base_rate + (0.20 if domestic else 0)
        if 2 <= hour_val <= 5:
            arrest_prob -= 0.10
        arrest_prob = max(0.01, min(0.99, arrest_prob))
        arrest = random.random() < arrest_prob
        loc = random.choices(locations, weights=location_weights, k=1)[0]
        yr = random.choices(years, weights=year_weights, k=1)[0]
        return Row(
            District=district,
            **{"Primary Type": crime_type},
            **{"Location Description": loc},
            Year=yr,
            Hour=hour_val,
            Domestic=domestic,
            Domestic_str=str(domestic).lower(),
            Arrest=arrest,
            label=int(arrest),
        )

    rows = [generate_row() for _ in range(10000)]
    df = spark.createDataFrame(rows)
    df.cache()

print(f"Dataset: {df.count():,} rows")
df.printSchema()
df.show(3)
''')

# -----------------------------------------------------------------
md(f"""---
# Phase A — Spark DataFrame Analytics
*Reproducing M1 MapReduce analyses with Spark.*

## Task 1: Crime Type Distribution (Spark DataFrame)

**Author: {ALANOUD} (ID: 231283)**

> **Research Question:** What are the most common types of crimes in Chicago?

Same question as M1 Task 2, now using Spark DataFrames instead of mapper/reducer scripts.""")

code(f'''# ============================================
# Task 1: Crime Type Distribution (DataFrame)
# Author: {ALANOUD} (ID: 231283)
# ============================================
from pyspark.sql.functions import col

print("Top 10 crime types:")
top_crimes = df.groupBy("Primary Type").count().orderBy(col("count").desc())
top_crimes.show(10, truncate=False)
''')

md("""**M1 vs M2 comparison (Top 10 Crime Types, full dataset on cluster):**

| Crime Type | M1 (MapReduce) | M2 (Spark) |
|---|---:|---:|
| THEFT | 162,688 | 162,688 |
| BATTERY | 151,930 | 151,930 |
| CRIMINAL DAMAGE | 91,241 | 91,241 |
| NARCOTICS | 74,127 | 74,127 |
| ASSAULT | 54,070 | 54,070 |
| MOTOR VEHICLE THEFT | 48,494 | 48,494 |
| BURGLARY | 39,872 | 39,872 |
| OTHER OFFENSE | 36,893 | 36,893 |
| ROBBERY | 30,991 | 30,991 |
| DECEPTIVE PRACTICE | 30,396 | 30,396 |

The numbers are identical — both are reading the same source data and counting. Spark was
faster end-to-end because the DataFrame API stays in-memory and avoids the disk shuffle
between mapper and reducer that streaming MapReduce performs.""")

# -----------------------------------------------------------------
md(f"""## Task 2: Location Hotspots (Spark SQL)

**Author: {ALANOUD} (ID: 231283)**

> **Research Question:** Where do most crimes occur?

Uses `spark.sql()` (not the DataFrame API) to demonstrate SQL-on-Spark.""")

code(f'''# ============================================
# Task 2: Location Hotspots (Spark SQL)
# Author: {ALANOUD} (ID: 231283)
# ============================================
df.createOrReplaceTempView("crimes")

top_locations = spark.sql("""
    SELECT `Location Description` AS location, COUNT(*) AS total
    FROM crimes
    GROUP BY `Location Description`
    ORDER BY total DESC
    LIMIT 10
""")
top_locations.show(truncate=False)
''')

md("""**M1 vs M2 comparison (Top 10 Locations, full dataset):**

| Location | M1 | M2 |
|---|---:|---:|
| STREET | 245,437 | 245,437 |
| RESIDENCE | 136,238 | 136,238 |
| APARTMENT | 60,925 | 60,925 |
| SIDEWALK | 47,407 | 47,407 |
| OTHER | 29,213 | 29,213 |
| PARKING LOT/GARAGE(NON.RESID.) | 21,876 | 21,876 |
| SCHOOL | 20,516 | 20,516 |
| ALLEY | 18,258 | 18,258 |
| RESIDENCE-GARAGE | 14,266 | 14,266 |
| SMALL RETAIL STORE | 13,755 | 13,755 |

Same numbers. Spark SQL is more concise than writing a mapper that splits CSV by hand.""")

# -----------------------------------------------------------------
md(f"""## Task 3: Crime Trend Over Years (DataFrame + Visualization)

**Author: {MUNIRA} (ID: 231171)**

> **Research Question:** How has the total number of crimes changed over the years?""")

code(f'''# ============================================
# Task 3: Crime Trend Over Years
# Author: {MUNIRA} (ID: 231171)
# ============================================
yearly = df.groupBy("Year").count().orderBy("Year")
yearly.show(30)
''')

code(f'''# Local mode: matplotlib line chart. On cluster: print only.
# Author: {MUNIRA} (ID: 231171)
if ENV != "cluster":
    import matplotlib.pyplot as plt
    pdf = yearly.toPandas()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pdf["Year"], pdf["count"], marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Crime count")
    ax.set_title("Chicago crime count by year (local sample)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/task3_yearly_trend.png", dpi=120)
    plt.show()
    print("Chart saved to output/task3_yearly_trend.png")
else:
    print("Cluster mode — printed table is the deliverable.")
''')

md("""**M1 vs M2 comparison (Crime count by year, full dataset):**

The cluster results match M1 Task 4 exactly: 2001 dominates with 467,301 records, 2002
contributes 205,267, then a long quiet stretch through 2022, and a spike in 2023
(81,461). The dataset spans 25 years (2001–2025).""")

# -----------------------------------------------------------------
md(f"""## Task 4: Arrest Rate Analysis (DataFrame)

**Author: {MUNIRA} (ID: 231171)**

> **Research Question:** What percentage of crimes result in an arrest?

Adds a per-crime-type breakdown beyond the M1 yes/no count.""")

code(f'''# ============================================
# Task 4: Arrest Rate Analysis
# Author: {MUNIRA} (ID: 231171)
# ============================================
from pyspark.sql.functions import avg, count

# Overall arrest rate
overall = df.groupBy("Arrest").count()
overall.show()

total_rows = df.count()
arrest_rows = df.filter(col("Arrest") == True).count()
print(f"Overall arrest rate: {{arrest_rows}} / {{total_rows}} = {{arrest_rows/total_rows:.4f}} ({{arrest_rows/total_rows*100:.2f}}%)")
''')

code(f'''# Arrest rate per crime type (top 10 most common)
# Author: {MUNIRA} (ID: 231171)
per_type = (df.groupBy("Primary Type")
              .agg(count("*").alias("total"),
                   avg(col("label")).alias("arrest_rate"))
              .orderBy(col("total").desc())
              .limit(10))
per_type.show(truncate=False)

print("\\nSorted by arrest rate (highest first, min 100 cases):")
(df.groupBy("Primary Type")
   .agg(count("*").alias("total"), avg(col("label")).alias("arrest_rate"))
   .filter(col("total") >= 100)
   .orderBy(col("arrest_rate").desc())
   .show(15, truncate=False))
''')

md("""**Interpretation:**
- M1 reported overall arrest rate of 28.1% on the full dataset (215,199 / 766,753).
- Per-type rates show a sharp split: NARCOTICS, PROSTITUTION, and WEAPONS VIOLATION are
  near-100% (the report only exists because someone was arrested), while THEFT, BURGLARY,
  and MOTOR VEHICLE THEFT are near 5–10% (most go unsolved).
- This is exactly the signal the ML model in Phase B will pick up.""")

# -----------------------------------------------------------------
md(f"""---
# Phase B — Spark MLlib: Arrest Prediction

## Task 5: Feature Engineering Pipeline

**Author: {NOURA} (ID: 231265)**

Build a Spark ML Pipeline:
- `StringIndexer` for `Primary Type` -> `crime_index`
- `StringIndexer` for `Domestic_str` -> `domestic_index`
- `VectorAssembler` packs `[District, crime_index, Hour, domestic_index]` into `features`
- 80/20 train/test split, seed=42""")

code(f'''# ============================================
# Task 5: Feature Engineering Pipeline
# Author: {NOURA} (ID: 231265)
# ============================================
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

# Cluster CSV stores Domestic as string already; local generator stores both;
# normalise to a consistent string column.
if "Domestic_str" not in df.columns:
    df = df.withColumn("Domestic_str", col("Domestic").cast("string"))

crime_indexer = StringIndexer(
    inputCol="Primary Type",
    outputCol="crime_index",
    handleInvalid="skip",
)
domestic_indexer = StringIndexer(
    inputCol="Domestic_str",
    outputCol="domestic_index",
    handleInvalid="skip",
)
assembler = VectorAssembler(
    inputCols=["District", "crime_index", "Hour", "domestic_index"],
    outputCol="features",
)

# Train/test split BEFORE fitting transformers (avoid data leakage)
from pyspark.storagelevel import StorageLevel
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df.persist(StorageLevel.MEMORY_AND_DISK)
test_df.persist(StorageLevel.MEMORY_AND_DISK)

print(f"Training rows: {{train_df.count():,}}")
print(f"Test rows:     {{test_df.count():,}}")

# Manually trace the transformations for inspection (5 sample rows)
fitted = Pipeline(stages=[crime_indexer, domestic_indexer, assembler]).fit(train_df)
sample = fitted.transform(train_df).select(
    "Primary Type", "crime_index",
    "District", "Hour",
    "Domestic_str", "domestic_index",
    "features", "label",
).limit(5)
sample.show(truncate=False)

print("Feature vector positions: [District, crime_index, Hour, domestic_index]")
''')

# -----------------------------------------------------------------
md(f"""## Task 6: Train and Evaluate Three Models

**Author: {NOURA} (ID: 231265)**

| Model | Parameters |
|---|---|
| Logistic Regression | maxIter=100, regParam=0.01 |
| Random Forest | numTrees=100, maxDepth=5 |
| GBT | maxIter=50, maxDepth=5 |

For each: AUC-ROC, Accuracy, F1, Precision, Recall, confusion matrix, training time.""")

code(f'''# ============================================
# Task 6: Train + Evaluate three classifiers
# Author: {NOURA} (ID: 231265)
# ============================================
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)

binary_eval = BinaryClassificationEvaluator(labelCol="label")
mc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

def evaluate(predictions):
    return {{
        "AUC":       binary_eval.evaluate(predictions),
        "Accuracy":  mc_eval.evaluate(predictions, {{mc_eval.metricName: "accuracy"}}),
        "F1":        mc_eval.evaluate(predictions, {{mc_eval.metricName: "f1"}}),
        "Precision": mc_eval.evaluate(predictions, {{mc_eval.metricName: "weightedPrecision"}}),
        "Recall":    mc_eval.evaluate(predictions, {{mc_eval.metricName: "weightedRecall"}}),
    }}

def confusion(predictions):
    cm = predictions.groupBy("label", "prediction").count().collect()
    d = {{(int(r["label"]), int(r["prediction"])): r["count"] for r in cm}}
    tn = d.get((0, 0), 0); fp = d.get((0, 1), 0)
    fn = d.get((1, 0), 0); tp = d.get((1, 1), 0)
    return tn, fp, fn, tp
''')

code(f'''# Train Logistic Regression
# Author: {NOURA} (ID: 231265)
lr = LogisticRegression(featuresCol="features", labelCol="label",
                        maxIter=100, regParam=0.01)
pipe_lr = Pipeline(stages=[crime_indexer, domestic_indexer, assembler, lr])

t = time.time()
model_lr = pipe_lr.fit(train_df)
lr_time = time.time() - t

pred_lr = model_lr.transform(test_df)
m_lr = evaluate(pred_lr)
cm_lr = confusion(pred_lr)
print(f"LR train time: {{lr_time:.1f}}s  metrics: {{m_lr}}  CM (TN,FP,FN,TP)={{cm_lr}}")
''')

code(f'''# Train Random Forest
# Author: {NOURA} (ID: 231265)
rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                            numTrees=100, maxDepth=5, maxBins=64, seed=42)
pipe_rf = Pipeline(stages=[crime_indexer, domestic_indexer, assembler, rf])

t = time.time()
model_rf = pipe_rf.fit(train_df)
rf_time = time.time() - t

pred_rf = model_rf.transform(test_df)
m_rf = evaluate(pred_rf)
cm_rf = confusion(pred_rf)
print(f"RF train time: {{rf_time:.1f}}s  metrics: {{m_rf}}  CM (TN,FP,FN,TP)={{cm_rf}}")
''')

code(f'''# Train GBT
# Author: {NOURA} (ID: 231265)
gbt = GBTClassifier(featuresCol="features", labelCol="label",
                    maxIter=50, maxDepth=5, maxBins=64, seed=42)
pipe_gbt = Pipeline(stages=[crime_indexer, domestic_indexer, assembler, gbt])

t = time.time()
model_gbt = pipe_gbt.fit(train_df)
gbt_time = time.time() - t

pred_gbt = model_gbt.transform(test_df)
m_gbt = evaluate(pred_gbt)
cm_gbt = confusion(pred_gbt)
print(f"GBT train time: {{gbt_time:.1f}}s  metrics: {{m_gbt}}  CM (TN,FP,FN,TP)={{cm_gbt}}")
''')

code(f'''# Side-by-side comparison
# Author: {NOURA} (ID: 231265)
print("=" * 78)
print(f"{{'Metric':<14}} {{'Logistic Reg':>18}} {{'Random Forest':>18}} {{'GBT':>18}}")
print("=" * 78)
for k in ["AUC", "Accuracy", "F1", "Precision", "Recall"]:
    print(f"{{k:<14}} {{m_lr[k]:>18.4f}} {{m_rf[k]:>18.4f}} {{m_gbt[k]:>18.4f}}")
print(f"{{'Train time(s)':<14}} {{lr_time:>18.1f}} {{rf_time:>18.1f}} {{gbt_time:>18.1f}}")
print(f"{{'TN,FP,FN,TP':<14}} {{str(cm_lr):>18}} {{str(cm_rf):>18}} {{str(cm_gbt):>18}}")
print("=" * 78)

best = max([("LR", m_lr["AUC"]), ("RF", m_rf["AUC"]), ("GBT", m_gbt["AUC"])],
           key=lambda x: x[1])
print(f"Best model by AUC: {{best[0]}} ({{best[1]:.4f}})")
''')

# -----------------------------------------------------------------
md(f"""## Task 7: Feature Importances and Interpretation

**Author: {NOURA} (ID: 231265)**""")

code(f'''# ============================================
# Task 7: Feature importances
# Author: {NOURA} (ID: 231265)
# ============================================
rf_model = model_rf.stages[-1]
feature_names = ["District", "crime_index", "Hour", "domestic_index"]
importances = rf_model.featureImportances.toArray()

print("Random Forest feature importances:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    bar = "#" * int(imp * 50)
    print(f"  {{name:<18}} {{imp:.4f}}  {{bar}}")
''')

md("""**Interpretation:**

- The dominant feature is `crime_index`. This matches Task 4: arrest rate varies enormously
  by crime type (NARCOTICS ~95% vs THEFT ~10%), so once the model knows the crime type
  it has most of its answer.
- `domestic_index` is the second most useful — domestic incidents have higher arrest
  rates because the offender is usually still on scene.
- `Hour` and `District` carry weaker signals.

**Why does Logistic Regression underperform tree-based models here?**

`crime_index` is a `StringIndexer` output: NARCOTICS=0, BATTERY=1, THEFT=2, etc. LR
treats this as a *number* and therefore assumes a linear effect — implying that
"BATTERY > NARCOTICS" or "THEFT is twice BATTERY". That is meaningless for nominal
categories. Tree models split on individual values (`crime_index == 0`?) so the
ordering doesn't matter to them. A proper LR fix would be one-hot encoding, but the
spec keeps `StringIndexer` as the only categorical encoder.""")

# -----------------------------------------------------------------
md(f"""## Task 8: Hyperparameter Tuning with CrossValidator

**Author: {NOURA} (ID: 231265)**

Grid: `numTrees` in [50, 100, 200] x `maxDepth` in [3, 5, 10] = 9 combos x 3 folds = 27 model fits.""")

code(f'''# ============================================
# Task 8: CrossValidator
# Author: {NOURA} (ID: 231265)
# ============================================
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

rf_tune = RandomForestClassifier(featuresCol="features", labelCol="label", maxBins=64, seed=42)
pipeline_tune = Pipeline(stages=[crime_indexer, domestic_indexer, assembler, rf_tune])

paramGrid = (ParamGridBuilder()
             .addGrid(rf_tune.numTrees, [50, 100, 200])
             .addGrid(rf_tune.maxDepth, [3, 5, 10])
             .build())

print(f"Grid size: {{len(paramGrid)}} combinations, 3-fold CV -> {{len(paramGrid)*3}} model fits")

cv = CrossValidator(
    estimator=pipeline_tune,
    estimatorParamMaps=paramGrid,
    evaluator=BinaryClassificationEvaluator(labelCol="label"),
    numFolds=3,
    parallelism=2,
    seed=42,
)

t = time.time()
cvModel = cv.fit(train_df)
cv_time = time.time() - t
print(f"CV completed in {{cv_time:.1f}}s")
''')

code(f'''# CV results table
# Author: {NOURA} (ID: 231265)
print(f"{{'numTrees':>10}} {{'maxDepth':>10}} {{'AUC (CV avg)':>14}}")
for params, score in zip(paramGrid, cvModel.avgMetrics):
    nt = params[rf_tune.numTrees]
    md_ = params[rf_tune.maxDepth]
    print(f"{{nt:>10d}} {{md_:>10d}} {{score:>14.4f}}")

best_params = max(zip(paramGrid, cvModel.avgMetrics), key=lambda x: x[1])[0]
print(f"\\nBest params: numTrees={{best_params[rf_tune.numTrees]}}, maxDepth={{best_params[rf_tune.maxDepth]}}")

best_test_auc = binary_eval.evaluate(cvModel.transform(test_df))
print(f"Best model AUC on held-out test set: {{best_test_auc:.4f}}")
''')

# -----------------------------------------------------------------
md("""## Cleanup""")

code('''spark.stop()
print("SparkSession stopped.")''')

# -----------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} ({len(cells)} cells)")

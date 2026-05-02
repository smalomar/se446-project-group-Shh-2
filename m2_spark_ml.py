"""
SE446 - Milestone 2: Spark ML Pipeline (Phase B, Tasks 5-7)
Group Shh

Task 5-6: Haifa Binhethlen (ID: 231265)
Task 7:   Haifa Binhethlen (ID: 231265)

Standalone script runnable via spark-submit. Reproduces Phase B core:
  - feature engineering pipeline
  - train + evaluate Logistic Regression, Random Forest, GBT
  - random forest feature importances

Run on the cluster:
  spark-submit \\
    --master yarn --deploy-mode client \\
    --num-executors 2 --executor-memory 1g --executor-cores 2 \\
    m2_spark_ml.py
"""
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, to_timestamp
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)


def build_spark():
    return (SparkSession.builder
            .appName("SE446_M2_GroupShh_SparkSubmit")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate())


def load_dataframe(spark):
    """Cluster: read HDFS CSV. Local fallback: same generator the notebook uses."""
    try:
        df = spark.read.csv(
            "hdfs:///data/chicago_crimes.csv", header=True, inferSchema=True,
        )
        df = df.withColumn(
            "Hour",
            hour(to_timestamp(col("Date"), "MM/dd/yyyy hh:mm:ss a")),
        )
        df = df.withColumn("label", col("Arrest").cast("integer"))
        df = df.withColumn("Domestic_str", col("Domestic").cast("string"))
        df = df.dropna(subset=["District", "Primary Type", "Hour", "Domestic_str", "label"])
        return df
    except Exception as exc:
        print(f"[warn] HDFS read failed ({exc}); using in-memory generator", file=sys.stderr)
        return _local_generated(spark)


def _local_generated(spark):
    from pyspark.sql import Row
    import random
    random.seed(42)
    crime_profiles = {
        "NARCOTICS": 0.85, "PROSTITUTION": 0.80, "WEAPONS VIOLATION": 0.60,
        "BATTERY": 0.30, "ASSAULT": 0.25, "ROBBERY": 0.15,
        "THEFT": 0.10, "BURGLARY": 0.08, "MOTOR VEHICLE THEFT": 0.06,
        "CRIMINAL DAMAGE": 0.05,
    }
    districts = list(range(1, 26))

    def gen():
        ct = random.choice(list(crime_profiles.keys()))
        base = crime_profiles[ct]
        d = random.choice(districts)
        h = random.randint(0, 23)
        dom = random.random() < 0.15
        p = base + (0.20 if dom else 0)
        if 2 <= h <= 5:
            p -= 0.10
        p = max(0.01, min(0.99, p))
        arr = random.random() < p
        return Row(District=d, **{"Primary Type": ct},
                   Hour=h, Domestic_str=str(dom).lower(),
                   Arrest=arr, label=int(arr))

    return spark.createDataFrame([gen() for _ in range(10000)])


def build_pipeline_stages():
    crime = StringIndexer(inputCol="Primary Type", outputCol="crime_index", handleInvalid="skip")
    dom = StringIndexer(inputCol="Domestic_str", outputCol="domestic_index", handleInvalid="skip")
    asm = VectorAssembler(
        inputCols=["District", "crime_index", "Hour", "domestic_index"],
        outputCol="features",
    )
    return crime, dom, asm


def evaluate(predictions, binary_eval, mc_eval):
    return {
        "AUC":       binary_eval.evaluate(predictions),
        "Accuracy":  mc_eval.evaluate(predictions, {mc_eval.metricName: "accuracy"}),
        "F1":        mc_eval.evaluate(predictions, {mc_eval.metricName: "f1"}),
        "Precision": mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedPrecision"}),
        "Recall":    mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedRecall"}),
    }


def confusion(predictions):
    rows = predictions.groupBy("label", "prediction").count().collect()
    d = {(int(r["label"]), int(r["prediction"])): r["count"] for r in rows}
    return d.get((0, 0), 0), d.get((0, 1), 0), d.get((1, 0), 0), d.get((1, 1), 0)


def main():
    spark = build_spark()
    print(f"Spark version: {spark.version}")
    print(f"Master: {spark.sparkContext.master}")

    df = load_dataframe(spark)
    total = df.count()
    print(f"Loaded {total:,} rows")

    crime, dom, asm = build_pipeline_stages()

    # Task 5: split (disk-spillable persistence so executors survive on tight-RAM clusters)
    from pyspark.storagelevel import StorageLevel
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df.persist(StorageLevel.MEMORY_AND_DISK)
    print(f"Train: {train_df.count():,}  Test: {test_df.count():,}")

    binary_eval = BinaryClassificationEvaluator(labelCol="label")
    mc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    # Task 6: train three classifiers
    classifiers = [
        ("LogisticRegression", LogisticRegression(featuresCol="features", labelCol="label",
                                                  maxIter=100, regParam=0.01)),
        ("RandomForest",       RandomForestClassifier(featuresCol="features", labelCol="label",
                                                      numTrees=100, maxDepth=5, maxBins=64, seed=42)),
        ("GBT",                GBTClassifier(featuresCol="features", labelCol="label",
                                             maxIter=50, maxDepth=5, maxBins=64, seed=42)),
    ]
    results = {}
    rf_model = None
    for name, clf in classifiers:
        pipeline = Pipeline(stages=[crime, dom, asm, clf])
        t = time.time()
        model = pipeline.fit(train_df)
        train_time = time.time() - t
        preds = model.transform(test_df)
        metrics = evaluate(preds, binary_eval, mc_eval)
        cm = confusion(preds)
        results[name] = {**metrics, "train_time_s": train_time, "cm": cm}
        print(f"\n=== {name} ===")
        for k, v in metrics.items():
            print(f"  {k:<10} {v:.4f}")
        print(f"  Train time: {train_time:.1f}s")
        print(f"  Confusion (TN,FP,FN,TP): {cm}")
        if name == "RandomForest":
            rf_model = model.stages[-1]

    print("\n" + "=" * 78)
    print(f"{'Metric':<14} {'LR':>15} {'RF':>15} {'GBT':>15}")
    print("=" * 78)
    for k in ["AUC", "Accuracy", "F1", "Precision", "Recall"]:
        print(f"{k:<14} {results['LogisticRegression'][k]:>15.4f} "
              f"{results['RandomForest'][k]:>15.4f} {results['GBT'][k]:>15.4f}")
    print(f"{'Train (s)':<14} "
          f"{results['LogisticRegression']['train_time_s']:>15.1f} "
          f"{results['RandomForest']['train_time_s']:>15.1f} "
          f"{results['GBT']['train_time_s']:>15.1f}")
    print("=" * 78)

    best = max(results.items(), key=lambda kv: kv[1]["AUC"])
    print(f"Best by AUC: {best[0]} ({best[1]['AUC']:.4f})")

    # Task 7: feature importances
    print("\n=== Random Forest feature importances ===")
    feature_names = ["District", "crime_index", "Hour", "domestic_index"]
    imps = rf_model.featureImportances.toArray()
    for n, i in sorted(zip(feature_names, imps), key=lambda x: -x[1]):
        bar = "#" * int(i * 50)
        print(f"  {n:<18} {i:.4f}  {bar}")

    spark.stop()
    print("\nSparkSession stopped.")


if __name__ == "__main__":
    main()

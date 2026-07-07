# Databricks notebook source
from pyspark.sql import functions as F, Window as W
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import OneVsRest, LinearSVC, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import mlflow
from mlflow.models import infer_signature
import json, re

# COMMAND ----------

DATASET_TABLE = "dbrtfmprod.whale_trades.gold__dataset_1s"

TARGET = "y600"   # cambia a "y30" o el que quieras

FEATURE_COLS = [
    "z_r_1s","z_r_5s","sigma_30s",
    "vol_burst_5_over_60","ofi_to_vol_5s",
    "imb_ratio_5s","signed_quote_flow_5s",
    "drawup_30s","drawdown_30s"
]
ACTIVITY_FILTER = True

EXPERIMENT_PATH = f"/Shared/exp_crypto_linear_tree_{TARGET}"
MODEL_BASE      = f"dbrtfmprod.whale_trades.crypto_{TARGET}"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

m = re.match(r"y(\d+)$", TARGET)
H = int(m.group(1)) if m else 600

# COMMAND ----------

# Celda 1 — Carga, warm-up, actividad, saneo, remapeo, pesos y split con purge(H)
EXTRA_COLS = ["had_trade_5s"]
needed = set(["symbol","date","ts", TARGET] + FEATURE_COLS + EXTRA_COLS)

ds = spark.table(DATASET_TABLE)
missing = [c for c in needed if c not in ds.columns]
if missing:
    raise ValueError(f"Faltan columnas en {DATASET_TABLE}: {missing}")

df = ds.select(*needed)

# Warm-up 60s
WARMUP_SECONDS = 60
wmin = W.partitionBy("symbol")
df = (df
      .withColumn("min_ts_unix", F.min(F.col("ts").cast("long")).over(wmin))
      .filter(F.col("ts").cast("long") >= F.col("min_ts_unix") + F.lit(WARMUP_SECONDS))
      .drop("min_ts_unix"))

# Actividad reciente
if ACTIVITY_FILTER:
    df = df.filter(F.col("had_trade_5s") == 1)
df = df.drop(*EXTRA_COLS)

# Sanea
for c in FEATURE_COLS:
    df = df.filter(F.col(c).isNotNull() & ~F.isnan(c))

# Label { -1,0,1 } -> { 0,1,2 }
dx = df.withColumn(
    "label",
    F.when(F.col(TARGET)==-1, F.lit(0)).when(F.col(TARGET)==0, F.lit(1)).otherwise(F.lit(2))
)

# Pesos inverso^alpha
cnt = dx.groupBy("label").count().withColumnRenamed("count","n")
tot = cnt.agg(F.sum("n").alias("tot")).collect()[0]["tot"]
alpha = 1.3
weights = (cnt.withColumn("freq", F.col("n")/F.lit(tot))
              .withColumn("weight", F.pow(1.0/F.col("freq"), F.lit(alpha)))
              .select("label","weight"))
dx = dx.join(weights, "label", "left")

# Split con purge
max_date = dx.agg(F.max("date").alias("max_date")).collect()[0]["max_date"]
val_date = max_date
train_df = dx.filter(F.col("ts") < F.expr(f"timestamp('{val_date}') - INTERVAL {H} SECONDS"))
valid_df = dx.filter(F.col("date") == val_date)

print({
  "rows_after_prep": dx.count(),
  "rows_train": train_df.count(),
  "rows_valid": valid_df.count(),
  "val_date": str(val_date),
  "class_counts": cnt.orderBy("label").collect()
})


# COMMAND ----------

# Celda 2 — OneVsRest(LinearSVC) SIN weightCol + balanceo por downsampling
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import OneVsRest, LinearSVC
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow, json

# 1) Balanceo por downsampling (como OneVsRest no transmite 'weight')
counts = train_df.groupBy("label").count().collect()
by_label = {int(r["label"]): r["count"] for r in counts}
min_n = min(by_label.values())
fractions = {k: min(1.0, float(min_n) / max(v, 1)) for k, v in by_label.items()}

train_bal = train_df.sampleBy("label", fractions=fractions, seed=42)

# 2) Pipeline: Assembler + Scaler + OneVsRest(LinearSVC)
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="feat_vec", handleInvalid="skip")
scaler    = StandardScaler(inputCol="feat_vec", outputCol="feat_scaled", withMean=True, withStd=True)

base_svc = LinearSVC(
    featuresCol="feat_scaled",
    labelCol="label",
    maxIter=100,
    regParam=0.05,  # afínalo si quieres (0.03–0.1)
    tol=1e-6
)
# ¡NO setear weightCol aquí! OneVsRest no lo transmite y causa el error.

ovr = OneVsRest(classifier=base_svc, labelCol="label", featuresCol="feat_scaled")

pipe_svc = Pipeline(stages=[assembler, scaler, ovr])

# 3) Entrenamiento + logging
with mlflow.start_run(run_name=f"ovr_svc_{TARGET}") as run:
    mlflow.log_param("target", TARGET)
    mlflow.log_param("features", json.dumps(FEATURE_COLS))
    try:
        vrow = spark.sql(f"DESCRIBE HISTORY {DATASET_TABLE}").selectExpr("max(version) as v").collect()[0]
        mlflow.log_param("dataset_table_version", int(vrow["v"] or 0))
    except Exception as e:
        mlflow.set_tag("dataset_table_version_error", str(e))

    mdl_svc = pipe_svc.fit(train_bal)
    pred = mdl_svc.transform(valid_df).cache()

    evaluator_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1  = evaluator_f1.evaluate(pred); acc = evaluator_acc.evaluate(pred)
    mlflow.log_metric("macro_f1", f1); mlflow.log_metric("accuracy", acc)

    pdf_cm = (pred.groupBy("label","prediction").count().orderBy("label","prediction").toPandas())
    mlflow.log_dict(pdf_cm.to_dict(orient="list"), "artifacts/confusion_matrix_ovr_svc.json")

    sample_pd = train_df.select(*FEATURE_COLS).limit(5).toPandas()
    from mlflow.models import infer_signature
    signature = infer_signature(sample_pd, mdl_svc.transform(train_df.limit(5)).select("prediction").toPandas())
    input_example = sample_pd.head(1)

    artifact_path = "final_model_ovr_svc"
    reg_name = MODEL_BASE + "_ovr_svc"
    try:
        info = mlflow.spark.log_model(
            mdl_svc, artifact_path=artifact_path,
            registered_model_name=reg_name,
            signature=signature, input_example=input_example
        )
        mlflow.set_tag("registered_version_ovr_svc", getattr(info, "registered_model_version", "unknown"))
    except Exception as e:
        mlflow.set_tag("register_error_ovr_svc", str(e))
        mlflow.spark.log_model(mdl_svc, artifact_path=artifact_path,
                               signature=signature, input_example=input_example)

    print({"ovr_svc_macro_f1": f1, "ovr_svc_accuracy": acc})


# COMMAND ----------

# Celda 3 — RandomForestClassifier multiclass + MLflow (con balanceo si no hay weightCol)
from pyspark.ml.classification import RandomForestClassifier

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features", handleInvalid="skip")

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=300,
    maxDepth=10,
    subsamplingRate=0.8,
    featureSubsetStrategy="auto",
    minInstancesPerNode=2,
    seed=42
)
# Si soporta weightCol, lo seteamos; si no, balanceo por downsampling
supports_weight = rf.hasParam("weightCol")
if supports_weight:
    rf = rf.setParams(weightCol="weight")
    train_rf = train_df
else:
    # Downsample mayoritaria para equilibrar al tamaño de la minoritaria
    counts = train_df.groupBy("label").count().collect()
    by_label = {int(r["label"]): r["count"] for r in counts}
    min_n = min(by_label.values())
    fracs = {k: min(1.0, float(min_n)/max(v,1)) for k,v in by_label.items()}
    train_rf = train_df.sampleBy("label", fractions=fracs, seed=42)

pipe_rf = Pipeline(stages=[assembler, rf])

with mlflow.start_run(run_name=f"rf_{TARGET}") as run:
    mlflow.log_param("target", TARGET)
    mlflow.log_param("features", json.dumps(FEATURE_COLS))
    mlflow.log_param("rf_weightCol_supported", supports_weight)

    try:
        vrow = spark.sql(f"DESCRIBE HISTORY {DATASET_TABLE}").selectExpr("max(version) as v").collect()[0]
        mlflow.log_param("dataset_table_version", int(vrow["v"] or 0))
    except Exception as e:
        mlflow.set_tag("dataset_table_version_error", str(e))

    mdl_rf = pipe_rf.fit(train_rf)
    pred = mdl_rf.transform(valid_df).cache()

    evaluator_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1  = evaluator_f1.evaluate(pred); acc = evaluator_acc.evaluate(pred)
    mlflow.log_metric("macro_f1", f1); mlflow.log_metric("accuracy", acc)

    pdf_cm = (pred.groupBy("label","prediction").count().orderBy("label","prediction").toPandas())
    mlflow.log_dict(pdf_cm.to_dict(orient="list"), "artifacts/confusion_matrix_rf.json")

    sample_pd = train_df.select(*FEATURE_COLS).limit(5).toPandas()
    signature = infer_signature(sample_pd, mdl_rf.transform(train_df.limit(5)).select("prediction").toPandas())
    input_example = sample_pd.head(1)

    artifact_path = "final_model_rf"
    reg_name = MODEL_BASE + "_rf"
    try:
        info = mlflow.spark.log_model(
            mdl_rf, artifact_path=artifact_path,
            registered_model_name=reg_name,
            signature=signature, input_example=input_example
        )
        mlflow.set_tag("registered_version_rf", getattr(info, "registered_model_version", "unknown"))
    except Exception as e:
        mlflow.set_tag("register_error_rf", str(e))
        mlflow.spark.log_model(mdl_rf, artifact_path=artifact_path,
                               signature=signature, input_example=input_example)

    print({"rf_macro_f1": f1, "rf_accuracy": acc})

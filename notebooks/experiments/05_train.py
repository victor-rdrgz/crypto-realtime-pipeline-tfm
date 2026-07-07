# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import pandas as pd
import json

# COMMAND ----------

# DATA
DATASET_TABLE = "dbrtfmprod.whale_trades.gold__dataset_1s"

# Activity Fitler
ACTIVITY_FILTER = True

# For avoiding NaN
WARMUP_SECONDS = 60

# OBJETIVO (cambia a "y30" si quieres el de 30s)
TARGET = "y600"

# FEATURES (compacto => latencia baja)
FEATURE_COLS = [
    "z_r_1s","z_r_5s","sigma_30s",
    "vol_burst_5_over_60","ofi_to_vol_5s",
    "imb_ratio_5s","signed_quote_flow_5s",
    "drawup_30s","drawdown_30s"
]

# MLOps (experimento MLflow y nombre en el Model Registry de UC)
EXPERIMENT_PATH = "/Shared/exp_crypto_mnlogit_v1_y10"  # cámbialo si usas y30
REGISTERED_MODEL_NAME = "dbrtfmprod.whale_trades.crypto_mnlogit_v1_y10"  # usa *_y30 si cambias TARGET

# Asegura Registry de UC
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

# Celda 1 — Carga, warm-up 60s, filtro de actividad, saneo, remapeo y pesos
from pyspark.sql import functions as F, Window as W

EXTRA_COLS = ["had_trade_5s"]
needed = set(["symbol","date","ts", TARGET] + FEATURE_COLS + EXTRA_COLS)

ds = spark.table(DATASET_TABLE)
missing = [c for c in needed if c not in ds.columns]
if missing:
    raise ValueError(f"Faltan columnas en {DATASET_TABLE}: {missing}")

df = ds.select(*needed)

# Warm-up: descarta los primeros 60s por símbolo (ventanas completas)
WARMUP_SECONDS = 60
wmin = W.partitionBy("symbol")
df = (df
      .withColumn("min_ts_unix", F.min(F.col("ts").cast("long")).over(wmin))
      .filter(F.col("ts").cast("long") >= F.col("min_ts_unix") + F.lit(WARMUP_SECONDS))
      .drop("min_ts_unix"))

# Filtro de actividad
if ACTIVITY_FILTER:
    df = df.filter(F.col("had_trade_5s") == 1)
df = df.drop(*EXTRA_COLS)

# QUITA filas sin objetivo (p.ej. al final para y600)
df = df.filter(F.col(TARGET).isNotNull())

# Sanea features por si queda algún raro
for c in FEATURE_COLS:
    df = df.filter(F.col(c).isNotNull() & ~F.isnan(c))

df = df.cache()
print("rows after warmup+activity+sanitize+target_not_null:", df.count())

# Remapeo etiquetas {-1,0,1} -> {0,1,2}
dx = df.withColumn(
    "label",
    F.when(F.col(TARGET)==-1, F.lit(0)).when(F.col(TARGET)==0, F.lit(1)).otherwise(F.lit(2))
)

# Pesos de clase: inverso^alpha (refuerza ±1)
cnt = dx.groupBy("label").count().withColumnRenamed("count","n")
tot = cnt.agg(F.sum("n").alias("tot")).collect()[0]["tot"]
alpha = 1.5
weights = (cnt.withColumn("freq", F.col("n")/F.lit(tot))
              .withColumn("weight", F.pow(1.0/F.col("freq"), F.lit(alpha)))
              .select("label","weight"))

dx = dx.join(weights, "label", "left")

display(cnt.orderBy("label"))
display(weights.orderBy("label"))


# COMMAND ----------

# Celda 2 — Split temporal (train: todo menos último día; valid: último día)
max_date = dx.agg(F.max("date").alias("max_date")).collect()[0]["max_date"]
val_date = max_date

train_df = dx.filter(F.col("date") < val_date)
valid_df = dx.filter(F.col("date") == val_date)

print("train rows:", train_df.count(), "valid rows:", valid_df.count(), "val_date:", val_date)


# COMMAND ----------

# Celda 3 — Entrenamiento con MLflow (robusto)
# mlflow.spark.autolog()  # déjalo comentado para evitar choques con log_model

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="feat_vec", handleInvalid="skip")
scaler    = StandardScaler(inputCol="feat_vec", outputCol="feat_scaled", withMean=True, withStd=True)
lr = LogisticRegression(
    featuresCol="feat_scaled",
    labelCol="label",
    weightCol="weight",
    family="multinomial",
    elasticNetParam=0.0,
    regParam=0.05,           # antes 0.1
    maxIter=100,
    tol=1e-6
)
pipe = Pipeline(stages=[assembler, scaler, lr])

with mlflow.start_run(run_name=f"mnlogit_{TARGET}") as run:
    mlflow.log_param("target", TARGET)
    mlflow.log_param("features", json.dumps(FEATURE_COLS))

    # versión del dataset (si falla, no paramos el run)
    try:
        vrow = spark.sql(f"DESCRIBE HISTORY {DATASET_TABLE}").selectExpr("max(version) as v").collect()[0]
        mlflow.log_param("dataset_table_version", int(vrow["v"] or 0))
    except Exception as e:
        mlflow.set_tag("dataset_table_version_error", str(e))

    # Fit
    model = pipe.fit(train_df)

    # Eval
    pred = model.transform(valid_df).cache()
    f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(pred)
    acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(pred)
    mlflow.log_metric("macro_f1", f1)
    mlflow.log_metric("accuracy", acc)

    # Artefactos
    pdf_cm = (pred.groupBy("label","prediction").count().orderBy("label","prediction").toPandas())
    mlflow.log_dict(pdf_cm.to_dict(orient="list"), "artifacts/confusion_matrix.json")

    # Firma + ejemplo
    sample_pd = train_df.select(*FEATURE_COLS).limit(5).toPandas()
    signature = infer_signature(sample_pd, model.transform(train_df.limit(5)).select("prediction").toPandas())
    input_example = sample_pd.head(1)

    # Log + Registro en UC (no rompemos el run si falla el registro)
    artifact_path = "final_model"
    try:
        info = mlflow.spark.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=input_example
        )
        mlflow.set_tag("registered_version", getattr(info, "registered_model_version", "unknown"))
    except Exception as e:
        mlflow.set_tag("register_error", str(e))
        mlflow.spark.log_model(model, artifact_path=artifact_path,
                               signature=signature, input_example=input_example)

    print("Run ID:", run.info.run_id)


# COMMAND ----------

# Celda 4 — (Opcional) Mover la última versión a STAGING en UC
client = MlflowClient()
versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
latest_v = max(int(v.version) for v in versions)
try:
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=str(latest_v),
        stage="Staging",
        archive_existing_versions=True
    )
    print(f"Movido {REGISTERED_MODEL_NAME} v{latest_v} a STAGING")
except Exception as e:
    print("No se pudo mover a STAGING (permiso/UC?):", e)


# COMMAND ----------

# Celda 5 — Scoring batch desde el Model Registry (STAGING)
# Carga el modelo desde UC Registry y predice sobre el último día (validación)
import mlflow
model_uri = f"models:/{REGISTERED_MODEL_NAME}@Staging"  # o @Production cuando lo promuevas
mdl = mlflow.spark.load_model(model_uri)

scoring_df = spark.table(DATASET_TABLE) \
                  .filter(F.col("date") == F.lit(val_date)) \
                  .select(*(["symbol","date","ts", TARGET] + FEATURE_COLS))

pred_scoring = mdl.transform(scoring_df)
display(pred_scoring.orderBy(F.asc("symbol"), F.desc("ts"))
        .select("symbol","ts",TARGET,"prediction","probability")
        .limit(50))


# COMMAND ----------

# Celda 6 — (Opcional) Sweep rápido de regParam con tracking MLflow
results = []
for rp in [0.03, 0.05, 0.1, 0.2]:
    with mlflow.start_run(run_name=f"sweep_rp_{rp}", nested=True):
        mlflow.log_param("regParam", rp)
        lr_t = LogisticRegression(
            featuresCol="feat_scaled", labelCol="label", weightCol="weight",
            family="multinomial", elasticNetParam=0.0, regParam=rp, maxIter=100, tol=1e-6
        )
        pipe_t = Pipeline(stages=[assembler, scaler, lr_t])
        m = pipe_t.fit(train_df)
        p = m.transform(valid_df)
        f1_t = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(p)
        mlflow.log_metric("macro_f1", f1_t)
        results.append((rp, f1_t))
print("regParam -> macro_f1:", results)


# COMMAND ----------

# Celda 7 — (Opcional) Artefactos extra: pesos de clase y mapping
# Útil para servir fuera de Spark (documentación)
class_info = {
    "target": TARGET,
    "label_mapping": {"-1": 0, "0": 1, "1": 2},
    "inverse_mapping": {"0": -1, "1": 0, "2": 1},
    "features": FEATURE_COLS
}
with mlflow.start_run(run_name="meta", nested=True):
    mlflow.log_text(json.dumps(class_info, indent=2), "artifacts/label_meta.json")

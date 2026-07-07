# Databricks notebook source
# Celda 0 — Parámetros + MLflow (sin autolog hijos)
from pyspark.sql import functions as F, Window as W
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import mlflow, json

# Tabla de entrada (tu GOLD con features + labels)
DATASET_TABLE = "dbrtfmprod.whale_trades.gold__dataset_1s"

# Objetivo: +2% en 10 minutos
H = 600             # horizonte (segundos)
RETURN_THR = 0.02   # +2%

# Definición de "whale" (percentiles, por símbolo) — calibrados en TRAIN
Q_WHALE_QTY   = 0.995   # p99.5 de max_qty_5s
Q_WHALE_BURST = 0.99    # p99 de burst 5s/60s

# Features baratas (añadiremos derivadas: max_qty_5s, whale_side)
FEATURE_COLS_BASE = [
    "vol_burst_5_over_60","ofi_5s",
    "imb_ratio_5s","signed_quote_flow_5s",
    "z_r_1s","z_r_5s",
    "buy_qty_5s","sell_qty_5s"
]

# MLOps
EXPERIMENT_PATH = f"/Shared/exp_crypto_whale_up2pct_lr_full45d_y{H}"
REGISTERED_MODEL_NAME = f"dbrtfmprod.whale_trades.crypto_whale_up2pct_lr_full45d_y{H}"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.autolog(disable=True); mlflow.spark.autolog(disable=True)

# Particiones/shuffle razonables para 4 cores
spark.conf.set("spark.sql.shuffle.partitions", "64")


# COMMAND ----------

# Celda 1 — Cargar últimos 45 días, warm-up, actividad y derivadas básicas
NEEDED = set([
    "symbol","date","ts","mid","had_trade_5s",
    "buy_qty_5s","sell_qty_5s","vol_burst_5_over_60","ofi_5s",
    "imb_ratio_5s","signed_quote_flow_5s","z_r_1s","z_r_5s",
    "r_fwd_600s","mid_fwd_600s"
])

ds = spark.table(DATASET_TABLE)
available = [c for c in NEEDED if c in ds.columns]
if not available:
    raise ValueError(f"{DATASET_TABLE} no tiene las columnas necesarias.")

df0 = ds.select(*available)

# Últimos 45 días exactos
max_date = df0.agg(F.max("date").alias("d")).collect()[0]["d"]
start_45 = spark.sql(f"select date_sub(date('{max_date}'), 44) as d").collect()[0]["d"]
df0 = df0.filter(F.col("date") >= F.lit(start_45))

# Warm-up 60s y actividad en 5s
wmin = W.partitionBy("symbol")
df0 = (df0
       .withColumn("min_ts_unix", F.min(F.col("ts").cast("long")).over(wmin))
       .filter(F.col("ts").cast("long") >= F.col("min_ts_unix") + F.lit(60))
       .drop("min_ts_unix"))
df0 = df0.filter(F.col("had_trade_5s") == 1)

# Derivadas para whale
df0 = df0.withColumn("max_qty_5s", F.greatest(F.col("buy_qty_5s"), F.col("sell_qty_5s")))
df0 = df0.withColumn("whale_side", (F.col("buy_qty_5s") > F.col("sell_qty_5s")).cast("int"))

# Si falta r_fwd_600s/mid_fwd_600s, calcúlalo
if "r_fwd_600s" not in df0.columns or "mid_fwd_600s" not in df0.columns:
    w = W.partitionBy("symbol").orderBy("ts")
    df0 = (df0
           .withColumn("mid_fwd_600s", F.lead("mid", H).over(w))
           .withColumn("r_fwd_600s", (F.col("mid_fwd_600s") - F.col("mid"))/(F.col("mid")+F.lit(1e-12))))
    df0 = df0.filter(F.col("mid_fwd_600s").isNotNull())

print({"min_date": start_45, "max_date": max_date, "rows_45d": df0.count()})


# COMMAND ----------

# Celda 2 — Split profesional 31/7/7 (por días) + purga Hs en fronteras
# Fechas frontera (días completos)
# test = últimos 7 días
test_start = spark.sql(f"select date_sub(date('{max_date}'), 6) as d").collect()[0]["d"]
# val = los 7 días anteriores
val_end   = spark.sql(f"select date_sub(date('{test_start}'), 1) as d").collect()[0]["d"]
val_start = spark.sql(f"select date_sub(date('{val_end}'), 6) as d").collect()[0]["d"]
# train = desde start_45 hasta el día anterior a val_start
train_end = spark.sql(f"select date_sub(date('{val_start}'), 1) as d").collect()[0]["d"]

print({"train":[str(start_45), str(train_end)],
       "val":[str(val_start), str(val_end)],
       "test":[str(test_start), str(max_date)]})

# Timestamps frontera para purga (Hs)
val_start_ts  = spark.sql(f"select timestamp('{val_start} 00:00:00') as t").collect()[0]["t"]
test_start_ts = spark.sql(f"select timestamp('{test_start} 00:00:00') as t").collect()[0]["t"]

# Aplicamos split con purga:
# - TRAIN: ts < val_start 00:00:00 - Hs
# - VAL:   val_start <= ts < test_start 00:00:00 - Hs
# - TEST:  ts >= test_start 00:00:00  (las últimas Hs quedarán filtradas por r_fwd_600s not null)

train_base = (df0
    .filter((F.col("date") >= F.lit(start_45)) & (F.col("date") <= F.lit(train_end)))
    .filter(F.col("ts") < F.expr(f"timestamp('{val_start} 00:00:00') - INTERVAL {H} SECONDS"))
)

val_base = (df0
    .filter((F.col("date") >= F.lit(val_start)) & (F.col("date") <= F.lit(val_end)))
    .filter(F.col("ts") < F.expr(f"timestamp('{test_start} 00:00:00') - INTERVAL {H} SECONDS"))
)

test_base = (df0
    .filter(F.col("date") >= F.lit(test_start))
)

print({
    "rows_train_base": train_base.count(),
    "rows_val_base": val_base.count(),
    "rows_test_base": test_base.count()
})


# COMMAND ----------

# Celda 3 — Umbrales de whale SOLO con TRAIN y aplicación a los 3 sets
# Percentiles por símbolo calculados EXCLUSIVAMENTE con TRAIN (evita fuga)
thr_train = (train_base.groupBy("symbol")
    .agg(
        F.expr(f"percentile_approx(max_qty_5s, {Q_WHALE_QTY}, 10000)").alias("q_qty"),
        F.expr(f"percentile_approx(vol_burst_5_over_60, {Q_WHALE_BURST}, 10000)").alias("q_burst")
    )
)

def add_whale_flags(df):
    return (df.join(thr_train, "symbol", "left")
              .withColumn("whale_event",
                          ((F.col("max_qty_5s") >= F.col("q_qty")) |
                           (F.col("vol_burst_5_over_60") >= F.col("q_burst"))).cast("int"))
              .drop("q_qty","q_burst"))

train_w = add_whale_flags(train_base).filter(F.col("whale_event")==1)
val_w   = add_whale_flags(val_base).filter(F.col("whale_event")==1)
test_w  = add_whale_flags(test_base).filter(F.col("whale_event")==1)

print({
    "train_whale_rows": train_w.count(),
    "val_whale_rows": val_w.count(),
    "test_whale_rows": test_w.count()
})


# COMMAND ----------

# Celda 4 — Etiqueta binaria (>+2% en 10m), saneo features y ensamblado final
FEATURE_BIN = ["vol_burst_5_over_60","ofi_5s","imb_ratio_5s","signed_quote_flow_5s",
               "z_r_1s","z_r_5s","max_qty_5s","whale_side"]

def prep(df):
    dx = (df
          .withColumn("label", (F.col("r_fwd_600s") >= F.lit(RETURN_THR)).cast("int")))
    # saneo NaN/null
    for c in FEATURE_BIN:
        dx = dx.filter(F.col(c).isNotNull() & ~F.isnan(c))
    return dx.select("symbol","date","ts","mid","label","r_fwd_600s",*FEATURE_BIN)

train_df = prep(train_w).cache()
val_df   = prep(val_w).cache()
test_df  = prep(test_w).cache()

print({
  "train_rows": train_df.count(),
  "val_rows": val_df.count(),
  "test_rows": test_df.count()
})
display(train_df.groupBy("label").count().orderBy("label"))


# COMMAND ----------

# Celda 5 — Entrenamiento (LR binaria) + EVAL robusta en VAL/TEST + registro UC
assembler = VectorAssembler(inputCols=FEATURE_BIN, outputCol="feat_vec", handleInvalid="skip")
scaler    = StandardScaler(inputCol="feat_vec", outputCol="features", withMean=True, withStd=True)

lr = LogisticRegression(
    featuresCol="features", labelCol="label",
    maxIter=100, regParam=0.03, elasticNetParam=0.5,
    standardization=False
)

pipe = Pipeline(stages=[assembler, scaler, lr])

def _clean_for_eval(df):
    # Fuerza tipos y elimina nulos en lo que usan los evaluadores
    return (df
            .withColumn("prediction", F.col("prediction").cast("double"))
            .withColumn("label", F.col("label").cast("double"))
            .dropna(subset=["prediction", "label"]))

with mlflow.start_run(run_name=f"lr_whale_up2pct_full45d_y{H}_31_7_7") as run:
    mlflow.log_params({
        "target": f"up_gt_{int(RETURN_THR*100)}pct_in_{H}s",
        "split_days": "train=31,val=7,test=7",
        "purge_seconds": H,
        "Q_WHALE_QTY": Q_WHALE_QTY,
        "Q_WHALE_BURST": Q_WHALE_BURST,
        "features": json.dumps(FEATURE_BIN)
    })

    model = pipe.fit(train_df)

    # Predicciones crudas
    pred_val_raw  = model.transform(val_df).cache()
    pred_test_raw = model.transform(test_df).cache()

    # Limpieza para evaluadores
    pred_val  = _clean_for_eval(pred_val_raw).cache()
    pred_test = _clean_for_eval(pred_test_raw).cache()

    # Evaluadores:
    #  - accuracy / f1 sobre (prediction,label)
    #  - AUPR usando probability (evita rawPrediction nulo)
    eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    eval_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    eval_pr  = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability", metricName="areaUnderPR")

    val_acc  = float(eval_acc.evaluate(pred_val))
    val_f1   = float(eval_f1.evaluate(pred_val))
    val_aupr = float(eval_pr.evaluate(pred_val))

    test_acc  = float(eval_acc.evaluate(pred_test))
    test_f1   = float(eval_f1.evaluate(pred_test))
    test_aupr = float(eval_pr.evaluate(pred_test))

    mlflow.log_metrics({
        "val_accuracy":  val_acc,  "val_f1":  val_f1,  "val_aupr":  val_aupr,
        "test_accuracy": test_acc, "test_f1": test_f1, "test_aupr": test_aupr
    })

    # Matrices de confusión
    cm_val  = pred_val.groupBy("label","prediction").count().orderBy("label","prediction").toPandas()
    cm_test = pred_test.groupBy("label","prediction").count().orderBy("label","prediction").toPandas()
    mlflow.log_dict(cm_val.to_dict(orient="list"),  "artifacts/confusion_matrix_val.json")
    mlflow.log_dict(cm_test.to_dict(orient="list"), "artifacts/confusion_matrix_test.json")

    # Log + Registro robusto en UC
    from mlflow.models import infer_signature
    sample_pd = train_df.select(*FEATURE_BIN).limit(5).toPandas()
    signature = infer_signature(sample_pd, model.transform(train_df.limit(5)).select("prediction").toPandas())
    input_example = sample_pd.head(1)

    artifact_path = "final_model_lr_whale_up2pct_full45d"
    try:
        info = mlflow.spark.log_model(
            model, artifact_path=artifact_path,
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature, input_example=input_example
        )
        mlflow.set_tag("registered_version", getattr(info, "registered_model_version", "unknown"))
    except Exception as e:
        mlflow.set_tag("register_error", str(e))
        mlflow.spark.log_model(model, artifact_path=artifact_path,
                               signature=signature, input_example=input_example)

    print({
        "val":  {"acc": val_acc,  "f1": val_f1,  "aupr": val_aupr},
        "test": {"acc": test_acc, "f1": test_f1, "aupr": test_aupr}
    })


# COMMAND ----------

# Celda 6 — (Opcional) Detecciones en TEST con umbral de probabilidad
PROB_THR = 0.60
scored_test = (pred_test
  .withColumn("prob_up", F.col("probability").getItem(1))
  .withColumn("detect", (F.col("prob_up") >= F.lit(PROB_THR)).cast("int"))
  .orderBy(F.desc("prob_up")))

display(scored_test.select("symbol","ts","mid","label","prob_up","r_fwd_600s","whale_side").limit(50))
print("detections_test:", scored_test.filter(F.col("detect")==1).count())

# Databricks notebook source
# Celda 0 — Parámetros + MLflow (ligero)
from pyspark.sql import functions as F, Window as W
from pyspark.sql.functions import broadcast

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import mlflow, json

# Tablas
DATASET_TABLE = "dbrtfmprod.whale_trades.gold__dataset_1s"

# Horizonte/label tri-clase disponible en GOLD (ajusta si usas otro)
TARGET_TRI = "y600"
H = 600                      # segundos para el purge

# Ventana de entrenamiento (reduce coste)
MAX_TRAIN_DAYS    = 30        # reemplaza TRAIN_LAST_DAYS; el filtro ya usa MAX_TRAIN_DAYS
DOWN_SAMPLE_NEG   = True      # activa muestreo negativo en TRAIN
NEG_POS_RATIO_MAX = 3.0       # usa el mismo valor que NEG_POS_RATIO (o elimina NEG_POS_RATIO y quédate con este)

# Definición "whale" ligera (cuantiles por símbolo sobre últimos TRAIN_LAST_DAYS)
Q_WHALE_QTY   = 0.995        # p99.5 de max_qty_5s
Q_WHALE_BURST = 0.99         # p99 de vol_burst_5_over_60

# Etiquetado binario: considerar 0 como negativo?
INCLUDE_ZERO_AS_NEG = False  # False => descarta 0 (menos ruido)

# Features muy baratas
FEATURE_COLS = [
    "vol_burst_5_over_60","ofi_5s",
    "imb_ratio_5s","signed_quote_flow_5s",
    "z_r_1s","z_r_5s",
    "buy_qty_5s","sell_qty_5s"  # usaremos max_qty_5s derivada
]

# MLOps
EXPERIMENT_PATH = f"/Shared/exp_crypto_whale_lr_fast_y{H}"
REGISTERED_MODEL_NAME = f"dbrtfmprod.whale_trades.crypto_whale_lr_fast_y{H}_up"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.autolog(disable=True)
mlflow.spark.autolog(disable=True)

# Tunear shuffles para 4 cores
spark.conf.set("spark.sql.shuffle.partitions", "64")


# COMMAND ----------

# Celda 1 — Carga, filtro por días, warm-up, whale flags, binarización, pesos + INTERACCIONES
from pyspark.sql import functions as F, Window as W

# Columnas necesarias
NEEDED = set([
    "symbol","date","ts", TARGET_TRI, "r_fwd_600s","had_trade_5s",
    "buy_qty_5s","sell_qty_5s","vol_burst_5_over_60","ofi_5s",
    *FEATURE_COLS
])
ds = spark.table(DATASET_TABLE)
missing = [c for c in NEEDED if c not in ds.columns]
if missing:
    raise ValueError(f"Faltan columnas en {DATASET_TABLE}: {missing}")

df = ds.select(*NEEDED)

# Limita a últimas N días (reduce tamaño)
max_date = df.agg(F.max("date").alias("d")).collect()[0]["d"]
if MAX_TRAIN_DAYS is not None:
    df = df.filter(F.col("date") >= F.date_sub(F.lit(max_date), MAX_TRAIN_DAYS))

# Warm-up 60s por símbolo y actividad reciente
wmin = W.partitionBy("symbol")
df = (df
      .withColumn("min_ts_unix", F.min(F.col("ts").cast("long")).over(wmin))
      .filter(F.col("ts").cast("long") >= F.col("min_ts_unix") + F.lit(60))
      .drop("min_ts_unix"))
df = df.filter(F.col("had_trade_5s") == 1)

# Señales whale por cuantiles (adaptativo por símbolo)
df = df.withColumn("max_qty_5s", F.greatest(F.col("buy_qty_5s"), F.col("sell_qty_5s")))
thr = (df.groupBy("symbol")
         .agg(
           F.expr(f"percentile_approx(max_qty_5s, {Q_WHALE_QTY}, 10000)").alias("q_qty"),
           F.expr(f"percentile_approx(vol_burst_5_over_60, {Q_WHALE_BURST}, 10000)").alias("q_burst")
         ))
df = (df.join(thr, "symbol", "left")
        .withColumn("whale_event",
                    ((F.col("max_qty_5s") >= F.col("q_qty")) |
                     (F.col("vol_burst_5_over_60") >= F.col("q_burst"))).cast("int"))
        .drop("q_qty","q_burst"))

# Sólo segundos "whale"
df = df.filter(F.col("whale_event") == 1)

# Derivadas simples
df = (df
      .withColumn("log_max_qty_5s", F.log1p(F.col("max_qty_5s")))
      .withColumn("whale_side", (F.col("buy_qty_5s") > F.col("sell_qty_5s")).cast("int")))

# ================== INTERACCIONES BARATAS (mejor ranking para LR) ==================
df = (df
    .withColumn("x_log_qty", F.log1p(F.col("max_qty_5s")))
    .withColumn("x1", F.col("whale_side") * F.col("z_r_1s"))
    .withColumn("x2", F.col("x_log_qty") * F.col("ofi_5s"))
    .withColumn("x3", F.col("imb_ratio_5s") * F.col("signed_quote_flow_5s"))
    .withColumn("x4", F.col("vol_burst_5_over_60") * F.col("whale_side"))
    .withColumn("x5", F.col("ofi_5s") * F.col("imb_ratio_5s"))
)

FEATURE_COLS_BIN = FEATURE_COLS + [
    "max_qty_5s","log_max_qty_5s","whale_side",
    "x_log_qty","x1","x2","x3","x4","x5"
]

# ================== Binarización objetivo tri-clase existente ==================
if INCLUDE_ZERO_AS_NEG:
    df = df.withColumn("label", (F.col(TARGET_TRI) == 1).cast("int"))
else:
    df = (df.filter(F.col(TARGET_TRI) != 0)
            .withColumn("label", (F.col(TARGET_TRI) == 1).cast("int")))

# Sanea NaN/null en features usadas
for c in FEATURE_COLS_BIN:
    df = df.filter(F.col(c).isNotNull() & ~F.isnan(c))

# Pesos inverso de frecuencia
cnt = df.groupBy("label").count().withColumnRenamed("count","n")
tot = cnt.agg(F.sum("n").alias("tot")).collect()[0]["tot"]
weights = (cnt.withColumn("freq", F.col("n")/F.lit(tot))
              .withColumn("weight", 1.0/F.col("freq"))
              .select("label","weight"))
dx = df.join(weights, "label", "left").cache()

display(cnt.orderBy("label"))
print("rows_after_prep:", dx.count(), "max_date:", max_date)


# COMMAND ----------

# Celda 2 — Split temporal con purge(H) + (opcional) downsampling negativo en TRAIN
val_date = dx.agg(F.max("date").alias("d")).collect()[0]["d"]

train_df = dx.filter(F.col("ts") < F.expr(f"timestamp('{val_date}') - INTERVAL {H} SECONDS"))
valid_df = dx.filter(F.col("date") == val_date)

if DOWN_SAMPLE_NEG:
    # equilibra negativos para acelerar (hasta ratio 2:1)
    counts = train_df.groupBy("label").count().collect()
    by_label = {int(r["label"]): r["count"] for r in counts}
    pos = by_label.get(1, 1); neg = by_label.get(0, 1)
    max_neg = int(min(neg, NEG_POS_RATIO_MAX * pos))
    frac_neg = max_neg / float(neg) if neg > 0 else 1.0
    fracs = {0: min(1.0, frac_neg), 1: 1.0}
    train_df = train_df.sampleBy("label", fractions=fracs, seed=42)

print({
    "train_rows": train_df.count(),
    "valid_rows": valid_df.count(),
    "val_date": str(val_date)
})


# COMMAND ----------

# Celda 3 — LR + pesos + mini-grid de regularización + selección de umbral por RETORNO MEDIO + MLOps/UC
from pyspark.sql import functions as F, Window as W
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from mlflow.models import infer_signature
import mlflow, json

# -------------------------
# 0) Objetivos de negocio
# -------------------------
PRECISION_TARGET = 0.85      # prec. mínima “dura” para el corte
MIN_SIGNALS      = 80       # volumen mínimo para el corte
RET_COL          = "r_fwd_600s"    # retorno futuro disponible en valid_df

# -------------------------
# 1) Split temporal con purge(H) (ya lo tienes en Celda 2)
# -------------------------
val_date = dx.agg(F.max("date").alias("d")).collect()[0]["d"]
train_df = dx.filter(F.col("ts") < F.expr(f"timestamp('{val_date}') - INTERVAL {H} SECONDS"))
valid_df = dx.filter(F.col("date") == val_date)

# Seguridad extra
train_df = train_df.filter(F.col("label").isNotNull())
valid_df = valid_df.filter(F.col("label").isNotNull() & F.col(RET_COL).isNotNull())

# -------------------------
# 2) Pesos de clase (TRAIN)
# -------------------------
cnt = train_df.groupBy("label").count().cache()
pos = cnt.filter("label=1").select("count").first()[0]
neg = cnt.filter("label=0").select("count").first()[0]
pos_w = float(neg) / float(pos)

train_df = train_df.withColumn("weight", F.when(F.col("label")==1, F.lit(pos_w)).otherwise(F.lit(1.0)))
valid_df = valid_df.withColumn("weight", F.lit(1.0))

# -------------------------
# 3) Pipeline base (assembler + scaler) y mini-grid de regularización
# -------------------------
assembler = VectorAssembler(inputCols=FEATURE_COLS_BIN, outputCol="feat_vec", handleInvalid="skip")
scaler    = StandardScaler(inputCol="feat_vec", outputCol="feat_scaled", withMean=True, withStd=True)

candidates = [
    (0.01, 0.0), (0.03, 0.0), (0.05, 0.0), (0.10, 0.0),   # L2
    (0.03, 0.5), (0.03, 1.0)                              # ElasticNet / L1
]

best_aupr = None
best = None
evaluator_pr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

with mlflow.start_run(run_name=f"lr_whale_up_y{H}_gridsearch") as run_gs:
    for reg, l1ratio in candidates:
        lr_tmp = LogisticRegression(
            featuresCol="feat_scaled", labelCol="label", weightCol="weight",
            family="binomial", elasticNetParam=l1ratio, regParam=reg, maxIter=80, tol=1e-6
        )
        model_tmp = Pipeline(stages=[assembler, scaler, lr_tmp]).fit(train_df)
        pred_val_tmp = model_tmp.transform(valid_df).cache()
        aupr_tmp = evaluator_pr.evaluate(pred_val_tmp)
        mlflow.log_metric(f"val_aupr_reg{reg}_l1{l1ratio}", float(aupr_tmp))
        if (best_aupr is None) or (aupr_tmp > best_aupr):
            best_aupr = aupr_tmp
            best = (reg, l1ratio, model_tmp, pred_val_tmp)

# Mejor modelo por AUPR en VALID
reg_best, l1_best, model, pred_valid = best
mlflow.log_params({"regParam_best": reg_best, "elasticNetParam_best": l1_best})
mlflow.log_metric("val_aupr_best", float(best_aupr))

# -------------------------
# 4) Selección de umbral que MAXIMIZA RETORNO MEDIO con suelo de precisión/volumen
# -------------------------
pv = (pred_valid
      .select(
          F.col("label").cast("int").alias("y"),
          vector_to_array("probability").getItem(1).alias("p"),
          F.col(RET_COL).alias("ret")
      )
      .filter(F.col("ret").isNotNull())
      .cache())

P = pv.agg(F.sum("y").alias("P")).first()["P"]
if not P or P == 0:
    raise ValueError("No hay positivos en validación.")

w = W.orderBy(F.col("p").desc())
curve = (pv
    .withColumn("one", F.lit(1))
    .withColumn("seen", F.sum("one").over(w))
    .withColumn("tp",   F.sum("y").over(w))
    .withColumn("sum_ret", F.sum("ret").over(w))
    .withColumn("prec", F.col("tp")/F.col("seen"))
    .withColumn("avg_ret_pred", F.col("sum_ret")/F.col("seen"))
)

# 1º: máximo retorno medio sujeto a PRECISION_TARGET y MIN_SIGNALS
rows = (curve
    .filter((F.col("prec") >= F.lit(PRECISION_TARGET)) & (F.col("seen") >= F.lit(MIN_SIGNALS)))
    .orderBy(F.col("avg_ret_pred").desc())
    .limit(1).collect())

if rows:
    cand = rows[0]
else:
    # Fallback: mejor precisión con suelo de señales
    rows = (curve
        .filter(F.col("seen") >= F.lit(MIN_SIGNALS))
        .orderBy(F.col("prec").desc(), F.col("p").desc())
        .limit(1).collect())
    cand = rows[0] if rows else (curve.orderBy(F.col("prec").desc(), F.col("p").desc()).limit(1).collect()[0])

best_t, best_prec, best_n, best_avg_ret = float(cand["p"]), float(cand["prec"]), int(cand["seen"]), float(cand["avg_ret_pred"])

mlflow.log_metrics({
    "best_threshold": best_t,
    "val_precision_pos_at_best_t": best_prec,
    "val_signals_at_best_t": best_n,
    "val_avg_ret_pred_at_best_t": best_avg_ret
})

# -------------------------
# 5) Aplica el umbral y calcula métricas finales
# -------------------------
pred_tuned = (pred_valid
  .withColumn("p1", vector_to_array("probability").getItem(1))
  .withColumn("prediction", (F.col("p1") >= F.lit(best_t)).cast("double"))
  .cache())

tp = pred_tuned.filter("label=1 and prediction=1").count()
fp = pred_tuned.filter("label=0 and prediction=1").count()
tn = pred_tuned.filter("label=0 and prediction=0").count()
fn = pred_tuned.filter("label=1 and prediction=0").count()

precision_pos = tp/(tp+fp) if (tp+fp)>0 else 0.0
recall_pos    = tp/(tp+fn) if (tp+fn)>0 else 0.0
f1_pos        = (2*precision_pos*recall_pos)/(precision_pos+recall_pos) if (precision_pos+recall_pos)>0 else 0.0
accuracy      = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0.0

mlflow.log_metrics({
    "val_precision_pos": float(precision_pos),
    "val_recall_pos": float(recall_pos),
    "val_f1_pos": float(f1_pos),
    "val_accuracy": float(accuracy),
    "val_signals": int(tp+fp)
})

# Artefactos
pdf_cm = (pred_tuned.groupBy("label","prediction").count().orderBy("label","prediction").toPandas())
mlflow.log_dict(pdf_cm.to_dict(orient="list"), "artifacts/confusion_matrix_at_best_t.json")

coefs = list(zip(FEATURE_COLS_BIN, model.stages[-1].coefficients.toArray().tolist()))
mlflow.log_dict({"coefficients": coefs, "intercept": float(model.stages[-1].intercept)}, "artifacts/lr_coefficients.json")

# -------------------------
# 6) Embebe umbral en el modelo y registra en UC
# -------------------------
tuned_pipe_model = model
last = tuned_pipe_model.stages[-1]
if isinstance(last, LogisticRegressionModel):
    tuned_pipe_model.stages[-1] = last.setThreshold(best_t)
    mlflow.set_tag("inference_threshold_embedded", True)
else:
    mlflow.set_tag("inference_threshold_embedded", False)

sample_pd = train_df.select(*FEATURE_COLS_BIN).limit(5).toPandas()
signature = infer_signature(sample_pd, tuned_pipe_model.transform(train_df.limit(5)).select("prediction").toPandas())
input_example = sample_pd.head(1)

artifact_path = "final_model_lr_whale_thresholded"
try:
    info = mlflow.spark.log_model(
        tuned_pipe_model, artifact_path=artifact_path,
        registered_model_name=REGISTERED_MODEL_NAME,
        signature=signature, input_example=input_example
    )
    mlflow.set_tag("registered_model", REGISTERED_MODEL_NAME)
    mlflow.set_tag("registered_version", getattr(info, "registered_model_version", "unknown"))
except Exception as e:
    mlflow.set_tag("register_error", str(e))
    mlflow.spark.log_model(tuned_pipe_model, artifact_path=artifact_path,
                           signature=signature, input_example=input_example)

print({
  "best_threshold": best_t,
  "val_precision_pos": precision_pos,
  "val_recall_pos": recall_pos,
  "val_f1_pos": f1_pos,
  "val_accuracy": accuracy,
  "val_signals": int(tp+fp),
  "val_aupr_best": best_aupr,
  "val_avg_ret_pred_at_best_t": best_avg_ret
})


# COMMAND ----------

# Celda 4 — Métricas de retorno sobre las señales positivas (pred=1)
from pyspark.sql import functions as F
import mlflow

LABEL_COL = "label"       # 1 si sube (según tu label)
PRED_COL  = "prediction"  # 1 si el modelo predice SUBE
RET_COL   = "r_fwd_600s"  # retorno futuro
RET_THRESHOLD = 0.01      # 1% (corregido)

if RET_COL not in pred_tuned.columns:
    raise ValueError(f"No encuentro la columna '{RET_COL}' en pred_tuned.")

pred_pos = (pred_tuned
            .filter(F.col(PRED_COL) == 1)
            .filter(F.col(RET_COL).isNotNull())
            .select(LABEL_COL, RET_COL))

agg = (pred_pos
    .agg(
        F.count(F.lit(1)).alias("signals"),
        F.avg(F.col(LABEL_COL).cast("double")).alias("precision_pos"),
        F.avg(F.col(RET_COL)).alias("avg_ret_all_predictions"),
        F.expr(f"percentile_approx({RET_COL}, 0.5)").alias("ret_p50"),
        F.expr(f"percentile_approx({RET_COL}, array(0.1,0.25,0.75,0.9))").alias("ret_percentiles"),
        F.avg(F.when(F.col(LABEL_COL)==1, F.col(RET_COL))).alias("avg_ret_hits"),
        F.avg(F.when(F.col(LABEL_COL)==0, F.col(RET_COL))).alias("avg_ret_misses"),
        F.avg(F.when(F.col(RET_COL) > 0, 1.0).otherwise(0.0)).alias("share_ret_gt_0"),
        F.avg(F.when(F.col(RET_COL) >= F.lit(RET_THRESHOLD), 1.0).otherwise(0.0)).alias("share_ret_ge_threshold")
    )
    .collect()[0]
)

result = {
    "signals":                int(agg["signals"]) if agg["signals"] is not None else 0,
    "precision_pos":          float(agg["precision_pos"]) if agg["precision_pos"] is not None else None,
    "avg_ret_all_predictions":float(agg["avg_ret_all_predictions"]) if agg["avg_ret_all_predictions"] is not None else None,
    "ret_p50":                float(agg["ret_p50"]) if agg["ret_p50"] is not None else None,
    "ret_percentiles(p10,p25,p75,p90)": [float(x) for x in agg["ret_percentiles"]] if agg["ret_percentiles"] is not None else None,
    "avg_ret_hits":           float(agg["avg_ret_hits"]) if agg["avg_ret_hits"] is not None else None,
    "avg_ret_misses":         float(agg["avg_ret_misses"]) if agg["avg_ret_misses"] is not None else None,
    "share_ret_gt_0":         float(agg["share_ret_gt_0"]) if agg["share_ret_gt_0"] is not None else None,
    f"share_ret_ge_{int(RET_THRESHOLD*100)}pct": float(agg["share_ret_ge_threshold"]) if agg["share_ret_ge_threshold"] is not None else None
}
print(result)

# (Opcional) Log a MLflow
try:
    with mlflow.start_run(run_name="ret_metrics_on_predictions", nested=True):
        mlflow.log_param("ret_threshold", RET_THRESHOLD)
        mlflow.set_tag("ret_col", RET_COL)
        mlflow.log_metrics({
            "signals_pred_pos": result["signals"],
            "precision_pos_eval": result["precision_pos"] or 0.0,
            "avg_ret_all_predictions": result["avg_ret_all_predictions"] or 0.0,
            "ret_p50": result["ret_p50"] or 0.0,
            "avg_ret_hits": result["avg_ret_hits"] or 0.0,
            "avg_ret_misses": result["avg_ret_misses"] or 0.0,
            "share_ret_gt_0": result["share_ret_gt_0"] or 0.0,
            f"share_ret_ge_{int(RET_THRESHOLD*100)}pct": result[f"share_ret_ge_{int(RET_THRESHOLD*100)}pct"] or 0.0,
        })
except Exception as e:
    print("MLflow logging skipped:", str(e))

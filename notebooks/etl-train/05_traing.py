# Databricks notebook source
# Databricks notebook source
# Celda 0 — Parámetros + MLflow / Spark  (MODIFICADA: SOLO FEATURE_COLS)
from pyspark.sql import functions as F, Window as W
from pyspark.sql.functions import broadcast
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow, json

# ========= Tablas =========
DATASET_TABLE = "dbrtfmprod.whale_trades.gold__dataset_1s"   # features base + flags
LABEL_TABLE   = "dbrtfmprod.whale_trades.silver__labels_1s"  # creado en 04-labels.py

# ========= Horizonte: 6 horas =========
H                = 21600
TEST_DAYS        = 6
VAL_DAYS         = 6   
RET_COL          = f"r_fwd_{H}s"      # retorno futuro a 6h
BINARY_LABEL_COL = f"y{H}_up"         # 1 si ret >= RET_THR_UP definido en 04-labels
TRI_LABEL_COL    = f"y{H}"            # por si quieres usar triclase -> binarizar luego
USE_TRI_LABEL    = False              # por defecto usamos la binaria y{H}_up

# ========= Ventana y muestreo =========
MAX_TRAIN_DAYS    = 180
DOWN_SAMPLE_NEG   = True
NEG_POS_RATIO_MAX = 2.0

# ========= Definición "whale" por cuantiles =========
Q_WHALE_QTY   = 0.995
Q_WHALE_BURST = 0.99

# ========= Features base (SOLO MODIFICADO: señales fáciles del WS) =========
FEATURE_COLS = [
    "buy_qty_5s",    # suma cantidades agresoras compra (últimos 5s)
    "sell_qty_5s",   # suma cantidades agresoras venta (últimos 5s)
    "z_r_1s",        # retorno corto (1s) ya calculado en tu dataset
    "z_r_5s"         # retorno a 5s ya calculado en tu dataset
]

# ========= Selección de umbral (negocio) =========
PRECISION_TARGET  = 0.70   # precisión mínima exigida
MIN_SIGNALS       = 200    # volumen mínimo de señales en valid
THRESHOLD_POLICY  = "fallback"   # "fallback" | "hard_stop" | "safe_no_signals"

# ========= MLOps =========
EXPERIMENT_PATH = f"/Shared/exp_crypto_whale_lr_fast_y{H}"
REGISTERED_MODEL_NAME = f"dbrtfmprod.whale_trades.crypto_whale_lr_fast_y{H}_up"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.autolog(disable=True); mlflow.spark.autolog(disable=True)

# ========= Spark =========
spark.conf.set("spark.sql.shuffle.partitions", "64")


# COMMAND ----------

# COMMAND ----------
# Celda 1 — Carga, join labels 6h, filtro días, whale, interacciones (MODIFICADA: SOLO paso 7), label, pesos

from pyspark.sql import functions as F, Window as W

# 1) Carga base y labels
ds  = spark.table(DATASET_TABLE)
lbl = spark.table(LABEL_TABLE)

# 1.a) Selecciona solo columnas de labels que necesitamos (si existen)
lbl_cols = ["symbol","ts"]
if RET_COL in lbl.columns:          lbl_cols.append(RET_COL)
if BINARY_LABEL_COL in lbl.columns: lbl_cols.append(BINARY_LABEL_COL)
if TRI_LABEL_COL in lbl.columns:    lbl_cols.append(TRI_LABEL_COL)
lbl = lbl.select(*lbl_cols)

# 1.b) Elimina de ds posibles columnas duplicadas (evita ambigüedad en el join)
dup_cols = [c for c in [RET_COL, BINARY_LABEL_COL, TRI_LABEL_COL] if c in ds.columns]
if dup_cols:
    ds = ds.drop(*dup_cols)

# 2) Join limpio (ya sin columnas duplicadas)
df = ds.join(lbl, ["symbol","ts"], "left")

# 3) Comprobaciones mínimas (NO TOCAR)
need_base = {"symbol","date","ts","had_trade_5s","buy_qty_5s","sell_qty_5s","vol_burst_5_over_60","ofi_5s", *FEATURE_COLS}
missing = [c for c in need_base if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas base tras el join: {missing}")
if RET_COL not in df.columns:
    raise ValueError(f"No encuentro {RET_COL}. Revisa que 04-labels haya corrido y LABEL_TABLE sea correcto.")
if (not USE_TRI_LABEL) and (BINARY_LABEL_COL not in df.columns):
    raise ValueError(f"No encuentro {BINARY_LABEL_COL}. Revisa LABEL_TABLE.")

# 4) Últimos N días (reduce tamaño)
max_date = df.agg(F.max("date").alias("d")).collect()[0]["d"]
if MAX_TRAIN_DAYS is not None:
    df = df.filter(F.col("date") >= F.date_sub(F.lit(max_date), MAX_TRAIN_DAYS))

# 5) Warm-up 60s por símbolo + actividad reciente
wmin = W.partitionBy("symbol")
df = (df
      .withColumn("min_ts_unix", F.min(F.col("ts").cast("long")).over(wmin))
      .filter(F.col("ts").cast("long") >= F.col("min_ts_unix") + F.lit(60))
      .drop("min_ts_unix"))
df = df.filter(F.col("had_trade_5s") == 1)

# 6) Whale por cuantiles (NO TOCAR)
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
df = df.filter(F.col("whale_event") == 1)

# 7) Derivadas e interacciones — ***ÚNICO CAMBIO DE LÓGICA***
df = (df
      .withColumn("qty_imb_5s",  (F.col("buy_qty_5s") - F.col("sell_qty_5s")) /
                                 (F.col("buy_qty_5s") + F.col("sell_qty_5s") + F.lit(1e-9)))
      .withColumn("log_max_qty_5s", F.log1p(F.col("max_qty_5s")))
      .withColumn("whale_side", (F.col("buy_qty_5s") > F.col("sell_qty_5s")).cast("int"))
      .withColumn("x1", F.col("whale_side")      * F.col("z_r_1s"))
      .withColumn("x2", F.col("log_max_qty_5s")  * F.col("z_r_1s"))
      .withColumn("x3", F.col("qty_imb_5s")      * F.col("z_r_5s"))
      .withColumn("x4", F.col("max_qty_5s")      * F.abs(F.col("z_r_1s")))
      .withColumn("x5", F.col("qty_imb_5s")      * F.col("log_max_qty_5s"))
)

FEATURE_COLS_BIN = FEATURE_COLS + [
    "qty_imb_5s","log_max_qty_5s","whale_side",
    "x1","x2","x3","x4","x5"
]

# 8) Label (NO TOCAR)
if USE_TRI_LABEL:
    df = df.withColumn("label", (F.col(TRI_LABEL_COL) == 1).cast("int"))
else:
    df = df.withColumn("label", F.col(BINARY_LABEL_COL).cast("int"))

# 9) Limpieza (NO TOCAR)
for c in FEATURE_COLS_BIN:
    df = df.filter(F.col(c).isNotNull() & ~F.isnan(c))
df = df.filter(F.col(RET_COL).isNotNull() & F.col("label").isNotNull())

# 10) Pesos (NO TOCAR)
cnt = df.groupBy("label").count().withColumnRenamed("count","n")
tot = cnt.agg(F.sum("n").alias("tot")).collect()[0]["tot"]
weights = (cnt.withColumn("freq", F.col("n")/F.lit(tot))
              .withColumn("weight", 1.0/F.col("freq"))
              .select("label","weight"))
dx = df.join(weights, "label", "left").cache()

display(cnt.orderBy("label"))
print("rows_after_prep:", dx.count(), "max_date:", max_date)


# COMMAND ----------

# Celda 2 — Split temporal con purge(H) + (opcional) downsampling negativo en TRAIN (SIN CAMBIOS)
from datetime import timedelta


val_date = dx.agg(F.max("date").alias("d")).collect()[0]["d"]

test_start = val_date - timedelta(days=TEST_DAYS - 1)
val_end    = test_start - timedelta(days=1)
val_start  = val_end - timedelta(days=VAL_DAYS - 1)


train_df = dx.filter(
    F.col("ts") < F.expr(f"timestamp('{val_start}') - INTERVAL {H} SECONDS")
)

valid_df = dx.filter(
    (F.col("date") >= F.lit(val_start)) & (F.col("date") <= F.lit(val_end))
)

test_df = dx.filter(
    (F.col("date") >= F.lit(test_start)) & (F.col("date") <= F.lit(val_date))
)

if DOWN_SAMPLE_NEG:
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
    "test_rows": test_df.count()
})


# COMMAND ----------

# COMMAND ----------
# Celda 3 — Entrenamiento LR + mini-grid + selección de umbral por retorno + logging MLflow (SIN CAMBIOS)

from pyspark.sql import functions as F, Window as W
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from mlflow.models import infer_signature
import mlflow, json

# ----- Pesos de clase -----
cnt = train_df.groupBy("label").count().cache()
pos = cnt.filter("label=1").select("count").first()[0]
neg = cnt.filter("label=0").select("count").first()[0]
pos_w = float(neg) / float(pos)

train_df = train_df.withColumn("weight", F.when(F.col("label")==1, F.lit(pos_w)).otherwise(F.lit(1.0)))
valid_df = valid_df.withColumn("weight", F.lit(1.0))

# ----- Pipeline base y mini-grid de regularización -----
assembler = VectorAssembler(inputCols=FEATURE_COLS_BIN, outputCol="feat_vec", handleInvalid="skip")
scaler    = StandardScaler(inputCol="feat_vec", outputCol="feat_scaled", withMean=False, withStd=True)

candidates = [
    (0.005, 0.0), (0.01, 0.0), (0.02, 0.0), (0.03, 0.0), (0.05, 0.0), (0.10, 0.0),
    (0.02, 0.5), (0.03, 0.5),
    (0.03, 1.0)
]
evaluator_pr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

# ----- MLflow: asegura un único run activo -----
try:
    if mlflow.active_run() is not None:
        mlflow.end_run()
except Exception:
    pass

with mlflow.start_run(run_name=f"lr_whale_up_y{H}_train") as run:
    mlflow.log_params({
        "target_horizon_sec": H,
        "precision_target": PRECISION_TARGET,
        "min_signals": MIN_SIGNALS,
        "threshold_policy": THRESHOLD_POLICY
    })

    # Grid search (entrena y evalúa en el MISMO run)
    best_aupr, best = None, None
    for reg, l1ratio in candidates:
        lr_tmp = LogisticRegression(
            featuresCol="feat_scaled", labelCol="label", weightCol="weight",
            family="binomial", elasticNetParam=l1ratio, regParam=reg,
            maxIter=80, tol=1e-6
        )
        model_tmp = Pipeline(stages=[assembler, scaler, lr_tmp]).fit(train_df)
        pred_val_tmp = model_tmp.transform(valid_df).cache()
        aupr_tmp = evaluator_pr.evaluate(pred_val_tmp)
        mlflow.log_metric(f"val_aupr_reg{reg}_l1{l1ratio}", float(aupr_tmp))
        if (best_aupr is None) or (aupr_tmp > best_aupr):
            best_aupr = aupr_tmp
            best = (reg, l1ratio, model_tmp, pred_val_tmp)

    reg_best, l1_best, model, pred_valid = best
    mlflow.log_params({"regParam_best": reg_best, "elasticNetParam_best": l1_best})
    mlflow.log_metric("val_aupr_best", float(best_aupr))

    # ----- Selección de umbral por retorno medio sujeto a PRECISION_TARGET y MIN_SIGNALS -----
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

    def pick_cand_fallback(curve, PRECISION_TARGET, MIN_SIGNALS):
        rowsA = (curve
            .filter((F.col("prec") >= F.lit(PRECISION_TARGET)) & (F.col("seen") >= F.lit(MIN_SIGNALS)))
            .orderBy(F.col("avg_ret_pred").desc(), F.col("seen").desc(), F.col("p").desc())
            .limit(1).collect())
        if rowsA: return rowsA[0]
        rowsB = (curve
            .filter(F.col("seen") >= F.lit(MIN_SIGNALS))
            .orderBy(F.col("prec").desc(), F.col("p").desc())
            .limit(1).collect())
        if rowsB: return rowsB[0]
        return (curve.orderBy(F.col("prec").desc(), F.col("p").desc()).limit(1).collect()[0])

    cand = None
    if THRESHOLD_POLICY == "hard_stop":
        rows = (curve
            .filter((F.col("prec") >= F.lit(PRECISION_TARGET)) & (F.col("seen") >= F.lit(MIN_SIGNALS)))
            .orderBy(F.col("avg_ret_pred").desc(), F.col("seen").desc(), F.col("p").desc())
            .limit(1).collect())
        if not rows:
            mlflow.set_tag("no_threshold_meets_target", True)
            raise ValueError(f"No hay umbral que cumpla PRECISION_TARGET={PRECISION_TARGET} y MIN_SIGNALS={MIN_SIGNALS}")
        cand = rows[0]
    elif THRESHOLD_POLICY == "safe_no_signals":
        rows = (curve
            .filter((F.col("prec") >= F.lit(PRECISION_TARGET)) & (F.col("seen") >= F.lit(MIN_SIGNALS)))
            .orderBy(F.col("avg_ret_pred").desc(), F.col("seen").desc(), F.col("p").desc())
            .limit(1).collect())
        if rows:
            cand = rows[0]
        else:
            mlflow.set_tag("no_threshold_meets_target", True)
            best_t, best_prec, best_n, best_avg_ret = 1.0, 0.0, 0, 0.0
    else:
        cand = pick_cand_fallback(curve, PRECISION_TARGET, MIN_SIGNALS)

    if cand is not None:
        best_t       = float(cand["p"])
        best_prec    = float(cand["prec"])
        best_n       = int(cand["seen"])
        best_avg_ret = float(cand["avg_ret_pred"])

    mlflow.log_metrics({
        "best_threshold": best_t,
        "val_precision_pos_at_best_t": best_prec,
        "val_signals_at_best_t": best_n,
        "val_avg_ret_pred_at_best_t": best_avg_ret
    })

    # ----- Aplica umbral y métricas finales -----
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

    # ----- Artefactos + registro de modelo -----
    pdf_cm = (pred_tuned.groupBy("label","prediction").count().orderBy("label","prediction").toPandas())
    mlflow.log_dict(pdf_cm.to_dict(orient="list"), "artifacts/confusion_matrix_at_best_t.json")

    coefs = list(zip(FEATURE_COLS_BIN, model.stages[-1].coefficients.toArray().tolist()))
    mlflow.log_dict({"coefficients": coefs, "intercept": float(model.stages[-1].intercept)}, "artifacts/lr_coefficients.json")

    tuned_pipe_model = model
    last = tuned_pipe_model.stages[-1]
    if isinstance(last, LogisticRegressionModel):
        tuned_pipe_model.stages[-1] = last.setThreshold(best_t)
        mlflow.set_tag("inference_threshold_embedded", True)
    else:
        mlflow.set_tag("inference_threshold_embedded", False)

    sample_pd = train_df.select(*FEATURE_COLS_BIN).limit(5).toPandas()
    from mlflow.models import infer_signature
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
# <- al salir del with, el run se cierra correctamente


# COMMAND ----------

# COMMAND ----------
# Celda 4 — Métricas de retorno sobre las señales positivas (pred=1) (SIN CAMBIOS)

from pyspark.sql import functions as F
import mlflow

LABEL_COL = "label"
PRED_COL  = "prediction"
RET_THRESHOLD = 0.01   # % de TP orientativo (1%)

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

# Log a MLflow (opcional)
try:
    with mlflow.start_run(run_name="ret_metrics_on_predictions_6h", nested=True):
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


# COMMAND ----------

# Celda 5 — Reempacar el Pipeline Spark a pyfunc (con limpieza para reiniciar a v1)
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from pyspark.ml import PipelineModel
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

TARGET_REGISTERED_MODEL = REGISTERED_MODEL_NAME + "_pyfunc"
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# === Limpieza previa: elimina TODAS las versiones y el modelo registrado (para volver a v1) ===
_client = MlflowClient()
try:
    for v in _client.search_model_versions(f"name = '{TARGET_REGISTERED_MODEL}'"):
        try:
            _client.delete_model_version(name=TARGET_REGISTERED_MODEL, version=v.version)
        except Exception:
            pass
    _client.delete_registered_model(name=TARGET_REGISTERED_MODEL)
except Exception:
    pass

# Pipeline entrenado con umbral embebido de la Celda 3
pm: PipelineModel = tuned_pipe_model

# Extrae VectorAssembler, StandardScalerModel y LogisticRegressionModel
assembler_cols = None
with_mean = False
with_std  = True
mean_vec = None
std_vec  = None
lr_coef  = None
lr_intercept = None
lr_threshold = None

for st in pm.stages:
    cname = st.__class__.__name__
    if "VectorAssembler" in cname:
        assembler_cols = list(st.getInputCols())
    elif "StandardScalerModel" in cname:
        with_mean = bool(st.getWithMean())
        with_std  = bool(st.getWithStd())
        # OJO: son ATRIBUTOS, no métodos
        mean_vec = st.mean.toArray() if with_mean else None
        std_vec  = st.std.toArray()  if with_std  else None
    elif "LogisticRegressionModel" in cname:
        lr_coef      = st.coefficients.toArray()
        lr_intercept = float(st.intercept)
        # El umbral quedó seteado en Celda 3 con setThreshold(best_t)
        lr_threshold = st.getThreshold()

# Seguridad mínima
if assembler_cols is None:
    assembler_cols = FEATURE_COLS_BIN
assert lr_coef is not None, "No encontré LogisticRegressionModel en el pipeline"
if with_std:
    assert std_vec is not None and len(std_vec) == len(assembler_cols), "std_vec ausente o tamaño incorrecto"
if with_mean:
    assert mean_vec is not None and len(mean_vec) == len(assembler_cols), "mean_vec ausente o tamaño incorrecto"

print({
    "features": assembler_cols,
    "with_mean": with_mean,
    "with_std": with_std,
    "coef_shape": np.shape(lr_coef),
    "intercept": lr_intercept,
    "threshold": lr_threshold
})

# PythonModel equivalente (StandardScaler + LR)
class LRWithStandardization(mlflow.pyfunc.PythonModel):
    def __init__(self, feature_names, with_mean, with_std, mean_vec, std_vec, coef, intercept, threshold=None):
        self.feature_names = list(feature_names)
        self.with_mean = bool(with_mean)
        self.with_std  = bool(with_std)
        self.mean_vec  = None if mean_vec is None else np.asarray(mean_vec, dtype="float64")
        self.std_vec   = None if std_vec  is None else np.asarray(std_vec,  dtype="float64")
        self.coef      = np.asarray(coef, dtype="float64")
        self.intercept = float(intercept)
        self.threshold = None if threshold is None else float(threshold)

    def _standardize(self, X):
        Z = X.astype("float64", copy=False)
        if self.with_mean and self.mean_vec is not None:
            Z = Z - self.mean_vec
        if self.with_std and self.std_vec is not None:
            denom = np.where(self.std_vec == 0.0, 1.0, self.std_vec)
            Z = Z / denom
        return Z

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            X = model_input[self.feature_names].to_numpy(dtype="float64", copy=False)
        else:
            X = pd.DataFrame(model_input, columns=self.feature_names, copy=False)[self.feature_names] \
                    .to_numpy(dtype="float64", copy=False)
        Z = self._standardize(X)
        logits = Z.dot(self.coef) + self.intercept
        logits = np.clip(logits, -50, 50)
        p1 = 1.0 / (1.0 + np.exp(-logits))
        pred = (p1 >= (0.5 if self.threshold is None else self.threshold)).astype("float64")
        p0 = 1.0 - p1
        prob = np.stack([p0, p1], axis=1).tolist()
        return pd.DataFrame({"prediction": pred, "probability": prob})

py_model = LRWithStandardization(
    feature_names=assembler_cols,
    with_mean=with_mean,
    with_std=with_std,
    mean_vec=mean_vec,
    std_vec=std_vec,
    coef=lr_coef,
    intercept=lr_intercept,
    threshold=lr_threshold
)

# Firma y requisitos
input_example = train_df.select(*assembler_cols).limit(1).toPandas()
signature = infer_signature(input_example, pd.DataFrame({"prediction":[0.0], "probability":[[0.5,0.5]]}))
pip_reqs = ["mlflow==3.1.1", "pandas>=2.1,<2.3", "numpy>=1.24,<2.0"]

# Log & Register en UC (tras la limpieza, este alta será v1)
with mlflow.start_run(run_name=f"repack_to_pyfunc_y{H}", nested=True):
    mlflow.set_tag("source_registered_model", REGISTERED_MODEL_NAME)
    mlflow.set_tag("repacked_flavor", "pyfunc_no_spark")
    mlflow.set_tag("features_order", ",".join(assembler_cols))
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=py_model,
        registered_model_name=TARGET_REGISTERED_MODEL,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_reqs
    )

print("OK → Registrado como:", TARGET_REGISTERED_MODEL, "(v1)")


# COMMAND ----------

import os
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.core.credentials import AccessToken

# --- Configuración ---
SUBSCRIPTION_ID = "b9c75d1a-b9c6-4b39-9a81-2d73d5a3cd22"   # tu sub ID
RESOURCE_GROUP  = "tfmprod-jpn"                            # RG donde está el ACI
ACI_NAME        = "predictor-ws"                           # nombre del container group

# --- Clase para usar un token directamente ---
class StaticTokenCredential:
    def __init__(self, token):
        self._token = token
    def get_token(self, *scopes):
        return AccessToken(self._token, float("inf"))

# --- Coge el token desde la variable de entorno del clúster ---
bearer_token = os.environ.get("AZURE_BEARER_TOKEN")
if not bearer_token:
    raise ValueError("No se encontró la variable AZURE_BEARER_TOKEN en el entorno del clúster")

# --- Inicializa cliente ACI ---
cred = StaticTokenCredential(bearer_token)
aci_client = ContainerInstanceManagementClient(cred, SUBSCRIPTION_ID)

# --- Reinicia el contenedor (equivale a arrancarlo) ---
aci_client.container_groups.begin_start(RESOURCE_GROUP, ACI_NAME)

'''.result()

# --- Consulta estado e IP ---
cg = aci_client.container_groups.get(RESOURCE_GROUP, ACI_NAME)
state = getattr(getattr(cg, "instance_view", None), "state", None)
ip    = getattr(getattr(cg, "ip_address", None), "ip", None)
fqdn  = getattr(getattr(cg, "ip_address", None), "fqdn", None)

print(f"✅ Contenedor reiniciado. Estado: {state}, IP: {ip}, FQDN: {fqdn}")

# --- (Opcional) Ver logs del primer contenedor ---
cname = cg.containers[0].name
logs = aci_client.containers.list_logs(RESOURCE_GROUP, ACI_NAME, cname, tail=50)
print("\n--- Últimos logs ---\n", logs.content or "<sin logs>")
'''
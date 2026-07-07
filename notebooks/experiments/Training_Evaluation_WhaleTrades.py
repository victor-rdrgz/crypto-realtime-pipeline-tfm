# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Databricks notebook source
# MAGIC Notebook 1 — Training & Evaluation (Whale Trades → ±1.5% @ +5m)
# MAGIC
# MAGIC **Stack**
# MAGIC
# MAGIC  PySpark · MLflow (Unity Catalog Model Registry) · Delta/Unity Catalog · LightGBM/XGBoost/LogReg.
# MAGIC
# MAGIC **Objetivo**
# MAGIC
# MAGIC  Por símbolo y lado (BUY/SELL), entrenar y evaluar modelos que predicen si el *mark price* se moverá ≥ +1.5% o ≤ −1.5% a +5m tras un *aggTrade* (P95). Seleccionar y registrar `Champion` por símbolo/lado. Guardar umbrales en `gold__model_thresholds`.
# MAGIC
# MAGIC **Buenas prácticas**
# MAGIC Datos, modelos y métricas gobernados en UC (Delta + Model Registry).
# MAGIC Entrenamiento reproducible (autolog, signature, input_example).
# MAGIC Selección de umbral con criterio de negocio (precision ≥ 0.80 y cobertura mínima).
# MAGIC Separación clara de Training vs Serving (este notebook es solo batch).
# MAGIC
# MAGIC **Libraries installed in cluster**
# MAGIC
# MAGIC => lightgbm==4.4.0
# MAGIC
# MAGIC => xgboost==2.1.1
# MAGIC
# MAGIC => mlflow-skinny==3.1.1

# COMMAND ----------

# Imports (post-restart)
import os
import json
import typing as t
from dataclasses import dataclass
from datetime import datetime

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from xgboost import XGBClassifier

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import DataFrame as SDF
from pyspark.sql.types import (
    StructType,StructField,StringType,
    DoubleType,LongType,TimestampType
)


# COMMAND ----------

symbols = [ r["symbol"] for r in spark.sql(
            """SELECT DISTINCT symbol
               FROM dbrtfmprod.whale_trades.silver__p95_trades
               ORDER BY symbol
               """).collect()
]

symbols

# COMMAND ----------

# CONFIG
spark.conf.set("spark.sql.shuffle.partitions", "12")

@dataclass
class Paths:
    catalog: str = "dbrtfmprod"
    schema: str = "whale_trades"

    # Tablas Silver necesarias
    table_p95_trades: str = "silver__p95_trades"        # eventos whale P95
    table_mark_1m: str = "silver__mark_klines_1m"       # velas 1m mark
    table_index_1m: str = "silver__index_klines_1m"     # velas 1m index
    table_kline_1m: str = "silver__klines_1m"           # velas 1m spot

    # Tabla Gold para thresholds
    table_model_thresh: str = "gold__model_thresholds"  # salida thresholds

@dataclass
class ModelingCfg:
    symbols: t.List[str]
    sides: t.List[str] = ("BUY", "SELL")
    positive_move_buy: float = 0.015
    positive_move_sell: float = -0.015

    # Ventanas temporales relativas al último timestamp disponible
    train_days: int = 30
    gap_days: int = 1
    val_days: int = 7
    test_days: int = 7

    min_weekly_signals: int = 30
    min_precision: float = 0.80
    threshold_grid: t.List[float] = tuple(np.round(np.linspace(0.99, 0.5, 50), 3))

@dataclass
class RegistryCfg:
    model_base_name: str = "whale_price_impact"
    use_uc: bool = True

PATHS = Paths()
CFG = ModelingCfg(symbols=symbols)
REG = RegistryCfg()

# MLflow ←→ UC
if REG.use_uc:
    mlflow.set_registry_uri("databricks-uc")

username = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{username}/whale_trades_training")

# Nombres UC
UC = PATHS.catalog
SC_WH = PATHS.schema
SC_GOLD = PATHS.schema

TBL_P95 = f"{UC}.{SC_WH}.{PATHS.table_p95_trades}"
TBL_MARK = f"{UC}.{SC_WH}.{PATHS.table_mark_1m}"
TBL_INDEX = f"{UC}.{SC_WH}.{PATHS.table_index_1m}"
TBL_KLINE = f"{UC}.{SC_WH}.{PATHS.table_kline_1m}"
TBL_THRESH = f"{UC}.{SC_GOLD}.{PATHS.table_model_thresh}"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1) Utilidades

# COMMAND ----------
FEATURE_COLS = [
    "trade_notional","trade_is_buy","notional_z",
    "buys_share_5s","buys_share_15s","buys_share_30s",
    "notional_sum_30s","runlen_same_side_30s","secs_since_last_whale",
    "kline_prev_close","ret_1m_prev","range_1m_prev_rel","vol_1m_prev_z",
    "mark_prev_close","ret_mark_1m_prev","index_prev_close",
    "premium","d_premium_1m","hour_of_day","day_of_week"
]

# COMMAND ----------

def _read_table_or_empty(fullname: str) -> SDF:
    try:
        return spark.table(fullname)
    except Exception:
        print(f"[ERROR] No se pudo leer {fullname}.")
        raise RuntimeError(f"No se pudo leer la tabla {fullname}: {e}")


def _ensure_thresholds_table():
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {TBL_THRESH} (
            model_key STRING,
            symbol STRING,
            side STRING,
            winner_type STRING,
            threshold DOUBLE,
            ap_val DOUBLE,
            precision_at_threshold_val DOUBLE,
            coverage_signals_val BIGINT,
            ap_test DOUBLE,
            precision_at_threshold_test DOUBLE,
            coverage_signals_test BIGINT,
            mean_return_test DOUBLE,
            trained_until TIMESTAMP,
            created_at TIMESTAMP
        ) USING DELTA
    """)


def _clean_pdf(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.replace([np.inf, -np.inf], np.nan)
    return pdf.dropna()


def evaluate_threshold_grid(y_true: np.ndarray, y_proba: np.ndarray, returns: np.ndarray,
                            thresholds: t.List[float], min_precision: float) -> dict:
    best = {"threshold": None, "precision": 0.0, "coverage": 0, "mean_return": None}
    for thr in thresholds:
        y_hat = (y_proba >= thr).astype(int)
        if y_hat.sum() == 0:
            continue
        prec = precision_score(y_true, y_hat, zero_division=0)
        if prec >= min_precision:
            cov = int(y_hat.sum())
            mean_ret = float(returns[y_hat.astype(bool)].mean()) if cov > 0 else float("nan")
            if cov > best["coverage"]:
                best = {"threshold": float(thr), "precision": float(prec), "coverage": cov, "mean_return": mean_ret}
    return best


# COMMAND ----------

# Carga de datos
p95 = _read_table_or_empty(TBL_P95)
mark1m = _read_table_or_empty(TBL_MARK)
index1m = _read_table_or_empty(TBL_INDEX)
kline1m = _read_table_or_empty(TBL_KLINE)

# COMMAND ----------

if not p95.columns:
    raise RuntimeError("La tabla silver__p95_trades es obligatoria y está vacía/inexistente.")

# Normalización de timestamps
if "ts_s" not in p95.columns:
    if "timestamp" in p95.columns:
        if isinstance(p95.schema["timestamp"].dataType, TimestampType):
            p95 = p95.withColumn("ts_s", F.col("timestamp").cast("timestamp"))
        else:
            p95 = p95.withColumn("ts_s", (F.col("timestamp")/F.lit(1000.0)).cast("timestamp"))
    else:
        raise RuntimeError("P95 debe tener 'timestamp' (ms o timestamp) o 'ts_s'.")


for var_name, df in (("mark1m", mark1m), ("index1m", index1m), ("kline1m", kline1m)):
    if df.columns and "ts_s" not in df.columns:
        colname = None
        if "timestamp" in df.columns:
            colname = "timestamp"
        elif "open_time" in df.columns:
            colname = "open_time"
        else:
            continue  # no hay columna temporal conocida

        if isinstance(df.schema[colname].dataType, TimestampType):
            locals()[var_name] = df.withColumn("ts_s", F.col(colname).cast("timestamp"))
        else:
            locals()[var_name] = df.withColumn("ts_s", (F.col(colname)/F.lit(1000.0)).cast("timestamp"))



# COMMAND ----------

# 3) Feature Engineering (batch)

# Campos base
p95 = (p95
       .withColumn("price", F.col("price").cast("double"))
       .withColumn("qty", F.col("qty").cast("double"))
       .withColumn("trade_notional", F.col("price") * F.col("qty"))
       .withColumn("trade_is_buy", (~F.col("is_buyer_maker")).cast("int"))
      )

# Z-score del notional (rolling)
win_prev = (Window
            .partitionBy("symbol")
            .orderBy(F.col("ts_s").cast("long"))
            .rowsBetween(-1440, -1))

p95 = (p95
       .withColumn("notional_mean", F.mean("trade_notional").over(win_prev))
       .withColumn("notional_std", F.stddev("trade_notional").over(win_prev))
       .withColumn("notional_z", (F.col("trade_notional") - F.col("notional_mean")) / F.col("notional_std"))
      )

# Ventanas 5s/15s/30s y suma 30s
for secs in (5, 15, 30):
    w = (Window.partitionBy("symbol").orderBy(F.col("ts_s").cast("long")).rangeBetween(-secs, -1))
    p95 = (p95
           .withColumn(f"notional_buy_{secs}s", F.sum(F.when(F.col("trade_is_buy") == 1, F.col("trade_notional")).otherwise(0.0)).over(w))
           .withColumn(f"notional_total_{secs}s", F.sum("trade_notional").over(w))
           .withColumn(f"buys_share_{secs}s", F.col(f"notional_buy_{secs}s")/F.col(f"notional_total_{secs}s"))
          )

w30 = (Window.partitionBy("symbol").orderBy(F.col("ts_s").cast("long")).rangeBetween(-30, -1))
p95 = p95.withColumn("notional_sum_30s", F.sum("trade_notional").over(w30))

# run-length aproximado del mismo lado (30s)
side_cur = F.when(F.col("trade_is_buy") == 1, F.lit(1)).otherwise(F.lit(-1))
p95 = p95.withColumn("_side", side_cur)
win30 = (Window.partitionBy("symbol").orderBy(F.col("ts_s").cast("long")).rangeBetween(-30, -1))
p95 = (p95
       .withColumn("runlen_same_side_30s", F.sum(F.when(F.col("_side") == F.last("_side").over(win30), 1).otherwise(0)).over(win30))
       .drop("_side")
      )

# secs since last whale
win_unbounded = (Window.partitionBy("symbol").orderBy(F.col("ts_s").cast("long")).rowsBetween(Window.unboundedPreceding, -1))
p95 = p95.withColumn("secs_since_last_whale", (F.col("ts_s").cast("long") - F.max(F.col("ts_s").cast("long")).over(win_unbounded)))

# Kline 1m (prev)
if kline1m.columns:
    k = (kline1m
         .select("symbol", F.date_trunc("minute", F.col("ts_s")).alias("ts_min"),
                 F.col("close").cast("double").alias("k_close"),
                 F.col("high").cast("double").alias("k_high"),
                 F.col("low").cast("double").alias("k_low"),
                 F.col("volume").cast("double").alias("k_vol"))
        )
    # join contra el minuto del evento
    p95 = p95.withColumn("ts_min", F.date_trunc("minute", F.col("ts_s")))
    p95 = (p95
           .join(k, ["symbol","ts_min"], "left")
           .withColumn("kline_prev_close", F.col("k_close"))
           .withColumn("range_1m_prev_rel", (F.col("k_high")-F.col("k_low"))/F.col("k_close"))
          )
    # ret_1m_prev + vol z-score por símbolo
    w_k = (Window.partitionBy("symbol").orderBy(F.col("ts_min").cast("long")))
    p95 = (p95
           .withColumn("k_close_prev", F.lag("k_close").over(w_k))
           .withColumn("ret_1m_prev", (F.col("k_close")/F.col("k_close_prev") - 1.0))
          )
    vol_sym = (k
               .groupBy("symbol").agg(F.avg("k_vol").alias("vol_mean"), F.stddev("k_vol").alias("vol_std")))
    p95 = (p95
           .join(vol_sym, ["symbol"], "left")
           .withColumn("vol_1m_prev_z", (F.col("k_vol")-F.col("vol_mean"))/F.col("vol_std"))
          )

# Mark/Index + premium
if mark1m.columns and index1m.columns:
    m = mark1m.select("symbol", F.date_trunc("minute", F.col("ts_s")).alias("ts_min"), F.col("close").cast("double").alias("mark_close"))
    i = index1m.select("symbol", F.date_trunc("minute", F.col("ts_s").cast("timestamp")).alias("ts_min"), F.col("close").cast("double").alias("index_close"))
    mi = (m.join(i, ["symbol","ts_min"], "inner")
            .withColumn("premium", (F.col("mark_close")-F.col("index_close"))/F.col("index_close")))
    wmi = (Window.partitionBy("symbol").orderBy(F.col("ts_min").cast("long")))
    mi = (mi
          .withColumn("premium_prev", F.lag("premium").over(wmi))
          .withColumn("d_premium_1m", F.col("premium")-F.col("premium_prev")))

    p95 = (p95
           .join(mi, ["symbol","ts_min"], "left")
           .withColumn("mark_prev_close", F.col("mark_close"))
           .withColumn("index_prev_close", F.col("index_close"))
           .withColumn("ret_mark_1m_prev", F.col("premium"))  # proxy si no hay lag explícito
          )

# Hora/día
p95 = (p95
       .withColumn("hour_of_day", F.hour("ts_s"))
       .withColumn("day_of_week", F.dayofweek("ts_s"))
      )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4) Etiquetado y splits temporales

# COMMAND ----------
# Minuto base y +5m

def _pick_price_source(mark1m, index1m, kline1m):
    # preferimos mark1m; si no, index1m; si no, kline1m
    for df, name in ((mark1m, "mark"), (index1m, "index"), (kline1m, "kline")):
        if df.columns and df.limit(2).count() >= 2:  # al menos 2 velas para poder tener t+5 alguna vez
            return df, name
    raise RuntimeError("No hay velas 1m suficientes en mark/index/kline para calcular ret_5m.")

src_df, src_name = _pick_price_source(mark1m, index1m, kline1m)

mm_base = (
    src_df
    .select(
        F.col("symbol").alias("px_symbol"),
        F.date_trunc("minute", F.col("ts_s")).alias("px_min"),
        F.col("close").cast("double").alias("px_close")
    )
)

# mark_t0 y mark_t5 con ventana (evitamos dobles joins y columnas duplicadas)
wpx = Window.partitionBy("px_symbol").orderBy("px_min")
mm2 = (
    mm_base
    .withColumn("mark_t0", F.col("px_close"))
    .withColumn("mark_t5", F.lead("px_close", 5).over(wpx))
    .select(
        F.col("px_symbol").alias("symbol"),
        F.col("px_min").alias("ts_min"),
        "mark_t0", "mark_t5"
    )
)

p95 = (
    p95
    .withColumn("ts_min", F.date_trunc("minute", F.col("ts_s")))
    .join(mm2, ["symbol", "ts_min"], "left")
    .withColumn("ret_5m", (F.col("mark_t5") / F.col("mark_t0") - 1.0))
)

# (Opcional) chequeo rápido
print(
    "Fuente precios:", src_name,
    "| t0 no nulos:", p95.filter(F.col("mark_t0").isNotNull()).count(),
    "| t5 no nulos:", p95.filter(F.col("mark_t5").isNotNull()).count()
)

# Límites temporales
max_ts = p95.agg(F.max("ts_s").alias("m")).first()["m"]
if max_ts is None:
    raise RuntimeError("Sin datos para definir ventanas.")

import pandas as _pd

def _days_ago(ts, days: int) -> str:
    return (_pd.to_datetime(ts) - _pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

end_test = max_ts
start_test = _days_ago(end_test, CFG.test_days)
end_val = _days_ago(_pd.to_datetime(start_test), CFG.gap_days)
start_val = _days_ago(_pd.to_datetime(start_test), CFG.val_days + CFG.gap_days)
end_train = _days_ago(_pd.to_datetime(start_val), CFG.gap_days)
start_train = _days_ago(_pd.to_datetime(start_val), CFG.train_days + CFG.gap_days)

# Construye dataset con side/target/split y crea la vista
dataset = (
    p95
    .withColumn("side", F.when(F.col("trade_is_buy")==1, F.lit("BUY")).otherwise(F.lit("SELL")))
    .withColumn(
        "target",
        F.when((F.col("side")=="BUY") & (F.col("ret_5m") >= F.lit(CFG.positive_move_buy)), 1)
         .when((F.col("side")=="SELL") & (F.col("ret_5m") <= F.lit(CFG.positive_move_sell)), 1)
         .otherwise(0)
    )
    .withColumn(
        "split",
        F.when((F.col("ts_s") >= F.lit(start_test)) & (F.col("ts_s") <= F.lit(end_test)), "test")
         .when((F.col("ts_s") >= F.lit(start_val))  & (F.col("ts_s") <= F.lit(end_val)),  "val")
         .when((F.col("ts_s") >= F.lit(start_train))& (F.col("ts_s") <= F.lit(end_train)),"train")
         .otherwise("drop")
    )
    .filter("mark_t0 IS NOT NULL AND mark_t5 IS NOT NULL AND split <> 'drop'")
    .select("symbol","side","ts_s","ret_5m","split", *FEATURE_COLS, "target")
)

dataset.createOrReplaceTempView("vw_whale_features_train")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5) Entrenamiento y selección por símbolo/lado
# === Training por (symbol, side) en paralelo con applyInPandas ===

'''from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType, TimestampType
)
from pyspark.sql.functions import lit, current_timestamp, to_timestamp
import mlflow, mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import average_precision_score, precision_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from xgboost import XGBClassifier'''

# 1) Schema de salida por grupo (symbol, side)
train_return_schema = StructType([
    StructField("symbol", StringType(), False),
    StructField("side", StringType(), False),
    StructField("winner_type", StringType(), False),
    StructField("threshold", DoubleType(), True),
    StructField("ap_val", DoubleType(), True),
    StructField("precision_val", DoubleType(), True),
    StructField("coverage_val", LongType(), True),
    StructField("ap_test", DoubleType(), True),
    StructField("precision_test", DoubleType(), True),
    StructField("coverage_test", LongType(), True),
    StructField("mean_return_test", DoubleType(), True),
    StructField("artifact_uri", StringType(), True),
    StructField("trained_until", TimestampType(), True),
    StructField("created_at", TimestampType(), True)
])

# 2) Función de entrenamiento por grupo (usa tu helper evaluate_threshold_grid)
def train_group(pdf: pd.DataFrame) -> pd.DataFrame:
    # Capturamos la config desde el entorno, no como argumentos
    threshold_grid = tuple(CFG.threshold_grid)
    min_precision  = float(CFG.min_precision)

    symbol = str(pdf["symbol"].iloc[0])
    side   = str(pdf["side"].iloc[0])

    # Splits
    train_pdf = pdf[pdf["split"]=="train"]
    val_pdf   = pdf[pdf["split"]=="val"]
    test_pdf  = pdf[pdf["split"]=="test"]

    if min(len(train_pdf), len(val_pdf), len(test_pdf)) < 50:
        return pd.DataFrame([{
            "symbol": symbol, "side": side, "winner_type": "NA",
            "threshold": np.nan, "ap_val": np.nan, "precision_val": np.nan, "coverage_val": 0,
            "ap_test": np.nan, "precision_test": np.nan, "coverage_test": 0, "mean_return_test": np.nan,
            "artifact_uri": None, "trained_until": pd.to_datetime(end_train), "created_at": datetime.utcnow()
        }])

    X_train, y_train = train_pdf[FEATURE_COLS], train_pdf["target"].values
    X_val,   y_val   = val_pdf[FEATURE_COLS],   val_pdf["target"].values
    X_test,  y_test  = test_pdf[FEATURE_COLS],  test_pdf["target"].values
    ret_val  = val_pdf["ret_5m"].values
    ret_test = test_pdf["ret_5m"].values

    mlflow.sklearn.autolog(disable=False)

    candidates = {}
    for mtype in ("lgbm","xgb","logreg"):
        with mlflow.start_run(run_name=f"{symbol}_{side}__{mtype}"):
            if mtype=="lgbm":
                model = lgb.LGBMClassifier(
                    n_estimators=600, learning_rate=0.05, num_leaves=31,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                )
            elif mtype=="xgb":
                model = XGBClassifier(
                    n_estimators=700, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
                )
            else:
                model = LogisticRegression(max_iter=2000, solver="lbfgs")

            model.fit(X_train, y_train)
            y_proba_val = model.predict_proba(X_val)[:,1]
            ap_val = average_precision_score(y_val, y_proba_val)

            sel = evaluate_threshold_grid(y_val, y_proba_val, ret_val, threshold_grid, min_precision)
            candidates[mtype] = {"model": model, "ap_val": ap_val, "sel": sel}

    # Ganador por AP(val) con desempate (lgbm > xgb > logreg)
    order = sorted(candidates.items(), key=lambda kv: (kv[1]["ap_val"], {"lgbm":3,"xgb":2,"logreg":1}[kv[0]]), reverse=True)
    winner_type, w = order[0]
    thr = w["sel"]["threshold"]

    if thr is None:
        return pd.DataFrame([{
            "symbol": symbol, "side": side, "winner_type": winner_type,
            "threshold": np.nan, "ap_val": float(w["ap_val"]), "precision_val": np.nan, "coverage_val": 0,
            "ap_test": np.nan, "precision_test": np.nan, "coverage_test": 0, "mean_return_test": np.nan,
            "artifact_uri": None, "trained_until": pd.to_datetime(end_train), "created_at": datetime.utcnow()
        }])

    # Evaluación en test
    y_proba_test = w["model"].predict_proba(X_test)[:,1]
    ap_test = average_precision_score(y_test, y_proba_test)
    y_hat_test = (y_proba_test >= thr).astype(int)
    prec_test = precision_score(y_test, y_hat_test, zero_division=0)
    cov_test  = int(y_hat_test.sum())
    mean_ret_test = float(ret_test[y_hat_test.astype(bool)].mean()) if cov_test>0 else float("nan")

    # Log del modelo ganador
    with mlflow.start_run(
        run_name=f"{symbol}_{side}__{mtype}",
        nested=True,
        experiment_id=experiment_id
    ) as run:
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_metric("ap_val", ap_val)
        mlflow.log_metric("precision_val", sel["precision"])
        mlflow.log_metric("coverage_val", sel["coverage"])
        mlflow.set_tag("symbol", symbol)
        mlflow.set_tag("side", side)
        mlflow.set_tag("model_type", mtype)
        artifact_uri = f"runs:/{run.info.run_id}/model"

    return pd.DataFrame([{
        "symbol": symbol, "side": side, "winner_type": winner_type,
        "threshold": float(thr), "ap_val": float(w["ap_val"]),
        "precision_val": float(w["sel"]["precision"]), "coverage_val": int(w["sel"]["coverage"]),
        "ap_test": float(ap_test), "precision_test": float(prec_test),
        "coverage_test": int(cov_test), "mean_return_test": float(mean_ret_test),
        "artifact_uri": artifact_uri,
        "trained_until": pd.to_datetime(end_train),
        "created_at": datetime.utcnow()
    }])


# 3) applyInPandas por (symbol, side)
with mlflow.start_run(run_name="training_per_symbol_side") as outer:
    results_df = (
        spark.table("vw_whale_features_train")  # o usa 'dataset' si ya lo creaste como SDF preparado
             .select("symbol","side","ts_s","ret_5m","split", *FEATURE_COLS, "target")
             .repartition("symbol","side")
             .groupBy("symbol","side")
             .applyInPandas(train_group, schema=train_return_schema)
             .cache()
    )

display(results_df)

# 4) Registrar en UC + persistir thresholds en GOLD
_ensure_thresholds_table()

client = MlflowClient()
for r in results_df.collect():
    if r["artifact_uri"] is None or r["threshold"] is None:
        continue

    uc_model_name = f"{UC}.{SC_GOLD}.{REG.model_base_name}__{r['symbol'].lower()}_{r['side'].lower()}"
    mv = mlflow.register_model(model_uri=r["artifact_uri"], name=uc_model_name)
    client.set_registered_model_alias(name=uc_model_name, alias="Champion", version=mv.version)
    client.set_registered_model_tag(uc_model_name, "symbol", r["symbol"])
    client.set_registered_model_tag(uc_model_name, "side", r["side"])
    client.set_registered_model_tag(uc_model_name, "model_type", r["winner_type"])
    client.set_registered_model_tag(uc_model_name, "feature_cols", json.dumps(FEATURE_COLS))

    (
      spark.range(1).select(
          F.lit(f"{r['symbol']}_{r['side']}").alias("model_key"),
          F.lit(r["symbol"]).alias("symbol"),
          F.lit(r["side"]).alias("side"),
          F.lit(r["winner_type"]).alias("winner_type"),
          F.lit(float(r["threshold"])).cast("double").alias("threshold"),
          F.lit(float(r["ap_val"])).cast("double").alias("ap_val"),
          F.lit(float(r["precision_val"])).cast("double").alias("precision_at_threshold_val"),
          F.lit(int(r["coverage_val"])).cast("long").alias("coverage_signals_val"),
          F.lit(float(r["ap_test"])).cast("double").alias("ap_test"),
          F.lit(float(r["precision_test"])).cast("double").alias("precision_at_threshold_test"),
          F.lit(int(r["coverage_test"])).cast("long").alias("coverage_signals_test"),
          F.lit(float(r["mean_return_test"])).cast("double").alias("mean_return_test"),
          F.lit(r["trained_until"]).cast("timestamp").alias("trained_until"),
          F.current_timestamp().alias("created_at")
      )
    ).write.mode("append").format("delta").saveAsTable(TBL_THRESH)
# MAGIC %md
# MAGIC ## 6) Resumen de resultados

# COMMAND ----------
display(results_df.orderBy("symbol","side"))


# COMMAND ----------
# MAGIC %md
# MAGIC ## 7) Notas de operación
# MAGIC 
# MAGIC - Este notebook se lanza como **Job** (semanal o diario). Actualiza `Champion` si mejora.
# MAGIC - El notebook de *Serving* consumirá el modelo `Champion` y el umbral desde `{TBL_THRESH}`.
# MAGIC - Ajusta `FEATURE_COLS` y ventanas si cambias la definición de *features*.


# COMMAND ----------

# 1) ¿Hay datos?
print("p95 rows:", p95.count())
print("mark1m rows:", mark1m.count())

# 2) Min/max por fuente (asegura solape)
p95_bounds = p95.agg(F.min("ts_s").alias("min"), F.max("ts_s").alias("max")).first()
mm = mark1m.select("symbol", F.date_trunc("minute", F.col("ts_s")).alias("ts_min"),
                   F.col("close").cast("double").alias("mark_close"))
mm_bounds = mm.agg(F.min("ts_min").alias("min"), F.max("ts_min").alias("max")).first()
print("p95 window:", p95_bounds)
print("mark1m window:", mm_bounds)

# 3) Cobertura tras los joins de t0/t5
mm = mark1m.select(
    "symbol",
    F.date_trunc("minute", F.col("ts_s")).alias("ts_min"),
    F.col("close").cast("double").alias("mark_close")
)

p95_tmp = p95.drop("mark_t0", "mark_t5").withColumn(  # quita posibles duplicados antes de unir
    "ts_min", F.date_trunc("minute", F.col("ts_s"))
)

mm_t0 = mm.select("symbol", "ts_min", F.col("mark_close").alias("mark_t0")).alias("t0")
mm_t5 = mm.select(
    "symbol",
    F.expr("timestampadd(MINUTE,-5, ts_min)").alias("ts_min"),
    F.col("mark_close").alias("mark_t5")
).alias("t5")

joined_diag = (
    p95_tmp.alias("p")
        .join(mm_t0, ["symbol", "ts_min"], "left")
        .join(mm_t5, ["symbol", "ts_min"], "left")
        .select("symbol", "ts_min",
                F.col("t0.mark_t0").alias("mark_t0"),
                F.col("t5.mark_t5").alias("mark_t5"))
)

print("t0 non-null:", joined_diag.filter(F.col("mark_t0").isNotNull()).count())
print("t5 non-null:", joined_diag.filter(F.col("mark_t5").isNotNull()).count())
print("t0&t5 non-null:", joined_diag.filter(F.col("mark_t0").isNotNull() & F.col("mark_t5").isNotNull()).count())

# 4) Símbolos que no existen en mark1m
p95_sym = p95.select("symbol").distinct()
mm_sym  = mm.select("symbol").distinct()
print("símbolos p95 sin mark:", p95_sym.join(mm_sym, "symbol", "left_anti").count())

# COMMAND ----------

# === INSPECTOR: esquema, muestra y perfiles de columnas de tiempo ===
import pyspark.sql.functions as F

def _has(df): 
    return bool(df is not None and getattr(df, "columns", []))

def _try_show(df, n=10):
    try:
        df.show(n, truncate=False)
    except Exception as e:
        print("show() failed:", e)

tables = [
    ("p95",     p95,     TBL_P95),
    ("mark1m",  mark1m,  TBL_MARK),
    ("index1m", index1m, TBL_INDEX),
    ("kline1m", kline1m, TBL_KLINE),
]

for name, df, tbl in tables:
    print("="*100)
    print(f"TABLE {name} -> {tbl}")
    exists = _has(df)
    rows = df.count() if exists else 0
    print(f"exists={exists}, rows={rows}")
    if not exists:
        continue

    # Esquema y dtypes
    print("\n-- printSchema() --")
    df.printSchema()
    print("\n-- dtypes --")
    print(df.dtypes)

    # Símbolos
    if "symbol" in df.columns:
        sc = df.select("symbol").distinct().count()
        sample = [r["symbol"] for r in df.select("symbol").distinct().limit(20).collect()]
        print(f"\n-- symbols --  distinct={sc} | sample={sample}")

    # Columnas de tiempo presentes y min/max crudo
    cands = [c for c in ["ts_s","timestamp","open_time","close_time","date","time","event_time"] if c in df.columns]
    print(f"\n-- time-like columns present --  {cands}")
    for c in cands:
        dt = dict(df.dtypes)[c]
        print(f"  · {c} ({dt}) min/max:")
        try:
            mm = df.select(F.min(F.col(c)).alias("min"), F.max(F.col(c)).alias("max")).first()
            print(f"    {mm.asDict()}")
        except Exception as e:
            print("    (cannot compute min/max):", e)

        # Si es numérica, muestro conversión ejemplo ms→timestamp y secs→timestamp
        if dt in ("bigint","long","int","double","float","decimal(38,18)","decimal(20,0)","decimal(19,0)"):
            conv = (df
                .select(
                    F.col(c).alias("raw"),
                    (F.col(c)/1000).cast("timestamp").alias("as_ts_div1000"),
                    F.from_unixtime(F.col(c)).cast("timestamp").alias("as_ts_secs")
                )
                .where(F.col(c).isNotNull())
                .limit(3)
            )
            print("    sample conversions (raw, /1000→ts, from_unixtime→ts):")
            _try_show(conv)

    # Muestra de filas
    print("\n-- sample rows --")
    _try_show(df.limit(10))

    # DESCRIBE DETAIL físico
    try:
        print("\n-- DESCRIBE DETAIL --")
        spark.sql(f"DESCRIBE DETAIL {tbl}").show(truncate=False)
    except Exception as e:
        print("DESCRIBE DETAIL failed:", e)


# === OPTIONAL: diagnóstico de ventana temporal con ts derivada (solo lectura) ===
import pyspark.sql.functions as F

def _derive_ts(df):
    if not _has(df): 
        return None, None
    dtypes = dict(df.dtypes)
    if "ts_s" in dtypes:
        return F.col("ts_s"), "ts_s"
    if "timestamp" in dtypes and dtypes["timestamp"] in ("bigint","long","int","double","float","decimal(38,18)"):
        return (F.col("timestamp")/1000).cast("timestamp"), "timestamp/1000→ts"
    if "open_time" in dtypes:
        if dtypes["open_time"] in ("bigint","long","int","double","float","decimal(38,18)"):
            return (F.col("open_time")/1000).cast("timestamp"), "open_time/1000→ts"
        elif dtypes["open_time"] == "timestamp":
            return F.col("open_time"), "open_time"
    return None, None

def _window_by_expr(df, expr):
    row = df.select(F.min(expr).alias("min"), F.max(expr).alias("max")).first()
    return {"min": row["min"], "max": row["max"]}

for name, df, _ in [
    ("p95", p95, None),
    ("mark1m", mark1m, None),
    ("index1m", index1m, None),
    ("kline1m", kline1m, None),
]:
    print("="*60, name)
    if not _has(df):
        print("no df")
        continue
    expr, how = _derive_ts(df)
    print("derived_ts:", how)
    if expr is None:
        print("no derived ts available")
        continue
    try:
        w = _window_by_expr(df, expr)
        print("window:", w)
    except Exception as e:
        print("window error:", e)



# COMMAND ----------

display(spark.sql("""
    SELECT * FROM dbrtfmprod.whale_trades.silver__mark_klines_1m"""))

# COMMAND ----------

display(spark.sql("""
    SELECT * FROM dbrtfmprod.whale_trades.silver__index_klines_1m"""))

# COMMAND ----------

display(spark.sql("""
    SELECT * FROM dbrtfmprod.whale_trades.silver__klines_1m"""))
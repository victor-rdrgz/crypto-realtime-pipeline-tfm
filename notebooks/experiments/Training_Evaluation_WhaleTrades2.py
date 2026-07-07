# Databricks notebook source
# Databricks notebook source
# MAGIC %md
# MAGIC # Whale Trades — Training & Evaluation (Mark ±1.5% @ +5m)
# MAGIC
# MAGIC Objetivo: entrenar por (symbol, side) y dejar TODO registrado en **Experiments** (runs por grupo, con jerarquía).
# MAGIC Fix clave: los runs se abren en workers con `experiment_id` y `mlflow.parentRunId`.

# COMMAND ----------
# Imports y setup

import os, json, typing as t
from dataclasses import dataclass
from datetime import datetime, timedelta

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, precision_score
from sklearn.linear_model import LogisticRegression

# LightGBM / XGBoost con fallback si no están instalados
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from xgboost import XGBClassifier as _XGBClassifier
    XGBClassifier = _XGBClassifier
except Exception:
    XGBClassifier = None

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import DataFrame as SDF
from pyspark.sql.types import (
    StructType,StructField,StringType,DoubleType,LongType,TimestampType
)

spark.conf.set("spark.sql.shuffle.partitions", "12")

# COMMAND ----------
# Configuración

@dataclass
class Paths:
    catalog: str = "dbrtfmprod"
    schema: str = "whale_trades"
    table_p95_trades: str = "silver__p95_trades"
    table_mark_1m: str = "silver__mark_klines_1m"
    table_index_1m: str = "silver__index_klines_1m"
    table_kline_1m: str = "silver__klines_1m"
    table_model_thresh: str = "gold__model_thresholds"

@dataclass
class ModelingCfg:
    sides: t.List[str] = ("BUY", "SELL")
    positive_move_buy: float = 0.015
    positive_move_sell: float = -0.015

    # Ajustado para trabajar con 30 días de datos totales
    train_days: int = 21
    val_days: int = 4
    test_days: int = 5
    gap_days: int = 0

    min_precision: float = 0.80
    threshold_grid: t.Tuple[float,...] = tuple(np.round(np.linspace(0.99, 0.5, 50), 3))


@dataclass
class RegistryCfg:
    model_base_name: str = "whale_price_impact"
    use_uc: bool = True

MIN_ROWS_PER_SPLIT = 10

PATHS = Paths()
CFG   = ModelingCfg()
REG   = RegistryCfg()

UC       = PATHS.catalog
SC_WH    = PATHS.schema
SC_GOLD  = PATHS.schema

TBL_P95   = f"{UC}.{SC_WH}.{PATHS.table_p95_trades}"
TBL_MARK  = f"{UC}.{SC_WH}.{PATHS.table_mark_1m}"
TBL_INDEX = f"{UC}.{SC_WH}.{PATHS.table_index_1m}"
TBL_KLINE = f"{UC}.{SC_WH}.{PATHS.table_kline_1m}"
TBL_THRESH= f"{UC}.{SC_GOLD}.{PATHS.table_model_thresh}"

if REG.use_uc:
    mlflow.set_registry_uri("databricks-uc")

username = spark.sql("SELECT current_user()").first()[0]
exp_name = f"/Users/{username}/whale_trades_training"
mlflow.set_experiment(exp_name)
exp = mlflow.get_experiment_by_name(exp_name)
if exp is None:
    raise RuntimeError(f"No se pudo resolver el experimento MLflow '{exp_name}'")
EXPERIMENT_ID = exp.experiment_id

# COMMAND ----------
# Utils

FEATURE_COLS = [
    "trade_notional","trade_is_buy","notional_z",
    "buys_share_5s","buys_share_15s","buys_share_30s",
    "notional_sum_30s","runlen_same_side_30s","secs_since_last_whale",
    "kline_prev_close","ret_1m_prev","range_1m_prev_rel","vol_1m_prev_z",
    "mark_prev_close","ret_mark_1m_prev","index_prev_close",
    "premium","d_premium_1m","hour_of_day","day_of_week"
]

def _read_table_or_fail(fullname: str) -> SDF:
    try:
        return spark.table(fullname)
    except Exception as e:
        raise RuntimeError(f"No se pudo leer {fullname}: {e}")

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

def evaluate_threshold_grid(y_true: np.ndarray, y_proba: np.ndarray, returns: np.ndarray,
                            thresholds: t.Iterable[float], min_precision: float) -> dict:
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

def ensure_numeric_cols(df: SDF, cols: t.List[str]) -> SDF:
    for c in cols:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("double"))
    return df

# COMMAND ----------
# Carga tablas

p95     = _read_table_or_fail(TBL_P95)
mark1m  = _read_table_or_fail(TBL_MARK)
index1m = _read_table_or_fail(TBL_INDEX)
kline1m = _read_table_or_fail(TBL_KLINE)

if not p95.columns:
    raise RuntimeError("silver__p95_trades vacío/inexistente.")

# Normalizar ts_s
if "ts_s" not in p95.columns:
    if "timestamp" in p95.columns:
        if isinstance(p95.schema["timestamp"].dataType, TimestampType):
            p95 = p95.withColumn("ts_s", F.col("timestamp").cast("timestamp"))
        else:
            p95 = p95.withColumn("ts_s", (F.col("timestamp")/F.lit(1000.0)).cast("timestamp"))
    else:
        raise RuntimeError("P95 debe tener 'timestamp' o 'ts_s'.")

for name in ("mark1m","index1m","kline1m"):
    df = locals()[name]
    if df.columns and "ts_s" not in df.columns:
        base = "open_time" if "open_time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
        if base is None: 
            continue
        if isinstance(df.schema[base].dataType, TimestampType):
            locals()[name] = df.withColumn("ts_s", F.col(base).cast("timestamp"))
        else:
            locals()[name] = df.withColumn("ts_s", (F.col(base)/F.lit(1000.0)).cast("timestamp"))

# COMMAND ----------
# Feature engineering

p95 = (p95
       .withColumn("price", F.col("price").cast("double"))
       .withColumn("qty", F.col("qty").cast("double"))
       .withColumn("trade_notional", F.col("price") * F.col("qty"))
       .withColumn("trade_is_buy", (~F.col("is_buyer_maker")).cast("int"))
)

win_prev = Window.partitionBy("symbol").orderBy(F.col("ts_s").cast("long")).rowsBetween(-1440, -1)
p95 = (p95
       .withColumn("notional_mean", F.mean("trade_notional").over(win_prev))
       .withColumn("notional_std",  F.stddev("trade_notional").over(win_prev))
       .withColumn("notional_z",    (F.col("trade_notional")-F.col("notional_mean"))/F.col("notional_std"))
)

for secs in (5, 15, 30):
    w = Window.partitionBy("symbol").orderBy(F.col("ts_s").cast("long")).rangeBetween(-secs, -1)
    p95 = (p95
           .withColumn(f"notional_buy_{secs}s",   F.sum(F.when(F.col("trade_is_buy")==1, F.col("trade_notional")).otherwise(0.0)).over(w))
           .withColumn(f"notional_total_{secs}s", F.sum("trade_notional").over(w))
           .withColumn(f"buys_share_{secs}s",     F.col(f"notional_buy_{secs}s")/F.col(f"notional_total_{secs}s"))
    )

w30 = Window.partitionBy("symbol").orderBy(F.col("ts_s").cast("long")).rangeBetween(-30, -1)
p95 = p95.withColumn("notional_sum_30s", F.sum("trade_notional").over(w30))

side_cur = F.when(F.col("trade_is_buy")==1, F.lit(1)).otherwise(F.lit(-1))
p95 = p95.withColumn("_side", side_cur)
win30 = Window.partitionBy("symbol").orderBy(F.col("ts_s").cast("long")).rangeBetween(-30, -1)
p95 = (p95
       .withColumn("runlen_same_side_30s", F.sum(F.when(F.col("_side")==F.last("_side").over(win30), 1).otherwise(0)).over(win30))
       .drop("_side")
)

win_unbounded = Window.partitionBy("symbol").orderBy(F.col("ts_s").cast("long")).rowsBetween(Window.unboundedPreceding, -1)
p95 = p95.withColumn("secs_since_last_whale", (F.col("ts_s").cast("long") - F.max(F.col("ts_s").cast("long")).over(win_unbounded)))

# Kline 1m (prev)
if kline1m.columns:
    k = (kline1m
         .select("symbol",
                 F.date_trunc("minute", F.col("ts_s")).alias("ts_min"),
                 F.col("close").cast("double").alias("k_close"),
                 F.col("high").cast("double").alias("k_high"),
                 F.col("low").cast("double").alias("k_low"),
                 F.col("volume").cast("double").alias("k_vol"))
    )
    p95 = p95.withColumn("ts_min", F.date_trunc("minute", F.col("ts_s")))
    p95 = (p95
           .join(k, ["symbol","ts_min"], "left")
           .withColumn("kline_prev_close", F.col("k_close"))
           .withColumn(
               "range_1m_prev_rel",
               (F.col("k_high")-F.col("k_low"))/F.when(F.col("k_close")!=0, F.col("k_close")).otherwise(F.lit(None).cast("double"))
           )
    )
    w_k = Window.partitionBy("symbol").orderBy(F.col("ts_min").cast("long"))
    p95 = (p95
           .withColumn("k_close_prev", F.lag("k_close").over(w_k))
           .withColumn("ret_1m_prev", (F.col("k_close")/F.col("k_close_prev") - 1.0))
    )
    vol_sym = (k.groupBy("symbol").agg(F.avg("k_vol").alias("vol_mean"), F.stddev("k_vol").alias("vol_std")))
    p95 = (p95
           .join(vol_sym, ["symbol"], "left")
           .withColumn(
               "vol_1m_prev_z",
               (F.col("k_vol")-F.col("vol_mean"))/F.when(F.col("vol_std")>0, F.col("vol_std")).otherwise(F.lit(None).cast("double"))
           )
    )

# Mark/Index + premium
if mark1m.columns and index1m.columns:
    m = mark1m.select("symbol", F.date_trunc("minute", F.col("ts_s")).alias("ts_min"),
                      F.col("close").cast("double").alias("mark_close"))
    i = index1m.select("symbol", F.date_trunc("minute", F.col("ts_s")).alias("ts_min"),
                       F.col("close").cast("double").alias("index_close"))
    mi = (m.join(i, ["symbol","ts_min"], "inner")
            .withColumn("premium", (F.col("mark_close")-F.col("index_close"))/F.col("index_close")))
    wmi = Window.partitionBy("symbol").orderBy(F.col("ts_min").cast("long"))
    mi = (mi
          .withColumn("premium_prev", F.lag("premium").over(wmi))
          .withColumn("d_premium_1m", F.col("premium")-F.col("premium_prev"))
    )
    p95 = (p95
           .join(mi, ["symbol","ts_min"], "left")
           .withColumn("mark_prev_close", F.col("mark_close"))
           .withColumn("index_prev_close", F.col("index_close"))
           .withColumn("ret_mark_1m_prev", F.col("premium"))
    )

p95 = (p95
       .withColumn("hour_of_day", F.hour("ts_s"))
       .withColumn("day_of_week", F.dayofweek("ts_s"))
)

p95 = ensure_numeric_cols(p95, FEATURE_COLS)

# COMMAND ----------
# Etiquetado y splits

def _pick_price_source(mark1m, index1m, kline1m):
    for df in (mark1m, index1m, kline1m):
        if df.columns and df.limit(2).count() >= 2:
            return df
    raise RuntimeError("No hay velas suficientes (mark/index/kline) para ret_5m.")

src_df = _pick_price_source(mark1m, index1m, kline1m)

mm_base = (src_df
    .select(
        F.col("symbol").alias("px_symbol"),
        F.date_trunc("minute", F.col("ts_s")).alias("px_min"),
        F.col("close").cast("double").alias("px_close"))
)

wpx = Window.partitionBy("px_symbol").orderBy("px_min")
mm2 = (mm_base
    .withColumn("mark_t0", F.col("px_close"))
    .withColumn("mark_t5", F.lead("px_close", 5).over(wpx))
    .select(F.col("px_symbol").alias("symbol"),
            F.col("px_min").alias("ts_min"), "mark_t0","mark_t5")
)

p95 = (p95
       .withColumn("ts_min", F.date_trunc("minute", F.col("ts_s")))
       .join(mm2, ["symbol","ts_min"], "left")
       .withColumn("ret_5m", (F.col("mark_t5")/F.col("mark_t0") - 1.0))
)

max_ts = p95.agg(F.max("ts_s").alias("m")).first()["m"]
if max_ts is None:
    raise RuntimeError("Sin datos para definir ventanas.")

import pandas as _pd
def _days_ago(ts, days: int) -> str:
    return (_pd.to_datetime(ts) - _pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

from datetime import timedelta

# Última fecha real de datos (robusto)
max_ts = p95.select(F.max("ts_s")).first()["max(ts_s)"]
if not max_ts:
    raise RuntimeError("No hay datos para calcular ventanas de split.")

# CFG: test/val/train en días, gap opcional
TEST_D  = int(CFG.test_days)
VAL_D   = int(CFG.val_days)
TRAIN_D = int(CFG.train_days)
GAP_D   = int(CFG.gap_days)

# Ventanas
start_test_dt     = max_ts - timedelta(days=TEST_D)
end_test_dt       = max_ts
end_val_dt        = start_test_dt - timedelta(days=GAP_D)
start_val_dt      = end_val_dt - timedelta(days=VAL_D)
end_train_dt      = start_val_dt - timedelta(days=GAP_D)
start_train_dt    = end_train_dt - timedelta(days=TRAIN_D)

# Cast explícito a timestamp
start_train_ts, end_train_ts = F.lit(start_train_dt).cast("timestamp"), F.lit(end_train_dt).cast("timestamp")
start_val_ts, end_val_ts     = F.lit(start_val_dt).cast("timestamp"),   F.lit(end_val_dt).cast("timestamp")
start_test_ts, end_test_ts   = F.lit(start_test_dt).cast("timestamp"),  F.lit(end_test_dt).cast("timestamp")

# DEBUG opcional: imprimir
print("Ventanas temporales:",
      f"\n   Train: {start_train_dt} → {end_train_dt}",
      f"\n   Val:   {start_val_dt} → {end_val_dt}",
      f"\n   Test:  {start_test_dt} → {end_test_dt}")


dataset = (p95
    .withColumn("side", F.when(F.col("trade_is_buy")==1, F.lit("BUY")).otherwise(F.lit("SELL")))
    .withColumn(
        "target",
        F.when((F.col("side")=="BUY")  & (F.col("ret_5m")>=F.lit(CFG.positive_move_buy)), 1)
         .when((F.col("side")=="SELL") & (F.col("ret_5m")<=F.lit(CFG.positive_move_sell)), 1)
         .otherwise(0)
    )
    .withColumn(
        "split",
        F.when(
            (F.col("ts_s") >= F.to_timestamp(F.lit(start_test))) & (F.col("ts_s") <= F.to_timestamp(F.lit(end_test))), "test"
        ).when(
            (F.col("ts_s") >= F.to_timestamp(F.lit(start_val)))  & (F.col("ts_s") <= F.to_timestamp(F.lit(end_val))),  "val"
        ).when(
            (F.col("ts_s") >= F.to_timestamp(F.lit(start_train)))& (F.col("ts_s") <= F.to_timestamp(F.lit(end_train))),"train"
        ).otherwise("drop")
    )
    .filter("mark_t0 IS NOT NULL AND mark_t5 IS NOT NULL AND split <> 'drop'")
    .select("symbol","side","ts_s","ret_5m","split", *FEATURE_COLS, "target")
)

# COMMAND ----------
# Schemas

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

# COMMAND ----------
# Entrenamiento por grupo — con runs abiertos en WORKERS
# (experiment_id explícito + tag mlflow.parentRunId)

def train_group(pdf: pd.DataFrame) -> pd.DataFrame:
    threshold_grid = tuple(CFG.threshold_grid)
    min_precision  = float(CFG.min_precision)

    symbol         = str(pdf["symbol"].iloc[0])
    side           = str(pdf["side"].iloc[0])
    parent_run_id  = pdf["parent_run_id"].iloc[0]
    experiment_id  = str(pdf["experiment_id"].iloc[0])

    feat = FEATURE_COLS
    pdf = pdf.copy()
    pdf[feat] = pdf[feat].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    train_pdf = pdf[pdf["split"]=="train"]
    val_pdf   = pdf[pdf["split"]=="val"]
    test_pdf  = pdf[pdf["split"]=="test"]

    # Si no hay suficiente data, AUN ASÍ creamos un run diagnóstico
    if min(len(train_pdf), len(val_pdf), len(test_pdf)) < MIN_ROWS_PER_SPLIT:
        with mlflow.start_run(run_name=f"{symbol}_{side}__insufficient",
                              experiment_id=experiment_id,
                              tags={"mlflow.parentRunId": parent_run_id,
                                    "symbol": symbol, "side": side, "reason": "insufficient_data"}):
            mlflow.log_metric("rows_train", int(len(train_pdf)))
            mlflow.log_metric("rows_val", int(len(val_pdf)))
            mlflow.log_metric("rows_test", int(len(test_pdf)))
            mlflow.log_param("reason", "insufficient_data")  # deja el motivo como parámetro
            mlflow.log_dict(
                {"symbol": symbol, "side": side,
                "rows": {"train": int(len(train_pdf)), "val": int(len(val_pdf)), "test": int(len(test_pdf))}},
                "insufficient_summary.json"
            ) 
        return pd.DataFrame([{
            "symbol": symbol, "side": side, "winner_type": "NA",
            "threshold": np.nan, "ap_val": np.nan, "precision_val": np.nan, "coverage_val": 0,
            "ap_test": np.nan, "precision_test": np.nan, "coverage_test": 0, "mean_return_test": np.nan,
            "artifact_uri": None, "trained_until": pd.to_datetime(end_train), "created_at": datetime.utcnow()
        }])

    X_train, y_train = train_pdf[feat], train_pdf["target"].values
    X_val,   y_val   = val_pdf[feat],   val_pdf["target"].values
    X_test,  y_test  = test_pdf[feat],  test_pdf["target"].values
    ret_val          = val_pdf["ret_5m"].values
    ret_test         = test_pdf["ret_5m"].values

    candidates = {}
    model_types = []
    if lgb is not None: model_types.append("lgbm")
    if XGBClassifier is not None: model_types.append("xgb")
    model_types.append("logreg")

    for mtype in model_types:
        with mlflow.start_run(run_name=f"{symbol}_{side}__{mtype}",
                              experiment_id=experiment_id,
                              tags={"mlflow.parentRunId": parent_run_id,
                                    "symbol": symbol, "side": side, "model_type": mtype} ) as run:
            mlflow.sklearn.autolog(log_models=False)

            if mtype=="lgbm":
                model = lgb.LGBMClassifier(
                    n_estimators=600, learning_rate=0.05, num_leaves=31,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                )
            elif mtype=="xgb":
                model = XGBClassifier(
                    n_estimators=700, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                    random_state=42
                )
            else:
                model = LogisticRegression(max_iter=2000, solver="lbfgs")

            model.fit(X_train, y_train)
            y_proba_val = model.predict_proba(X_val)[:,1]
            ap_val = average_precision_score(y_val, y_proba_val)
            sel = evaluate_threshold_grid(y_val, y_proba_val, ret_val, threshold_grid, min_precision)

            mlflow.log_metric("ap_val", ap_val)
            if sel["threshold"] is not None:
                mlflow.log_metric("precision_val", sel["precision"])
                mlflow.log_metric("coverage_val", sel["coverage"])
                mlflow.sklearn.log_model(model, "model")
                artifact_uri = f"runs:/{run.info.run_id}/model"
            else:
                artifact_uri = None

            candidates[mtype] = {
                "model": model,
                "ap_val": ap_val,
                "sel": sel,
                "artifact_uri": artifact_uri,
                "run_id": run.info.run_id
            }

    valid = [(mt,info) for mt,info in candidates.items() if info["sel"]["threshold"] is not None]
    if valid:
        order = sorted(valid, key=lambda kv: (kv[1]["ap_val"], {"lgbm":3,"xgb":2,"logreg":1}[kv[0]]), reverse=True)
    else:
        order = sorted(candidates.items(), key=lambda kv: (kv[1]["ap_val"], {"lgbm":3,"xgb":2,"logreg":1}[kv[0]]), reverse=True)

    winner_type, w = order[0]
    thr = w["sel"]["threshold"]

    if thr is None:
        # Aun así deja constancia de ganador sin threshold
        with mlflow.start_run(run_id=w["run_id"]):
            mlflow.set_tag("winner_without_threshold", "true")
        return pd.DataFrame([{
            "symbol": symbol, "side": side, "winner_type": winner_type,
            "threshold": np.nan, "ap_val": float(w["ap_val"]),
            "precision_val": np.nan, "coverage_val": 0,
            "ap_test": np.nan, "precision_test": np.nan,
            "coverage_test": 0, "mean_return_test": np.nan,
            "artifact_uri": None,
            "trained_until": pd.to_datetime(end_train),
            "created_at": datetime.utcnow()
        }])

    # TEST en run ganador
    y_proba_test = w["model"].predict_proba(X_test)[:,1]
    ap_test = average_precision_score(y_test, y_proba_test)
    y_hat_test = (y_proba_test >= thr).astype(int)
    prec_test = precision_score(y_test, y_hat_test, zero_division=0)
    cov_test  = int(y_hat_test.sum())
    mean_ret_test = float(ret_test[y_hat_test.astype(bool)].mean()) if cov_test>0 else float("nan")

    with mlflow.start_run(run_id=w["run_id"]):
        mlflow.log_metric("ap_test", ap_test)
        mlflow.log_metric("precision_test", prec_test)
        mlflow.log_metric("coverage_test", cov_test)
        mlflow.log_metric("mean_return_test", 0.0 if np.isnan(mean_ret_test) else mean_ret_test)
        mlflow.log_param("selected_threshold", float(thr))
        mlflow.set_tag("winner", "true")

    return pd.DataFrame([{
        "symbol": symbol, "side": side, "winner_type": winner_type,
        "threshold": float(thr), "ap_val": float(w["ap_val"]),
        "precision_val": float(w["sel"]["precision"]), "coverage_val": int(w["sel"]["coverage"]),
        "ap_test": float(ap_test), "precision_test": float(prec_test),
        "coverage_test": int(cov_test), "mean_return_test": float(mean_ret_test),
        "artifact_uri": w["artifact_uri"],
        "trained_until": pd.to_datetime(end_train),
        "created_at": datetime.utcnow()
    }])

# COMMAND ----------
# Ejecución + Persistencia + Registro

split_counts = {r["split"]: r["count"] for r in dataset.groupBy("split").count().collect()}

with mlflow.start_run(run_name="training_per_symbol_side", experiment_id=EXPERIMENT_ID) as outer_run:
    parent_run_id = outer_run.info.run_id
    for k,v in split_counts.items():
        mlflow.log_param(f"rows_{k}", int(v))
    mlflow.set_tag("symbols_total", dataset.select("symbol").distinct().count())

    # añade experiment_id+parent_run_id para los workers
    ds2 = (dataset
           .withColumn("parent_run_id", F.lit(parent_run_id))
           .withColumn("experiment_id", F.lit(EXPERIMENT_ID))
          )

    results_df = (ds2
        .repartition("symbol","side")
        .groupBy("symbol","side")
        .applyInPandas(train_group, schema=train_return_schema)
        .cache()
    )

display(results_df)

# GOLD thresholds
_ensure_thresholds_table()
for r in results_df.collect():
    if r["threshold"] is None:
        continue
    (spark.range(1).select(
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
    )).write.mode("append").format("delta").saveAsTable(TBL_THRESH)

# Registro en UC Model Registry
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

# COMMAND ----------
# Resumen
display(results_df.orderBy("symbol","side"))


# COMMAND ----------

# ==== REPORTING CONSOLA — TODOS LOS MODELOS (robusto a columnas faltantes) ====

import mlflow
import pandas as pd
import numpy as np

# --- 0) Resolver EXPERIMENT_ID si no está en el entorno
try:
    EXPERIMENT_ID
except NameError:
    username = spark.sql("SELECT current_user()").first()[0]
    exp_name = f"/Users/{username}/whale_trades_training"
    mlflow.set_experiment(exp_name)
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        raise RuntimeError(f"No encuentro experimento MLflow '{exp_name}'")
    EXPERIMENT_ID = exp.experiment_id

# --- 1) Obtener runs
runs = mlflow.search_runs(
    experiment_ids=[EXPERIMENT_ID],
    filter_string=None,
    max_results=100000,
    output_format="pandas"
)

if runs.empty:
    print("No hay runs en el experimento.")
else:
    # Ayudas
    def has(col): 
        return col in runs.columns

    def to_num(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- 2) Detectar CANDIDATOS
    # Preferimos filtrar por tags.model_type/symbol/side si existen; si no, caemos a ap_val.
    mask = pd.Series(True, index=runs.index)

    if has("metrics.ap_val"):
        mask &= runs["metrics.ap_val"].notna()
    else:
        # Sin ap_val no podemos identificar candidatos de forma fiable
        print("Aviso: no existe 'metrics.ap_val' en los runs. No puedo listar candidatos.")
        print("Columnas disponibles:", list(runs.columns))
        cand = pd.DataFrame(columns=runs.columns)

    if 'cand' not in locals():
        cand = runs[mask].copy()

    # Normaliza columnas esperadas (si faltan, crea vacías)
    for col, default in [
        ("tags.symbol", None),
        ("tags.side", None),
        ("tags.model_type", None),
        ("metrics.ap_val", np.nan),
        ("metrics.precision_val", np.nan),
        ("metrics.coverage_val", np.nan),
        ("metrics.ap_test", np.nan),
        ("metrics.precision_test", np.nan),
        ("metrics.coverage_test", np.nan),
        ("metrics.mean_return_test", np.nan),
        ("params.selected_threshold", np.nan),
        ("tags.winner", None),
    ]:
        if col not in cand.columns:
            cand[col] = default

    # Asegurar numéricos
    to_num(cand, [
        "metrics.ap_val","metrics.precision_val","metrics.coverage_val",
        "metrics.ap_test","metrics.precision_test","metrics.coverage_test",
        "metrics.mean_return_test","params.selected_threshold"
    ])

    # Derivados
    cand["artifact_uri"] = "runs:/" + cand["run_id"] + "/model"
    cand["winner_tag"] = (cand["tags.winner"] == "true").astype(int)

    # Si no hay symbol/side, rellena con 'unknown' para que no explote el groupby
    cand["tags.symbol"] = cand["tags.symbol"].fillna("unknown_symbol")
    cand["tags.side"]   = cand["tags.side"].fillna("unknown_side")
    cand["tags.model_type"] = cand["tags.model_type"].fillna("unknown_model")

    # --- 3) Conteo por (symbol, side)
    print("\n=== MODELOS POR (symbol, side) ===")
    if cand.empty:
        print("No hay candidatos detectados (revisa que los runs tengan 'metrics.ap_val').")
    else:
        counts = (cand
                  .groupby(["tags.symbol","tags.side"], dropna=False)
                  .size()
                  .reset_index(name="n_models")
                  .sort_values(["tags.symbol","tags.side"]))
        print(counts.to_string(index=False))

        # --- 4) Todos los modelos con métricas
        print("\n=== TODOS LOS MODELOS (ordenados por ap_val desc) ===")
        cols = [
            "tags.symbol","tags.side","tags.model_type",
            "metrics.ap_val","metrics.precision_val","metrics.coverage_val",
            "metrics.ap_test","metrics.precision_test","metrics.coverage_test","metrics.mean_return_test",
            "params.selected_threshold","run_id","artifact_uri"
        ]
        have = [c for c in cols if c in cand.columns]
        cand_sorted = cand.sort_values(["tags.symbol","tags.side","metrics.ap_val"], ascending=[True, True, False])
        print(cand_sorted[have].to_string(index=False, na_rep="NA"))

        # --- 5) Winners por tag; si no, por mejor ap_val
        cand["_order_ap"] = cand["metrics.ap_val"].fillna(-1)
        winners = (cand
                   .sort_values(["tags.symbol","tags.side","winner_tag","_order_ap"],
                                ascending=[True, True, False, False])
                   .groupby(["tags.symbol","tags.side"], as_index=False)
                   .head(1)
                  )

        print("\n=== WINNERS POR (symbol, side) ===")
        wcols = [
            "tags.symbol","tags.side","tags.model_type",
            "metrics.ap_val","metrics.precision_val","metrics.coverage_val",
            "metrics.ap_test","metrics.precision_test","metrics.coverage_test","metrics.mean_return_test",
            "params.selected_threshold","run_id","artifact_uri"
        ]
        whave = [c for c in wcols if c in winners.columns]
        print(winners[whave].sort_values(["tags.symbol","tags.side"]).to_string(index=False, na_rep="NA"))

    # --- 6) Diagnóstico de insuficientes (si existen)
    insuf = runs[(has("tags.reason") and runs["tags.reason"]=="insufficient_data")].copy() if has("tags.reason") else pd.DataFrame()
    if not insuf.empty:
        for c in ["metrics.rows_train","metrics.rows_val","metrics.rows_test"]:
            if c in insuf.columns:
                insuf[c] = pd.to_numeric(insuf[c], errors="coerce").astype("Int64")
        keep = ["tags.symbol","tags.side","metrics.rows_train","metrics.rows_val","metrics.rows_test","run_id"]
        keep = [c for c in keep if c in insuf.columns]
        print("\n=== GRUPOS INSUFICIENTES (diagnóstico) ===")
        print(insuf[keep].sort_values(["tags.symbol","tags.side"]).to_string(index=False, na_rep="NA"))


# COMMAND ----------

print("Test rows:")
print(df.filter((F.col("open_time") >= "2025-09-06") & (F.col("open_time") < "2025-09-11")).count())

print("Val rows:")
df.filter((F.col("open_time") >= "2025-09-01") & (F.col("open_time") < "2025-09-06")).count()

# COMMAND ----------

display(df)

# COMMAND ----------


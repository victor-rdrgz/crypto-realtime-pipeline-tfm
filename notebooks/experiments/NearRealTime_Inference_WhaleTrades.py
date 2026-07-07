# Databricks notebook source
# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 2 — Near Real-Time Inference (Whale Trades)
# MAGIC 
# MAGIC **Objetivo**: consumir en *streaming* los *whale trades* (aggTrades P95) desde la zona *landing* (Kafka/Auto Loader), generar las mismas *features* que en training, aplicar el modelo `Champion` registrado en UC con MLflow, usar el umbral seleccionado en `gold__model_thresholds`, aplicar cooldown de 300 s y registrar señales en `gold__signals` y *shadow signals*.
# MAGIC 
# MAGIC ---
# MAGIC **Buenas prácticas aplicadas**
# MAGIC - Features en streaming con **Structured Streaming**.
# MAGIC - Modelos desde **MLflow Model Registry (UC)** con alias `Champion`.
# MAGIC - **Estado de cooldown** con `mapGroupsWithState`.
# MAGIC - Señales reales → `gold__signals`, *shadow* → `gold__shadow_signals` (Delta, UC).
# MAGIC - Umbrales distribuidos con **broadcast variable**.
# MAGIC - Job en modo **continuous** o *trigger(availableNow)* según SLA.
# MAGIC 
# COMMAND ----------
# Imports
import json
import typing as t
import mlflow
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.streaming import GroupState, GroupStateTimeout

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
# Configuración
CATALOG = "dbrtfmprod"
SC_WH = "whale_trades"
SC_GOLD = "gold"

STREAM_FEATURES_TBL = f"{CATALOG}.{SC_GOLD}.gold__ws_features_1m"   # tabla/stream de features en tiempo real
THRESH_TBL = f"{CATALOG}.{SC_GOLD}.gold__model_thresholds"
SIGNALS_TBL = f"{CATALOG}.{SC_GOLD}.gold__signals"
SHADOW_TBL = f"{CATALOG}.{SC_GOLD}.gold__shadow_signals"
MODEL_BASE = "whale_price_impact"

COOLDOWN_SECS = 300

FEATURE_COLS = [
    "trade_notional","trade_is_buy","notional_z",
    "buys_share_5s","buys_share_15s","buys_share_30s",
    "notional_sum_30s","runlen_same_side_30s","secs_since_last_whale",
    "kline_prev_close","ret_1m_prev","range_1m_prev_rel","vol_1m_prev_z",
    "mark_prev_close","ret_mark_1m_prev","index_prev_close",
    "premium","d_premium_1m","hour_of_day","day_of_week"
]

# Cargar thresholds UC
if spark._jsparkSession.catalog().tableExists(THRESH_TBL):
    thr_df = spark.table(THRESH_TBL)
    thr_map = {(r.symbol, r.side): (r.threshold, r.winner_type) for r in thr_df.collect()}
else:
    thr_map = {}

broadcast_thr = spark.sparkContext.broadcast(thr_map)

# COMMAND ----------
# Esquema de salida para señales
signal_schema = StructType([
    StructField("symbol", StringType()),
    StructField("ts_min", TimestampType()),
    StructField("side", StringType()),
    StructField("score", DoubleType()),
    StructField("signal", IntegerType()),
    StructField("is_shadow", IntegerType()),
    StructField("created_at", TimestampType()),
])

# COMMAND ----------
# Función de estado para aplicar cooldown
from datetime import datetime

def cooldown_update(key: t.Tuple[str,str], rows: t.Iterator[pd.DataFrame], state: GroupState) -> t.Iterator[pd.DataFrame]:
    symbol, side = key
    last_ts = state.get("last_ts") if state.exists else None
    thr, model_type = broadcast_thr.value.get((symbol, side), (0.5, None))
    if model_type is None:
        return iter([])

    uc_model_name = f"{CATALOG}.{SC_GOLD}.{MODEL_BASE}__{symbol.lower()}_{side.lower()}@Champion"
    mdl = mlflow.pyfunc.load_model(f"models:/{uc_model_name}")

    for pdf in rows:
        if pdf.empty:
            continue
        X = pdf[FEATURE_COLS].copy()
        y_score = mdl.predict(X)
        if isinstance(y_score, (pd.DataFrame, pd.Series)):
            y_score = np.array(y_score).reshape(-1)
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:,1]

        ts_vals = pd.to_datetime(pdf["ts_min"])
        outputs = []
        for idx, score in enumerate(y_score):
            thr_val = thr
            sig = int(score >= thr_val)
            ts_evt = ts_vals.iloc[idx]
            now = pd.Timestamp.utcnow()

            if sig == 1:
                if last_ts is None or (ts_evt.to_pydatetime().timestamp() - last_ts >= COOLDOWN_SECS):
                    # Señal real
                    outputs.append({
                        "symbol": symbol,
                        "ts_min": ts_evt,
                        "side": side,
                        "score": float(score),
                        "signal": 1,
                        "is_shadow": 0,
                        "created_at": now
                    })
                    last_ts = ts_evt.to_pydatetime().timestamp()
                else:
                    # Shadow signal (en cooldown)
                    outputs.append({
                        "symbol": symbol,
                        "ts_min": ts_evt,
                        "side": side,
                        "score": float(score),
                        "signal": 1,
                        "is_shadow": 1,
                        "created_at": now
                    })
        if last_ts:
            state.update({"last_ts": last_ts})
        if outputs:
            yield pd.DataFrame(outputs)

# COMMAND ----------
# Lectura y scoring en streaming
if spark._jsparkSession.catalog().tableExists(STREAM_FEATURES_TBL):
    features_stream = (spark.table(STREAM_FEATURES_TBL)
                       .withWatermark("ts_min", "10 minutes"))

    state_schema = StructType([StructField("last_ts", DoubleType())])

    signals_stream = (features_stream
                      .groupByKey(lambda row: (row.symbol, str(int(row.trade_is_buy))), schema=StructType([
                          StructField("symbol", StringType()),
                          StructField("side", StringType())
                      ]))
                      .mapGroupsWithState(
                          stateSchema=state_schema,
                          outputSchema=signal_schema,
                          stateTimeout=GroupStateTimeout.ProcessingTimeTimeout
                      )(cooldown_update))

    # Escribir señales (reales + shadow) a GOLD
    (signals_stream
     .writeStream
     .option("checkpointLocation", "/mnt/adls2/checkpoints/whale_signals_with_cooldown")
     .outputMode("append")
     .toTable(SIGNALS_TBL))

    # Shadow: filtrar y escribir aparte
    (signals_stream.filter("is_shadow = 1")
     .writeStream
     .option("checkpointLocation", "/mnt/adls2/checkpoints/whale_shadow_signals")
     .outputMode("append")
     .toTable(SHADOW_TBL))
else:
    print(f"[INFO] No existe {STREAM_FEATURES_TBL}. Revisa tu pipeline de features en streaming.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10) Notas de operación
# MAGIC 
# MAGIC - El estado de cooldown se mantiene en memoria por clave `(symbol, side)` y se actualiza con `mapGroupsWithState`.
# MAGIC - Para persistir y recuperar estado tras reinicios, Databricks Structured Streaming ya gestiona el **state store** en el checkpoint. 
# MAGIC - En producción, alternativa más realista: **usar un topic compactado en Kafka** para mantener `last_ts` por `(symbol, side)` y asegurar baja latencia.
# MAGIC - Señales reales → `{SIGNALS_TBL}`; shadow → `{SHADOW_TBL}`.
# MAGIC - Conectar `{SIGNALS_TBL}` a **Inference Tables** para monitorización de *precision*, *coverage* y *drift*.
# MAGIC - Este notebook se ejecuta como **streaming job 24/7** en **Databricks Workflows**.

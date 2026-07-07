# Databricks notebook source
import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone

from delta.tables import DeltaTable
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, TimestampType, 
    DoubleType, BooleanType, IntegerType, MapType, DateType
)

# COMMAND ----------

DELTA_PATH_KLINES = (
    "abfss://datalake@statfmprod.dfs.core.windows.net/silver/crypto_klines")
DELTA_PATH_MARK = (
    "abfss://datalake@statfmprod.dfs.core.windows.net/silver/crypto_mark_klines")
DELTA_PATH_INDEX = (
    "abfss://datalake@statfmprod.dfs.core.windows.net/silver/crypto_index_klines")
DELTA_PATH_PREMIUM = (
    "abfss://datalake@statfmprod.dfs.core.windows.net/silver/crypto_premium_klines")
DELTA_PATH_OI_5M = (
    "abfss://datalake@statfmprod.dfs.core.windows.net/silver/crypto_open_interest_5m")
DELTA_PATH_TAKER_5M = (
    "abfss://datalake@statfmprod.dfs.core.windows.net/silver/crypto_taker_5m")
DELTA_PATH_FUNDING = (
    "abfss://datalake@statfmprod.dfs.core.windows.net/silver/crypto_funding")

# COMMAND ----------

BATCH_SIZE = 500
interval = "1m"
limit_kl = 1000           # /fapi/v1/klines
limit_px_kl = 1500        # /fapi/v1/*PriceKlines
days = 30

SLEEP = 0.15  # throttling leve para evitar 429
TIMEOUT = 15

# Límites por llamada
KL_LIMIT = 1000            # /fapi/v1/klines
PRICE_KL_LIMIT = 1500      # /fapi/v1/*PriceKlines (mark/index/premium)
STAT_LIMIT = 500           # /futures/data/* (OI, taker) — 30 días

catalog = "dbrtfmprod"
schema  = "whale_trades"

end_time = datetime.utcnow().replace(tzinfo=timezone.utc)
start_time = end_time - timedelta(days=days)

START_MS = int(start_time.timestamp() * 1000)
END_MS   = int(end_time.timestamp() * 1000)

# COMMAND ----------

schema_klines_full = StructType([
    StructField("open_time", TimestampType(), False),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", DoubleType(), True),
    StructField("close_time", TimestampType(), False),
    StructField("quote_asset_volume", DoubleType(), True),
    StructField("n_trades", IntegerType(), True),
    StructField("taker_base_vol", DoubleType(), True),
    StructField("taker_quote_vol", DoubleType(), True),
    StructField("symbol", StringType(), False),
    StructField("date", DateType(), False)
])

# Para Mark/Index/Premium solo necesitamos OHLC + close_time
schema_px_klines_ohlc = StructType([
    StructField("open_time", TimestampType(), False),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("close_time", TimestampType(), False),
    StructField("symbol", StringType(), False),
    StructField("date", DateType(), False)
])

schema_oi = StructType([
    StructField("ts", TimestampType(), False),
    StructField("oi", DoubleType(), True),
    StructField("oi_value_usd", DoubleType(), True),
    StructField("symbol", StringType(), False),
    StructField("date", DateType(), False)
])

schema_taker = StructType([
    StructField("ts", TimestampType(), False),
    StructField("buy_vol", DoubleType(), True),
    StructField("sell_vol", DoubleType(), True),
    StructField("buy_sell_ratio", DoubleType(), True),
    StructField("symbol", StringType(), False),
    StructField("date", DateType(), False)
])

schema_funding = StructType([
    StructField("funding_time", TimestampType(), False),
    StructField("funding_rate", DoubleType(), True),
    StructField("symbol", StringType(), False),
    StructField("date", DateType(), False)
])




# COMMAND ----------

def _req(url: str, params: dict):
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    time.sleep(SLEEP)
    return data

def _empty_sdf(schema):
    return spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)

def _write_delta(sdf, path, keys, partitions=("symbol", "date"), mode="append"):
    if sdf.limit(1).count() == 0:
        return
    (sdf.write.format("delta")
        .mode(mode)
        .partitionBy(*partitions)
        .save(path))

# COMMAND ----------

def get_klines_last_month(symbol: str):
    """ OHLCV 1m — /fapi/v1/klines """
    all_rows = []
    cur = START_MS
    while cur < END_MS:
        params = {"symbol": symbol, "interval": "1m", "startTime": cur, "limit": KL_LIMIT}
        try:
            batch = _req("https://fapi.binance.com/fapi/v1/klines", params)
        except Exception as e:
            print(f"[KLINES][{symbol}] ERROR: {e}"); break
        if not batch: break
        all_rows.extend(batch)
        cur = batch[-1][0] + 60_000

    if not all_rows:
        return _empty_sdf(schema_klines_full)

    df = pd.DataFrame(all_rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","n_trades",
        "taker_base_vol","taker_quote_vol","ignore"
    ])
    df = df.drop(columns=["ignore"])
    df["symbol"] = symbol
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["date"] = df["open_time"].dt.date
    df = df.astype({
        "open":"float64","high":"float64","low":"float64","close":"float64",
        "volume":"float64","quote_asset_volume":"float64",
        "taker_base_vol":"float64","taker_quote_vol":"float64",
        "n_trades":"int32"
    })
    return spark.createDataFrame(df, schema=schema_klines_full)

# COMMAND ----------

def get_mark_klines_last_month(symbol: str):
    """ Mark Price 1m — /fapi/v1/markPriceKlines """
    all_rows = []
    cur = START_MS
    while cur < END_MS:
        params = {"symbol": symbol, "interval": "1m", "startTime": cur, "limit": PRICE_KL_LIMIT}
        try:
            batch = _req("https://fapi.binance.com/fapi/v1/markPriceKlines", params)
        except Exception as e:
            print(f"[MARK][{symbol}] ERROR: {e}"); break
        if not batch: break
        all_rows.extend(batch)
        cur = batch[-1][0] + 60_000

    if not all_rows:
        return _empty_sdf(schema_px_klines_ohlc)

    tmp = pd.DataFrame(all_rows)
    df = tmp.iloc[:, [0,1,2,3,4,6]].copy()
    df.columns = ["open_time","open","high","low","close","close_time"]
    df["symbol"] = symbol
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["date"] = df["open_time"].dt.date
    df = df.astype({"open":"float64","high":"float64","low":"float64","close":"float64"})
    return spark.createDataFrame(df, schema=schema_px_klines_ohlc)

# COMMAND ----------

def get_index_klines_last_month(symbol: str):
    """Index Price 1m — /fapi/v1/indexPriceKlines (usa 'pair', no 'symbol')."""
    all_rows = []
    cur = START_MS
    while cur < END_MS:
        params = {
            "pair": symbol,          # <- CLAVE: usar 'pair'
            "interval": "1m",
            "startTime": cur,
            "limit": PRICE_KL_LIMIT  # máx 1500
        }
        try:
            batch = _req("https://fapi.binance.com/fapi/v1/indexPriceKlines", params)
        except Exception as e:
            print(f"[INDEX][{symbol}] ERROR: {e}"); break
        if not batch:
            break
        all_rows.extend(batch)
        cur = batch[-1][0] + 60_000  # avanza 1 min

    if not all_rows:
        return _empty_sdf(schema_px_klines_ohlc)

    tmp = pd.DataFrame(all_rows)
    df = tmp.iloc[:, [0,1,2,3,4,6]].copy()  # openTime, O, H, L, C, closeTime
    df.columns = ["open_time","open","high","low","close","close_time"]
    df["symbol"] = symbol
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["date"] = df["open_time"].dt.date
    df = df.astype({"open":"float64","high":"float64","low":"float64","close":"float64"})
    return spark.createDataFrame(df, schema=schema_px_klines_ohlc)

# COMMAND ----------

def get_premium_klines_last_month(symbol: str):
    """ Premium Index 1m — /fapi/v1/premiumIndexKlines """
    all_rows = []
    cur = START_MS
    while cur < END_MS:
        params = {"symbol": symbol, "interval": "1m", "startTime": cur, "limit": PRICE_KL_LIMIT}
        try:
            batch = _req("https://fapi.binance.com/fapi/v1/premiumIndexKlines", params)
        except Exception as e:
            print(f"[PREMIUM][{symbol}] ERROR: {e}"); break
        if not batch: break
        all_rows.extend(batch)
        cur = batch[-1][0] + 60_000

    if not all_rows:
        return _empty_sdf(schema_px_klines_ohlc)

    tmp = pd.DataFrame(all_rows)
    df = tmp.iloc[:, [0,1,2,3,4,6]].copy()
    df.columns = ["open_time","open","high","low","close","close_time"]
    df["symbol"] = symbol
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["date"] = df["open_time"].dt.date
    df = df.astype({"open":"float64","high":"float64","low":"float64","close":"float64"})
    return spark.createDataFrame(df, schema=schema_px_klines_ohlc)

# COMMAND ----------

def get_oi_5m_last_month(symbol: str):
    """ Open Interest 5m — /futures/data/openInterestHist """
    all_rows = []
    cur = START_MS
    while cur < END_MS:
        params = {
            "symbol": symbol,
            "period": "5m",
            "limit": STAT_LIMIT,
            "startTime": cur,
            "endTime": END_MS
        }
        try:
            batch = _req("https://fapi.binance.com/futures/data/openInterestHist", params)
        except Exception as e:
            print(f"[OI][{symbol}] ERROR: {e}"); break
        if not batch: break
        all_rows.extend(batch)
        last_ts = int(batch[-1].get("timestamp", batch[-1].get("time")))
        cur = last_ts + 5*60_000

    if not all_rows:
        return _empty_sdf(schema_oi)

    df = pd.DataFrame(all_rows)
    ts_col = "timestamp" if "timestamp" in df.columns else ("time" if "time" in df.columns else None)
    if ts_col is None:
        return _empty_sdf(schema_oi)

    df = df[[ts_col, "sumOpenInterest", "sumOpenInterestValue"]].copy()
    df.columns = ["ts", "oi", "oi_value_usd"]
    df["symbol"] = symbol
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["date"] = df["ts"].dt.date
    df = df.astype({"oi":"float64","oi_value_usd":"float64"})
    return spark.createDataFrame(df, schema=schema_oi)

# COMMAND ----------

def get_funding_last_month(symbol: str):
    """ Funding Rate (~8h) — /fapi/v1/fundingRate """
    try:
        batch = _req("https://fapi.binance.com/fapi/v1/fundingRate", {
            "symbol": symbol,
            "startTime": START_MS,
            "endTime": END_MS,
            "limit": 1000
        })
    except Exception as e:
        print(f"[FUNDING][{symbol}] ERROR: {e}")
        return _empty_sdf(schema_funding)

    if not batch:
        return _empty_sdf(schema_funding)

    df = pd.DataFrame(batch)
    if not {"fundingTime","fundingRate"}.issubset(df.columns):
        return _empty_sdf(schema_funding)

    df = df[["fundingTime","fundingRate"]].copy()
    df.columns = ["funding_time","funding_rate"]
    df["symbol"] = symbol
    df["funding_time"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df["date"] = df["funding_time"].dt.date
    return spark.createDataFrame(df, schema=schema_funding)

# COMMAND ----------

def get_taker_5m_last_month(symbol: str):
    """
    Taker Buy/Sell 5m — /futures/data/takerlongshortRatio (USDⓈ-M)
    Descarga el último mes paginando de forma segura.
    """
    # Alinear START_MS al múltiplo inferior de 5 minutos
    FIVE_MIN_MS = 5 * 60_000
    start_aligned = START_MS - (START_MS % FIVE_MIN_MS)

    all_rows = []
    cur = start_aligned
    last_progress = None

    while cur < END_MS:
        params = {
            "symbol": symbol,
            "period": "5m",
            "limit": min(STAT_LIMIT, 500),  # máx 500
            "startTime": cur                 # SIN endTime para paginar hacia delante
        }
        try:
            batch = _req("https://fapi.binance.com/futures/data/takerlongshortRatio", params)
        except Exception as e:
            print(f"[TAKER][{symbol}] ERROR: {e}")
            break

        if not batch:
            break

        all_rows.extend(batch)

        # timestamp viene en ms (string o int)
        last_ts = int(batch[-1].get("timestamp") or batch[-1].get("time"))

        # Proteger de bucle si la API repite la última página
        if last_progress is not None and last_ts <= last_progress:
            cur += FIVE_MIN_MS
        else:
            cur = last_ts + FIVE_MIN_MS
            last_progress = last_ts

        if cur >= END_MS:
            break

    if not all_rows:
        return _empty_sdf(schema_taker)

    df = pd.DataFrame(all_rows)

    # Validación mínima de columnas esperadas
    ts_col = "timestamp" if "timestamp" in df.columns else ("time" if "time" in df.columns else None)
    needed = {"buyVol", "sellVol", "buySellRatio"}
    if ts_col is None or not needed.issubset(df.columns):
        print(f"[TAKER][{symbol}] Respuesta inesperada: cols={list(df.columns)}")
        return _empty_sdf(schema_taker)

    # Selección/renombrado y tipos
    df = df[[ts_col, "buyVol", "sellVol", "buySellRatio"]].copy()
    df.columns = ["ts", "buy_vol", "sell_vol", "buy_sell_ratio"]
    df["symbol"] = symbol
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["date"] = df["ts"].dt.date
    for c in ["buy_vol", "sell_vol", "buy_sell_ratio"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Quitar solapes/duplicados que puede devolver la API
    df.drop_duplicates(subset=["symbol", "ts"], inplace=True)

    return spark.createDataFrame(df, schema=schema_taker)

# COMMAND ----------

symbols = [ r["symbol"] for r in spark.sql(
            """SELECT DISTINCT symbol
               FROM dbrtfmprod.whale_trades.silver__p95_trades
               ORDER BY symbol
               """).collect()
]

print(f"[INFO] {len(symbols)} símbolos")

for symbol in symbols:
    print(f"\n=== {symbol} ===")

    sdf_kl = get_klines_last_month(symbol)
    print(sdf_kl)
    _write_delta(sdf_kl, DELTA_PATH_KLINES, keys=["symbol"])

    sdf_mark = get_mark_klines_last_month(symbol)
    display(sdf_mark)
    _write_delta(sdf_mark, DELTA_PATH_MARK, keys=["symbol"])

    sdf_index = get_index_klines_last_month(symbol)
    print(sdf_index)
    _write_delta(sdf_index, DELTA_PATH_INDEX, keys=["symbol"])

    sdf_premium = get_premium_klines_last_month(symbol)
    print(sdf_premium)
    _write_delta(sdf_premium, DELTA_PATH_PREMIUM, keys=["symbol"])

    sdf_oi = get_oi_5m_last_month(symbol)
    print(sdf_oi)
    _write_delta(sdf_oi, DELTA_PATH_OI_5M, keys=["symbol"])

    sdf_taker = get_taker_5m_last_month(symbol)
    print(sdf_taker)
    _write_delta(sdf_taker, DELTA_PATH_TAKER_5M, keys=["symbol"])

    sdf_funding = get_funding_last_month(symbol)
    print(sdf_funding)
    _write_delta(sdf_funding, DELTA_PATH_FUNDING, keys=["symbol"])

print("\n[DONE] Descarga y guardado del último mes completados.")

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# === Definición de tablas → ubicación ===
tables = {
    "silver__klines_1m"        : DELTA_PATH_KLINES,
    "silver__mark_klines_1m"   : DELTA_PATH_MARK,
    "silver__index_klines_1m"  : DELTA_PATH_INDEX,
    "silver__premium_klines_1m": DELTA_PATH_PREMIUM,
    "silver__open_interest_5m" : DELTA_PATH_OI_5M,
    "silver__taker_5m"         : DELTA_PATH_TAKER_5M,
    "silver__funding_8h"       : DELTA_PATH_FUNDING,
}

# === Crear tablas externas USING DELTA con LOCATION ===
for tbl, path in tables.items():
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {catalog}.{schema}.{tbl}
        USING DELTA
        LOCATION '{path}'
    """)
    print(f"[OK] {catalog}.{schema}.{tbl} -> {path}")

# COMMAND ----------

display(spark.sql("""
    SELECT COUNT(*), symbol
    FROM dbrtfmprod.whale_trades.silver__mark_klines_1m
    GROUP BY symbol
                  """))
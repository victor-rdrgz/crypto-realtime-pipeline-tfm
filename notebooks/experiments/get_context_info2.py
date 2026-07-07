# Databricks notebook source
import requests
import pandas as pd
import time
from datetime import datetime, timedelta

from pyspark.sql import types as T
from pyspark.sql import functions as F

# COMMAND ----------

# =========================
# CONFIG
# =========================
CATALOG = "dbrtfmprod"
SCHEMA  = "whale_trades"
TRADES  = f"{CATALOG}.{SCHEMA}.silver__p95_trades"
OUTTAB  = f"{CATALOG}.{SCHEMA}.gold__big_trades_price_evolution"

KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"               # 1m
OI_URL     = "https://fapi.binance.com/futures/data/openInterestHist" # 5m
FUND_URL   = "https://fapi.binance.com/fapi/v1/fundingRate"          # 8h
TIMEOUT    = 10
PAGE_LIMIT = 1000
PAUSE      = 0.02
MAX_THREADS= 8

# COMMAND ----------

# Carga de trades
all_trades = spark.table("dbrtfmprod.whale_trades.silver__p95_trades")

tr = (
  all_trades
    # TIPOS CORRECTOS
    .withColumn("price", F.col("price").cast("double"))
    .withColumn("qty",   F.col("qty").cast("double"))
    .withColumn("trade_value", F.col("trade_value").cast("double"))
    # DERIVADAS
    .withColumn("notional", (F.col("price") * F.col("qty")))
    .withColumn("ts_s", (F.col("timestamp")/1000).cast("timestamp"))
    .withColumn("t0",   F.date_trunc("minute", F.col("ts_s")))
    # LIMPIEZA BÁSICA
    .filter(F.col("price").isNotNull() & F.col("qty").isNotNull())
    .select("symbol","agg_trade_id","price","qty","first_trade_id","last_trade_id",
            "ts_s","timestamp","is_buyer_maker","date","trade_value","notional","t0")
)

tr.printSchema()

all_trades.printSchema()

# COMMAND ----------


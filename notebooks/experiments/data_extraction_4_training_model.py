# Databricks notebook source
from datetime import datetime, timezone, timedelta
from pyspark.sql import functions as F, types as T

# COMMAND ----------

CATALOG = "dbrtfmprod"
SCHEMA  = "whale_trades"
T_SYMBOLS = f"{CATALOG}{SCHEMA}.silver__p95_trades"
T_TRADES  = f"{CATALOG}.{DB}.silver__agg_trades"

# Ventana dinámica
DAYS_BACK = 45
# Opcional: restar unos segundos para evitar ticks "en vuelo"
SAFETY_SECONDS = 30

END_DT   = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=SAFETY_SECONDS)
START_DT = END_DT - timedelta(days=DAYS_BACK)

START = START_DT.isoformat()
END   = END_DT.isoformat()

STEP_HOURS = 6

# COMMAND ----------

symbols = [ r["symbol"] for r in spark.sql(
            f"""SELECT DISTINCT symbol
               FROM {T_SYMBOLS}
               ORDER BY symbol
               """).collect()
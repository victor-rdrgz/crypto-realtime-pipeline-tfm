# Databricks notebook source
from pyspark.sql import functions as F, types as T

# Rutas Silver
SILVER_FUTURES_PATH = "abfss://datalake@statfmprod.dfs.core.windows.net/silver/criptolist/futures"
SILVER_SPOT_PATH    = "abfss://datalake@statfmprod.dfs.core.windows.net/silver/criptolist/spot"

# Rutas gold y nombres de tabla
GOLD_PATH_CRIPTOLIST_CURRENT = "abfss://datalake@statfmprod.dfs.core.windows.net/gold/criptolist/current"
GOLD_PATH_CRIPTOLIST_QUARANTINE = "abfss://datalake@statfmprod.dfs.core.windows.net/gold/criptolist/quarantine"

GOLD_TABLE_CRIPTOLIST_CURRENT = "dbrtfmprod.whale_trades.gold__criptolist_current"
GOLD_TABLE_CRIPTOLIST_QUARANTINE = "dbrtfmprod.whale_trades.gold__criptolist_quarantine"

GOLD_COLUMNS_CRIPTOLIST = [
    "symbol", "base_coin", "quote_coin", "settlement_coin", "status",
    "max_leverage", "tick_size", "min_order_qty", "max_order_qty",
    "processed_at", "_ingested_at", "price_precision", "exchange"
]



# COMMAND ----------

df_silver_fut = spark.read.format("delta").load(SILVER_FUTURES_PATH)
df_silver_spo = spark.read.format("delta").load(SILVER_SPOT_PATH)


# COMMAND ----------

expected_futures = set(GOLD_COLUMNS_CRIPTOLIST)
expected_spot = expected_futures - {"settlement_coin", "max_leverage", "price_precision"}

missing_fut = sorted(list(expected_futures - set(df_silver_fut.columns)))
missing_spo = sorted(list(expected_spot - set(df_silver_spo.columns)))

errors = []
if missing_fut:
    errors.append(f"FUTURES: faltan columnas {missing_fut}")
if missing_spo:
    errors.append(f"SPOT:    faltan columnas {missing_spo}")

if errors:
    raise Exception("[SCHEMA MISMATCH] " + " | ".join(errors))


# COMMAND ----------

df_silver_fut = df_silver_fut.withColumn("market_type", F.lit("futures"))
df_silver_spo = df_silver_spo.withColumn("market_type", F.lit("spot"))

df_silver_spo_fixed = (
    df_silver_spo
    .withColumn("settlement_coin", F.lit(None).cast("string"))
    .withColumn("max_leverage", F.lit(None).cast("double"))
    .withColumn("price_precision", F.lit(None).cast("int"))
)

df_clean = (
    df_silver_fut.select(*GOLD_COLUMNS_CRIPTOLIST)
    .unionByName(df_silver_spo_fixed.select(*GOLD_COLUMNS_CRIPTOLIST))
    .withColumn("market_type", F.when(F.col("settlement_coin").isNull(), "spot").otherwise("futures"))
)


# COMMAND ----------

valid_cond = (
    F.col("base_coin").isNotNull() &
    F.col("quote_coin").isNotNull() &
    (F.col("base_coin") != F.col("quote_coin")) &
    F.col("tick_size").isNotNull()
)

df_gold_criptolist_current = df_clean.filter(valid_cond).select(*GOLD_COLUMNS_CRIPTOLIST, "market_type")

df_gold_criptolist_quarantine = (
    df_clean
    .filter(~valid_cond)
    .withColumn("reason", F.lit("DQ_FAILED"))
    .withColumn("payload", F.to_json(F.struct(*df_clean.columns)))
    .select("reason", "exchange", "market_type", "payload", "processed_at")
)


# COMMAND ----------

(df_gold_criptolist_current.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("exchange", "market_type")
    .save(GOLD_PATH_CRIPTOLIST_CURRENT)
)

(df_gold_criptolist_quarantine.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("exchange", "market_type")
    .save(GOLD_PATH_CRIPTOLIST_QUARANTINE)
)


# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {GOLD_TABLE_CRIPTOLIST_CURRENT}
USING DELTA
LOCATION '{GOLD_PATH_CRIPTOLIST_CURRENT}'
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {GOLD_TABLE_CRIPTOLIST_QUARANTINE}
USING DELTA
LOCATION '{GOLD_PATH_CRIPTOLIST_QUARANTINE}'
""")

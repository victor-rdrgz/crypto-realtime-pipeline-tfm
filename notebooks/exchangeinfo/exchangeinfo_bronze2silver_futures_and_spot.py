# Databricks notebook source
from pyspark.sql import functions as F

# COMMAND ----------

exchanges= [
    "okx",
    "binance",
    "bybit"
]

base_bronze_path = "abfss://datalake@statfmprod.dfs.core.windows.net/bronze/exchangeinfo" 
base_silver_path = "abfss://datalake@statfmprod.dfs.core.windows.net/silver/criptolist"
checkpoint_path = "abfss://datalake@statfmprod.dfs.core.windows.net/checkpoints"

processed_at = F.current_timestamp()

# COMMAND ----------

# MAGIC %md
# MAGIC # PROCESAMIENTO DE CRIPTOMONEDAS DE MERCADOS DE FUTUROS

# COMMAND ----------

# -------------------
# OKX
# -------------------
df_bronze_okx = (
    spark.readStream
    .option("cloudFiles.format", "delta")
    .load(f"{base_bronze_path}/okx_futures")
)

df_okx = (
    df_bronze_okx
    .select(F.explode(F.col("data")).alias("item"), F.col("_ingested_at"))
    .filter(F.col("item.instType") == "SWAP")
    .select(
        F.col("item.instId").alias("symbol"),
        F.split(F.col("item.instId"), "-").getItem(0).alias("base_coin"),
        F.split(F.col("item.instId"), "-").getItem(1).alias("quote_coin"),
        F.col("item.settleCcy").alias("settlement_coin"),
        F.col("item.state").alias("status"),

        F.when(F.col("item.lever") == "", F.lit(None))
         .otherwise(F.col("item.lever").cast("double"))
         .alias("max_leverage"),

        F.when(F.col("item.tickSz") == "", F.lit(None))
         .otherwise(F.col("item.tickSz").cast("double"))
         .alias("tick_size"),

        F.when(F.col("item.lotSz") == "", F.lit(None))
         .otherwise(F.col("item.lotSz").cast("double"))
         .alias("step_size"),

        F.when(F.col("item.minSz") == "", F.lit(None))
         .otherwise(F.col("item.minSz").cast("double"))
         .alias("min_order_qty"),

        F.when(F.col("item.maxLmtSz") == "", F.lit(None))
         .otherwise(F.col("item.maxLmtSz").cast("double"))
         .alias("max_order_qty"),

        F.from_unixtime((F.col("item.listTime") / 1000).cast("long")).alias("listed_at"),
        F.col("_ingested_at"),
        F.lit(processed_at).alias("processed_at"),
        F.lit(None).cast("int").alias("price_precision"),
        F.lit("okx").alias("exchange"),
    )
)


df_okx = (
    df_okx
        .withColumn("status",F.when(df_okx["status"] != "live", "break").otherwise(df_okx["status"]))
)



# COMMAND ----------

# -------------------
# BINANCE
# -------------------
df_bronze_binance = (
    spark.readStream
    .option("cloudFiles.format", "delta")
    .load(f"{base_bronze_path}/binance_futures")
)

df_binance = (
    df_bronze_binance
    .select(F.explode(F.col("symbols")).alias("item"))
    .filter(F.col("item.contractType") == "PERPETUAL")
    .select(
        F.col("item.symbol").alias("symbol"),
        F.col("item.baseAsset").alias("base_coin"),
        F.col("item.quoteAsset").alias("quote_coin"),
        F.col("item.marginAsset").alias("settlement_coin"),
        F.col("item.status").alias("status"),
        F.lit(None).cast("double").alias("max_leverage"),
        F.expr("filter(item.filters, x -> x.filterType = 'PRICE_FILTER')[0].tickSize").cast("double").alias("tick_size"),
        F.expr("filter(item.filters, x -> x.filterType = 'LOT_SIZE')[0].stepSize").cast("double").alias("step_size"),
        F.expr("filter(item.filters, x -> x.filterType = 'LOT_SIZE')[0].minQty").cast("double").alias("min_order_qty"),
        F.expr("filter(item.filters, x -> x.filterType = 'LOT_SIZE')[0].maxQty").cast("double").alias("max_order_qty"),
        F.lit(None).cast("timestamp").alias("listed_at"),
        F.lit(processed_at).alias("processed_at"),
        F.lit(None).cast("timestamp").alias("_ingested_at"),
        F.col("item.pricePrecision").cast("int").alias("price_precision"),
        F.lit("binance").alias("exchange"),
    )
)

df_binance = (
    df_binance
        .withColumn("status",F.when(df_binance["status"] == "TRADING", "live").otherwise("break"))

)


# COMMAND ----------

# -------------------
# BYBIT
# -------------------
df_bronze_bybit = (
    spark.readStream
    .option("cloudFiles.format", "delta")
    .load(f"{base_bronze_path}/bybit_futures")
)

df_bybit = (
    df_bronze_bybit
    .select(F.explode(F.col("result.list")).alias("item"), F.col("_ingested_at"))
    .select(
        F.col("item.symbol").alias("symbol"),
        F.col("item.baseCoin").alias("base_coin"),
        F.col("item.quoteCoin").alias("quote_coin"),
        F.col("item.settleCoin").alias("settlement_coin"),
        F.col("item.status").alias("status"),
        F.col("item.leverageFilter.maxLeverage").cast("double").alias("max_leverage"),
        F.col("item.priceFilter.tickSize").cast("double").alias("tick_size"),
        F.col("item.lotSizeFilter.qtyStep").cast("double").alias("step_size"),
        F.col("item.lotSizeFilter.minOrderQty").cast("double").alias("min_order_qty"),
        F.col("item.lotSizeFilter.maxOrderQty").cast("double").alias("max_order_qty"),
        F.from_unixtime((F.col("item.launchTime") / 1000).cast("long")).alias("listed_at"),
        F.col("_ingested_at"),
        F.lit(processed_at).alias("processed_at"),
        F.col("item.priceScale").cast("int").alias("price_precision"),
        F.lit("bybit").alias("exchange"),
    )
)

df_bybit = df_bybit.withColumn(
    "status", F.when(F.col("status") == "Trading", "live").otherwise("break")
)


# COMMAND ----------

import traceback
from pyspark.sql import functions as F

try:
    # -------------------
    # UNIÓN Y ESCRITURA
    # -------------------

    df_combined = df_binance.unionByName(df_bybit).unionByName(df_okx)

    def overwrite_to_silver(batch_df, batch_id):
        (
            batch_df.write
            .format("delta")
            .mode("overwrite")
            .partitionBy("exchange")
            .option("mergeSchema", "false")
            .option("path", f"{base_silver_path}/futures")
            .save()
        )

    (df_combined.writeStream
        .foreachBatch(overwrite_to_silver)
        .option("checkpointLocation", f"{checkpoint_path}/silver/criptolist/futures")
        .queryName("silver_criptolist_futures")
        .trigger(availableNow=True)
        .start()
        .awaitTermination()
    )

except Exception as e:
    print("❌ Error dentro de overwrite_to_silver():")
    traceback.print_exc()
    raise e


# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS dbrtfmprod.whale_trades.silver__criptolist_futures
# MAGIC USING DELTA
# MAGIC LOCATION 'abfss://datalake@statfmprod.dfs.core.windows.net/silver/criptolist/futures';

# COMMAND ----------

# MAGIC %md
# MAGIC # PROCESAMIENTO DE CRIPTOMONEDAS DE MERCADOS SPOT

# COMMAND ----------

# -------------------
# OKX
# -------------------
df_bronze_okx = (
    spark.readStream
    .option("cloudFiles.format", "delta")
    .load(f"{base_bronze_path}/okx_spot")
)

df_okx = (
    df_bronze_okx
    .select(F.explode(F.col("data")).alias("item"), F.col("_ingested_at"))
    .select(
        F.col("item.instId").alias("symbol"),
        F.col("item.baseCcy").alias("base_coin"),
        F.col("item.quoteCcy").alias("quote_coin"),
        F.col("item.state").alias("status"),

        F.when(F.col("item.minSz") == "", F.lit(None))
         .otherwise(F.col("item.minSz").cast("double"))
         .alias("min_order_qty"),

        F.when(F.col("item.maxLmtSz") == "", F.lit(None))
         .otherwise(F.col("item.maxLmtSz").cast("double"))
         .alias("max_order_qty"),

        F.when(F.col("item.tickSz") == "", F.lit(None))
         .otherwise(F.col("item.tickSz").cast("double"))
         .alias("tick_size"),

        F.col("_ingested_at"),
        F.lit(processed_at).alias("processed_at"),
    )
    .withColumn("exchange", F.lit("okx"))
)

df_okx = (
    df_okx
        .withColumn("status",F.when(df_okx["status"] != "live", "break").otherwise(df_okx["status"]))
        .withColumn("symbol", F.concat_ws("-", "base_coin", "quote_coin"))
)
display(df_okx)

# COMMAND ----------

# -------------------
# BINANCE
# -------------------

df_bronze_binance = (
    spark.readStream
    .option("cloudFiles.format", "delta")
    .load(f"{base_bronze_path}/binance_spot")
)

df_binance = (
    df_bronze_binance
    .withColumn("symbol_data", F.explode(F.col("symbols")))
    .select(
        F.col("symbol_data.symbol").alias("symbol"),
        F.col("symbol_data.baseAsset").alias("base_coin"),
        F.col("symbol_data.quoteAsset").alias("quote_coin"),
        F.col("symbol_data.status").alias("status"),
        F.expr("filter(symbol_data.filters, x -> x.filterType = 'LOT_SIZE')[0].minQty").cast("double").alias("min_order_qty"),
        F.expr("filter(symbol_data.filters, x -> x.filterType = 'LOT_SIZE')[0].maxQty").cast("double").alias("max_order_qty"),
        F.expr("filter(symbol_data.filters, x -> x.filterType = 'PRICE_FILTER')[0].tickSize").cast("double").alias("tick_size"),
        F.col("_ingested_at"),
        F.lit(processed_at).alias("processed_at"),
        F.lit("binance").alias("exchange")
    )
)

df_binance = (
    df_binance
        .withColumn("status",F.when(df_binance["status"] == "TRADING", "live").otherwise("break"))
        .withColumn("symbol", F.concat_ws("-", "base_coin", "quote_coin"))

)
display(df_binance)

# COMMAND ----------

# -------------------
# BYBIT
# -------------------
df_bronze_bybit = (
    spark.readStream
    .option("cloudFiles.format", "delta")
    .load(f"{base_bronze_path}/bybit_spot")
)

df_bybit = (
    df_bronze_bybit
        .select(F.explode(F.col("result.list")).alias("pair"), F.col("_ingested_at"))
        .select(
            F.col("pair.symbol"),
            F.col("pair.baseCoin").alias("base_coin"),
            F.col("pair.quoteCoin").alias("quote_coin"),
            F.col("pair.status"),
            F.col("pair.lotSizeFilter.minOrderQty").cast("double").alias("min_order_qty"),
            F.col("pair.lotSizeFilter.maxOrderQty").cast("double").alias("max_order_qty"),
            F.col("pair.priceFilter.tickSize").cast("double").alias("tick_size"),
            F.col("_ingested_at"),
            F.lit(processed_at).alias("processed_at"),
            F.lit("bybit").alias("exchange")
        )
    )

df_bybit = (
    df_bybit
        .withColumn("status", F.when(df_bybit["status"] == "Trading", "live").otherwise("break"))
        .withColumn("symbol", F.concat_ws("-", "base_coin", "quote_coin"))
)

display(df_bybit)

# COMMAND ----------

import traceback

try:
    # -------------------
    # UNIÓN Y ESCRITURA
    # -------------------

    df_combined = df_binance.unionByName(df_bybit).unionByName(df_okx)

    def overwrite_to_silver(batch_df, batch_id):
        (
            batch_df.write
            .format("delta")
            .mode("overwrite")
            .partitionBy("_ingested_at")
            .option("mergeSchema", "false")
            .option("path", f"{base_silver_path}/spot")
            .save()
        )

    (df_combined.writeStream
        .foreachBatch(overwrite_to_silver)
        .option("checkpointLocation", f"{checkpoint_path}/silver/criptolist/spot")
        .queryName("silver_criptolist")
        .trigger(availableNow=True)
        .start()
        .awaitTermination()
    )

except Exception as e:
    print("❌ Error dentro de overwrite_to_silver():")
    traceback.print_exc()
    raise e

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS dbrtfmprod.whale_trades.silver__criptolist_spot
# MAGIC USING DELTA
# MAGIC LOCATION 'abfss://datalake@statfmprod.dfs.core.windows.net/silver/criptolist/spot';
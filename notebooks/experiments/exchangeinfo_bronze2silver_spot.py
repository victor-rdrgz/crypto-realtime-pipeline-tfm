# Databricks notebook source
from pyspark.sql import functions as F

# COMMAND ----------

exchanges= [
    "okx",
    "binance",
    "bybit"
]

# base_raw_path = "abfss://datalake@statfmprod.dfs.core.windows.net/raw/exchangeinfo"
base_bronze_path = "abfss://datalake@statfmprod.dfs.core.windows.net/bronze/exchangeinfo" 
base_silver_path = "abfss://datalake@statfmprod.dfs.core.windows.net/silver"
checkpoint_path = "abfss://datalake@statfmprod.dfs.core.windows.net/checkpoints"

processed_at = F.current_timestamp()

# COMMAND ----------

# -------------------
# OKX
# -------------------
df_bronze_okx = (
    spark.readStream
    .option("cloudFiles.format", "delta")
    .load(f"{base_bronze_path}/okx")
)

df_okx = (
    df_bronze_okx
    .select(F.explode(F.col("data")).alias("item"), F.col("_ingested_at"))
    .select(
        F.col("item.instId").alias("symbol"),
        F.col("item.baseCcy").alias("base_coin"),
        F.col("item.quoteCcy").alias("quote_coin"),
        F.col("item.state").alias("status"),
        F.col("item.minSz").cast("double").alias("min_order_qty"),
        F.col("item.maxLmtSz").cast("double").alias("max_order_qty"),
        F.col("item.tickSz").cast("double").alias("tick_size"),
        F.col("_ingested_at"),
        F.lit(processed_at).alias("processed_at"),

).withColumn("exchange", F.lit("okx"))
)

df_okx = (
    df_okx
        .withColumn("status",F.when(df_okx["status"] != "live", "break").otherwise(df_okx["status"]))
        .withColumn("symbol", F.concat_ws("-", "base_coin", "quote_coin"))
)

# df_okx.select("status").distinct().show()
display(df_okx)


# COMMAND ----------

# -------------------
# BINANCE
# -------------------
'''df_binance_final = df_bronze_binance.select(
    F.col("symbol"),
    F.col("baseAsset").alias("base_coin"),
    F.col("quoteAsset").alias("quote_coin"),
    F.col("status"),
    F.expr("filter(filters, x -> x.filterType = 'LOT_SIZE')[0].minQty").cast("double").alias("min_order_qty"),
    F.expr("filter(filters, x -> x.filterType = 'LOT_SIZE')[0].maxQty").cast("double").alias("max_order_qty"),
    F.expr("filter(filters, x -> x.filterType = 'PRICE_FILTER')[0].tickSize").cast("double").alias("tick_size"),
    F.current_timestamp().alias("_ingested_at"),
    F.lit("binance").alias("exchange")
)'''

df_bronze_binance = (
    spark.readStream
    .option("cloudFiles.format", "delta")
    .load(f"{base_bronze_path}/binance")
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

# df_binance.select("status").distinct().show()
display(df_binance)

# COMMAND ----------

# -------------------
# BYBIT
# -------------------
df_bronze_bybit = (
    spark.readStream
    .option("cloudFiles.format", "delta")
    .load(f"{base_bronze_path}/bybit")
)

# display(df_bronze_bybit)
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

#df_bybit.select("status").distinct().show()
display(df_bybit)

# COMMAND ----------

import traceback

try:
    # -------------------
    # UNIÓN Y ESCRITURA
    # -------------------
    df_combined = df_binance.unionByName(df_bybit).unionByName(df_okx)

    '''df_combined.write.format("delta") \
        .mode("append") \
        .option("mergeSchema", "false") \
        .partitionBy("_ingested_at") \
        .save("abfss://datalake@statfmprod.dfs.core.windows.net/silver/exchangeinfo")'''

    def append_to_silver(batch_df, batch_id):
        (
            batch_df.write
            .format("delta")
            .mode("append")
            .partitionBy("_ingested_at")
            .option("mergeSchema", "false")
            .option("path", f"{base_silver_path}/criptolist")
            .save()
        )

    (df_combined.writeStream
        .foreachBatch(append_to_silver)
        .option("checkpointLocation", f"{checkpoint_path}/silver/criptolist/")
        .queryName("silver_criptolist")
        .trigger(availableNow=True)
        .start()
        .awaitTermination()
    )

except Exception as e:
    print("❌ Error dentro de append_to_silver():")
    traceback.print_exc()
    raise e  # para que Spark lo capture también

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS dbrtfmprod.whale_trades;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS dbrtfmprod.whale_trades.silver__cripto_list
# MAGIC USING DELTA
# MAGIC LOCATION 'abfss://datalake@statfmprod.dfs.core.windows.net/silver/criptolist';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM dbrtfmprod.whale_trades.silver__cripto_list
# MAGIC LIMIT 10;
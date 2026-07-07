# Databricks notebook source
from pyspark.sql import functions as F

# COMMAND ----------

exchanges= [
    "okx",
    "binance",
    "bybit"
]

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
    .load(f"{base_bronze_path}/okx_futures")
)

'''display(df_bronze_okx
    .select(F.explode(F.col("data")).alias("item"), F.col("_ingested_at")))'''

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
        F.col("item.lever").cast("double").alias("max_leverage"),
        F.col("item.tickSz").cast("double").alias("tick_size"),
        F.col("item.lotSz").cast("double").alias("step_size"),
        F.col("item.minSz").cast("double").alias("min_order_qty"),
        F.col("item.maxLmtSz").cast("double").alias("max_order_qty"),
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

# df_okx.select("status").distinct().show()
display(df_okx)


# COMMAND ----------

# -------------------
# BINANCE
# -------------------
df_bronze_binance = (
    spark.readStream
    .option("cloudFiles.format", "delta")
    .load(f"{base_bronze_path}/binance_futures")
)

# display(df_bronze_binance.select(F.explode(F.col("symbols")).alias("item")))

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

# df_binance.select("status").distinct().show()
display(df_binance)


# COMMAND ----------

# -------------------
# BYBIT
# -------------------
df_bronze_bybit = (
    spark.readStream
    .option("cloudFiles.format", "delta")
    .load(f"{base_bronze_path}/bybit_futures")
)

# display(df_bronze_bybit.select(F.explode("result").alias("instrument")))

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

    print(f"➡️ Guardando en ruta: {base_silver_path}/criptolist_futures")
    
    def append_to_silver(batch_df, batch_id):
        (
            batch_df.write
            .format("delta")
            .mode("append")
            .partitionBy("exchange")
            .option("mergeSchema", "false")
            .option("path", f"{base_silver_path}/criptolist_futures")
            .save()
        )
    print('antes de guardar')
    (df_combined.writeStream
        .foreachBatch(append_to_silver)
        .option("checkpointLocation", f"{checkpoint_path}/silver/criptolist_futures")
        .queryName("silver_criptolist_futures")
        .trigger(availableNow=True)
        .start()
        .awaitTermination()
    )
    print('despues de guardar')

except Exception as e:
    print("❌ Error dentro de append_to_silver():")
    traceback.print_exc()
    raise e 

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS dbrtfmprod.whale_trades;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS dbrtfmprod.whale_trades.silver__cripto_list_futures
# MAGIC USING DELTA
# MAGIC LOCATION 'abfss://datalake@statfmprod.dfs.core.windows.net/silver/criptolist_futures';

# COMMAND ----------

display(spark.sql("SELECT * FROM dbrtfmprod.whale_trades.silver__cripto_list_futures LIMIT 10;"))

# COMMAND ----------

count = spark.sql("SELECT count(*) as count FROM dbrtfmprod.whale_trades.silver__cripto_list_futures").collect()[0]['count']
print(count)

# COMMAND ----------

count = spark.sql("SELECT count(*) as count FROM dbrtfmprod.whale_trades.silver__cripto_list").collect()[0]['count']
print(count)
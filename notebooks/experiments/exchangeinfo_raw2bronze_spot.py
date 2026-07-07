# Databricks notebook source
from pyspark.sql.functions import input_file_name, current_timestamp
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType,
    IntegerType, BooleanType, ArrayType, MapType
)
from pyspark.sql import functions as F

# COMMAND ----------

# Esquema Binance
schema_binance = StructType([
    StructField("timezone", StringType(), True),
    StructField("serverTime", LongType(), True),
    StructField("rateLimits", ArrayType(
        StructType([
            StructField("rateLimitType", StringType(), True),
            StructField("interval", StringType(), True),
            StructField("intervalNum", IntegerType(), True),
            StructField("limit", IntegerType(), True)
        ])
    ), True),
    StructField("exchangeFilters", ArrayType(MapType(StringType(), StringType())), True),
    StructField("symbols", ArrayType(
        StructType([
            StructField("symbol", StringType(), True),
            StructField("status", StringType(), True),
            StructField("baseAsset", StringType(), True),
            StructField("baseAssetPrecision", IntegerType(), True),
            StructField("quoteAsset", StringType(), True),
            StructField("quotePrecision", IntegerType(), True),
            StructField("quoteAssetPrecision", IntegerType(), True),
            StructField("baseCommissionPrecision", IntegerType(), True),
            StructField("quoteCommissionPrecision", IntegerType(), True),
            StructField("orderTypes", ArrayType(StringType()), True),
            StructField("icebergAllowed", BooleanType(), True),
            StructField("ocoAllowed", BooleanType(), True),
            StructField("otoAllowed", BooleanType(), True),
            StructField("quoteOrderQtyMarketAllowed", BooleanType(), True),
            StructField("allowTrailingStop", BooleanType(), True),
            StructField("cancelReplaceAllowed", BooleanType(), True),
            StructField("amendAllowed", BooleanType(), True),
            StructField("isSpotTradingAllowed", BooleanType(), True),
            StructField("isMarginTradingAllowed", BooleanType(), True),
            StructField("filters", ArrayType(MapType(StringType(), StringType())), True),
            StructField("permissions", ArrayType(StringType()), True),
            StructField("permissionSets", ArrayType(ArrayType(StringType())), True),
            StructField("defaultSelfTradePreventionMode", StringType(), True),
            StructField("allowedSelfTradePreventionModes", ArrayType(StringType()), True)
        ])
    ), True)
])

# COMMAND ----------

schema_bybit = StructType([
    StructField("retCode", IntegerType(), True),
    StructField("retMsg", StringType(), True),
    StructField("result", StructType([
        StructField("category", StringType(), True),
        StructField("list", ArrayType(StructType([
            StructField("symbol", StringType(), True),
            StructField("baseCoin", StringType(), True),
            StructField("quoteCoin", StringType(), True),
            StructField("innovation", StringType(), True),
            StructField("status", StringType(), True),
            StructField("marginTrading", StringType(), True),
            StructField("stTag", StringType(), True),
            StructField("lotSizeFilter", StructType([
                StructField("basePrecision", StringType(), True),
                StructField("quotePrecision", StringType(), True),
                StructField("minOrderQty", StringType(), True),
                StructField("maxOrderQty", StringType(), True),
                StructField("minOrderAmt", StringType(), True),
                StructField("maxOrderAmt", StringType(), True)
            ])),
            StructField("priceFilter", StructType([
                StructField("tickSize", StringType(), True)
            ])),
            StructField("riskParameters", StructType([
                StructField("priceLimitRatioX", StringType(), True),
                StructField("priceLimitRatioY", StringType(), True)
            ]))
        ])), True)
    ])),
    StructField("retExtInfo", MapType(StringType(), StringType()), True)
])

# COMMAND ----------

# Esquema OKX
schema_okx = StructType([
    StructField("code", StringType(), True),
    StructField("data", ArrayType(StructType([
        StructField("alias", StringType(), True),
        StructField("auctionEndTime", StringType(), True),
        StructField("baseCcy", StringType(), True),
        StructField("category", StringType(), True),
        StructField("contTdSwTime", StringType(), True),
        StructField("ctMult", StringType(), True),
        StructField("ctType", StringType(), True),
        StructField("ctVal", StringType(), True),
        StructField("ctValCcy", StringType(), True),
        StructField("expTime", StringType(), True),
        StructField("futureSettlement", BooleanType(), True),
        StructField("instFamily", StringType(), True),
        StructField("instId", StringType(), True),
        StructField("instIdCode", StringType(), True),
        StructField("instType", StringType(), True),
        StructField("lever", StringType(), True),
        StructField("listTime", StringType(), True),
        StructField("lotSz", StringType(), True),
        StructField("maxIcebergSz", StringType(), True),
        StructField("maxLmtAmt", StringType(), True),
        StructField("maxLmtSz", StringType(), True),
        StructField("maxMktAmt", StringType(), True),
        StructField("maxMktSz", StringType(), True),
        StructField("maxStopSz", StringType(), True),
        StructField("maxTriggerSz", StringType(), True),
        StructField("maxTwapSz", StringType(), True),
        StructField("minSz", StringType(), True),
        StructField("openType", StringType(), True),
        StructField("optType", StringType(), True),
        StructField("quoteCcy", StringType(), True),
        StructField("ruleType", StringType(), True),
        StructField("settleCcy", StringType(), True),
        StructField("state", StringType(), True),
        StructField("stk", StringType(), True),
        StructField("tickSz", StringType(), True),
        StructField("tradeQuoteCcyList", ArrayType(StringType()), True),
        StructField("uly", StringType(), True)
    ]), True))
])

# COMMAND ----------

# Diccionario de esquemas
exchangeinfo_schemas = {
    "binance": schema_binance,
    "bybit": schema_bybit,
    "okx": schema_okx
}

# COMMAND ----------

exchanges= [
    "binance",
    "bybit",
    "okx"
]

base_raw_path = "abfss://datalake@statfmprod.dfs.core.windows.net/raw/exchangeinfo"
base_bronze_path = "abfss://datalake@statfmprod.dfs.core.windows.net/bronze/exchangeinfo"
checkpoint_path = "abfss://datalake@statfmprod.dfs.core.windows.net/checkpoints"

def get_append_fn(exchange):
    def append_to_bronze(batch_df, batch_id):
        (
            batch_df.write
            .format("delta")
            .mode("append")
            .partitionBy("_ingested_filename")
            .option("mergeSchema", "true")
            .option("path", f"{base_bronze_path}/{exchange}")
            .save()
        )
    return append_to_bronze

timestamp = current_timestamp()
for exchange in exchanges:
    df = (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .schema(exchangeinfo_schemas[exchange])
        .option("cloudFiles.schemaLocation", f"{base_raw_path}/schema/{exchange}")
        .load(f"{base_raw_path}/{exchange}")
    )
    # metadatos
    df = (
        df.withColumn("_ingested_at", timestamp)
          .withColumn("_ingested_filename", F.col("_metadata.file_path"))
    )

    (
        df.writeStream
        .foreachBatch(get_append_fn(exchange))
        .option("checkpointLocation", f"{checkpoint_path}/bronze/exchangeinfo/{exchange}")
        .queryName(f"bronze_{exchange}")
        .trigger(availableNow=True)
        .start()
        .awaitTermination()
    )
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS dbrtfmprod.whale_trades.bronze__{exchange}_exchangeinfo
        USING DELTA
        LOCATION 'abfss://datalake@statfmprod.dfs.core.windows.net/bronze/exchangeinfo/{exchange}'
    """)



# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT slice(symbols, 1, 10) AS first_10_symbols
# MAGIC FROM dbrtfmprod.whale_trades.bronze__binance_exchangeinfo
# MAGIC LIMIT 10;
# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    NumericType,
    DoubleType,
    LongType,
    TimestampType,
    ArrayType,
    MapType
)

# COMMAND ----------

schema_exchangeinfo_futures = StructType([
    StructField("symbol", StringType(), False),
    StructField("pair", StringType(), True),
    StructField("contractType", StringType(), True),
    StructField("delivery_date", TimestampType(), True),
    StructField("onboard_date", TimestampType(), True),
    StructField("status", StringType(), True),
    StructField("maintMarginPercent", DoubleType(), True),
    StructField("requiredMarginPercent", DoubleType(), True),
    StructField("baseAsset", StringType(), True),
    StructField("quoteAsset", StringType(), True),
    StructField("marginAsset", StringType(), True),
    StructField("pricePrecision", IntegerType(), True),
    StructField("quantityPrecision", IntegerType(), True),
    StructField("baseAssetPrecision", IntegerType(), True),
    StructField("quotePrecision", IntegerType(), True),
    StructField("underlyingType", StringType(), True),
    StructField("underlyingSubType", ArrayType(StringType()), True),
    StructField("settlePlan", StringType(), True),
    StructField("triggerProtect", DoubleType(), True),
    StructField("liquidationFee", DoubleType(), True),
    StructField("marketTakeBound", DoubleType(), True),
    StructField("maxMoveOrderLimit", IntegerType(), True),
    StructField("filters", ArrayType(MapType(StringType(), StringType())), True),
    StructField("orderTypes", ArrayType(StringType()), True),
    StructField("timeInForce", ArrayType(StringType()), True),
    StructField("permissionSets", ArrayType(StringType()), True),
    StructField("_ingested_at", TimestampType(), True)
])

# COMMAND ----------

cols_to_drop = [
    "maintMarginPercent",
    "requiredMarginPercent",
    "baseAssetPrecision",
    "quotePrecision",
    "maxMoveOrderLimit",
    "orderTypes",
    "timeInForce",
    "permissionSets"
]

criptos_df = (
    spark.sql("SELECT symbols FROM dbrtfmprod.whale_trades.bronze__binance_exchangeinfo_futures")
    .select(F.explode("symbols").alias("symbol_struct"))
    .select("symbol_struct.*")
)

df_cryptos_clean = (
    criptos_df
    .withColumn("delivery_date", F.from_unixtime(F.col("deliveryDate") / 1000).cast("timestamp"))
    .withColumn("onboard_date", F.from_unixtime(F.col("onboardDate") / 1000).cast("timestamp"))
    .withColumn(
        "underlyingSubType",
        F.when(F.size("underlyingSubType") > 0, F.col("underlyingSubType").getItem(0))
    )    
    .drop("deliveryDate", "onboardDate", "settlePlan")
    .drop(*cols_to_drop)
)

df_cryptos_clean.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("abfss://datalake@statfmprod.dfs.core.windows.net/silver/exchangeinfo/binance_futures")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS dbrtfmprod.whale_trades.silver__binance_exchangeinfo_futures
    USING DELTA
    LOCATION 'abfss://datalake@statfmprod.dfs.core.windows.net/silver/exchangeinfo/binance_futures'
""")


# COMMAND ----------

criptos_df = spark.table("dbrtfmprod.whale_trades.silver__binance_exchangeinfo_futures")
print(criptos_df.count())
display(criptos_df)  

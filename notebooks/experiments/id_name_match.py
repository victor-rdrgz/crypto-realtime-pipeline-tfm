# Databricks notebook source
import copy
import requests
import time

from datetime import datetime, timezone
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType,
    BooleanType, DateType, TimestampType
)

# COMMAND ----------

bronze_path_base = "abfss://datalake@statfmprod.dfs.core.windows.net/bronze"

# COMMAND ----------

symbols_df = spark.sql("""
    SELECT DISTINCT base_coin
    FROM dbrtfmprod.whale_trades.silver__cripto_list
""")
symbols_list = [row["base_coin"].lower() for row in symbols_df.collect()]

# COMMAND ----------

coins = requests.get("https://api.coingecko.com/api/v3/coins/list").json()
coins_by_symbol = {}

for coin in coins:
    key = coin["symbol"].lower()
    coins_by_symbol.setdefault(key, []).append(coin)
print(len(coins_by_symbol))


# COMMAND ----------

def get_coin_id(symbol):
    matches = coins_by_symbol.get(symbol, [])
    if len(matches) == 1:
        result = matches[0]
        result["_ingested_at"] = datetime.now(timezone.utc)

        return result

# COMMAND ----------

result = []

for symbol in symbols_list:
    try:
        coin_info = get_coin_id(symbol)
        if not coin_info:
            continue
        result.append(copy.deepcopy(coin_info))
    except Exception as e:
        print(e)
print(len(result))

# COMMAND ----------

schema = StructType([
    StructField("id", StringType(), True),
    StructField("symbol", StringType(), True),
    StructField("name", StringType(), True),
    StructField("_ingested_at", TimestampType(), True) 
])

# COMMAND ----------

df = spark.createDataFrame(result, schema=schema)
df.write.format("delta").mode("overwrite").save(f"{bronze_path_base}/id_name_match")


# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS dbrtfmprod.whale_trades.bronze__id_name_match
# MAGIC USING DELTA
# MAGIC LOCATION 'abfss://datalake@statfmprod.dfs.core.windows.net/bronze/id_name_match';

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType,
    BooleanType, DateType
)

schema = StructType([
    # Identificadores base
    StructField("id", StringType(), True),
    StructField("symbol", StringType(), True),
    StructField("name", StringType(), True),

    # Categorías
    StructField("categories", ArrayType(StringType()), True),

    # Gobernanza
    StructField("governance_type", StringType(), True),
    StructField("governance_stack", StringType(), True),
    StructField("governance_url", StringType(), True),

    # Seguridad / auditorías
    StructField("audited", BooleanType(), True),
    StructField("audit_firm", StringType(), True),
    StructField("audit_date", DateType(), True),
    StructField("audit_report_url", StringType(), True),
    StructField("code_verified", BooleanType(), True),
    StructField("bug_bounty", BooleanType(), True),
    StructField("bug_bounty_url", StringType(), True),

    # Tipo de activo
    StructField("asset_type", StringType(), True),

    # Enlace oficial a Twitter
    StructField("twitter_url", StringType(), True),
])
# Databricks notebook source
import requests
import time
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType,
    BooleanType, DateType, TimestampType
)

# COMMAND ----------

cripto_id_name = spark.sql("""
    SELECT id
    FROM dbrtfmprod.whale_trades.bronze__id_name_match
""")

# COMMAND ----------

def safe_get(url, params=None, max_retries=7):
    retries = 0
    print(url)
    while retries < max_retries:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            time.sleep(2)
            return response.json()

        elif response.status_code == 429:
            time.sleep(2)
            retries += 1
        else:
            raise Exception
    raise Exception("ha fallado una cripto por alcanzar max_retries, empezamos de nuevo con mayor sleep")

# COMMAND ----------

bronze_criptoinfo_path= 'abfss://datalake@statfmprod.dfs.core.windows.net/bronze/cripto_info'

# COMMAND ----------

info_schema = StructType([
    StructField("id", StringType(), True),
    StructField("symbol", StringType(), True),
    StructField("name", StringType(), True),
    StructField("description", StringType(), True),
    StructField("links", StructType([
        StructField("whitepaper", StringType(), True),
        StructField("official_forum", ArrayType(StringType()), True),
        StructField("subreddit", StringType(), True),
        StructField("twitter_url", StringType(), True),
    ]), True),
    StructField("genesis_date", DateType(), True),
    StructField("last_update", TimestampType(), True),
    StructField("_ingested_at", TimestampType(), True)
])

# COMMAND ----------

error_id = [
{"id": "nexus"},
{"id": "oreswap-2"},
{"id": "cornermarket"},
{"id": "usual"}
]

# COMMAND ----------

result = []

# Recolectar datos
for row in cripto_id_name.collect():
    try:

        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false"
        }
        data = safe_get(
            f"https://api.coingecko.com/api/v3/coins/{row['id']}",
            params=params
        )
        if not data:
            print("no hemos recabado datos")
            raise()
        info = {
            "id": data.get("id"),
            "symbol": data.get("symbol"),
            "name": data.get("name"),
            "description": data.get("description", {}).get("en","").strip(),
            "links":{
                "whitepaper": data.get("links", {}).get("whitepaper"),
                "official_forum": data.get("links", {}).get("official_forum_url",[]),
                "subreddit": data.get("links", {}).get("subreddit_url",[]),
                "twitter_url": (
                    f'https://twitter.com/{data.get("links", {}).get("twitter_screen_name")}'
                    if data.get("links", {}).get("twitter_screen_name") else None
                )
            },
            "genesis_date": data.get("genesis_date"),
                        "last_update": (
                datetime.fromisoformat(data.get("last_updated")
                    .replace("Z", "+00:00")) 
                if data.get("last_updated")
                else None
            )
        }
        info['genesis_date'] = (
            datetime.strptime(data.get("genesis_date"), "%Y-%m-%d").date()
                if data.get("genesis_date") 
                else None
        )
        result.append(info)
    except Exception as e:
        print(f"Error procesando {row['id']}: {e}")


# Crear DataFrame de los encontrados
cripto_info_list = spark.createDataFrame(result, info_schema)
cripto_info_list = cripto_info_list.withColumn("_ingested_at", F.current_timestamp())
display(cripto_info_list)


# COMMAND ----------

result_error = []

# Recolectar datos
for row in error_id:
    try:

        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false"
        }
        data = safe_get(
            f"https://api.coingecko.com/api/v3/coins/{row['id']}",
            params=params
        )
        if not data:
            print("no hemos recabado datos")
            raise()
        info = {
            "id": data.get("id"),
            "symbol": data.get("symbol"),
            "name": data.get("name"),
            "description": data.get("description", {}).get("en","").strip(),
            "links":{
                "whitepaper": data.get("links", {}).get("whitepaper"),
                "official_forum": data.get("links", {}).get("official_forum_url",[]),
                "subreddit": data.get("links", {}).get("subreddit_url",[]),
                "twitter_url": (
                    f'https://twitter.com/{data.get("links", {}).get("twitter_screen_name")}'
                    if data.get("links", {}).get("twitter_screen_name") else None
                )
            },
            "genesis_date": data.get("genesis_date"),
            "last_update": (
                datetime.fromisoformat(data.get("last_updated")
                    .replace("Z", "+00:00")) 
                if data.get("last_updated")
                else None
            )
        }
        
        info['genesis_date'] = (
            datetime.strptime(data.get("genesis_date"), "%Y-%m-%d").date()
                if data.get("genesis_date") 
                else None
        )

        print(data.get("genesis_date"))
        result_error.append(info)
    except Exception as e:
        print(f"Error procesando {row['id']}: {e}")


# Crear DataFrame de los encontrados
'''cripto_info_list = spark.createDataFrame(result, info_schema)
cripto_info_list = cripto_info_list.withColumn("_ingested_at", F.current_timestamp())
display(cripto_info_list)'''


# COMMAND ----------

cripto_info_list_error = spark.createDataFrame(result_error, info_schema)
cripto_info_list = cripto_info_list.dropDuplicates()
cripto_info_list_error = cripto_info_list_error.withColumn("_ingested_at", F.current_timestamp())
cripto_info_list = cripto_info_list.unionByName(cripto_info_list_error)
display(cripto_info_list)

# COMMAND ----------

bronze_criptoinfo_path = 'abfss://datalake@statfmprod.dfs.core.windows.net/bronze/cripto_info'

cripto_info_list.write.format("delta").mode("append").save(f"{bronze_criptoinfo_path}")

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS dbrtfmprod.whale_trades.bronze__cripto_info
    USING DELTA
    LOCATION '{bronze_criptoinfo_path}'
""")

# COMMAND ----------

df = spark.table("dbrtfmprod.whale_trades.bronze__cripto_info")
cols = df.columns

null_counts = df.select([
    F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in cols
])

null_counts_long = null_counts.selectExpr(
    [f"stack({len(cols)}, " + ", ".join([f"'{c}', {c}" for c in cols]) + ") as (column, null_count)"]
)

null_counts_long.orderBy("null_count", ascending=False).show(truncate=False)
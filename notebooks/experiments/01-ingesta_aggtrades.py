# Databricks notebook source
import requests
import time
from datetime import datetime, timedelta, timezone

from pyspark.sql import Row
from pyspark.sql import types as T, functions as F, Window as W
from pyspark.sql import functions as F, Window as W

# COMMAND ----------

TABLE_BRONZE__TOP_ALL_TRADES = 'dbrtfmprod.whale_trades.bronze__top_kyles_all_trades'
BRONZE_PATH  = "abfss://datalake@statfmprod.dfs.core.windows.net/bronze/tradesv2"
LIMIT = 1000
BUFFER_TARGET = 800000

DAYS_BACK       = 45
SAFETY_SECONDS  = 30
STEP_HOURS      = 6

END_MS   = int((datetime.now(timezone.utc) - timedelta(seconds=SAFETY_SECONDS)).timestamp() * 1000)
START_MS = int((datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).timestamp() * 1000)
STEP_MS  = STEP_HOURS * 60 * 60 * 1000

# COMMAND ----------

spark.sql("""
    UPDATE dbrtfmprod.whale_trades.gold__kyles_top
    SET interesting = true,
        analyzed = false
""")

# COMMAND ----------

top_kyles_crypto = spark.sql("""
    SELECT symbol
    FROM dbrtfmprod.whale_trades.gold__kyles_top
    WHERE interesting IS TRUE
    AND analyzed IS False
    ORDER BY score_final DESC
    LIMIT 3;
"""
)

top_symbols_list= [row["symbol"] for row in top_kyles_crypto.collect()]

print(top_symbols_list)

# COMMAND ----------

schema = T.StructType([
    T.StructField("symbol",         T.StringType(),   True),
    T.StructField("agg_trade_id",   T.LongType(),     True),
    T.StructField("price",          T.DoubleType(),   True),
    T.StructField("qty",            T.DoubleType(),   True),
    T.StructField("first_trade_id", T.LongType(),     True),
    T.StructField("last_trade_id",  T.LongType(),     True),
    T.StructField("timestamp",      T.LongType(),     True),
    T.StructField("is_buyer_maker", T.BooleanType(),  True),
    T.StructField("ts",             T.TimestampType(), True),
    T.StructField("sec",            T.TimestampType(), True),
    T.StructField("date",           T.DateType(),      True),
    T.StructField("trade_value",    T.DoubleType(),    True),
])


# COMMAND ----------

def write_buffer(buffer):
    trades_df = (spark.createDataFrame(buffer, schema)
                        .withColumn("ts",  F.to_timestamp(F.from_unixtime(F.col("timestamp")/1000)))
                        .withColumn("sec", F.date_trunc("second", F.col("ts")))
                        .withColumn("date", F.to_date("ts"))
                        .withColumn("trade_value", F.col("price") * F.col("qty"))
                )

    (trades_df.write.mode("append")
        .format("delta")
        .partitionBy("symbol","date")
        .option("mergeSchema","true")
        .save(BRONZE_PATH)
    )
    buffer.clear()


# COMMAND ----------

sleep_counter = 0
for symbol in top_symbols_list:
    print(f"Descargando {symbol} …")
    rows_ingested = 0
    trades_count = 0
    buffer = []
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).timestamp() * 1000)

    while True:
        try:
            url = (
                f"https://fapi.binance.com/fapi/v1/aggTrades"
                f"?symbol={symbol}"
                f"&startTime={start_ms}"
                f"&limit={LIMIT}"
            )
            response = requests.get(
                url,
                timeout=30
            )

            if response.status_code == 429:
                sleep_counter += 1
                print(f"⏱️ Rate limited (429). Intento {sleep_counter}/10. Esperando 2 segundos...")
                time.sleep(10)

                if sleep_counter >= 30:
                    print(f"💥 Demasiados 429 consecutivos. Último startTime = {start_ms}")
                    raise Exception("Rate limit alcanzado 10 veces seguidas. Abortando.")
                continue  # reintenta el mismo start_time

            # Si no es 429, se reinicia el contador
            sleep_counter = 0
            response.raise_for_status()

            data = response.json()
            if not data:
                write_buffer(buffer)
                print("✅ No hay más operaciones.")
                break

            # Convertimos los dicts en Rows con campos renombrados
            buffer.extend([
                Row(
                    symbol         = symbol,
                    agg_trade_id   = int(tr["a"]),
                    price          = float(tr["p"]),
                    qty            = float(tr["q"]),
                    first_trade_id = int(tr["f"]),
                    last_trade_id  = int(tr["l"]),
                    timestamp      = int(tr["T"]),
                    is_buyer_maker = bool(tr["m"]),
                    ts=None,
                    sec=None,
                    date=None,
                    trade_value=None 
                ) for tr in data]
            )
            if len(buffer) >= BUFFER_TARGET:
                write_buffer(buffer)

            trades_count += LIMIT
            if len(data) < LIMIT:
                write_buffer(buffer)
                trades_count += len(data)
                print(f"✅ Último lote recibido. Número de filas = {trades_count}")
                break

            # Avanzamos al timestamp del último trade
            last_trade_time = data[-1]["T"]
            start_ms = last_trade_time
        except Exception as e:
            print(f"❌ Error: {e}")
            break  
    print(f"{symbol}: {trades_count} rows ingested")


# COMMAND ----------

# Celda 4 — Deduplicación (idempotencia): (symbol, agg_trade_id)
sdf = spark.read.format("delta").load(BRONZE_PATH)

print("rows before dedup:", sdf.count())

w = W.partitionBy("symbol","agg_trade_id").orderBy(F.col("timestamp").desc())
dedup = sdf.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")

print("rows after dedup:", dedup.count())

(dedup.write
    .mode("overwrite")
    .option("overwriteSchema","true")
    .format("delta")
    .partitionBy("symbol","date")
    .save(BRONZE_PATH))

print("rows after dedup:", dedup.count())

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TABLE_BRONZE__TOP_ALL_TRADES}
USING DELTA
LOCATION '{BRONZE_PATH}'
""")

# COMMAND ----------

TABLE_BRONZE__TOP_ALL_TRADES

# COMMAND ----------

df = spark.table(TABLE_BRONZE__TOP_ALL_TRADES)

rows = df.count()
dups = (df.groupBy("symbol","agg_trade_id").count().filter("count>1").count())
neg  = {
    "price<=0": df.filter("price <= 0 OR price IS NULL").count(),
    "qty<=0":   df.filter("qty <= 0 OR qty IS NULL").count(),
}

by_sym = (df.groupBy("symbol")
            .agg(F.count("*").alias("n"),
                 F.min("ts").alias("min_ts"),
                 F.max("ts").alias("max_ts"))
            .orderBy("symbol"))

print({"rows": rows, "dups_pk": dups, "nonpositive": neg})
display(by_sym)


# COMMAND ----------

try:
    spark.sql(f"OPTIMIZE {TABLE_BRONZE__TOP_ALL_TRADES} ZORDER BY (symbol, ts)")
except Exception as e:
    print("OPTIMIZE no disponible o no necesario:", e)

# COMMAND ----------

sdf = spark.read.format("delta").load(BRONZE_PATH)

w = W.partitionBy("symbol","agg_trade_id").orderBy(F.col("timestamp").desc())
dedup = (sdf.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn"))

# sobrescribe ya limpio
dedup.write.mode("overwrite").option("overwriteSchema","true").format("delta").save(BRONZE_PATH)


# COMMAND ----------

def fetch_aggtrades_range(symbol, start_ms, end_ms):
    out = []
    # 1) primera page acotada por tiempo
    page = requests.get(
        "https://fapi.binance.com/fapi/v1/aggTrades",
        params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": LIMIT}
    ).json()
    if not page:
        return out
    out.extend(page)
    last_id = page[-1]["a"]

    # 2) continuar por fromId para no perder nada si hay >1000 trades
    while True:
        page = requests.get(
            "https://fapi.binance.com/fapi/v1/aggTrades",
            params={"symbol": symbol, "fromId": last_id + 1, "limit": LIMIT}
        ).json()
        if not page:
            break
        batch = [t for t in page if t["T"] <= end_ms]
        if not batch:
            break
        out.extend(batch)
        last_id = page[-1]["a"]
        if len(page) < LIMIT:
            break
        time.sleep(0.05)
    return out


# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS dbrtfmprod.whale_trades.{TABLE_BRONZE__TOP_ALL_TRADES}
USING DELTA
LOCATION 'abfss://datalake@statfmprod.dfs.core.windows.net/bronze/trades/'
""")

# COMMAND ----------

spark.sql(f"""
    UPDATE dbrtfmprod.whale_trades.gold__kyles_top
    SET interesting = false
    WHERE symbol IN (
        SELECT symbol
        FROM dbrtfmprod.whale_trades.{TABLE_BRONZE__TOP_ALL_TRADES}
        GROUP BY symbol
        HAVING COUNT(*) < 100000
    )
""")

spark.sql(f"""
    UPDATE dbrtfmprod.whale_trades.gold__kyles_top
    SET analyzed = true
    WHERE symbol IN (
        SELECT symbol
        FROM dbrtfmprod.whale_trades.{TABLE_BRONZE__TOP_ALL_TRADES}
        GROUP BY symbol
    )
""")

# COMMAND ----------

spark.sql(f"""
    SELECT DISTINCT symbol
    FROM dbrtfmprod.whale_trades.{TABLE_BRONZE__TOP_ALL_TRADES}
    GROUP BY symbol""").show()

# COMMAND ----------

spark.sql("""
    SELECT COUNT(*) AS num_not_interesting
    FROM dbrtfmprod.whale_trades.gold__kyles_top
    WHERE interesting = false
""").show()

# COMMAND ----------

spark.sql(f"""
    SELECT DISTINCT symbol
    FROM dbrtfmprod.whale_trades.gold__kyles_top
    WHERE interesting IS TRUE
""").show()

# COMMAND ----------

spark.sql(f"""
DELETE FROM dbrtfmprod.whale_trades.{TABLE_BRONZE__TOP_ALL_TRADES}
WHERE symbol IN (
    SELECT symbol
    FROM dbrtfmprod.whale_trades.{TABLE_BRONZE__TOP_ALL_TRADES}
    GROUP BY symbol
    HAVING COUNT(*) < 100000
)
""")

# COMMAND ----------

display(spark.sql("""
    SELECT COUNT(*) AS trades, symbol
    FROM dbrtfmprod.whale_trades.bronze__top_kyles_all_trades
    GROUP BY symbol
"""))

# COMMAND ----------

spark.sql(f"""
    SELECT * 
    FROM dbrtfmprod.whale_trades.{TABLE_BRONZE__TOP_ALL_TRADES}
    Limit 10
""").show()

# COMMAND ----------

from pyspark.sql.functions import (
    col, 
    expr, 
    percentile_approx, 
    round
)

trades_df = spark.sql(f"""
    SELECT * 
    FROM dbrtfmprod.whale_trades.{TABLE_BRONZE__TOP_ALL_TRADES}
    WHERE is_buyer_maker IS FALSE
""")

trades_df = trades_df.withColumn(
    "trade_value",
    round(
        col("price").cast("double") 
            * 
        col("qty").cast("double"),
        2
    )
)

# Calcular percentil 95 de trade_value por símbolo
percentiles_df = trades_df.groupBy("symbol").agg(
    percentile_approx("trade_value", 0.95, 100).alias("p95_value")
)

# COMMAND ----------

display(percentiles_df)

# COMMAND ----------

# Unir con el percentil 95
filtered_trades = trades_df.join(percentiles_df, on="symbol") \
    .filter(col("trade_value") >= col("p95_value")) \
    .drop("p95_value")

# COMMAND ----------

silver_base_path = "abfss://datalake@statfmprod.dfs.core.windows.net/silver/p95_trades"


filtered_trades.write.mode("append")\
    .format("delta") \
    .partitionBy("symbol") \
    .save(silver_base_path)


# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS dbrtfmprod.whale_trades.silver__p95_trades
    USING DELTA
    LOCATION 'abfss://datalake@statfmprod.dfs.core.windows.net/silver/p95_trades'
""")
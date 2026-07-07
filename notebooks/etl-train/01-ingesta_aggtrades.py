# Databricks notebook source
import requests
import time
from datetime import datetime, timedelta, timezone

from pyspark.sql import Row
from pyspark.sql import types as T, functions as F, Window as W

# COMMAND ----------

# Ubicaciones para lectura y escritura
TABLE_BRONZE__TOP_ALL_TRADES = 'dbrtfmprod.whale_trades.bronze__top_kyles_all_trades'
BRONZE_PATH  = "abfss://datalake@statfmprod.dfs.core.windows.net/bronze/tradesv2"
GOLD_TABLE_WITH_SCORES = "dbrtfmprod.whale_trades.gold__kyles_top"

# Parámetros de configuración
LIMIT = 1000
BUFFER_TARGET = 800000
MIN_TRADES = 110000
SYMBOLS_REQUIRED = 1
DAYS_BACK       = 180
SAFETY_SECONDS  = 10
NUM_RETRIES     = 30

# COMMAND ----------

# importamos las criptomonedas filtradas y ordenadas por interés
top_kyles_crypto = spark.sql(f"""
    SELECT symbol
    FROM {GOLD_TABLE_WITH_SCORES}
    ORDER BY score_final DESC
    LIMIT 1;
"""
)

top_symbols_list= [row["symbol"] for row in top_kyles_crypto.collect()]

# COMMAND ----------

# Definición de la estructura del dataframe que vamos a escribir
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

# Función auxiliar que escribe en ADLS2
def write_buffer(buffer):
    if not buffer:
        return
    trades_df = (spark.createDataFrame(buffer, schema)
                        .withColumn("ts",  F.to_timestamp(F.from_unixtime(F.col("timestamp")/1000)))
                        .withColumn("sec", F.date_trunc("second", F.col("ts")))
                        .withColumn("date", F.to_date("ts"))
                        .withColumn("trade_value", F.col("price") * F.col("qty"))
                )
    
    trades_df = trades_df.dropDuplicates(["symbol", "agg_trade_id"])


    (trades_df.write.mode("append")
        .format("delta")
        .partitionBy("symbol")
        .option("mergeSchema","true")
        .save(BRONZE_PATH)
    )
    buffer.clear()


# COMMAND ----------

# comprobamos el último timestamp escrito en la tabla y
# empezamos a hacer peticiones a la api a partir de ese ts.
try:
    # Consulta para obtener el último timestamp por símbolo
    df_max_ts = spark.sql(f"""
        SELECT symbol, MAX(timestamp) AS max_ts
        FROM {TABLE_BRONZE__TOP_ALL_TRADES}
        WHERE symbol IN ({','.join(f"'{s}'" for s in top_symbols_list)})
        GROUP BY symbol
    """)

    # Convertimos el resultado a un diccionario
    max_ts_per_symbol = {
        row["symbol"]: (row["max_ts"] if row["max_ts"] is not None else 0)
        for row in df_max_ts.collect()
    }
except:
    max_ts_per_symbol = 0

# COMMAND ----------

# Lógica para hacer peticiones a la API, con reintentos
# en caso de exceder el límite de la API
sleep_counter = 0
valid_symbols = 0
for symbol in top_symbols_list:
    print(f"Descargando {symbol} …")
    rows_ingested = 0
    trades_count = 0
    buffer = []
    if max_ts_per_symbol and symbol in max_ts_per_symbol.keys():
        start_ms = (max_ts_per_symbol[symbol]+1)
    else:
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
                print(f"⏱️ Rate limited (429). Intento {sleep_counter}/10. Esperando {SAFETY_SECONDS} segundos...")
                time.sleep(SAFETY_SECONDS)

                if sleep_counter >= NUM_RETRIES:
                    print(f"💥 Demasiados 429 consecutivos. Último startTime = {start_ms}")
                    raise Exception(f"Rate limit alcanzado {NUM_RETRIES} veces seguidas. Abortando.")
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
                if trades_count > MIN_TRADES or not max_ts_per_symbol:
                    valid_symbols += 1

                break

            # Avanzamos al timestamp del último trade
            last_trade_time = data[-1]["T"]
            start_ms = last_trade_time+1
        except Exception as e:
            print(f"❌ Error: {e}")
            break  
    if valid_symbols >= SYMBOLS_REQUIRED:
        break
    print(f"{symbol}: {trades_count} rows ingested")


# COMMAND ----------

# En caso de ser la primera ejecución, 
# creamos la tabla en Unity Catalog
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {TABLE_BRONZE__TOP_ALL_TRADES}
USING DELTA
LOCATION '{BRONZE_PATH}'
""")

# COMMAND ----------

# borramos symbols con menos de MIN_TRADES trades
spark.sql(f"""
DELETE FROM {TABLE_BRONZE__TOP_ALL_TRADES}
WHERE symbol IN (
    SELECT symbol
    FROM {TABLE_BRONZE__TOP_ALL_TRADES}
    GROUP BY symbol
    HAVING COUNT(*) < 100000
)
""")

# COMMAND ----------

# Optimizamos la tabla en base a ts
try:
    spark.sql(f"OPTIMIZE {TABLE_BRONZE__TOP_ALL_TRADES} ZORDER BY (ts)")
except Exception as e:
    print("OPTIMIZE no disponible o no necesario:", e)
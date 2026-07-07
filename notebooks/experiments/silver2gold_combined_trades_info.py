# Databricks notebook source
from datetime import timedelta
import pandas as pd

from pyspark.sql import functions as F, Window
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, DoubleType, BooleanType
)

# COMMAND ----------

spark.conf.set("spark.sql.session.timeZone", "UTC")  # Binance opera en UTC

CAT = "dbrtfmprod"
SCH = "whale_trades"
GOLD_PATH = "abfss://datalake@statfmprod.dfs.core.windows.net/gold/whale_trades/context_5m_plus60m"
GOLD_TABLE = f"{CAT}.{SCH}.gold__context_5m_plus60m"

# ===== 1) Base p95: mantener `timestamp` (long, ms) y crear `ts` (timestamp) =====
p95 = spark.table(f"{CAT}.{SCH}.silver__p95_trades")

# COMMAND ----------

p95 = (
    p95
    .withColumn("ts", F.from_unixtime((F.col("timestamp")/1000).cast("double")).cast("timestamp"))
    .withColumn("start_ts", F.col("ts") - F.expr("INTERVAL 5 MINUTES"))
    .withColumn("end_ts",   F.col("ts") + F.expr("INTERVAL 60 MINUTES"))
    .withColumn("date", F.to_date("ts"))
)

BASE_KEYS = ["symbol","agg_trade_id","timestamp","ts","start_ts","end_ts","date"]

# COMMAND ----------

# =========================
# 2) Tablas secundarias (ajusta nombres si cambian)
# =========================
klines = spark.table(f"{CAT}.{SCH}.silver__klines_1m").select(
    "symbol","open_time","open","high","low","close","volume","n_trades","taker_base_vol","taker_quote_vol","close_time"
)
mark_kl = spark.table(f"{CAT}.{SCH}.silver__mark_klines_1m").select(
    "symbol","open_time","open","high","low","close","close_time"
)
index_kl = spark.table(f"{CAT}.{SCH}.silver__index_klines_1m").select(
    "symbol","open_time","open","high","low","close","close_time"
)
premium_kl = spark.table(f"{CAT}.{SCH}.silver__premium_klines_1m").select(
    "symbol","open_time","open","high","low","close","close_time"
)
oi_5m = spark.table(f"{CAT}.{SCH}.silver__open_interest_5m").select(
    "symbol","ts","oi","oi_value_usd"
)
taker_5m = spark.table(f"{CAT}.{SCH}.silver__taker_5m").select(
    "symbol","ts","buy_vol","sell_vol","buy_sell_ratio"
)
funding_8h = spark.table(f"{CAT}.{SCH}.silver__funding_8h").select(
    "symbol","funding_time","funding_rate"
)

# COMMAND ----------

kl_joined = (
    p95.alias("b")
    .join(
        klines.alias("k"),
        (F.col("b.symbol") == F.col("k.symbol")) &
        (F.col("k.open_time").between(F.col("b.start_ts"), F.col("b.end_ts"))),
        "left"
    )
    # offset en minutos respecto al trade
    .withColumn(
        "minutes_offset",
        F.floor((F.unix_timestamp("k.open_time") - F.unix_timestamp("b.ts")) / 60).cast("int")
    )
)

# COMMAND ----------

kl_features = (
    kl_joined
    .groupBy(*[F.col("b."+k) for k in BASE_KEYS])
    .pivot("minutes_offset")  # genera columnas -5 … +60
    .agg(
        F.first("k.open").alias("open"),
        F.first("k.high").alias("high"),
        F.first("k.low").alias("low"),
        F.first("k.close").alias("close"),
        F.first("k.volume").alias("volume"),
        F.first("k.n_trades").alias("n_trades"),
        F.first("k.taker_base_vol").alias("taker_base_vol"),
        F.first("k.taker_quote_vol").alias("taker_quote_vol")
    )
)

# COMMAND ----------

for i in range(-5, 61):
    for col in ["open","high","low","close","volume","n_trades","taker_base_vol","taker_quote_vol"]:
        old_name = f"{i}_{col}"
        new_name = f"{col}_t{i:+}"  # ejemplo: close_t-5, volume_t+15
        if old_name in kl_features.columns:
            kl_features = kl_features.withColumnRenamed(old_name, new_name)



out = (
    p95
    .join(kl_features, BASE_KEYS, "left")
)

display(out)

# COMMAND ----------

schema = StructType([
    # columnas originales de p95
    StructField("symbol", StringType(), True),
    StructField("agg_trade_id", LongType(), True),
    StructField("price", StringType(), True),
    StructField("qty", StringType(), True),
    StructField("first_trade_id", LongType(), True),
    StructField("last_trade_id", LongType(), True),
    StructField("timestamp", LongType(), True),
    StructField("is_buyer_maker", BooleanType(), True),
    StructField("date", StringType(), True),
    StructField("trade_value", DoubleType(), True),

    # columnas de contexto (14 en total, tipo MapType[String, Double])
    StructField("klines_5m_before",  MapType(StringType(), DoubleType()), True),
    StructField("klines_60m_after",  MapType(StringType(), DoubleType()), True),

    StructField("mark_kl_5m_before", MapType(StringType(), DoubleType()), True),
    StructField("mark_kl_60m_after", MapType(StringType(), DoubleType()), True),

    StructField("index_kl_5m_before", MapType(StringType(), DoubleType()), True),
    StructField("index_kl_60m_after", MapType(StringType(), DoubleType()), True),

    StructField("premium_kl_5m_before", MapType(StringType(), DoubleType()), True),
    StructField("premium_kl_60m_after",  MapType(StringType(), DoubleType()), True),

    StructField("oi_5m_before",  MapType(StringType(), DoubleType()), True),
    StructField("oi_60m_after",   MapType(StringType(), DoubleType()), True),

    StructField("taker_5m_before", MapType(StringType(), DoubleType()), True),
    StructField("taker_60m_after",  MapType(StringType(), DoubleType()), True),

    StructField("funding_before", MapType(StringType(), DoubleType()), True),
    StructField("funding_after",  MapType(StringType(), DoubleType()), True),
])

# COMMAND ----------

def filter_context_by_time(df_ctx, symbol, start, finish, col_start, col_end):
    # filtramos df_ctx de ese símbolo en el rango [start, finish]
    kl_subset = df_ctx.filter(
        (df_ctx.symbol == symbol &
        (df_ctx.col_start >= start) &
        (df_ctx.col_end <= finish))
    )
        
    return klines_5m_before.to_dict()
        



    # construimos un dict {timestamp_str: close}
    kl_map = {
        str(pd.to_datetime(r["open_time"])): float(r["close"])
        for _, r in kl_subset.iterrows()
    }

# COMMAND ----------

# Función que recibe un microbatch (pandas.DataFrame con filas de p95)
def process_batch(pdf_iter):

    for pdf in pdf_iter:
        out_rows = []
        for _, row in pdf.iterrows():
            ts = pd.to_datetime(row["timestamp"], unit="ms")
            start = ts - timedelta(minutes=5)
            finish = ts + timedelta(minutes=60)
            out_rows.append({
                "symbol": row["symbol"],
                "agg_trade_id": row["agg_trade_id"],
                "price": row["price"],
                "qty": row["qty"],
                "first_trade_id": row["first_trade_id"],
                "last_trade_id": row["last_trade_id"],
                "timestamp": row["timestamp"],
                "is_buyer_maker": row["is_buyer_maker"],
                "date": row["date"],
                "trade_value": float(row["qty"]) * float(row["price"]),

                # columnas de contexto (aquí simuladas)
                "klines_5m_before":  filter_context_by_time(
                    klines, row["symbol"], start, finish, "open_time", "close_time"),
                "klines_60m_after":    {key: 125.67},
                "mark_kl_5m_before":   {key: 223.11},
                "mark_kl_60m_after":   {key: 224.22},
                "index_kl_5m_before": {},
                "index_kl_60m_after": {}.
                "premium_kl_5m_before":  {key: 101.11},
                "premium_kl_60m_after":  {key: 102.22},
                "oi_5m_before":   {key: 0.0011},
                "oi_60m_after":    {key: 0.0015},
                "taker_5m_before":   {key: 124.11},
                "taker_60m_after":    {key: 124.55},
                "funding_before":  {key: 225.33},
                "funding_after":   {key: 226.44},
            })
        yield pd.DataFrame(out_rows)

# Ejecutar en paralelo en todos los workers
result = p95.mapInPandas(process_batch, schema=schema)

# Ver resultado
result.show(truncate=False)

# COMMAND ----------

# =========================
# 3) Función para añadir 2 arrays (prev_5m y next_60m) a cada fila de base
# =========================
def attach_events(base_df, side_df, time_col, fields, out_prefix, alias):
    side = side_df.alias(alias)
    base = base_df.alias("base")

    side_ts = F.col(f"{alias}.{time_col}").cast("timestamp")
    event_struct = F.struct(*([side_ts.alias(time_col)] + [
        F.col(f"{alias}.{c}") for c in fields if c != time_col
    ]))

    joined = (
        base.join(
            side,
            (base.symbol == side.symbol) &
            (side_ts >= base.start_ts) &
            (side_ts <= base.end_ts),
            "left"
        )
    )
    
    agg = (
        joined.groupBy(*[F.col("base."+k) for k in BASE_KEYS])
        .agg(
            F.coalesce(
                F.array_sort(
                    F.filter(
                        F.collect_list(
                            F.when(
                                (side_ts >= F.col("base.start_ts")) & (side_ts < F.col("base.ts")),
                                event_struct
                            )
                        ),
                        lambda x: x.isNotNull()
                    )
                ),
                F.array()
            ).alias(f"{out_prefix}_prev5m"),
            F.coalesce(
                F.array_sort(
                    F.filter(
                        F.collect_list(
                            F.when(
                                (side_ts > F.col("base.ts")) & (side_ts <= F.col("base.end_ts")),
                                event_struct
                            )
                        ),
                        lambda x: x.isNotNull()
                    )
                ),
                F.array()
            ).alias(f"{out_prefix}_next60m")
        )
    )

    return base_df.join(agg, BASE_KEYS, "left")



# =========================
# 4) Encadenar los 7 joins
# =========================
out = p95

out = attach_events(
    out, klines,
    time_col="open_time",
    fields=["open_time","open","high","low","close","volume","n_trades","taker_base_vol","taker_quote_vol","close_time"],
    out_prefix="klines", alias="kl"
)

out = attach_events(
    out, mark_kl,
    time_col="open_time",
    fields=["open_time","open","high","low","close","close_time"],
    out_prefix="mark_klines", alias="mk"
)

out = attach_events(
    out, index_kl,
    time_col="open_time",
    fields=["open_time","open","high","low","close","close_time"],
    out_prefix="index_klines", alias="ix"
)

out = attach_events(
    out, premium_kl,
    time_col="open_time",
    fields=["open_time","high","low","close"],
    out_prefix="premium_klines", alias="pm"
)

out = attach_events(
    out, oi_5m,
    time_col="ts",
    fields=["ts","oi","oi_value_usd"],
    out_prefix="oi", alias="oi"
)

out = attach_events(
    out, taker_5m,
    time_col="ts",
    fields=["ts","buy_vol","sell_vol","buy_sell_ratio"],
    out_prefix="taker", alias="tk"
)

out = attach_events(
    out, funding_8h,
    time_col="funding_time",
    fields=["funding_time","funding_rate"],
    out_prefix="funding", alias="fd"
)

# =========================
# 5) Verificación de tamaños
# =========================
print("p95 count:", p95.count())
print("out count:", out.count())   # debería dar ~45578

# COMMAND ----------

display(taker_5m)

# COMMAND ----------

# Define los límites como timestamps
start_ts = F.lit("2025-08-14T20:00:03").cast("timestamp")
end_ts   = F.lit("2025-08-14T21:30:03").cast("timestamp")

filtered_oi = (
    oi_5m
    .filter((F.col("ts") >= start_ts) & (F.col("ts") <= end_ts))
)

display(filtered_oi)

# COMMAND ----------

display(out.limit(45578))



# COMMAND ----------

print("OI con matches:", out.filter(F.size("oi_prev5m") > 0).count())
print("Taker con matches:", out.filter(F.size("taker_prev5m") > 0).count())
print("Funding con matches:", out.filter(F.size("funding_prev5m") > 0).count())

# COMMAND ----------

# --- Escribir a GOLD ---
GOLD_PATH = "abfss://datalake@statfmprod.dfs.core.windows.net/gold/whale_trades/context_5m_plus60m"

(out
 .write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .partitionBy("symbol")       # particiona por símbolo; opcional añadir por fecha del trade si la tienes
 .save(GOLD_PATH)
)

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CAT}.{SCH}.gold__context_5m_plus60m
USING DELTA
LOCATION '{GOLD_PATH}'
""")

# (Opcional) Optimización Databricks
spark.sql(f"OPTIMIZE {CAT}.{SCH}.gold__context_5m_plus60m ZORDER BY (symbol, ts)")

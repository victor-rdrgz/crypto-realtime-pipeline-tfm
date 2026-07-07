# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql import Window

# COMMAND ----------

CAT = "dbrtfmprod"
SCH = "whale_trades"
BASE_PATH = "abfss://datalake@statfmprod.dfs.core.windows.net/"

# COMMAND ----------

p95.count()

# COMMAND ----------

p95 = spark.table(f"{CAT}.{SCH}.silver__p95_trades")

def floor_to_min(col, minutes=1):
    sec = 60*minutes
    return F.from_unixtime((F.floor(F.unix_timestamp(F.col(col))/sec)*sec).cast("bigint")).cast("timestamp")

p95 = (
    p95
    .withColumn("ts", F.from_unixtime((F.col("timestamp")/1000).cast("double")).cast("timestamp"))
    # anclas alineadas:
    .withColumn("ts_m1",  floor_to_min("ts", 1))
    .withColumn("start_m1", F.col("ts_m1") - F.expr("INTERVAL 5 MINUTES"))
    .withColumn("end_m1",   F.col("ts_m1") + F.expr("INTERVAL 60 MINUTES"))
    .withColumn("ts_m5",  floor_to_min("ts", 5))
    .withColumn("start_m5", F.col("ts_m5") - F.expr("INTERVAL 5 MINUTES"))
    .withColumn("end_m5",   F.col("ts_m5") + F.expr("INTERVAL 60 MINUTES"))
    .withColumn("date", F.to_date("ts"))
)

BASE_KEYS = ["symbol","agg_trade_id","timestamp","ts","ts_m1","start_m1","end_m1","ts_m5","start_m5","end_m5","date"]


# Rango de fechas para podar lectura de tablas secundarias
tmp = p95.select(F.min("start_m1").alias("min_ts"), F.max("end_m1").alias("max_ts")).first()
min_date = tmp["min_ts"].date()
max_date = tmp["max_ts"].date()

# COMMAND ----------

p95.count()

# COMMAND ----------

klines = (
    spark.table(f"{CAT}.{SCH}.silver__klines_1m")
    .select("symbol", "open_time", "close", "volume", "n_trades", "taker_base_vol", "taker_quote_vol", "close_time")
    .withColumn("date", F.to_date("open_time"))
    .filter((F.col("date") >= F.lit(min_date)) & (F.col("date") <= F.lit(max_date)))
    .cache()
)

mark_kl = (
    spark.table(f"{CAT}.{SCH}.silver__mark_klines_1m")
    .select("symbol", "open_time", "close", "close_time")
    .withColumn("date", F.to_date("open_time"))
    .filter((F.col("date") >= F.lit(min_date)) & (F.col("date") <= F.lit(max_date)))
)

index_kl = (
    spark.table(f"{CAT}.{SCH}.silver__index_klines_1m")
    .select("symbol", "open_time", "close", "close_time")
    .withColumn("date", F.to_date("open_time"))
    .filter((F.col("date") >= F.lit(min_date)) & (F.col("date") <= F.lit(max_date)))
)

premium_kl = (
    spark.table(f"{CAT}.{SCH}.silver__premium_klines_1m")
    .select("symbol", "open_time", "close", "close_time")
    .withColumn("date", F.to_date("open_time"))
    .filter((F.col("date") >= F.lit(min_date)) & (F.col("date") <= F.lit(max_date)))
)

oi_5m = (
    spark.table(f"{CAT}.{SCH}.silver__open_interest_5m")
    .select("symbol", "ts", "oi", "oi_value_usd")
    .withColumn("date", F.to_date("ts"))
    .filter((F.col("date") >= F.lit(min_date)) & (F.col("date") <= F.lit(max_date)))
)

taker_5m = (
    spark.table(f"{CAT}.{SCH}.silver__taker_5m")
    .select("symbol", "ts", "buy_vol", "sell_vol", "buy_sell_ratio")
    .withColumn("date", F.to_date("ts"))
    .filter((F.col("date") >= F.lit(min_date)) & (F.col("date") <= F.lit(max_date)))
)

funding_8h = (
    spark.table(f"{CAT}.{SCH}.silver__funding_8h")
    .select("symbol", "funding_time", "funding_rate")
    .withColumn("date", F.to_date("funding_time"))
    .filter((F.col("date") >= F.lit(min_date)) & (F.col("date") <= F.lit(max_date)))
)

# COMMAND ----------

OFFSETS = [-5, 0, 5, 10, 30, 60]

def _resolve_base_keys(df):
    cols = set(df.columns)
    for keys in (["symbol","agg_trade_id"], ["symbol","ts"], ["symbol","timestamp"]):
        if all(k in cols for k in keys):
            return keys
    raise ValueError("No encuentro claves: necesito ('symbol','agg_trade_id') o ('symbol','ts'/'timestamp').")

def pivot_window(base_df, side_df, time_col, metrics, prefix, granularity="1m"):
    """
    Devuelve un DF con exactamente 1 fila por clave del base_df y columnas pivoteadas
    SOLO para los offsets en OFFSETS. Preparado para join 1:1 contra base_df.
    """
    base_keys = _resolve_base_keys(base_df)

    b, s = base_df.alias("b"), side_df.alias("s")
    if granularity == "1m":
        cond   = (F.col("b.symbol")==F.col("s.symbol")) & F.col(f"s.{time_col}").between(F.col("b.start_m1"), F.col("b.end_m1"))
        ref_ts = F.col("b.ts_m1")
    elif granularity == "5m":
        cond   = (F.col("b.symbol")==F.col("s.symbol")) & F.col(f"s.{time_col}").between(F.col("b.start_m5"), F.col("b.end_m5"))
        ref_ts = F.col("b.ts_m5")
    else:
        raise ValueError("granularity debe ser '1m' o '5m'")

    # Join + offset exacto (no 'between' de offsets)
    joined = (
        b.join(s, cond, "left")
         .withColumn(
             "minutes_offset",
             ((F.unix_timestamp(F.col(f"s.{time_col}")) - F.unix_timestamp(ref_ts))/60).cast("int")
         )
         .filter(F.col("minutes_offset").isin(OFFSETS))
         .select(
             *[F.col(f"b.{k}").alias(k) for k in base_keys],
             F.col("minutes_offset"),
             *[F.col(f"s.{m}").alias(m) for m in metrics]
         )
         # Enforce: como mucho 1 fila por (clave base, offset)
         .dropDuplicates(base_keys + ["minutes_offset"])
    )

    # Pivot sólo para OFFSETS. Agregamos con first(ignorenulls=True)
    agg_exprs = [F.first(F.col(m), ignorenulls=True).alias(m) for m in metrics]
    piv = (joined
           .groupBy(*[F.col(k) for k in base_keys])
           .pivot("minutes_offset", OFFSETS)
           .agg(*agg_exprs))

    # Renombrado:
    # - si metrics == ["close"] -> prefix_{offset}  (e.g., mark_-5)
    # - si hay varias métricas  -> prefix_{offset}_{metric} (e.g., kl_-5_close)
    if len(metrics) == 1 and metrics[0] == "close":
        for i in OFFSETS:
            old = f"close_{i}"
            new = f"{prefix}_{i}"
            if old in piv.columns:
                piv = piv.withColumnRenamed(old, new)
    else:
        for i in OFFSETS:
            for m in metrics:
                old = f"{m}_{i}"
                new = f"{prefix}_{i}_{m}"
                if old in piv.columns:
                    piv = piv.withColumnRenamed(old, new)

    return piv


# COMMAND ----------

# 1m
kl_features = pivot_window(p95, klines, "open_time",
    ["close", "taker_base_vol", "taker_quote_vol"],
    prefix="kl", granularity="1m")

mark_features = pivot_window(p95, mark_kl, "open_time",
    ["close"],
    prefix="mark", granularity="1m")

index_features = pivot_window(p95, index_kl, "open_time",
    ["close"],
    prefix="index", granularity="1m")

premium_features = pivot_window(p95, premium_kl, "open_time",
    ["close"],
    prefix="premium", granularity="1m")  # opcional

# 5m
oi_features = pivot_window(p95, oi_5m, "ts",
    ["oi", "oi_value_usd"],
    prefix="oi", granularity="5m")

taker_features = pivot_window(p95, taker_5m, "ts",
    ["buy_vol", "sell_vol", "buy_sell_ratio"],
    prefix="taker", granularity="5m")


# FUNDING 8h (sparse; mayormente nulos salvo coincidencias exactas)
funding_features = pivot_window(
    p95, funding_8h,
    time_col="funding_time",
    metrics=["funding_rate"],
    prefix="funding"
)

# COMMAND ----------

def add_funding_prev_next_from_8h(base_df, funding_8h, *, broadcast_funding=True):
    """
    base_df: necesita 'symbol' y 'ts' (timestamp). Si no hay 'ts' pero hay 'timestamp' (ms), lo crea.
    funding_8h: columnas ['symbol','funding_time','funding_rate'] (y opcional 'date').
    Devuelve base_df + ['funding_prev','funding_next'].
    """
    # Asegura 'ts' en base
    b = base_df
    if "ts" not in b.columns:
        if "timestamp" not in b.columns:
            raise ValueError("base_df necesita 'ts' o 'timestamp' (ms).")
        b = b.withColumn("ts", F.from_unixtime(F.col("timestamp")/1000).cast("timestamp"))

    # Claves para mantener cardinalidad (si hay agg_trade_id, se usa)
    keys = ["symbol", "ts"] + (["agg_trade_id"] if "agg_trade_id" in b.columns else [])

    # Normaliza funding_8h
    f = (funding_8h
         .select(
             "symbol",
             F.col("funding_time").cast("timestamp").alias("f_time"),
             F.col("funding_rate").cast("double").alias("f_rate"),
         )
         .dropna(subset=["symbol","f_time"])
         .dropDuplicates(["symbol","f_time"])
    )

    fj = F.broadcast(f) if broadcast_funding else f
    j = (b.alias("b")
         .join(fj.alias("f"), F.col("b.symbol")==F.col("f.symbol"), "left")
         .select(
             *[F.col(f"b.{c}").alias(c) for c in b.columns],
             F.col("f.f_time").alias("f_time"),
             F.col("f.f_rate").alias("f_rate"),
         )
    )

    # Un único valor previo y siguiente por fila base
    prev_struct = F.max(F.when(F.col("f_time") <= F.col("ts"),
                               F.struct(F.col("f_time").alias("t"), F.col("f_rate").alias("r")))).alias("prev")
    next_struct = F.min(F.when(F.col("f_time") >  F.col("ts"),
                               F.struct(F.col("f_time").alias("t"), F.col("f_rate").alias("r")))).alias("next")

    agg = (j.groupBy(*[F.col(k) for k in keys])
             .agg(prev_struct, next_struct)
             .select(
                 *[F.col(k) for k in keys],
                 F.col("prev.r").alias("funding_prev"),
                 F.col("next.r").alias("funding_next"),
             ))

    return b.join(agg, on=keys, how="left")

# COMMAND ----------

# Helper para evitar duplicados en los joins
def prefix_columns(df, prefix, base_keys=BASE_KEYS):
    return df.select(
        *[F.col(c) for c in base_keys if c in df.columns],
        *[F.col(c).alias(f"{prefix}_{c}") for c in df.columns if c not in base_keys]
    )

# =========================
# Prefijar columnas
# =========================
kl_features      = prefix_columns(kl_features,      "kl")
mark_features    = prefix_columns(mark_features,    "mark")
index_features   = prefix_columns(index_features,   "index")
premium_features = prefix_columns(premium_features, "premium")
oi_features      = prefix_columns(oi_features,      "oi")
taker_features   = prefix_columns(taker_features,   "taker")

gold_tabular = (
    p95
    .join(kl_features,      ["symbol", "agg_trade_id"], "left")
    .join(mark_features,    ["symbol", "agg_trade_id"], "left")
    .join(index_features,   ["symbol", "agg_trade_id"], "left")
    .join(premium_features, ["symbol", "agg_trade_id"], "left")
    .join(oi_features,      ["symbol", "agg_trade_id"], "left")
    .join(taker_features,   ["symbol", "agg_trade_id"], "left")
)

gold_tabular = add_funding_prev_next_from_8h(gold_tabular, funding_8h)


# COMMAND ----------

gold_tabular.count()

# COMMAND ----------

gold_tabular.printSchema()


# COMMAND ----------

cols0 = [c for c in gold_tabular.columns if "_0_" in c or c.endswith("_p0")]
gold_tabular = gold_tabular.drop(*cols0)

cols_to_drop = [c for c in gold_tabular.columns if "start" in c or "end" in c]
gold_tabular = gold_tabular.drop(*cols_to_drop)

# COMMAND ----------


gold_tabular \
.write.format("delta") \
.mode("overwrite") \
.option("overwriteSchema", "true") \
.partitionBy("symbol") \
.save(f"{BASE_PATH}/gold/whale_trades_tabular") \




# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CAT}.{SCH}.gold__whale_trades_tabular
USING DELTA
LOCATION '{BASE_PATH}/gold/whale_trades_tabular'
""")


# COMMAND ----------

spark.sql(f"""OPTIMIZE dbrtfmprod.whale_trades.gold__whale_trades_tabular
ZORDER BY (ts);""")

# COMMAND ----------

display(gold_tabular)

# COMMAND ----------

for column in gold_tabular.columns:
    count = gold_tabular.groupBy(column).count().filter("count > 1").count()
    print(f"{column}: {count}")


# COMMAND ----------

# =========================
# VALIDACIÓN GOLD_TABULAR
# =========================

# 1. Conteo de filas y columnas
print("👉 Número de filas:", gold_tabular.count())
print("👉 Número de columnas:", len(gold_tabular.columns))

# 2. Preview de nombres de columnas
print("👉 Primeras 30 columnas:")
print(gold_tabular.columns[:30])

# 3. Comprobar duplicados en las claves base
dup_count = (
    gold_tabular
    .groupBy("symbol", "agg_trade_id", "ts")
    .count()
    .filter("count > 1")
    .count()
)
print("👉 Duplicados en las claves (symbol, agg_trade_id, ts):", dup_count)

# 4. Porcentaje de nulos por familia de columnas
familias = {
    "klines":    [c for c in gold_tabular.columns if c.startswith("kl_")],
    "mark":      [c for c in gold_tabular.columns if c.startswith("mark_")],
    "index":     [c for c in gold_tabular.columns if c.startswith("index_")],
    "premium":   [c for c in gold_tabular.columns if c.startswith("premium_")],
    "oi":        [c for c in gold_tabular.columns if c.startswith("oi_")],
    "taker":     [c for c in gold_tabular.columns if c.startswith("taker_")],
    "funding":   [c for c in gold_tabular.columns if c.startswith("funding_")],
    "labels":    [c for c in gold_tabular.columns if c.startswith("y_fwd_")]
}

n_rows = gold_tabular.count()
for fam, cols in familias.items():
    if cols:
        nulls = gold_tabular.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in cols]).first().asDict()
        null_pct = {c: round(nulls[c] / n_rows * 100, 2) for c in cols}
        non_empty = sum(1 for v in nulls.values() if v < n_rows)
        print(f"\n🔎 Familia {fam}: {len(cols)} columnas, {non_empty}/{len(cols)} con datos")
        # opcional: mostrar top 5 columnas con más nulos
        top_nulls = sorted(null_pct.items(), key=lambda x: -x[1])[:5]
        for col, pct in top_nulls:
            print(f"   {col}: {pct}% nulos")

# 5. Vista rápida de algunas filas
display(gold_tabular.select(
    "symbol",
    "ts",
    "price",
    "y_fwd_5m",
    "kl_0_close",
    "oi_0_oi",
    "taker_0_buy_vol"
).limit(10))

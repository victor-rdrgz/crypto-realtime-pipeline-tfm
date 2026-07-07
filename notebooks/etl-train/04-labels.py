# Databricks notebook source
# Celda 0 — Parámetros
from pyspark.sql import functions as F, Window as W

# Tablas/paths
FEAT_TABLE   = "dbrtfmprod.whale_trades.silver__features_1s"
LABEL_TABLE  = "dbrtfmprod.whale_trades.silver__labels_1s"
LABEL_PATH   = "abfss://datalake@statfmprod.dfs.core.windows.net/silver/labels_1s"

DATASET_TABLE = "dbrtfmprod.whale_trades.gold__dataset_1s"
DATASET_PATH  = "abfss://datalake@statfmprod.dfs.core.windows.net/gold/dataset_1s"

# Umbrales (en bps; 1 bp = 0.0001)
THR_BPS = 3
THR = THR_BPS / 10000.0

# binaria "UP": ret ≥ 100 bps (1%) → 1; si no → 0
RET_BPS_UP = 100
RET_THR_UP = RET_BPS_UP / 10000.0

# evita división por cero
EPS = 1e-12                

# HORIZONTES (segundos): 30m, 1h, 3h, 6h
HORIZONS = [1800, 3600, 10800, 21600]


# COMMAND ----------

# Celda 1 — Carga features y saneo mínimo
f_raw = spark.table(FEAT_TABLE)

# Aseguramos la presencia de las columnas basicas
need_base = {"symbol", "ts"}
missing = [c for c in need_base if c not in f_raw.columns]
if missing:
    raise ValueError(f"Faltan columnas base en {FEAT_TABLE}: {missing}")

# TS a timestamp y DATE (si no existe)
f = (f_raw
     .withColumn("ts", F.col("ts").cast("timestamp"))
    )
if "date" not in f.columns:
    f = f.withColumn("date", F.to_date("ts"))

# Asegura columna de precio 'mid'
if "mid" in f.columns:
    f = f.withColumn("mid", F.col("mid").cast("double"))
else:
    # Intentamos construir 'mid' a partir de otras columnas habituales
    candidates = [c for c in ["mid_px", "price", "close", "kl_0_close"] if c in f.columns]
    if len(candidates) > 0:
        f = f.withColumn("mid", F.col(candidates[0]).cast("double"))
    elif ("best_ask" in f.columns) and ("best_bid" in f.columns):
        f = f.withColumn("mid", (F.col("best_ask").cast("double") + F.col("best_bid").cast("double"))/2.0)
    else:
        raise ValueError("No encuentro columna de precio para construir 'mid' (busqué: mid, mid_px, price, close, kl_0_close o best_ask/best_bid).")

# Orden y partición por símbolo para ventanas
f = f.select("symbol", "date", "ts", "mid").repartition("symbol")


# COMMAND ----------

# Celda 2 — Futuros y retornos (sin leakage) para todos los horizontes
w = W.partitionBy("symbol").orderBy("ts")

labels = f
for h in HORIZONS:
    labels = (labels
        .withColumn(f"mid_fwd_{h}s", F.lead("mid", h).over(w))
        .withColumn(f"r_fwd_{h}s",
                    (F.col(f"mid_fwd_{h}s") - F.col("mid")) / (F.col("mid") + EPS))
    )


# COMMAND ----------

# Celda 3 — Etiquetas triclase y binaria (UP)
for h in HORIZONS:
    r = F.col(f"r_fwd_{h}s")
    # Triclase: +1, 0, -1 según umbral THR
    labels = (labels
        .withColumn(f"y{h}",
            F.when(r >  THR,  1)
             .when(r < -THR, -1)
             .otherwise(0))
    )
    # Binaria "UP": 1 si retorno ≥ RET_THR_UP; 0 si no
    labels = labels.withColumn(f"y{h}_up", (r >= F.lit(RET_THR_UP)).cast("int"))


# COMMAND ----------

# Celda 4 — Guardar labels en Delta (tabla Silver)
mid_cols = [f"mid_fwd_{h}s" for h in HORIZONS]
r_cols   = [f"r_fwd_{h}s"  for h in HORIZONS]
y_tri    = [f"y{h}"        for h in HORIZONS]
y_up     = [f"y{h}_up"     for h in HORIZONS]

labels_out = labels.select(
    "symbol","date","ts",
    *mid_cols,
    *r_cols,
    *y_tri,
    *y_up
)

(labels_out
  .write.mode("overwrite")
  .format("delta")
  .option("overwriteSchema","true")
  .save(LABEL_PATH))

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {LABEL_TABLE}
USING DELTA
LOCATION '{LABEL_PATH}'
""")


# COMMAND ----------

# Celda 5 — Dataset de entrenamiento (join 1:1 features + labels, sin duplicados)
FEAT_ALL = spark.table(FEAT_TABLE)

dataset = (FEAT_ALL.alias("x")
  .join(spark.table(LABEL_TABLE).alias("y"), ["symbol","date","ts"], "inner")
)

# Check de columnas duplicadas (debería estar vacío)
dupes = [c for c in dataset.columns if dataset.columns.count(c) > 1]
if dupes:
    raise ValueError(f"Columnas duplicadas tras el join: {sorted(set(dupes))}")

(dataset
  .write.mode("overwrite")
  .format("delta")
  .option("overwriteSchema","true")
  .save(DATASET_PATH))

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {DATASET_TABLE}
USING DELTA
LOCATION '{DATASET_PATH}'
""")

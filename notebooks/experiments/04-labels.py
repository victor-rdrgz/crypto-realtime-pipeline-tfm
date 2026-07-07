# Databricks notebook source
from pyspark.sql import functions as F, Window as W

# COMMAND ----------

FEAT_TABLE   = "dbrtfmprod.whale_trades.silver__features_1s"
LABEL_TABLE  = "dbrtfmprod.whale_trades.silver__labels_1s"
LABEL_PATH   = "abfss://datalake@statfmprod.dfs.core.windows.net/silver/labels_1s"

DATASET_TABLE = "dbrtfmprod.whale_trades.gold__dataset_1s"
DATASET_PATH  = "abfss://datalake@statfmprod.dfs.core.windows.net/gold/dataset_1s"

# Umbral de clasificación en bps (1 bp = 0.0001). Ejemplo: 3 bps = 0.0003
THR_BPS = 3
THR = THR_BPS / 10000.0

EPS = 1e-12

# COMMAND ----------

# Celda 1 — Cargar features y preparar ventanas
f = spark.table(FEAT_TABLE)

needed = ["symbol","date","ts","mid","sigma_30s","vol_5s","had_trade_1s","had_trade_5s"]
missing = [c for c in needed if c not in f.columns]
if missing:
    raise ValueError(f"Faltan columnas en {FEAT_TABLE}: {missing}")

f = f.select(*needed)

w = W.partitionBy("symbol").orderBy("ts")


# COMMAND ----------

# Celda 2 — Futuros y retornos futuros (sin leakage) — 10s, 30s, 120s, 300s, 600s
w = W.partitionBy("symbol").orderBy("ts")

labels = (f
  .withColumn("mid_fwd_10s",  F.lead("mid", 10).over(w))
  .withColumn("mid_fwd_30s",  F.lead("mid", 30).over(w))
  .withColumn("mid_fwd_120s", F.lead("mid",120).over(w))
  .withColumn("mid_fwd_300s", F.lead("mid",300).over(w))
  .withColumn("mid_fwd_600s", F.lead("mid",600).over(w))

  .withColumn("r_fwd_10s",  (F.col("mid_fwd_10s")  - F.col("mid"))/(F.col("mid")+EPS))
  .withColumn("r_fwd_30s",  (F.col("mid_fwd_30s")  - F.col("mid"))/(F.col("mid")+EPS))
  .withColumn("r_fwd_120s", (F.col("mid_fwd_120s") - F.col("mid"))/(F.col("mid")+EPS))
  .withColumn("r_fwd_300s", (F.col("mid_fwd_300s") - F.col("mid"))/(F.col("mid")+EPS))
  .withColumn("r_fwd_600s", (F.col("mid_fwd_600s") - F.col("mid"))/(F.col("mid")+EPS))
)


# COMMAND ----------

# Celda 3 — Etiquetas triclase por umbral en bps (10s, 30s, 120s, 300s, 600s)
labels = (labels
  .withColumn("y10",
      F.when(F.col("r_fwd_10s")  >  THR, 1).when(F.col("r_fwd_10s")  < -THR, -1).otherwise(0))
  .withColumn("y30",
      F.when(F.col("r_fwd_30s")  >  THR, 1).when(F.col("r_fwd_30s")  < -THR, -1).otherwise(0))
  .withColumn("y120",
      F.when(F.col("r_fwd_120s") >  THR, 1).when(F.col("r_fwd_120s") < -THR, -1).otherwise(0))
  .withColumn("y300",
      F.when(F.col("r_fwd_300s") >  THR, 1).when(F.col("r_fwd_300s") < -THR, -1).otherwise(0))
  .withColumn("y600",
      F.when(F.col("r_fwd_600s") >  THR, 1).when(F.col("r_fwd_600s") < -THR, -1).otherwise(0))
)


# COMMAND ----------

# Celda 4 — Guardar labels en Delta (Silver)
labels_out = (labels
  .select(
      "symbol","date","ts",
      # futuros y retornos
      "mid_fwd_10s","mid_fwd_30s","mid_fwd_120s","mid_fwd_300s","mid_fwd_600s",
      "r_fwd_10s","r_fwd_30s","r_fwd_120s","r_fwd_300s","r_fwd_600s",
      # etiquetas
      "y10","y30","y120","y300","y600"
  )
)

(labels_out
  .repartition("symbol","date")
  .write.mode("overwrite")
  .format("delta")
  .option("overwriteSchema","true")
  .partitionBy("symbol","date")
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
  .repartition("symbol","date")
  .write.mode("overwrite")
  .format("delta")
  .option("overwriteSchema","true")
  .partitionBy("symbol","date")
  .save(DATASET_PATH))

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {DATASET_TABLE}
USING DELTA
LOCATION '{DATASET_PATH}'
""")


# COMMAND ----------

# Celda 6 — QA: distribución de etiquetas y sanity (para todos los horizontes)
ds = spark.table(DATASET_TABLE).cache()

def dist(col):
    return (ds.filter(F.col(col).isNotNull())
             .groupBy("symbol", col).count().orderBy("symbol", col))

display(dist("y10"))
display(dist("y30"))
display(dist("y120"))
display(dist("y300"))
display(dist("y600"))

# nulls por target
for col in ["y10","y30","y120","y300","y600"]:
    n_all  = ds.count()
    n_null = ds.filter(F.col(col).isNull()).count()
    print(col, {"nulls": n_null, "rows": n_all})

# muestra reciente
display(ds.orderBy(F.asc("symbol"), F.desc("ts")).limit(30))
ds.unpersist()

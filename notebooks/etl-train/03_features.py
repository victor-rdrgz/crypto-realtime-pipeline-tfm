# Databricks notebook source
from pyspark.sql import functions as F, Window as W

# COMMAND ----------

# Ubicaciones para lectura y escritura
BASE_TABLE = "dbrtfmprod.whale_trades.silver__base_1s_trades"
FEAT_TABLE = "dbrtfmprod.whale_trades.silver__features_1s"
FEAT_PATH  = "abfss://datalake@statfmprod.dfs.core.windows.net/silver/features_1s"

# valor para evitar divisiones entre 0
EPS = 1e-12

# días que vamos a procesar +1 ya que en test, 
# tenemos retraso en ejecutar la siguiente consulta
DAYS_BACK = 181

# COMMAND ----------

# Celda 1 — Cargar base y validar columnas + definir ventanas
needed = [
    "symbol","date","ts","mid",
    "n_trades_1s","vol_base_1s","vol_quote_1s",
    "ofi_trades_1s","buy_aggr_qty_1s","sell_aggr_qty_1s","had_trade_1s"
]

cutoff_s = F.unix_timestamp(F.current_timestamp()) - (DAYS_BACK * 86400)

bdf = (spark.table(BASE_TABLE)
      .where(F.unix_timestamp("ts") >= cutoff_s)
      .select(*needed)
)


missing = [c for c in needed if c not in bdf.columns]
if missing:
    raise ValueError(f"Faltan columnas en {BASE_TABLE}: {missing}")

b = bdf.select(*needed)

# Ventanas por símbolo y día
w   = W.partitionBy("symbol","date").orderBy("ts")
w5  = w.rowsBetween(-4, 0)
w10 = w.rowsBetween(-9, 0)
w30 = w.rowsBetween(-29, 0)
w60 = w.rowsBetween(-59, 0)


# COMMAND ----------

# Celda 2 — Momentum y volatilidad (solo pasado)
feat = (b
  .withColumn("mid_lag_1s",  F.lag("mid", 1).over(w))
  .withColumn("mid_lag_5s",  F.lag("mid", 5).over(w))
  .withColumn("mid_lag_10s", F.lag("mid",10).over(w))
  .withColumn("mid_lag_30s", F.lag("mid",30).over(w))

  .withColumn("r_1s",  (F.col("mid")-F.col("mid_lag_1s"))  /(F.col("mid_lag_1s")+EPS))
  .withColumn("r_5s",  (F.col("mid")-F.col("mid_lag_5s"))  /(F.col("mid_lag_5s")+EPS))
  .withColumn("r_10s", (F.col("mid")-F.col("mid_lag_10s")) /(F.col("mid_lag_10s")+EPS))
  .withColumn("r_30s", (F.col("mid")-F.col("mid_lag_30s")) /(F.col("mid_lag_30s")+EPS))

  .withColumn("r_1s_sq", F.pow(F.col("r_1s"), 2))
  .withColumn("var_30s", F.avg("r_1s_sq").over(w30))
  .withColumn("sigma_30s", F.sqrt(F.col("var_30s")))

  .withColumn("z_r_1s", F.col("r_1s")/(F.col("sigma_30s")+EPS))
  .withColumn("z_r_5s", F.col("r_5s")/(F.col("sigma_30s")+EPS))
)


# COMMAND ----------

# Celda 3 — Volumen, OFI y bursts
feat = (feat
  .withColumn("vol_5s",      F.sum("vol_base_1s").over(w5))
  .withColumn("vol_60s",     F.sum("vol_base_1s").over(w60))
  .withColumn("n_trades_5s", F.sum("n_trades_1s").over(w5))
  .withColumn("ofi_5s",      F.sum("ofi_trades_1s").over(w5))

  .withColumn("avg_vol_per_s_60s",   F.col("vol_60s")/F.lit(60.0))
  .withColumn("vol_burst_5_over_60", F.col("vol_5s")/((F.col("avg_vol_per_s_60s")*F.lit(5.0))+EPS))

  .withColumn("ofi_to_vol_5s", F.col("ofi_5s")/(F.col("vol_5s")+EPS))
)


# COMMAND ----------

# Celda 4 — Compras/ventas agresoras (qty y notional) + desequilibrio
feat = (feat
  .withColumn("buy_quote_1s",  F.col("buy_aggr_qty_1s")  * F.col("mid"))
  .withColumn("sell_quote_1s", F.col("sell_aggr_qty_1s") * F.col("mid"))

  .withColumn("buy_qty_5s",    F.sum("buy_aggr_qty_1s").over(w5))
  .withColumn("sell_qty_5s",   F.sum("sell_aggr_qty_1s").over(w5))
  .withColumn("buy_quote_5s",  F.sum("buy_quote_1s").over(w5))
  .withColumn("sell_quote_5s", F.sum("sell_quote_1s").over(w5))

  .withColumn("imb_qty_5s",   F.col("buy_qty_5s") - F.col("sell_qty_5s"))
  .withColumn("imb_ratio_5s", (F.col("buy_qty_5s") - F.col("sell_qty_5s"))/
                              (F.col("buy_qty_5s") + F.col("sell_qty_5s") + EPS))
  .withColumn("buy_share_5s", F.col("buy_qty_5s")/(F.col("buy_qty_5s")+F.col("sell_qty_5s")+EPS))

  .withColumn("signed_quote_flow_5s", F.col("buy_quote_5s") - F.col("sell_quote_5s"))
)


# COMMAND ----------

# Celda 5 — Rachas 30s y actividad
feat = (feat
  .withColumn("min_mid_30s", F.min("mid").over(w30))
  .withColumn("max_mid_30s", F.max("mid").over(w30))
  .withColumn("drawup_30s",   (F.col("mid")-F.col("min_mid_30s"))/(F.col("min_mid_30s")+EPS))
  .withColumn("drawdown_30s", (F.col("max_mid_30s")-F.col("mid"))/(F.col("max_mid_30s")+EPS))

  .withColumn("had_trade_5s", (F.sum("had_trade_1s").over(w5) > 0).cast("int"))
)


# COMMAND ----------

# Celda 6 — Guardado de features

(feat
  .write.mode("overwrite")
  .format("delta")
  .option("overwriteSchema","true")
  .save(FEAT_PATH))

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {FEAT_TABLE}
USING DELTA
LOCATION '{FEAT_PATH}'
""")


# COMMAND ----------

# Celda 7 — OPTIMIZE
try:
    spark.sql(f"OPTIMIZE {FEAT_TABLE} ZORDER BY (ts)")
except Exception as e:
    print("OPTIMIZE no disponible o no necesario:", e)


# COMMAND ----------

# Celda 8 — QA de features (cobertura y consistencia)
'''feat = spark.table(FEAT_TABLE).cache()
base = spark.table(BASE_TABLE).select("symbol","ts")  # para comparar cobertura

# 1) Cobertura
rows_feat = feat.count()
rows_base = base.count()
miss_in_feat = (base.join(feat.select("symbol","ts"), ["symbol","ts"], "left_anti").count())

# 2) Columnas esperadas
expected = {
  "symbol","date","ts","mid",
  "r_1s","r_5s","r_10s","r_30s",
  "sigma_30s","z_r_1s","z_r_5s",
  "vol_5s","vol_60s","n_trades_5s","ofi_5s","ofi_to_vol_5s","vol_burst_5_over_60",
  "buy_qty_5s","sell_qty_5s","imb_qty_5s","imb_ratio_5s","buy_share_5s",
  "buy_quote_5s","sell_quote_5s","signed_quote_flow_5s",
  "drawup_30s","drawdown_30s","had_trade_1s","had_trade_5s"
}
missing_cols = sorted(list(expected - set(feat.columns)))

# 3) Checks de valores
checks = {
  "sigma_30s_neg":      feat.filter(F.col("sigma_30s") < 0).count(),
  "buy_share_outside":  feat.filter(~(F.col("buy_share_5s").between(0,1))).count(),
  "imb_ratio_outside":  feat.filter(~(F.col("imb_ratio_5s").between(-1,1))).count(),
  "ofi_to_vol_outside": feat.filter((F.col("vol_5s") > 0) & (~F.col("ofi_to_vol_5s").between(-1,1))).count(),
  "vol5_gt_vol60":      feat.filter(F.col("vol_5s") > F.col("vol_60s") + EPS).count(),
  "had_trade_5s_not01": feat.filter(~(F.col("had_trade_5s").isin(0,1))).count(),
}

agree_sign = (feat
  .withColumn("sign_imb", F.signum(F.col("imb_qty_5s")))
  .withColumn("sign_flow", F.signum(F.col("signed_quote_flow_5s")))
  .filter((F.col("sign_imb") != 0) & (F.col("sign_flow") != 0))
  .select((F.col("sign_imb") == F.col("sign_flow")).cast("int").alias("ok"))
  .agg(F.avg("ok").alias("agree_rate"))
  .collect()[0]["agree_rate"]
)

print({
  "rows_feat": rows_feat,
  "rows_base": rows_base,
  "missing_pairs_in_feat": miss_in_feat,
  "missing_cols": missing_cols,
  "checks": checks,
  "agree_sign(imb_vs_flow)_0-1": agree_sign
})

display(
  feat.groupBy("symbol").agg(
    F.count("*").alias("n"),
    F.min("ts").alias("min_ts"),
    F.max("ts").alias("max_ts")
  ).orderBy("symbol")
)

display(feat.orderBy(F.asc("symbol"), F.desc("ts")).limit(30))
feat.unpersist()'''

# Databricks notebook source
from pyspark.sql import functions as F, Window as W

# COMMAND ----------

# Ubicaciones para lectura y escritura
BRONZE_PATH  = "abfss://datalake@statfmprod.dfs.core.windows.net/bronze/tradesv2"
BRONZE_TABLE = "dbrtfmprod.whale_trades.bronze__top_kyles_all_trades"
TABLE_BRONZE__TOP_ALL_TRADES = 'bronze__top_kyles_all_trades'


# Tabla y ruta de salida (base 1s desde trades)
SILVER_TABLE = "dbrtfmprod.whale_trades.silver__base_1s_trades"
SILVER_PATH  = "abfss://datalake@statfmprod.dfs.core.windows.net/silver/base_1s_trades"

# días que vamos a procesar. uno más ya que en test, 
# tenemos retraso en ejecutar la siguiente consulta
DAYS_BACK = 181

# COMMAND ----------

# Celda 1 — Cargar bronze y preparar límites temporales por símbolo
cutoff_s = F.unix_timestamp(F.current_timestamp()) - (DAYS_BACK * 86400)

tr = (spark.table(BRONZE_TABLE)
      .where(F.unix_timestamp("ts") >= cutoff_s)
      .select("symbol","price","qty","is_buyer_maker","ts","sec","trade_value"))
print(tr.count())

# límites de calendario por símbolo (entre el primer y el último trade)
bounds = (tr.groupBy("symbol")
            .agg(F.min("ts").alias("min_ts"), F.max("ts").alias("max_ts")))
display(bounds.orderBy("symbol"))

# COMMAND ----------

# Celda 2 — Calendario a 1s por símbolo (sin huecos)
calendar = (bounds
    .withColumn("ts", F.explode(F.sequence("min_ts", "max_ts", F.expr("INTERVAL 1 SECOND"))))
    .select("symbol","ts")
)
# Para joins eficientes
calendar = calendar.repartition("symbol")
display(calendar.limit(5))

# COMMAND ----------

# Celda 3 — Agregados por segundo y último precio del segundo
# Nota: en Binance 'is_buyer_maker = False' => comprador agresivo (BUY); True => vendedor agresivo (SELL)

# Último precio del segundo (trade más cercano al final del segundo)
w_last = W.partitionBy("symbol","sec").orderBy(F.col("ts").desc())
last_price_1s = (tr
    .withColumn("rn", F.row_number().over(w_last))
    .filter("rn = 1")
    .select(F.col("symbol"), F.col("sec").alias("ts"),
            F.col("price").alias("last_price_1s"))
)

# Volúmenes y conteo por segundo
agg_1s = (tr.groupBy("symbol","sec")
    .agg(F.sum("qty").alias("vol_base_1s"),
         F.sum("trade_value").alias("vol_quote_1s"),
         F.count(F.lit(1)).alias("n_trades_1s"),
         F.sum(F.when(F.col("is_buyer_maker")==F.lit(False), F.col("qty")).otherwise(0.0)).alias("buy_aggr_qty_1s"),
         F.sum(F.when(F.col("is_buyer_maker")==F.lit(True),  F.col("qty")).otherwise(0.0)).alias("sell_aggr_qty_1s"))
    .withColumnRenamed("sec","ts")
)

display(agg_1s.orderBy(F.desc("ts")).limit(5))


# COMMAND ----------

# Celda 4 — Unir calendario + agregados + último precio y rellenar huecos
base = (calendar
    .join(agg_1s,       ["symbol","ts"], "left")
    .join(last_price_1s,["symbol","ts"], "left")
)

# Reparticionar por símbolo para mejorar distribución
base = base.repartition("symbol")

# Forward-fill por símbolo (arrastra último precio y contadores no nulos)
w_ff = W.partitionBy("symbol").orderBy("ts").rowsBetween(W.unboundedPreceding, 0)

# Precio continuo (proxy de mid = último trade conocido)
base = base.withColumn("mid", F.last(F.col("last_price_1s"), ignorenulls=True).over(w_ff))

# Volúmenes y conteos: si no hubo trades en ese segundo => 0
base = (base
  .fillna({"vol_base_1s":0.0, "vol_quote_1s":0.0,
           "n_trades_1s":0, "buy_aggr_qty_1s":0.0, "sell_aggr_qty_1s":0.0})
)

# Presión neta de flujo por trades (compras agresivas - ventas agresivas)
base = base.withColumn("ofi_trades_1s", F.col("buy_aggr_qty_1s") - F.col("sell_aggr_qty_1s"))

# Orden y columnas finales
base_1s = base.select(
    "symbol","ts",
    "mid",
    "last_price_1s",
    "vol_base_1s","vol_quote_1s","n_trades_1s",
    "buy_aggr_qty_1s","sell_aggr_qty_1s","ofi_trades_1s"
)

display(base_1s.orderBy(F.asc("symbol"), F.desc("ts")).limit(20))

# COMMAND ----------

# Celda 5 — Guardamos en ADLS2

full = (
    base_1s
      .withColumn("had_trade_1s", (F.col("n_trades_1s") > 0).cast("int"))
      .withColumn("price_1s", F.coalesce(F.col("last_price_1s"), F.col("mid")))
      .withColumn("date", F.to_date("ts"))
)

# Escribe la base FULL (todas las columnas)
(full
  .write
  .mode("overwrite")
  .format("delta")
  .option("overwriteSchema","true")
  .partitionBy("symbol")
  .save(SILVER_PATH)
)

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {SILVER_TABLE}
USING DELTA
LOCATION '{SILVER_PATH}'
""")

# COMMAND ----------

# Checks de salud de los datos
'''print("bad_mid:", full.filter((F.col("mid").isNull()) | (F.col("mid") <= 0)).count())
print("bad_price_1s:", full.filter((F.col("price_1s").isNull()) | (F.col("price_1s") <= 0)).count())
print("neg_vol_base:", full.filter(F.col("vol_base_1s") < 0).count())

print("bad_had_trade_zeros:", full.filter(
    (F.col("had_trade_1s") == 0) & ((F.col("n_trades_1s") != 0) | (F.col("vol_base_1s") != 0))
).count())

print("bad_had_trade_ones:", full.filter(
    (F.col("had_trade_1s") == 1) & (F.col("n_trades_1s") == 0)
).count())

print("bad_ofi_bounds:", full.filter(
    F.abs(F.col("ofi_trades_1s")) > F.col("vol_base_1s")
).count())

print("bad_aggr_sum:", full.filter(
    F.abs(F.col("buy_aggr_qty_1s") + F.col("sell_aggr_qty_1s") - F.col("vol_base_1s")) > 0
).count())

print("neg_buy_or_sell:", full.filter(
    (F.col("buy_aggr_qty_1s") < 0) | (F.col("sell_aggr_qty_1s") < 0)
).count())

print("bad_price_when_no_trade:", full.filter(
    (F.col("had_trade_1s") == 0) & (F.col("price_1s") != F.col("mid"))
).count())

print("bad_last_price_when_trade:", full.filter(
    (F.col("had_trade_1s") == 1) & (F.col("last_price_1s").isNull())
).count())'''
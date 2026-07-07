# Databricks notebook source
import requests

from datetime import datetime

from functools import reduce
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql import Window as W
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, 
    StructField, 
    StringType, 
    IntegerType, 
    LongType,
    DoubleType,
    BooleanType
)


# COMMAND ----------

schema_klines = StructType([
    StructField("symbol", StringType(), True),
    StructField("openTime", LongType(), True),           # timestamp en ms
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume_base", DoubleType(), True),         # volumen en moneda base
    StructField("closeTime", LongType(), True),          # timestamp en ms
    StructField("volume_quote", DoubleType(), True),        # volumen en moneda cotizada (USDT en la mayoría)
    StructField("trades", IntegerType(), True),             # nº de trades
    StructField("takerBuyBase", DoubleType(), True),        # volumen compradora (moneda base)
    StructField("takerBuyQuote", DoubleType(), True)        # volumen compradora (en USDT)
    StructField("interesting", BooleanType(), True),
    StructField("analyzed", BooleanType(), True)
])

# COMMAND ----------

symbols = spark.sql("""
    SELECT * 
    FROM dbrtfmprod.whale_trades.silver__binance_exchangeinfo_futures
""")

# display(symbols)

# COMMAND ----------

market = "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol="

# 1) Extraer symbols desde tu silver
symbols = [r["symbol"] for r in spark.sql("""
    SELECT DISTINCT symbol 
    FROM dbrtfmprod.whale_trades.silver__binance_exchangeinfo_futures
    WHERE status = 'TRADING'
""").collect()]

# 2) Función para traer velas 1D
def get_klines(symbol, limit=1500):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1d&limit={limit}"
    data = requests.get(url, timeout=10).json()

    if not isinstance(data, list) or len(data) == 0 or not isinstance(data[0], list):
        raise ValueError(f"Respuesta inesperada de Binance para {symbol}: {data}")

    rows = []
    for k in data:
        rows.append(Row(
            symbol=symbol,
            openTime=int(k[0]),
            open=float(k[1]),
            high=float(k[2]),
            low=float(k[3]),
            close=float(k[4]),
            volume_base=float(k[5]),
            closeTime=int(k[6]),
            volume_quote=float(k[7]),
            trades=int(k[8]),
            takerBuyBase=float(k[9]),
            takerBuyQuote=float(k[10])
        ))
    return rows

# 3) Recorrer todos los symbols
all_rows = []
for s in symbols:
    try:
        all_rows.extend(get_klines(s, limit=1500))
    except Exception as e:
        print(f"Error con {s}: {e}")
        raise

# 4) Crear DataFrame Spark
df_klines = spark.createDataFrame(all_rows, schema=schema_klines)

# 5) Normalizar fechas y seleccionar columnas útiles
df_klines = (
    df_klines
    .withColumn("date", F.to_date(F.from_unixtime(F.col("closeTime")/1000)))
    .drop("closeTime", "openTime")
)

# COMMAND ----------

df_klines.groupBy("symbol").agg(F.count("*").alias("n_dias")).filter("n_dias >= 1200").count()


# COMMAND ----------

# === Parámetros de ventana y umbrales ===
DAYS_WINDOW = 1500             # ventana de evaluación
MIN_VALID_DAYS = 1200          # días válidos mínimos dentro de los 1500
MAX_GAP_DAYS = 3               # huecos consecutivos máximos permitidos
BAD_RATE_MAX = 0.02            # 2% de días "malos" máximo
MED_VOLQ_90_FLOOR = 1_000_000  # mediana 90d de volumen_quote (USDT) mínima
MED_TRADES_90_FLOOR = 2000     # mediana 90d de trades mínima
STABLE_BAND = (0.98, 1.02)     # banda de precio para detectar "stable"
STABLE_BAND_SHARE = 0.80       # 80% de días en banda => estable
STABLE_ASSETS = {"USDC","USDT","FDUSD","TUSD","USDP","DAI","USD"}

# === Fechas de referencia ===
today = F.current_date()
start_1500 = F.date_sub(today, DAYS_WINDOW - 1)
start_1200  = F.date_sub(today, 1200 - 1)

# === Tablas de entrada ===
'''klines = spark.table("dbrtfmprod.whale_trades.silver__binance_futures_klines") \
    .select("symbol","date","open","high","low","close","volume_quote","trades","takerBuyQuote")'''

exchangeinfo = spark.table("dbrtfmprod.whale_trades.silver__binance_exchangeinfo_futures") \
    .select("symbol","status","baseAsset","quoteAsset")

# === Recorta a los últimos 1500 días ===
df1500 = df_klines.filter(F.col("date").between(start_1500, today))
# Día válido: volumen y trades positivos
df1500 = df1500.withColumn("is_valid_day",
                         (F.col("volume_quote") > 0) & (F.col("trades") > 0))

# ==== GAPS: máximo hueco de días consecutivos perdidos por símbolo en la ventana ====
w_seq = W.partitionBy("symbol").orderBy("date")
df_gap = (
    df1500
    .select("symbol","date")
    .withColumn("lag_date", F.lag("date").over(w_seq))
    # gap interno entre filas consecutivas; para la primera fila, compara con el inicio de ventana
    .withColumn(
        "gap_days",
        F.when(F.col("lag_date").isNull(),
            F.datediff(F.col("date"), start_1500)  # <-- usa fecha de inicio real
        ).otherwise(F.datediff(F.col("date"), F.col("lag_date")) - 1)
    )
)

# Para contar bien el gap inicial respecto al inicio de la ventana:
df_first = (
    df1500.groupBy("symbol")
    .agg(F.min("date").alias("first_date"))
    .withColumn("first_gap", F.datediff(F.col("first_date"), start_1500))
)

df_gap_max = (
    df_gap.groupBy("symbol").agg(F.max(F.coalesce(F.col("gap_days"), F.lit(0))).alias("max_internal_gap"))
    .join(df_first, "symbol", "left")
    .withColumn("max_gap_days", F.greatest(F.col("max_internal_gap"), F.col("first_gap")))
    .select("symbol","max_gap_days")
)

# ==== Agregados por símbolo en 1500d (calidad, estable, válidos) ====
agg1500 = (
    df1500.groupBy("symbol")
    .agg(
        F.count("*").alias("days_total_1500"),
        F.sum(F.when(F.col("is_valid_day"), 1).otherwise(0)).alias("valid_days_1500"),
        F.sum(F.when((F.col("volume_quote").isNull()) | (F.col("volume_quote") <= 0), 1).otherwise(0)).alias("days_volq_bad"),
        F.sum(F.when((F.col("trades").isNull()) | (F.col("trades") <= 0), 1).otherwise(0)).alias("days_trades_bad"),
        F.sum(F.when((F.col("takerBuyQuote").isNull()) | (F.col("takerBuyQuote") < 0) | (F.col("takerBuyQuote") > F.col("volume_quote")), 1).otherwise(0)).alias("days_taker_bad"),
        F.sum(F.when(F.col("close").between(*STABLE_BAND), 1).otherwise(0)).alias("days_close_in_band")
    )
    .withColumn("rate_volq_bad", F.col("days_volq_bad")/F.col("days_total_1500"))
    .withColumn("rate_trades_bad", F.col("days_trades_bad")/F.col("days_total_1500"))
    .withColumn("rate_taker_bad", F.col("days_taker_bad")/F.col("days_total_1500"))
    .withColumn("share_close_in_band", F.col("days_close_in_band")/F.col("days_total_1500"))
)

# ==== Medianas en 90d (liquidez y actividad) ====
df1500 = df1500.filter(F.col("date") >= start_1200)
agg90 = (
    df1500.groupBy("symbol")
    .agg(
        F.expr("percentile_approx(volume_quote, 0.5, 500)").alias("med_volq_90"),
        F.expr("percentile_approx(trades, 0.5, 500)").alias("med_trades_90"),
        F.count("*").alias("days_total_90")
    )
)

# ==== Join con exchangeinfo para estado/trading y baseAsset ====
df_status = exchangeinfo

# ==== Ensamble de métricas ====
metrics = (agg1500
           .join(agg90, "symbol", "left")
           .join(df_gap_max, "symbol", "left")
           .join(df_status, "symbol", "left"))

# ==== Flags de exclusión (TRUE = excluir) ====
metrics = (
    metrics
    # 1) Histórico mínimo
    .withColumn("ex_hist_min", F.col("valid_days_1500") < F.lit(MIN_VALID_DAYS))
    # 2) Huecos > 3 días
    .withColumn("ex_gaps", F.col("max_gap_days") > F.lit(MAX_GAP_DAYS))
    # 3) Calidad de datos (>2% de días con problemas)
    .withColumn("ex_data_quality",
                (F.col("rate_volq_bad") > BAD_RATE_MAX) |
                (F.col("rate_trades_bad") > BAD_RATE_MAX) |
                (F.col("rate_taker_bad") > BAD_RATE_MAX))
    # 4) Estables: baseAsset estable o 80% de días precio ~1
    .withColumn("ex_stable_by_asset", F.col("baseAsset").isin(list(STABLE_ASSETS)))
    .withColumn("ex_stable_by_price", F.col("share_close_in_band") >= F.lit(STABLE_BAND_SHARE))
    .withColumn("ex_stable", F.col("ex_stable_by_asset") | F.col("ex_stable_by_price"))
    # 6) Piso de liquidez
    .withColumn("ex_liquidity", F.col("med_volq_90") < F.lit(MED_VOLQ_90_FLOOR))
    # 7) Piso de actividad
    .withColumn("ex_activity", F.col("med_trades_90") < F.lit(MED_TRADES_90_FLOOR))
    # 8) Estado no operativo
    .withColumn("ex_status", (F.col("status").isNull()) | (F.col("status") != F.lit("TRADING")))
)

# ==== Síntesis final ====
exclude_cols = ["ex_hist_min","ex_gaps","ex_data_quality","ex_stable","ex_liquidity","ex_activity","ex_status"]
metrics = metrics.withColumn(
    "exclude_any",
    reduce(lambda a, b: a | b, [F.col(c) for c in exclude_cols])
)
aptos = metrics.filter(~F.col("exclude_any")) \
               .select("symbol","valid_days_1500","med_volq_90","med_trades_90","status")

excluidos = metrics.filter(F.col("exclude_any")) \
    .select("symbol","status","valid_days_1500","max_gap_days",
            "rate_volq_bad","rate_trades_bad","rate_taker_bad","share_close_in_band",
            "med_volq_90","med_trades_90",
            *exclude_cols)

# === Mostrar resultados ===
print("APTOS (universo limpio):")
display(aptos.orderBy(F.desc("med_volq_90")))

print("EXCLUIDOS (con motivo):")
display(excluidos.orderBy("symbol"))


# COMMAND ----------

# Peso más alto para impacto real, pero CRIT_1 y CRIT_2 también suman
WEIGHTS = {
    "lambda": 0.6,
    "crit2": 0.25,
    "crit1": 0.15
}

# COMMAND ----------

aptos_symbols = [r["symbol"] for r in aptos.select("symbol").collect()]

aptos = (
    df_klines
    .filter(
        (F.col("symbol").isin(aptos_symbols)) &
        (F.col("date").between(start_1500, today))
    )
)

df_kyles_base = (
    aptos
    .filter(
        (F.col("date").between(start_1500, today)) &
        (F.col("volume_quote") > 0)
    )
    .withColumn("takerBuyQuote", F.least(F.col("takerBuyQuote"), F.col("volume_quote")))  # aseguramos consistencia
    .withColumn("r", F.log("close") - F.log("open"))  # retorno logarítmico
    .withColumn("q", 2 * F.col("takerBuyQuote") - F.col("volume_quote"))  # flujo firmado neto en quote
)

# === Agrega por símbolo para obtener λ ===
kyles_por_simbolo = (
    df_kyles_base
    .groupBy("symbol")
    .agg(
        F.covar_samp("r", "q").alias("cov_r_q"),
        F.var_samp("q").alias("var_q"),
        F.count("*").alias("n_dias_validos")
    )
    .withColumn(
        "kyles_lambda",
        F.when(F.col("var_q") > 0, F.abs(F.col("cov_r_q") / F.col("var_q")))
         .otherwise(F.lit(None))
    )
    .select("symbol", "kyles_lambda", "n_dias_validos")
    .orderBy(F.col("kyles_lambda").desc_nulls_last())
)

# === Mostrar resultados ===
display(kyles_por_simbolo)

# COMMAND ----------

# === Crea flags diarios ===
CRIT_1 = (F.abs(F.col("close") - F.col("open")) / F.col("open") > 0.05)
CRIT_2 = ((F.col("high") - F.col("low")) / F.col("open") > 0.20)

# === Datos diarios de los símbolos aptos en los últimos 120 días ===
df_diario = (
    df_klines
    .join(aptos.select("symbol").distinct(), on="symbol", how="left_semi")
    .filter(
        (F.col("date").between(F.lit(start_1200), F.lit(today))) &
        (F.col("volume_quote") > F.lit(0))
    )
    .withColumn("takerBuyQuote", F.least(F.col("takerBuyQuote"), F.col("volume_quote")))
    .withColumn("r", F.log("close") - F.log("open"))
    .withColumn("q", 2 * F.col("takerBuyQuote") - F.col("volume_quote"))
    .withColumn("c1", CRIT_1.cast("double"))
    .withColumn("c2", CRIT_2.cast("double"))
)

# === Agregados por símbolo ===
agg_por_symbol = (
    df_diario.groupBy("symbol")
    .agg(
        F.covar_samp("r", "q").alias("cov_r_q"),
        F.var_samp("q").alias("var_q"),
        F.avg("c1").alias("pct_cumple_crit1"),
        F.avg("c2").alias("pct_cumple_crit2"),
        F.count("*").alias("n_dias_validos")
    )
    .withColumn(
        "kyles_lambda",
        F.when(F.col("var_q") > 0, F.abs(F.col("cov_r_q") / F.col("var_q")))
         .otherwise(F.lit(None))
    )
    .filter(F.col("kyles_lambda").isNotNull())
)

'''# === Normalizar lambda entre 0..1
lambda_bounds = agg_por_symbol.agg(
    F.min("kyles_lambda").alias("lambda_min"),
    F.max("kyles_lambda").alias("lambda_max")
).collect()[0]

lambda_min = lambda_bounds["lambda_min"]
lambda_max = lambda_bounds["lambda_max"]
lambda_range = lambda_max - lambda_min if lambda_max != lambda_min else 1.0

agg_por_symbol = agg_por_symbol.withColumn(
    "lambda_normalizada",
    (F.col("kyles_lambda") - F.lit(lambda_min)) / F.lit(lambda_range)
)'''


w_all = Window.orderBy(F.lit(1)).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

agg_por_symbol = (
    agg_por_symbol
    .withColumn("lambda_min", F.min("kyles_lambda").over(w_all))
    .withColumn("lambda_max", F.max("kyles_lambda").over(w_all))
    .withColumn(
        "lambda_range",
        F.when(F.col("lambda_max") != F.col("lambda_min"),
               F.col("lambda_max") - F.col("lambda_min")).otherwise(F.lit(1.0))
    )
    .withColumn(
        "lambda_normalizada",
        (F.col("kyles_lambda") - F.col("lambda_min")) / F.col("lambda_range")
    )
    .drop("lambda_min", "lambda_max", "lambda_range")
)

# === Score final ponderado
agg_por_symbol = agg_por_symbol.withColumn(
    "score_final",
    WEIGHTS["lambda"] * F.col("lambda_normalizada") +
    WEIGHTS["crit2"]  * F.col("pct_cumple_crit2") +
    WEIGHTS["crit1"]  * F.col("pct_cumple_crit1")
)



# COMMAND ----------

aptos_metrics = (
    df_diario.groupBy("symbol")
    .agg(
        F.expr("percentile_approx(volume_quote, 0.5)").alias("med_volq_90"),
        F.expr("percentile_approx(trades, 0.5)").alias("med_trades_90"),
        F.count("*").alias("valid_days_1500"),
    )
)

# COMMAND ----------

# === Join con info asociada desde aptos
aptos_con_score = (
    agg_por_symbol.join(df_klines, on="symbol", how="inner")
    .orderBy(F.col("score_final").desc())
)

# === TOP 10 FINAL CON TODA LA INFO ===
# top10_criptos_sensibles = aptos_con_score.limit(10)

# COMMAND ----------

display(df_klines)

# COMMAND ----------

# === Mostrar en pantalla ===
display(aptos_con_score.limit(10))

# COMMAND ----------

aptos_con_score = (
    aptos_con_score
        .select(
            "symbol",
            "score_final",
            "open",
            "high",
            "low",
            "close",
            "volume_base",
            "volume_quote",
            "trades",
            "takerBuyBase",
            "takerBuyQuote",
            "date"
        )
        .dropDuplicates(["symbol"])
        .withColumn("interesting", F.lit(True))
        .withColumn("analyzed", F.lit(False))
)

aptos_con_score = aptos_con_score.withColumn("date", F.current_date())

aptos_con_score.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("date") \
    .save("abfss://datalake@statfmprod.dfs.core.windows.net/gold/kyles_top")

# Crear tabla si no existe (opcional)
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS dbrtfmprod.whale_trades.gold__kyles_top
    USING DELTA
    LOCATION 'abfss://datalake@statfmprod.dfs.core.windows.net/gold/kyles_top'
""")

# COMMAND ----------

display(spark.sql(f"""SELECT * FROM dbrtfmprod.whale_trades.gold__kyles_top"""))

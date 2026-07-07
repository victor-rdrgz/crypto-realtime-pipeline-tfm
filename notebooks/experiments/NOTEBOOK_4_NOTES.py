# Databricks notebook source
display((spark.sql("SELECT DISTINCT symbol FROM dbrtfmprod.whale_trades.bronze__top_kyles_all_trades")))

# COMMAND ----------

display((spark.sql("SELECT * FROM dbrtfmprod.whale_trades.bronze__top_kyles_all_trades")).columns)

# COMMAND ----------

from pyspark.sql import functions as F

df = spark.sql("SELECT * FROM dbrtfmprod.whale_trades.bronze__top_kyles_all_trades")
df = df.withColumn('amount', F.col("price")*F.col("qty"))
display(
    df_full.groupBy("symbol")
      .agg(
          F.round(
            F.expr("percentile_approx(amount, 0.95, 10)").alias("p95"), 5
          )
      )
)

# COMMAND ----------

result = (
    df_full.groupBy("symbol")
      .agg(
          F.expr("percentile_approx(amount, 0.95, 10)").alias("p95")
      )
)

# COMMAND ----------

display(result)

# COMMAND ----------


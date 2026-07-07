# Databricks notebook source
# Celda 0 — Parámetros + MLflow
from pyspark.sql import functions as F, Window as W
from pyspark.sql import types as T
import mlflow, json

# Tabla GOLD con tus columnas
DATASET_TABLE = "dbrtfmprod.whale_trades.gold__dataset_1s"

# Objetivo: +2% en 10 minutos
H = 600
RETURN_THR = 0.01

# MLOps
EXPERIMENT_PATH = f"/Shared/exp_crypto_size_only_up2pct_y{H}"
REGISTERED_MODEL_NAME = f"dbrtfmprod.whale_trades.crypto_size_only_lr_y{H}"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.autolog(disable=True); mlflow.spark.autolog(disable=True)

spark.conf.set("spark.sql.shuffle.partitions", "64")


# COMMAND ----------

# Celda 1 — Cargar 45 días y preparar tamaño + label (>2% en 10m)
from pyspark.sql import functions as F, Window as W

NEEDED = [
    "symbol","date","ts","mid",
    "had_trade_5s",
    "buy_qty_5s","sell_qty_5s",
    "r_fwd_600s","mid_fwd_600s"
]
ds = spark.table(DATASET_TABLE)
missing = [c for c in NEEDED if c not in ds.columns]
if missing:
    # si falta r_fwd o mid_fwd, las calculamos luego; si faltan buy/sell qty no podemos seguir
    soft = set(["r_fwd_600s","mid_fwd_600s"])
    hard_missing = [m for m in missing if m not in soft]
    if hard_missing:
        raise ValueError(f"Faltan columnas obligatorias en {DATASET_TABLE}: {hard_missing}")
sel_cols = [c for c in NEEDED if c in ds.columns]
df0 = ds.select(*sel_cols)

# 45 días exactos
max_date = df0.agg(F.max("date").alias("d")).collect()[0]["d"]
start_45 = spark.sql(f"select date_sub(date('{max_date}'), 44) as d").collect()[0]["d"]
df0 = df0.filter(F.col("date") >= F.lit(start_45))

# Warm-up y actividad
wmin = W.partitionBy("symbol")
df0 = (df0
       .withColumn("min_ts_unix", F.min(F.col("ts").cast("long")).over(wmin))
       .filter(F.col("ts").cast("long") >= F.col("min_ts_unix") + F.lit(60))
       .drop("min_ts_unix"))
df0 = df0.filter(F.col("had_trade_5s") == 1)

# Tamaño (SOLO tamaño): suma por lado en 5s y máxima de los dos
df0 = df0.withColumn("size_5s", F.greatest(F.col("buy_qty_5s"), F.col("sell_qty_5s")).cast("double"))

# Si falta r_fwd_600s lo calculamos
if "r_fwd_600s" not in df0.columns or "mid_fwd_600s" not in df0.columns:
    w = W.partitionBy("symbol").orderBy("ts")
    df0 = (df0
           .withColumn("mid_fwd_600s", F.lead("mid", H).over(w))
           .withColumn("r_fwd_600s", (F.col("mid_fwd_600s") - F.col("mid"))/(F.col("mid")+F.lit(1e-12))))
    df0 = df0.filter(F.col("mid_fwd_600s").isNotNull())

# Label: sube > 2% en 10 min
df0 = df0.withColumn("label", (F.col("r_fwd_600s") >= F.lit(RETURN_THR)).cast("int"))

# Sanea
df0 = df0.filter(F.col("size_5s").isNotNull() & ~F.isnan("size_5s"))

print({"min_date": str(start_45), "max_date": str(max_date), "rows_45d": df0.count()})
display(df0.groupBy("symbol").agg(F.count("*").alias("rows")).orderBy("symbol"))


# COMMAND ----------

# Celda 2 — Split temporal 31/7/7 con purga de 600s
test_start = spark.sql(f"select date_sub(date('{max_date}'), 6) as d").collect()[0]["d"]
val_end    = spark.sql(f"select date_sub(date('{test_start}'), 1) as d").collect()[0]["d"]
val_start  = spark.sql(f"select date_sub(date('{val_end}'),   6) as d").collect()[0]["d"]
train_end  = spark.sql(f"select date_sub(date('{val_start}'), 1) as d").collect()[0]["d"]

print({"train":[str(start_45), str(train_end)],
       "val":[str(val_start), str(val_end)],
       "test":[str(test_start), str(max_date)]})

train_df = (df0
    .filter((F.col("date") >= F.lit(start_45)) & (F.col("date") <= F.lit(train_end)))
    .filter(F.col("ts") < F.expr(f"timestamp('{val_start} 00:00:00') - INTERVAL {H} SECONDS"))
    .select("symbol","date","ts","size_5s","label"))

val_df = (df0
    .filter((F.col("date") >= F.lit(val_start)) & (F.col("date") <= F.lit(val_end)))
    .filter(F.col("ts") < F.expr(f"timestamp('{test_start} 00:00:00') - INTERVAL {H} SECONDS"))
    .select("symbol","date","ts","size_5s","label"))

test_df = (df0
    .filter(F.col("date") >= F.lit(test_start))
    .select("symbol","date","ts","size_5s","label"))

print({
    "train_rows": train_df.count(),
    "val_rows":   val_df.count(),
    "test_rows":  test_df.count()
})
display(train_df.groupBy("label").count().orderBy("label"))


# COMMAND ----------

# Celda 3 — Candidatos de umbral SOLO con TRAIN (percentiles altos)
from pyspark.sql import functions as F, Window as W

# Parrilla más extrema (añadimos p99.95 y p99.995)
probs = [0.99, 0.995, 0.9975, 0.999, 0.9995, 0.9997, 0.9999, 0.99995]

exprs = [F.expr(f"percentile_approx(size_5s, {p}, 10000)").alias(f"q_{str(p).replace('.', '')}") for p in probs]
qdf_train = train_df.groupBy("symbol").agg(*exprs)

# “Stack” a filas (symbol, thr)
cand_cols = [c for c in qdf_train.columns if c != "symbol"]
stack_expr = "stack({n}, {pairs}) as (thr_name, thr)".format(
    n=len(cand_cols),
    pairs=",".join([f"'{c}', {c}" for c in cand_cols])
)
thr_train = (qdf_train
             .select("symbol", F.expr(stack_expr))
             .select("symbol","thr")
             .filter(F.col("thr").isNotNull())
             .cache())

print("N candidatos de umbral por símbolo:", thr_train.groupBy("symbol").count().orderBy("symbol").collect())


# COMMAND ----------

# Celda 4 — Selección de umbral por símbolo con VALID, evaluación en TEST
from pyspark.sql import functions as F, Window as W

def pick_best_on_valid(val_df, thr_df):
    cand = (val_df.alias("x")
            .join(thr_df.alias("t"), "symbol", "inner")
            .withColumn("pred", (F.col("x.size_5s") >= F.col("t.thr")).cast("int")))

    agg = (cand.groupBy("symbol","thr")
           .agg(
             F.sum(F.when((F.col("label")==1) & (F.col("pred")==1), 1).otherwise(0)).alias("TP"),
             F.sum(F.when((F.col("label")==0) & (F.col("pred")==1), 1).otherwise(0)).alias("FP"),
             F.sum(F.when((F.col("label")==1) & (F.col("pred")==0), 1).otherwise(0)).alias("FN"),
             F.count("*").alias("N")
           )
           .withColumn("f1", F.when((F.col("TP")+F.col("FP")+F.col("FN"))>0,
                                    2*F.col("TP")/(2*F.col("TP")+F.col("FP")+F.col("FN"))).otherwise(F.lit(0.0)))
           # desempate: a igual f1, coge umbral más ALTO (más conservador) y menos FP
           .withColumn("fp_neg", -F.col("FP"))
          )

    w = W.partitionBy("symbol").orderBy(F.desc("f1"), F.desc("thr"), F.desc("fp_neg"))
    best = (agg
            .withColumn("rk", F.row_number().over(w))
            .filter(F.col("rk")==1)
            .select("symbol","thr","f1","TP","FP","FN","N"))
    return best

def eval_split(split_df, best_thr):
    pred = (split_df.alias("x")
            .join(best_thr.select("symbol","thr").alias("b"), "symbol", "inner")
            .withColumn("pred", (F.col("x.size_5s") >= F.col("b.thr")).cast("int")))

    m = (pred.agg(
        F.sum(F.when((F.col("label")==1) & (F.col("pred")==1), 1).otherwise(0)).alias("TP"),
        F.sum(F.when((F.col("label")==0) & (F.col("pred")==1), 1).otherwise(0)).alias("FP"),
        F.sum(F.when((F.col("label")==1) & (F.col("pred")==0), 1).otherwise(0)).alias("FN"),
        F.sum(F.when((F.col("label")==0) & (F.col("pred")==0), 1).otherwise(0)).alias("TN")
    ).collect()[0])
    TP,FP,FN,TN = [m[c] or 0 for c in ["TP","FP","FN","TN"]]
    f1 = (2*TP)/(2*TP+FP+FN) if (2*TP+FP+FN)>0 else 0.0
    acc = (TP+TN)/max(TP+FP+FN+TN,1)
    prec = TP/max(TP+FP,1)
    rec  = TP/max(TP+FN,1)
    return {"TP":TP,"FP":FP,"FN":FN,"TN":TN,"precision":prec,"recall":rec,"f1":f1,"accuracy":acc}

# 1) Elegimos umbral por símbolo con VALID
best_thr_valid = pick_best_on_valid(val_df, thr_train).cache()
print("Umbrales elegidos en VALID:")
display(best_thr_valid.orderBy("symbol"))

# 2) Métricas en VALID (con esos umbrales)
val_metrics  = eval_split(val_df,  best_thr_valid)

# 3) Métricas en TEST (holdout final)
test_metrics = eval_split(test_df, best_thr_valid)

print({"VAL_threshold": val_metrics, "TEST_threshold": test_metrics})


# COMMAND ----------

# Celda 5 — B) Logistic Regression con UNA sola feature (log1p(size_5s))
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Feature única
train_lr = (train_df
            .withColumn("feat1", F.log1p(F.col("size_5s")).cast("double"))
            .select("symbol","date","ts","label","feat1")).cache()
val_lr   = (val_df
            .withColumn("feat1", F.log1p(F.col("size_5s")).cast("double"))
            .select("symbol","date","ts","label","feat1")).cache()
test_lr  = (test_df
            .withColumn("feat1", F.log1p(F.col("size_5s")).cast("double"))
            .select("symbol","date","ts","label","feat1")).cache()

assembler = VectorAssembler(inputCols=["feat1"], outputCol="feat_vec", handleInvalid="skip")
scaler    = StandardScaler(inputCol="feat_vec", outputCol="features", withMean=True, withStd=True)

lr = LogisticRegression(
    featuresCol="features", labelCol="label",
    maxIter=100, regParam=0.03, elasticNetParam=0.0,  # L2
    standardization=False
)

pipe = Pipeline(stages=[assembler, scaler, lr])

with mlflow.start_run(run_name=f"size_only_lr_y{H}_31_7_7") as run:
    mlflow.log_params({"model":"logreg_size_only", "feature":"log1p(size_5s)", "H":H, "return_thr":RETURN_THR})

    mdl = pipe.fit(train_lr)

    pred_val  = mdl.transform(val_lr).withColumn("prediction", F.col("prediction").cast("double")).withColumn("label", F.col("label").cast("double")).dropna(subset=["prediction","label"]).cache()
    pred_test = mdl.transform(test_lr).withColumn("prediction", F.col("prediction").cast("double")).withColumn("label", F.col("label").cast("double")).dropna(subset=["prediction","label"]).cache()

    eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    eval_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    eval_pr  = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability", metricName="areaUnderPR")

    val_acc = float(eval_acc.evaluate(pred_val));  val_f1 = float(eval_f1.evaluate(pred_val));  val_aupr = float(eval_pr.evaluate(pred_val))
    test_acc = float(eval_acc.evaluate(pred_test)); test_f1 = float(eval_f1.evaluate(pred_test)); test_aupr = float(eval_pr.evaluate(pred_test))

    mlflow.log_metrics({
        "val_accuracy": val_acc, "val_f1": val_f1, "val_aupr": val_aupr,
        "test_accuracy": test_acc, "test_f1": test_f1, "test_aupr": test_aupr
    })

    # Registro del modelo (pipeline)
    from mlflow.models import infer_signature
    sample_pd = train_lr.select("feat1").limit(5).toPandas()
    signature = infer_signature(sample_pd, mdl.transform(train_lr.limit(5)).select("prediction").toPandas())
    input_example = sample_pd.head(1)

    try:
        info = mlflow.spark.log_model(mdl, artifact_path="final_model_size_only_lr",
                                      registered_model_name=REGISTERED_MODEL_NAME,
                                      signature=signature, input_example=input_example)
        mlflow.set_tag("registered_version", getattr(info, "registered_model_version", "unknown"))
    except Exception as e:
        mlflow.set_tag("register_error", str(e))
        mlflow.spark.log_model(mdl, artifact_path="final_model_size_only_lr",
                               signature=signature, input_example=input_example)

    print({"VAL":{"acc":val_acc,"f1":val_f1,"aupr":val_aupr},
           "TEST":{"acc":test_acc,"f1":test_f1,"aupr":test_aupr}})

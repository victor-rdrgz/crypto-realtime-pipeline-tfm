# Databricks notebook source
from pyspark.sql.functions import input_file_name, current_timestamp
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType,
    IntegerType, BooleanType, ArrayType, MapType
)
from pyspark.sql import functions as F

# COMMAND ----------

# Esquema Binance futures
schema_binance_futures = StructType([
    StructField("timezone", StringType(), True),
    StructField("serverTime", LongType(), True),
    StructField("futuresType", StringType(), True),  # p.ej. "UMFUTURE" (opcional)
    StructField("rateLimits", ArrayType(StructType([
        StructField("rateLimitType", StringType(), True),   # REQUEST_WEIGHT, ORDERS, RAW_REQUESTS
        StructField("interval", StringType(), True),        # SECOND, MINUTE, DAY...
        StructField("intervalNum", IntegerType(), True),
        StructField("limit", IntegerType(), True),
    ])), True),
    StructField("exchangeFilters", ArrayType(StringType()), True),

    # A veces aparece; déjalo opcional por si no viene
    StructField("assets", ArrayType(StructType([
        StructField("asset", StringType(), True),
        StructField("marginAvailable", BooleanType(), True),
        StructField("autoAssetExchange", StringType(), True),
    ])), True),

    StructField("symbols", ArrayType(StructType([
        StructField("symbol", StringType(), True),
        StructField("pair", StringType(), True),
        StructField("contractType", StringType(), True),       # PERPETUAL | CURRENT_QUARTER | NEXT_QUARTER...
        StructField("deliveryDate", LongType(), True),         # epoch ms
        StructField("onboardDate", LongType(), True),          # epoch ms
        StructField("status", StringType(), True),             # TRADING...
        StructField("maintMarginPercent", StringType(), True),
        StructField("requiredMarginPercent", StringType(), True),
        StructField("baseAsset", StringType(), True),
        StructField("quoteAsset", StringType(), True),
        StructField("marginAsset", StringType(), True),
        StructField("pricePrecision", IntegerType(), True),
        StructField("quantityPrecision", IntegerType(), True),
        StructField("baseAssetPrecision", IntegerType(), True),
        StructField("quotePrecision", IntegerType(), True),
        StructField("underlyingType", StringType(), True),     # COIN | INDEX | PREMARKET...
        StructField("underlyingSubType", ArrayType(StringType()), True),  # a veces vacío o ausente
        StructField("settlePlan", IntegerType(), True),
        StructField("triggerProtect", StringType(), True),     # definido en ExchangeInfo
        StructField("liquidationFee", StringType(), True),
        StructField("marketTakeBound", StringType(), True),
        StructField("maxMoveOrderLimit", IntegerType(), True),

        StructField("filters", ArrayType(StructType([
            StructField("filterType", StringType(), True),

            # PRICE_FILTER
            StructField("minPrice", StringType(), True),
            StructField("maxPrice", StringType(), True),
            StructField("tickSize", StringType(), True),

            # LOT_SIZE / MARKET_LOT_SIZE
            StructField("minQty", StringType(), True),
            StructField("maxQty", StringType(), True),
            StructField("stepSize", StringType(), True),

            # MAX_NUM_ORDERS / MAX_NUM_ALGO_ORDERS
            StructField("limit", IntegerType(), True),

            # MIN_NOTIONAL (en USDⓈ-M)
            StructField("notional", StringType(), True),

            # PERCENT_PRICE
            StructField("multiplierUp", StringType(), True),
            StructField("multiplierDown", StringType(), True),
            StructField("multiplierDecimal", StringType(), True),

            # POSITION_RISK_CONTROL (si aparece)
            StructField("positionControlSide", StringType(), True),

            # TRAILING_DELTA (si aparece en algún símbolo)
            StructField("minTrailingAboveDelta", StringType(), True),
            StructField("maxTrailingAboveDelta", StringType(), True),
            StructField("minTrailingBelowDelta", StringType(), True),
            StructField("maxTrailingBelowDelta", StringType(), True),
        ])), True),

        StructField("orderTypes", ArrayType(StringType()), True),
        StructField("timeInForce", ArrayType(StringType()), True),
        StructField("permissionSets", ArrayType(StringType()), True),
    ])), True),
])

# Schema bybit futures
schema_bybit_futures = StructType([
    StructField("retCode", IntegerType(), True),
    StructField("retMsg", StringType(), True),
    StructField("result", StructType([
        StructField("category", StringType(), True),
        StructField("nextPageCursor", StringType(), True),
        StructField("list", ArrayType(StructType([
            StructField("symbol", StringType(), True),
            StructField("contractType", StringType(), True),   # p.ej. LinearPerpetual / LinearFutures
            StructField("status", StringType(), True),         # Trading / ...
            StructField("baseCoin", StringType(), True),
            StructField("quoteCoin", StringType(), True),
            # Bybit devuelve estos ms como STRING en la API v5 → los dejamos en StringType
            StructField("launchTime", StringType(), True),
            StructField("deliveryTime", StringType(), True),
            StructField("deliveryFeeRate", StringType(), True),
            StructField("priceScale", StringType(), True),

            StructField("leverageFilter", StructType([
                StructField("minLeverage", StringType(), True),
                StructField("maxLeverage", StringType(), True),
                StructField("leverageStep", StringType(), True),
            ]), True),

            StructField("priceFilter", StructType([
                StructField("minPrice", StringType(), True),
                StructField("maxPrice", StringType(), True),
                StructField("tickSize", StringType(), True),
            ]), True),

            StructField("lotSizeFilter", StructType([
                StructField("minNotionalValue", StringType(), True),
                StructField("maxOrderQty", StringType(), True),
                StructField("maxMktOrderQty", StringType(), True),
                StructField("minOrderQty", StringType(), True),
                StructField("qtyStep", StringType(), True),
                StructField("postOnlyMaxOrderQty", StringType(), True),  # deprecado pero aún aparece
            ]), True),

            StructField("unifiedMarginTrade", BooleanType(), True),
            StructField("fundingInterval", IntegerType(), True),
            StructField("settleCoin", StringType(), True),

            StructField("copyTrading", StringType(), True),     # "none" | "both" | ...
            StructField("upperFundingRate", StringType(), True),
            StructField("lowerFundingRate", StringType(), True),
            StructField("displayName", StringType(), True),     # más común en USDC

            StructField("riskParameters", StructType([
                StructField("priceLimitRatioX", StringType(), True),
                StructField("priceLimitRatioY", StringType(), True),
            ]), True),

            StructField("isPreListing", BooleanType(), True),
            StructField("preListingInfo", StructType([
                StructField("curAuctionPhase", StringType(), True),
                StructField("phases", ArrayType(StructType([
                    StructField("phase", StringType(), True),
                    StructField("startTime", StringType(), True),  # ms en string
                    StructField("endTime", StringType(), True),    # ms en string
                ])), True),
                StructField("auctionFeeInfo", StructType([
                    StructField("auctionFeeRate", StringType(), True),
                    StructField("takerFeeRate", StringType(), True),
                    StructField("makerFeeRate", StringType(), True),
                ]), True),
            ]), True),
        ]), True)),
    ]), True),
    # Suele venir {}, lo dejamos flexible como mapa de strings
    StructField("retExtInfo", MapType(StringType(), StringType()), True),
    # En top-level viene numérico (no entrecomillado) → LongType
    StructField("time", LongType(), True),
])

# Esquema OKX Futures
schema_okx_futures = StructType([
    StructField("code", StringType(), True),
    StructField("msg",  StringType(), True),
    StructField("data", ArrayType(StructType([
        StructField("instType",   StringType(), True),  # "SWAP"
        StructField("instId",     StringType(), True),  # p.ej. "BTC-USDT-SWAP"
        StructField("instFamily", StringType(), True),  # p.ej. "BTC-USDT"
        StructField("uly",        StringType(), True),  # subyacente: "BTC-USDT" / "BTC-USD"
        StructField("ctType",     StringType(), True),  # "linear" | "inverse"
        StructField("ctVal",      StringType(), True),
        StructField("ctMult",     StringType(), True),
        StructField("ctValCcy",   StringType(), True),
        StructField("settleCcy",  StringType(), True),  # "USDT", "USD", "BTC", ...
        StructField("baseCcy",    StringType(), True),  # suele ser "" en SWAP
        StructField("quoteCcy",   StringType(), True),  # suele ser "" en SWAP
        StructField("alias",      StringType(), True),  # "", "this_week", etc. (en SWAP normalmente "")
        StructField("state",      StringType(), True),  # "live" | "suspend" | "preopen" | "test"
        StructField("category",   StringType(), True),  # string (OKX lo expone como texto)
        StructField("listTime",   StringType(), True),  # epoch ms como string
        StructField("expTime",    StringType(), True),  # "" en SWAP
        StructField("lever",      StringType(), True),  # máx. apalancamiento (string)
        StructField("tickSz",     StringType(), True),
        StructField("lotSz",      StringType(), True),
        StructField("minSz",      StringType(), True),

        # Límites máximos de tamaño (llegan como string; algunos pueden venir vacíos según instrumento)
        StructField("maxLmtSz",      StringType(), True),
        StructField("maxMktSz",      StringType(), True),
        StructField("maxTwapSz",     StringType(), True),
        StructField("maxIcebergSz",  StringType(), True),
        StructField("maxStopSz",     StringType(), True),
        StructField("maxTriggerSz",  StringType(), True),

        # Normalmente vacíos en SWAP, pero pueden aparecer como ""
        StructField("optType",    StringType(), True),
        StructField("stk",        StringType(), True),

        # Puede aparecer en respuestas recientes; lo dejamos opcional como string
        StructField("ruleType",   StringType(), True)   # "normal" | "pre_market" (si aplica)
    ]), True), True)
])

# COMMAND ----------

# Esquema Binance
schema_binance_spot = StructType([
    StructField("timezone", StringType(), True),
    StructField("serverTime", LongType(), True),
    StructField("rateLimits", ArrayType(
        StructType([
            StructField("rateLimitType", StringType(), True),
            StructField("interval", StringType(), True),
            StructField("intervalNum", IntegerType(), True),
            StructField("limit", IntegerType(), True)
        ])
    ), True),
    StructField("exchangeFilters", ArrayType(MapType(StringType(), StringType())), True),
    StructField("symbols", ArrayType(
        StructType([
            StructField("symbol", StringType(), True),
            StructField("status", StringType(), True),
            StructField("baseAsset", StringType(), True),
            StructField("baseAssetPrecision", IntegerType(), True),
            StructField("quoteAsset", StringType(), True),
            StructField("quotePrecision", IntegerType(), True),
            StructField("quoteAssetPrecision", IntegerType(), True),
            StructField("baseCommissionPrecision", IntegerType(), True),
            StructField("quoteCommissionPrecision", IntegerType(), True),
            StructField("orderTypes", ArrayType(StringType()), True),
            StructField("icebergAllowed", BooleanType(), True),
            StructField("ocoAllowed", BooleanType(), True),
            StructField("otoAllowed", BooleanType(), True),
            StructField("quoteOrderQtyMarketAllowed", BooleanType(), True),
            StructField("allowTrailingStop", BooleanType(), True),
            StructField("cancelReplaceAllowed", BooleanType(), True),
            StructField("amendAllowed", BooleanType(), True),
            StructField("isSpotTradingAllowed", BooleanType(), True),
            StructField("isMarginTradingAllowed", BooleanType(), True),
            StructField("filters", ArrayType(MapType(StringType(), StringType())), True),
            StructField("permissions", ArrayType(StringType()), True),
            StructField("permissionSets", ArrayType(ArrayType(StringType())), True),
            StructField("defaultSelfTradePreventionMode", StringType(), True),
            StructField("allowedSelfTradePreventionModes", ArrayType(StringType()), True)
        ])
    ), True)
])

schema_bybit_spot = StructType([
    StructField("retCode", IntegerType(), True),
    StructField("retMsg", StringType(), True),
    StructField("result", StructType([
        StructField("category", StringType(), True),
        StructField("list", ArrayType(StructType([
            StructField("symbol", StringType(), True),
            StructField("baseCoin", StringType(), True),
            StructField("quoteCoin", StringType(), True),
            StructField("innovation", StringType(), True),
            StructField("status", StringType(), True),
            StructField("marginTrading", StringType(), True),
            StructField("stTag", StringType(), True),
            StructField("lotSizeFilter", StructType([
                StructField("basePrecision", StringType(), True),
                StructField("quotePrecision", StringType(), True),
                StructField("minOrderQty", StringType(), True),
                StructField("maxOrderQty", StringType(), True),
                StructField("minOrderAmt", StringType(), True),
                StructField("maxOrderAmt", StringType(), True)
            ])),
            StructField("priceFilter", StructType([
                StructField("tickSize", StringType(), True)
            ])),
            StructField("riskParameters", StructType([
                StructField("priceLimitRatioX", StringType(), True),
                StructField("priceLimitRatioY", StringType(), True)
            ]))
        ])), True)
    ])),
    StructField("retExtInfo", MapType(StringType(), StringType()), True)
])

# Esquema OKX
schema_okx_spot = StructType([
    StructField("code", StringType(), True),
    StructField("data", ArrayType(StructType([
        StructField("alias", StringType(), True),
        StructField("auctionEndTime", StringType(), True),
        StructField("baseCcy", StringType(), True),
        StructField("category", StringType(), True),
        StructField("contTdSwTime", StringType(), True),
        StructField("ctMult", StringType(), True),
        StructField("ctType", StringType(), True),
        StructField("ctVal", StringType(), True),
        StructField("ctValCcy", StringType(), True),
        StructField("expTime", StringType(), True),
        StructField("futureSettlement", BooleanType(), True),
        StructField("instFamily", StringType(), True),
        StructField("instId", StringType(), True),
        StructField("instIdCode", StringType(), True),
        StructField("instType", StringType(), True),
        StructField("lever", StringType(), True),
        StructField("listTime", StringType(), True),
        StructField("lotSz", StringType(), True),
        StructField("maxIcebergSz", StringType(), True),
        StructField("maxLmtAmt", StringType(), True),
        StructField("maxLmtSz", StringType(), True),
        StructField("maxMktAmt", StringType(), True),
        StructField("maxMktSz", StringType(), True),
        StructField("maxStopSz", StringType(), True),
        StructField("maxTriggerSz", StringType(), True),
        StructField("maxTwapSz", StringType(), True),
        StructField("minSz", StringType(), True),
        StructField("openType", StringType(), True),
        StructField("optType", StringType(), True),
        StructField("quoteCcy", StringType(), True),
        StructField("ruleType", StringType(), True),
        StructField("settleCcy", StringType(), True),
        StructField("state", StringType(), True),
        StructField("stk", StringType(), True),
        StructField("tickSz", StringType(), True),
        StructField("tradeQuoteCcyList", ArrayType(StringType()), True),
        StructField("uly", StringType(), True)
    ]), True))
])

# COMMAND ----------

# Diccionario de esquemas
exchangeinfo_schemas_futures = {
    "binance": schema_binance_futures,
    "bybit": schema_bybit_futures ,
    "okx": schema_okx_futures
}

# Diccionario de esquemas
exchangeinfo_schemas_spot = {
    "binance": schema_binance_spot,
    "bybit": schema_bybit_spot,
    "okx": schema_okx_spot
}

# COMMAND ----------

exchanges= [
    "binance",
    "bybit",
    "okx"
]

base_raw_path = "abfss://datalake@statfmprod.dfs.core.windows.net/raw/exchangeinfo"
base_bronze_path = "abfss://datalake@statfmprod.dfs.core.windows.net/bronze/exchangeinfo"
checkpoint_path = "abfss://datalake@statfmprod.dfs.core.windows.net/checkpoints"

def get_append_fn_futures(exchange):
    def append_to_bronze_futures(batch_df, batch_id):
        (
            batch_df.write
            .format("delta")
            .mode("append")
            .partitionBy("_ingested_filename")
            .option("mergeSchema", "true")
            .option("path", f"{base_bronze_path}/{exchange}_futures")
            .save()
        )
    return append_to_bronze_futures

def get_append_fn_spot(exchange):
    def append_to_bronze_spot(batch_df, batch_id):
        (
            batch_df.write
            .format("delta")
            .mode("append")
            .partitionBy("_ingested_filename")
            .option("mergeSchema", "true")
            .option("path", f"{base_bronze_path}/{exchange}_spot")
            .save()
        )
    return append_to_bronze_spot

timestamp = current_timestamp()
for exchange in exchanges:

    # Futures
    df = (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .schema(exchangeinfo_schemas_futures[exchange])
        .option("cloudFiles.schemaLocation", f"{base_raw_path}/schema/{exchange}_futures")
        .load(f"{base_raw_path}/{exchange}_futures")
    )
    # metadatos
    df = (
        df.withColumn("_ingested_at", timestamp)
          .withColumn("_ingested_filename", F.col("_metadata.file_path"))
    )

    (
        df.writeStream
        .foreachBatch(get_append_fn_futures(exchange))
        .option("checkpointLocation", f"{checkpoint_path}/bronze/exchangeinfo/{exchange}_futures")
        .queryName(f"bronze_{exchange}_futures")
        .trigger(availableNow=True)
        .start()
        .awaitTermination()
    )
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS dbrtfmprod.whale_trades.bronze__{exchange}_exchangeinfo_futures
        USING DELTA
        LOCATION 'abfss://datalake@statfmprod.dfs.core.windows.net/bronze/exchangeinfo/{exchange}_futures'
    """)

    # spot
    df = (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .schema(exchangeinfo_schemas_spot[exchange])
        .option("cloudFiles.schemaLocation", f"{base_raw_path}/schema/{exchange}")
        .load(f"{base_raw_path}/{exchange}")
    )
    # metadatos
    df = (
        df.withColumn("_ingested_at", timestamp)
          .withColumn("_ingested_filename", F.col("_metadata.file_path"))
    )

    (
        df.writeStream
        .foreachBatch(get_append_fn_spot(exchange))
        .option("checkpointLocation", f"{checkpoint_path}/bronze/exchangeinfo/{exchange}_spot")
        .queryName(f"bronze_{exchange}")
        .trigger(availableNow=True)
        .start()
        .awaitTermination()
    )
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS dbrtfmprod.whale_trades.bronze__{exchange}_exchangeinfo
        USING DELTA
        LOCATION 'abfss://datalake@statfmprod.dfs.core.windows.net/bronze/exchangeinfo/{exchange}_spot'
    """)



# COMMAND ----------

# Lista de carpetas en /raw/exchangeinfo
all_dirs = [f.path for f in dbutils.fs.ls(base_raw_path)]

# Filtra las carpetas que NO sean schema
dirs_to_delete = [path for path in all_dirs if not path.rstrip('/').endswith('schema')]

# Borra cada carpeta recursivamente
for path in dirs_to_delete:
    print(f"Borrando: {path}")
    dbutils.fs.rm(path, recurse=True)
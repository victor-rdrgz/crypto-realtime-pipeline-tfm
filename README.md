<p align="center">
  <a href="https://www.youtube.com/watch?v=2-JEcO-fkCI">
    <img src="https://img.youtube.com/vi/2-JEcO-fkCI/hqdefault.jpg" alt="Watch the demo on YouTube" width="600">
  </a>
</p>
</br></br></br></br>


# Predictive Analytics and Trading Architecture for Crypto Futures Markets

Master's thesis (Big Data & Data Engineering, UCM) — awarded 2nd Best Master's Thesis: an end-to-end real-time data pipeline on Microsoft Azure that analyzes price variability in crypto futures markets, detects "whale trades" (large-volume operations) and predicts short-term price increases.

## Summary

- Processes and transforms data in **Azure Databricks** from the **Binance, OKX and Bybit APIs**.
- Applies a **Medallion architecture (raw, bronze, silver, gold)** for storage in **Azure Data Lake Storage Gen2**.
- Trains and registers models in **MLflow**, integrated with **Unity Catalog**.
- Deploys a prototype on **Azure Container Instances (ACI)** that consumes the Binance WebSocket and makes real-time predictions.

## Architecture

![Architecture diagram](docs/imagenes/arquitectura.jpg)

**Main components:**
- **Data sources**: Binance (REST/WebSocket), Bybit and OKX.
- **Storage**: ADLS Gen2 with a Medallion architecture (raw → bronze → silver → gold).
- **Ingestion/Orchestration**: Azure Data Factory + Azure Databricks.
- **Processing**: Azure Databricks.
- **Governance**: Unity Catalog for lineage and access control.
- **Deployment**: Docker + Azure Container Instances (ACI).
- **Modeling**: MLflow for experiments, metrics and model registry.

![Model registered in Unity Catalog](docs/imagenes/mlflow.jpeg)

## Key results

- **Latency reduction:** 59% by deploying in the Japan East region.
- **Optimized production cost:** €0.89/day on ACI.
- **Average prediction time**: 6 ms.
- **Average return per signal**: +0.81%.
- **Median return (p50)**: +0.93%.
- **Winning trades (>0)**: 73.2%.
- **Signals ≥ 1%**: 49.48%.

## Repository structure

```
docs/           Full thesis report (PDF, in Spanish), diagrams and screenshots
notebooks/
  etl-train/     Ingestion, features, labels and training pipeline (Databricks)
  exchangeinfo/  Market metadata ingestion (raw → bronze → silver → gold)
  experiments/   Experimentation notebooks: different models (boosting, light, linear tree) and near-real-time inference
streaming/      Binance WebSocket client + Dockerfile for deployment on ACI
pipelines/      Azure Data Factory definitions (pipeline, datasets, linked services)
aux/            Binance metrics export script
```

**Note:** demo videos, raw exchange datasets/dumps and any files containing tokens or credentials (`.env`, Databricks/Azure/Confluent keys) have been excluded from the repository. The demo video is linked in the thesis report.

## How to run

1. **Ingestion and training**: run the Data Factory pipeline (`pipelines/pipeline/exchangeinfo_pipeline_futures_and_spot.json`) and the notebooks in `notebooks/etl-train/` on Databricks.
2. **Real-time deployment**: set the environment variables (Databricks host and token, symbol to track) and deploy `streaming/` as a container on Azure Container Instances.

## Stack

Azure Databricks, Azure Data Factory, ADLS Gen2, Unity Catalog, MLflow, Docker, Azure Container Instances, Python, PySpark.

## Future improvements

- Integrate the ACI deployment into the Azure Data Factory pipeline.
- Build a `.whl` package to reuse the pipeline.
- Implement automatic deletion/archiving of old data.
- Build Power BI dashboards or notebooks for advanced reporting.

---
Master's in Big Data & Data Engineering (UCM) — **Author:** Víctor Daniel Rodríguez Pérez

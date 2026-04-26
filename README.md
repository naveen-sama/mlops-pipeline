# MLOps Pipeline 🚀

A production-grade **end-to-end MLOps pipeline** for training, tracking, registering, and deploying machine learning models. Built with MLflow, Docker, FastAPI, and AWS — designed to be reproducible and CI/CD-ready.

## Features

- **Experiment tracking** — MLflow tracks all runs, parameters, metrics, and artifacts
- **Model registry** — staging → production promotion workflow with model versioning
- **Automated retraining** — triggers on data drift detection (evidently AI)
- **Dockerized serving** — FastAPI model server with health checks and versioned endpoints
- **CI/CD pipeline** — GitHub Actions automates test → train → deploy
- **Monitoring** — Prometheus + Grafana dashboard for prediction latency and drift

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           GitHub Actions CI/CD       │
                    └──────┬──────────────────────┬───────┘
                           │                      │
                    ┌──────▼──────┐        ┌──────▼──────┐
                    │   Training  │        │   Testing   │
                    │   Pipeline  │        │   Suite     │
                    └──────┬──────┘        └─────────────┘
                           │
              ┌────────────▼────────────┐
              │    MLflow Tracking      │
              │  (experiments, models)  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    Model Registry       │
              │  Staging → Production   │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   FastAPI Model Server  │
              │   Docker + AWS ECS      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Prometheus + Grafana   │
              │  Drift Detection        │
              └─────────────────────────┘
```

## Tech Stack

- **MLflow** — experiment tracking and model registry
- **FastAPI** — model serving REST API
- **Docker + Docker Compose** — containerization
- **AWS ECS + ECR** — cloud deployment
- **GitHub Actions** — CI/CD automation
- **Evidently AI** — data drift and model performance monitoring
- **Prometheus + Grafana** — metrics and alerting

## Quick Start

```bash
git clone https://github.com/naveen-sama/mlops-pipeline.git
cd mlops-pipeline

# Start all services (MLflow, FastAPI, monitoring)
docker-compose up -d

# Train a model and log to MLflow
python src/train.py --config configs/xgboost.yaml

# View MLflow UI
open http://localhost:5000

# Promote best model to production
python src/promote.py --run-id <mlflow-run-id>

# Query the model API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.2, 0.5, 3.1, ...]}'
```

## Pipeline Stages

| Stage | Tool | Description |
|-------|------|-------------|
| Data Validation | Great Expectations | Schema and quality checks |
| Feature Engineering | scikit-learn Pipelines | Reproducible transformations |
| Training | MLflow | Tracked experiments |
| Evaluation | MLflow + custom metrics | Model comparison |
| Registration | MLflow Model Registry | Version and stage management |
| Serving | FastAPI + Docker | REST API endpoint |
| Monitoring | Evidently + Grafana | Drift and latency tracking |

## Project Structure

```
mlops-pipeline/
├── src/
│   ├── train.py              # Training entry point
│   ├── evaluate.py           # Model evaluation
│   ├── promote.py            # Registry promotion script
│   └── drift_detector.py     # Evidently drift checks
├── serving/
│   ├── api.py                # FastAPI server
│   └── Dockerfile
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── configs/
│   ├── xgboost.yaml
│   └── lightgbm.yaml
├── .github/workflows/
│   └── ml_pipeline.yml       # CI/CD workflow
├── docker-compose.yml
└── requirements.txt
```

---

*Part of my AI/ML portfolio — [github.com/naveen-sama](https://github.com/naveen-sama)*

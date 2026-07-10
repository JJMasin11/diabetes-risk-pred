# Diabetes Risk Prediction

End-to-end diabetes risk prediction with a production-style architecture and SHAP-powered explainability.

## Live Demo

SCREENSHOT/GIF GOES HERE

[Try it live](LINK GOES HERE)

> **Note:** First visit may take 30–60 seconds to load, since min replicas is set to 0 on Azure Container Apps to keep hosting costs down on a portfolio project. In production, at least one warm replica would be maintained.

## Project Overview

- What problem this solves and why (diabetes risk prediction as a portfolio-worthy end-to-end ML system)
- High-level summary: data → training → tracked experiments → deployed API → web frontend
- Who this is for (recruiters/reviewers browsing your portfolio) — keep this section skimmable

## Architecture

- Flask web frontend (`Dockerfile.web`) — collects user input, renders results
- FastAPI backend (`Dockerfile.api`) — serves `/predict`, returns risk, probability, SHAP values
- Training pipeline (`Dockerfile.train`) — retrains models, logs to MLflow, uploads artifacts
- PostgreSQL database with two tables: raw BRFSS health indicators, and logged predictions (with the inputs that produced them) — logging predictions is designed so that, in a real healthcare deployment, true patient outcomes could later be joined against past predictions to monitor model drift and support retraining
- Azure services used and why: Key Vault (DB + Blob Storage secrets), Blob Storage (model artifacts), Azure Container Apps (hosting), Azure ML MLflow (experiment tracking)
- Why this design: mirrors a real production ML system (separate training/serving, secrets management, artifact versioning) rather than a single notebook-to-app script

## Data

- Source: BRFSS (Behavioral Risk Factor Surveillance System) health indicators dataset
- Brief description of features used and target variable
- Feature engineering highlight: BMI × Age interaction term (and why it helps)
- Note on class balance / any preprocessing decisions worth mentioning

## Modeling

- Models evaluated: CatBoost, XGBoost, LightGBM, scikit-learn baselines — why CatBoost was selected
- Two production models: primary diabetes risk model + auxiliary cholesterol prediction model — briefly explain why the auxiliary model exists
- Metrics table for both models: precision, recall, ROC-AUC (and any other relevant metrics)
- Explainability: SHAP values surfaced per-prediction in the app, not just a global feature importance chart
- Experiment tracking: MLflow (via Azure ML), including how historical notebook experiments were backfilled via `log_historical_experiments.py`

## Known Model Limitation

**Heavy alcohol consumption may appear as a risk-reducing factor** due to reverse causality in the BRFSS training data. People who already have diabetes are commonly advised to reduce or stop drinking, so diabetic respondents skew toward non-heavy-drinkers in the survey. The model learns this spurious correlation.

**Results should not be interpreted as clinical guidance.**

## Tech Stack

- Language/libraries: Python, pandas, scikit-learn, CatBoost/XGBoost/LightGBM, SHAP
- Backend/frontend: FastAPI, Flask
- Data: PostgreSQL, SQLAlchemy
- MLOps: MLflow, Azure ML, Docker
- Cloud: Azure Container Apps, Azure Key Vault, Azure Blob Storage, Azure Container Registry
- CI/CD: GitHub Actions

## CI/CD Pipeline

- `ci.yml`: runs tests and linting (ruff), builds Docker images, pushes to ACR, updates FastAPI + Flask container apps
- `train.yml`: a manually-triggered, one-click pipeline that builds a training image, pushes it to ACR, runs the training job, and restarts FastAPI with the newly trained models

## Repository Structure

- `notebook/` — EDA and model development
- `etl/` — data loading (`load_health_indicators.py`) and MLflow experiment logging (`log_historical_experiments.py`)
- `sql/` — database/table creation scripts
- `src/` — core application/model code
- `templates/` — Flask HTML templates
- `tests/` — test suite
- `.github/workflows/` — CI/CD pipelines

## Running Locally

- Prerequisites (Docker, Python version, Azure credentials if needed for full functionality)
- Steps to build/run each container (train, api, web) locally
- Note on any local fallback if Azure services aren't configured (or state that Azure access is required)

## Future Improvements

- Using logged predictions + eventual ground-truth outcomes to implement actual drift monitoring and outcome-based retraining (the logging schema already supports this; the evaluation/monitoring layer itself is not yet built)
- Scheduled or drift-triggered retraining (currently manually triggered via GitHub Actions)
- Addressing the alcohol/reverse-causality issue directly (feature removal, causal reweighting)
- Additional ideas: authentication, expanded auxiliary model, monitoring dashboard

## License / Contact

- License (if any)
- Link to your portfolio/LinkedIn/contact info

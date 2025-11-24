# Satellite Image Colorization
[![DVC](https://img.shields.io/badge/DVC-Tracked-blue)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)

This project develops a deep learning model for colorizing satellite images using the EuroSAT dataset. It employs a **ResNet34 encoder** and **UNet decoder** to predict color channels in LAB space from grayscale inputs. The project integrates *DVC* for data versioning, *MLflow* for experiment tracking, *GitHub Actions* for CI/CD, a *FastAPI-Streamlit* app for inference, and Docker for deployment.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#inference)
- [Data Pipeline (DVC)](#data-pipeline-dvc)
- [Model Training](#model-training)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [Authors](#authors)

## Project Overview
Satellite images are often grayscale, limiting their utility. This project colorizes them using DL, improving visualization for analysis. Key components:
- **Data Management**: DVC with Yandex Cloud S3.
- **Model**: ResNet34 + UNet in PyTorch.
- **Tracking**: MLflow for experiments.
- **Deployment**: FastAPI backend + Streamlit UI, Dockerized.
- **Automation**: GitHub Actions CI/CD.

### Why This Project?
- Addresses DeOldify's poor performance on satellite data.
- Uses LAB color space for better results.
- Full MLOps: Versioning, tracking, deployment.

## Features
- Automatic colorization of grayscale satellite images.
- Reproducible data pipeline with DVC.
- Experiment logging with MLflow.
- Web app for inference (FastAPI + Streamlit).
- Docker containers for easy deployment.
- CI/CD for tests and builds.

## Inference
1. Clone the repo:
   ```bash
   git clone https://github.com/WorriedSeat/image_colorizer.git
   cd image_colorizer
   ```

2. Build Docker:
   ```bash
   ./docker-build.local.sh
   ```

3. Run Docker:
    ```bash
    ./docker-run.local.sh
    ```

You got running Streamlit app where you can upload gray satellite image and ontain colored results.

## Data
All information about live here: `/data/data_desription.md`. 

## Model
- Run training script:
  ```bash
  python -m src.models.model_train.py
  ```
- Logs metrics/models to MLflow/Dagshub.
- Best model saved to `models/best.pt`.

The model architecture combines a **ResNet34 encoder** for feature extraction and a **UNet decoder** for color reconstruction. The encoder processes the repeated L channel (to match RGB input) and extracts multi-level features through five downsampling layers. The decoder upsamples these features with skip connections across five corresponding layers to predict ab channels. Training uses a combined L1 and pretrained VGG-16 perceptual loss, where VGG-16 extracts features from predicted and ground truth RGB images, computing L1 loss on these features to ensure perceptual similarity beyond pixel-level accuracy.

## CI/CD with GitHub Actions
- Workflows in `.github/workflows/`:
  - `docker-build.yaml`: Tests code changes in `/src` on push/PR by building deploying Docker.

## Authors
- Vasilev Ivan
- Sarantsev Stepan
- Shariev Marat

---
# SLM Training Container
#
# Build:  docker build -t chart-analysis-trainer services/trainer/
# Push:   docker tag chart-analysis-trainer gcr.io/PROJECT_ID/chart-analysis-trainer:latest
#         docker push gcr.io/PROJECT_ID/chart-analysis-trainer:latest

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY services/trainer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source code and config
COPY src/ src/
COPY config/ config/
COPY scripts/training/ scripts/training/

# Copy training data (or mount as volume)
# Data is typically uploaded to GCS and downloaded at runtime

ENTRYPOINT ["python", "scripts/training/train_slm_lora.py"]
CMD ["--config", "config/training.yaml"]

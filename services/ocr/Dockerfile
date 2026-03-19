FROM python:3.10-slim

WORKDIR /app

# System dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .

# Model weights are volume-mounted at runtime (not baked into image)
# Mount: ./models/paddleocr_vl -> /app/models/paddleocr_vl

ENV MODEL_PATH=/app/models/paddleocr_vl
ENV PADDLE_DEVICE=cpu
ENV PORT=8001

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]

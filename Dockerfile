FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first to avoid NVIDIA CUDA dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install the remaining project dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: start the API server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
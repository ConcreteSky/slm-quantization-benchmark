FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /benchmark
COPY requirements.txt .
RUN python3.10 -m pip install --upgrade pip && python3.10 -m pip install -r requirements.txt
COPY README.md config.yaml ./
COPY src ./src
COPY app ./app
EXPOSE 8501
CMD ["python3.10", "-m", "src.benchmark_cli", "--config", "config.yaml"]

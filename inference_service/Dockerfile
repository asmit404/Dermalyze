FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U huggingface_hub
COPY . .

ENV HF_TOKEN=""
ENV MODEL_CHECKPOINT=/app/models/checkpoint_best.pt
ENV TRUST_CONFIG_PATH=/app/models/trust_config.json
ENV USE_TTA=false
ENV CORS_ORIGINS=https://www.dermalyze.tech

EXPOSE 7860

CMD hf download asmit404/dermalyze-model \
      checkpoint_best.pt trust_config.json \
      --local-dir /app/models \
      --token ${HF_TOKEN} && \
    uvicorn app:app --host 0.0.0.0 --port 7860
FROM python:3.11-slim

WORKDIR /app

# wget needed to download the model checkpoint
RUN apt-get update && apt-get install -y --no-install-recommends wget && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (model file is gitignored — downloaded below instead)
COPY . .

# Download model checkpoint from HF Hub model repo (cached Docker layer)
RUN mkdir -p /app/models && \
    wget -q -O /app/models/checkpoint_best.pt \
    https://huggingface.co/asmit404/dermalyze-model/resolve/main/checkpoint_best.pt

ENV MODEL_CHECKPOINT=/app/models/checkpoint_best.pt
ENV USE_TTA=false
ENV CORS_ORIGINS=https://www.dermalyze.tech

# Hugging Face Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

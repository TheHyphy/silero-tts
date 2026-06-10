# Silero TTS — RunPod Serverless
# =================================
# Сборка:
#   docker build -t ghcr.io/thehyphy/silero-tts:latest .
#   docker push ghcr.io/thehyphy/silero-tts:latest

FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_PREFER_BINARY=1

# Системные зависимости
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python-зависимости
RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    silero \
    numpy \
    runpod \
    requests

# Handler
COPY runpod_silero_handler.py /handler.py

# Pre-download model при сборке (ускоряет cold start)
RUN python3 -c "\
import torch; \
torch.hub.set_dir('/tmp/silero_models'); \
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', \
    model='v4_ru', source='github', device='cpu', trust_repo=True); \
print('Model cached OK'); \
" 2>&1 | tail -5

CMD ["python3", "-u", "/handler.py"]

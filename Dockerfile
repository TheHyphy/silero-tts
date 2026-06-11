# Silero TTS -- RunPod Serverless
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3-pip wget libsndfile1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy runpod requests omegaconf scipy soundfile

# Download model at build time (URL passed from CI)
ARG MODEL_URL
RUN mkdir -p /app/silero && wget -q -O /app/silero/v2_kseniya.pt "$MODEL_URL"

COPY runpod_silero_handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]

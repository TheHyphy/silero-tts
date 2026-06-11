# Silero TTS -- RunPod Serverless
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3-pip wget libsndfile1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy runpod requests omegaconf scipy soundfile

# Pre-download model into torch hub cache (so first request is fast)
RUN python3 -c "import torch; torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='kseniya_v2', source='github', trust_repo=True, device='cpu'); print('Model pre-loaded OK')"

COPY runpod_silero_handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]

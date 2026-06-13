# Silero TTS -- RunPod Serverless
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3-pip wget libsndfile1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy runpod requests omegaconf scipy soundfile num2words g2p_en nltk

# Pre-download nltk data for g2p_en
RUN python3 -c "import nltk; nltk.download('cmudict', quiet=True)"
# Pre-download model into torch hub cache (v5_ru with xenia voice)
RUN python3 -c "\
import torch; \
model, _ = torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='v5_ru', source='github', trust_repo=True, device='cpu'); \
model.apply_tts('Тест.', speaker='xenia', sample_rate=48000); \
print('v5_ru + warm-up OK')"

COPY runpod_silero_handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]

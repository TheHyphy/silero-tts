# Silero TTS -- RunPod Serverless
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y python3-pip wget libsndfile1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy runpod requests omegaconf scipy soundfile

# Pre-download model into torch hub cache (v5_ru with xenia voice)
RUN python3 -c "\
import torch; \
model, _ = torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='v5_ru', source='github', trust_repo=True, device='cpu'); \
print(f'v5_ru loaded OK, voices: {model.get_speakers()}'); \
model.apply_tts('Тест.', speaker='xenia', sample_rate=48000); \
print('Warm-up OK')"

COPY runpod_silero_handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]

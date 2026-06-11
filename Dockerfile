# Silero TTS -- RunPod Serverless
FROM runpod/base:0.7.0-cuda12.4.1

ENV PYTHONUNBUFFERED=1

# Install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Install other deps
RUN pip install --no-cache-dir numpy runpod requests omegaconf scipy soundfile

# Pre-download Silero model into torch hub cache
RUN python3 -c "import torch; torch.hub.load('snakers4/silero-models', 'silero_tts', language='ru', speaker='kseniya_v2', source='github', trust_repo=True, device='cpu'); print('Model pre-loaded OK')"

COPY runpod_silero_handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]

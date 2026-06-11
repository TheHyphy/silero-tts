# Silero TTS -- RunPod Serverless (fast pull: runpod/base cached on all nodes)
FROM runpod/base:0.7.0-cuda12.4.1

ENV PYTHONUNBUFFERED=1

# Install torch and deps (model loaded from /runpod-volume)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir numpy runpod requests omegaconf scipy soundfile

COPY runpod_silero_handler.py /handler.py

# Model is on network volume at /runpod-volume/silero/v2_kseniya.pt
CMD ["python3", "-u", "/handler.py"]

#!/usr/bin/env python3
"""
RunPod Serverless Handler — Silero TTS (Russian voice xenia)
============================================================
Принимает текст → синтезирует речь через Silero → возвращает WAV.

Формат запроса:
{
    "text": "Привет, мир!",
    "voice": "xenia",       # опционально
    "speaker": "xenia",     # для совместимости с laptop-сервером
    "sample_rate": 24000,   # опционально, по умолч. 24000
    "put_intonation": true  # опционально, знаки препинания → паузы
}

Формат ответа:
{
    "audio": "<base64-wav>",
    "sample_rate": 24000,
    "duration_sec": 3.5
}
"""
import base64
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch
import numpy as np

# ─── Конфиг ────────────────────────────────────────────────────────────────────

DEFAULT_VOICE = "xenia"
DEFAULT_SAMPLE_RATE = 24000
MODEL_CACHE_DIR = Path("/tmp/silero_models")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Репозиторий Silero
SILERO_REPO = "snakers4/silero-models"
SILERO_MODEL = "v4_ru"  # последняя русская модель

# ─── Загрузка модели (lazy, один раз) ──────────────────────────────────────────

_model = None
_model_device = None


def load_model(voice: str = DEFAULT_VOICE):
    """Загрузить Silero model + speaker. Кешируется в /tmp для холодного старта."""
    global _model, _model_device

    if _model is not None:
        return _model, _model_device

    torch.backends.quantized.engine = "qnnpack"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Silero] Loading model on {device}...", flush=True)

    t0 = time.time()

    try:
        # Попробовать загрузить через silero API
        import silero

        model, _ = silero.load_model(
            SILERO_REPO,
            model_id=SILERO_MODEL,
            device=device,
            cache_dir=str(MODEL_CACHE_DIR),
        )
    except (ImportError, Exception) as e:
        print(f"[Silero] silero API failed ({e}), trying direct torch.hub...", flush=True)
        try:
            model, _ = torch.hub.load(
                repo_or_dir=SILERO_REPO,
                model=SILERO_MODEL,
                source="github",
                device=device,
                trust_repo=True,
            )
        except Exception as e2:
            print(f"[Silero] torch.hub also failed: {e2}", flush=True)
            raise

    _model = model
    _model_device = device

    elapsed = time.time() - t0
    print(f"[Silero] Model loaded in {elapsed:.1f}s on {device}", flush=True)
    return _model, _model_device


def split_text(text: str, max_chars: int = 450) -> list[str]:
    """
    Разбить длинный текст на куски по границам предложений.
    Silero стабилен до ~500 символов, выше — артефакты.
    """
    if len(text) <= max_chars:
        return [text]

    # Разбить по .!? с учётом пробелов
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            # Если одно предложение длиннее max_chars — режем принудительно
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i : i + max_chars])
                current = ""
            else:
                current = sent

    if current:
        chunks.append(current)

    return chunks


def synthesize(text: str, voice: str = DEFAULT_VOICE,
               sample_rate: int = DEFAULT_SAMPLE_RATE,
               put_intonation: bool = True) -> tuple[bytes, float]:
    """
    Синтезировать речь через Silero.
    Возвращает (wav_bytes, duration_sec).
    """
    model, device = load_model(voice)

    # Подготовка текста
    if put_intonation:
        text = text.replace("\n", " ").replace("  ", " ").strip()
        if text and not text[-1] in ".!?":
            text += "."

    # Выбор speaker
    speaker = model.speakers[0] if hasattr(model, "speakers") else voice
    try:
        speaker_id = model.speakers.index(voice) if hasattr(model, "speakers") else 0
    except ValueError:
        speaker_id = 0

    # Разбивка на чанки для длинных текстов
    chunks = split_text(text)

    all_audio = []
    total_duration = 0.0

    for i, chunk in enumerate(chunks):
        print(f"[Silero] Chunk {i+1}/{len(chunks)}: {len(chunk)} chars", flush=True)
        try:
            audio = model.save_wav(
                text=chunk,
                speaker=speaker,
                sample_rate=sample_rate,
                audio_path=None,  # вернуть тензор, не сохранять
            )
            # audio — тензор формы (1, samples)
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy().flatten()
            else:
                audio_np = np.array(audio, dtype=np.float32).flatten()

            duration = len(audio_np) / sample_rate
            total_duration += duration
            all_audio.append(audio_np)

        except Exception as e:
            print(f"[Silero] Chunk {i+1} failed: {e}", flush=True)
            continue

    if not all_audio:
        raise RuntimeError("No audio generated")

    # Склеить чанки
    combined = np.concatenate(all_audio)

    # Конвертировать float32 → int16 WAV
    combined_int16 = (combined * 32767).astype(np.int16)

    # Собрать WAV в BytesIO
    import struct

    buf = io.BytesIO()
    n_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * n_channels * bits_per_sample // 8
    block_align = n_channels * bits_per_sample // 8
    data_size = len(combined_int16) * bits_per_sample // 8

    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate, byte_rate, block_align, bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(combined_int16.tobytes())

    wav_bytes = buf.getvalue()
    return wav_bytes, total_duration


# ─── RunPod Handler ─────────────────────────────────────────────────────────────

def handler(job):
    """RunPod serverless handler entrypoint."""
    job_input = job.get("input", {})

    text = job_input.get("text", "")
    if not text:
        return {"error": "No text provided"}

    voice = job_input.get("voice") or job_input.get("speaker") or DEFAULT_VOICE
    sample_rate = job_input.get("sample_rate", DEFAULT_SAMPLE_RATE)
    put_intonation = job_input.get("put_intonation", True)

    print(f"[Handler] voice={voice}, sample_rate={sample_rate}, text_len={len(text)}", flush=True)

    try:
        wav_bytes, duration = synthesize(
            text=text,
            voice=voice,
            sample_rate=sample_rate,
            put_intonation=put_intonation,
        )

        audio_b64 = base64.b64encode(wav_bytes).decode()

        print(f"[Handler] Generated {len(wav_bytes)} bytes, {duration:.1f}s", flush=True)
        return {
            "audio": audio_b64,
            "sample_rate": sample_rate,
            "duration_sec": round(duration, 2),
            "format": "wav",
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ─── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # RunPod serverless mode
    try:
        import runpod
        print("[Silero] Starting RunPod serverless...", flush=True)
        runpod.serverless.start({"handler": handler})
    except ImportError:
        # Тестовый режим: чтение из stdin
        print("[Silero] RunPod SDK not found, test mode. Reading from stdin...", flush=True)
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                resp = handler({"input": req})
                print(json.dumps(resp, ensure_ascii=False), flush=True)
            except json.JSONDecodeError:
                pass

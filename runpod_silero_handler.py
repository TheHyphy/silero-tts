#!/usr/bin/env python3
"""
RunPod Serverless Handler — Silero TTS (Russian voice kseniya_v2)
===============================================================
Model is loaded from local path (baked into Docker image).
"""
import base64, io, json, os, re, struct, sys, time, traceback
from pathlib import Path

import torch
import numpy as np

DEFAULT_VOICE = "kseniya_v2"
DEFAULT_SAMPLE_RATE = 24000
MODEL_PATH = Path("/runpod-volume/silero/v2_kseniya.pt")

_model = None
_model_device = None
_model_speakers = DEFAULT_VOICE


def load_model():
    global _model, _model_device
    if _model is not None:
        return _model, _model_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        print(f"[Silero] Loading model from {MODEL_PATH} on {device}...", flush=True)
        t0 = time.time()
        model = torch.package.PackageImporter(MODEL_PATH).load_pickle("tts_models", "model")
        model = model.to(device)
        model.eval()
        _model = model
        _model_device = device
        print(f"[Silero] Loaded in {time.time()-t0:.1f}s", flush=True)
        return _model, _model_device
    except Exception as e:
        print(f"[Silero] Local load failed ({e}), falling back to torch.hub...", flush=True)
        # Fallback
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="kseniya_v2",
            source="github",
            trust_repo=True,
            device=device,
        )
        _model = model
        _model_device = device
        return _model, _model_device


def split_text(text: str, max_chars: int = 140) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i:i + max_chars])
                current = ""
            else:
                current = sent
    if current:
        chunks.append(current)
    return chunks


def synthesize(text: str, voice: str = DEFAULT_VOICE,
               sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[bytes, float]:
    import soundfile as sf
    
    model, device = load_model()
    
    chunks = split_text(text)
    all_wav_files = []
    total_duration = 0.0
    
    for i, chunk in enumerate(chunks):
        print(f"[Silero] Chunk {i+1}/{len(chunks)}: {len(chunk)} chars", flush=True)
        try:
            paths = model.save_wav(
                texts=chunk,
                audio_pathes='',
                sample_rate=sample_rate,
            )
            wav_path = paths[0] if isinstance(paths, list) else paths
            all_wav_files.append(wav_path)
        except Exception as e:
            print(f"[Silero] Chunk {i+1} failed: {e}", flush=True)
            continue
    
    if not all_wav_files:
        raise RuntimeError("No audio generated")
    
    combined_parts = []
    for wav_path in all_wav_files:
        data, sr = sf.read(wav_path)
        total_duration += len(data) / sr
        combined_parts.append(data)
        try:
            os.remove(wav_path)
        except OSError:
            pass
    
    combined = np.concatenate(combined_parts)
    combined_int16 = (combined * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    n_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * n_channels * bits_per_sample // 8
    block_align = n_channels * bits_per_sample // 8
    data_size = len(combined_int16) * bits_per_sample // 8
    
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate, byte_rate, block_align, bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(combined_int16.tobytes())
    
    return buf.getvalue(), total_duration


def handler(job):
    job_input = job.get("input", {})
    text = job_input.get("text", "")
    if not text:
        return {"error": "No text provided"}
    voice = job_input.get("voice") or job_input.get("speaker") or DEFAULT_VOICE
    sample_rate = job_input.get("sample_rate", DEFAULT_SAMPLE_RATE)
    
    print(f"[Handler] voice={voice}, sr={sample_rate}, text_len={len(text)}", flush=True)
    try:
        wav_bytes, duration = synthesize(text=text, voice=voice, sample_rate=sample_rate)
        audio_b64 = base64.b64encode(wav_bytes).decode()
        print(f"[Handler] {len(wav_bytes)}b, {duration:.1f}s", flush=True)
        return {"audio": audio_b64, "sample_rate": sample_rate, "duration_sec": round(duration, 2), "format": "wav"}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    try:
        import runpod
        print("[Silero] Starting RunPod serverless...", flush=True)
        runpod.serverless.start({"handler": handler})
    except ImportError:
        print("[Silero] Test mode. Reading from stdin...", flush=True)
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                resp = handler({"input": req})
                print(json.dumps(resp, ensure_ascii=False), flush=True)
            except json.JSONDecodeError:
                pass

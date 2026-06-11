#!/usr/bin/env python3
"""RunPod Serverless Handler — Silero TTS (Russian voice kseniya_v2)"""

import base64, io, json, os, re, struct, sys, time, traceback
from pathlib import Path
import torch
import numpy as np

# v2 — RunPod serverless, L40/L40S GPUs
DEFAULT_VOICE = "kseniya_v2"
DEFAULT_SAMPLE_RATE = 24000

_model = None
_model_device = None


def load_model():
    global _model, _model_device
    if _model is not None:
        return _model, _model_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Silero] Loading via torch.hub on {device}...", flush=True)
    t0 = time.time()
    model, _ = torch.hub.load(
        "snakers4/silero-models", "silero_tts",
        language="ru", speaker="kseniya_v2",
        source="github", trust_repo=True, device=device,
    )
    _model, _model_device = model, device
    print(f"[Silero] Loaded in {time.time()-t0:.1f}s", flush=True)
    return _model, _model_device


def split_text(text, max_chars=140):
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            chunks.extend(sent[i:i+max_chars] for i in range(0, len(sent), max_chars))
            current = ""
    if current:
        chunks.append(current)
    return chunks


def synthesize(text, voice=DEFAULT_VOICE, sample_rate=DEFAULT_SAMPLE_RATE):
    import soundfile as sf
    model, device = load_model()
    
    chunks = split_text(text)
    parts, total_dur = [], 0.0
    
    for i, chunk in enumerate(chunks):
        print(f"[Silero] Chunk {i+1}/{len(chunks)}: {len(chunk)} chars", flush=True)
        try:
            paths = model.save_wav(texts=chunk, audio_pathes='', sample_rate=sample_rate)
            wav = paths[0] if isinstance(paths, list) else paths
            data, sr = sf.read(wav)
            total_dur += len(data) / sr
            parts.append(data)
            os.remove(wav)
        except Exception as e:
            print(f"[Silero] Chunk failed: {e}", flush=True)
    
    if not parts:
        raise RuntimeError("No audio generated")
    
    combined = np.concatenate(parts).astype(np.float64)
    combined_int16 = (combined * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    data_size = len(combined_int16) * 2
    fmt = struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    buf.write(b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE" + b"fmt ")
    buf.write(fmt + b"data" + struct.pack("<I", data_size) + combined_int16.tobytes())
    
    return buf.getvalue(), total_dur


def handler(job):
    inp = job.get("input", {})
    text = inp.get("text", "")
    if not text:
        return {"error": "No text provided"}
    voice = inp.get("voice") or inp.get("speaker") or DEFAULT_VOICE
    sr = inp.get("sample_rate", DEFAULT_SAMPLE_RATE)
    
    print(f"[Handler] voice={voice}, text_len={len(text)}", flush=True)
    try:
        wav, dur = synthesize(text, voice, sr)
        return {"audio": base64.b64encode(wav).decode(), "sample_rate": sr, "duration_sec": round(dur, 2), "format": "wav"}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    try:
        import runpod
        runpod.serverless.start({"handler": handler})
    except ImportError:
        for line in sys.stdin:
            if line.strip():
                resp = handler({"input": json.loads(line)})
                print(json.dumps(resp, ensure_ascii=False), flush=True)

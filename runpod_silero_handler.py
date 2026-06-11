#!/usr/bin/env python3
"""
RunPod Serverless Handler — Silero TTS (Russian voice xenia)
============================================================
Fixed: torch.hub API without silero package dependency.
"""
import base64, io, json, os, re, struct, sys, time, traceback
from pathlib import Path

import torch
import numpy as np
from num2words import num2words

DEFAULT_VOICE = "xenia"
DEFAULT_SAMPLE_RATE = 24000

_model = None
_model_device = None
_model_speakers = []


def load_model():
    global _model, _model_device, _model_speakers
    if _model is not None:
        return _model, _model_device, _model_speakers

    torch.backends.quantized.engine = "qnnpack"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try multiple approaches to load model
    errors = []
    
    # Approach 1: torch.hub with silero_tts
    try:
        print(f"[Silero] Loading via torch.hub (silero_tts, ru, xenia) on {device}...", flush=True)
        t0 = time.time()
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v5_ru",
            source="github",
            trust_repo=True,
            device=device,
        )
        _model = model
        _model_device = device
        if hasattr(model, "speakers"):
            _model_speakers = model.speakers
        print(f"[Silero] Loaded in {time.time()-t0:.1f}s, speakers={_model_speakers[:3]}", flush=True)
        return _model, _model_device, _model_speakers
    except Exception as e:
        errors.append(f"torch.hub: {e}")
    
    # Approach 2: torch.hub with different model name
    try:
        print(f"[Silero] Trying silero_tts_ru...", flush=True)
        t0 = time.time()
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts_ru",
            source="github",
            trust_repo=True,
            device=device,
        )
        _model = model
        _model_device = device
        _model_speakers = model.speakers if hasattr(model, "speakers") else ["xenia"]
        print(f"[Silero] Loaded in {time.time()-t0:.1f}s", flush=True)
        return _model, _model_device, _model_speakers
    except Exception as e:
        errors.append(f"torch.hub v2: {e}")

    # Approach 3: direct download from Silero releases
    try:
        print(f"[Silero] Downloading from GitHub releases...", flush=True)
        import urllib.request
        import zipfile
        
        url = "https://github.com/snakers4/silero-models/releases/download/v4.0/silero_tts_ru_v4.pt"
        model_path = Path("/tmp/silero_models/silero_tts_ru_v4.pt")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not model_path.exists():
            t0 = time.time()
            urllib.request.urlretrieve(url, model_path)
            print(f"[Silero] Downloaded in {time.time()-t0:.1f}s", flush=True)
        
        model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
        model = model.to(device)
        _model = model
        _model_device = device
        _model_speakers = ["xenia", "baya", "kseniya", "natasha", "random"]
        print(f"[Silero] Loaded from direct download", flush=True)
        return _model, _model_device, _model_speakers
    except Exception as e:
        errors.append(f"direct: {e}")

    raise RuntimeError(f"All load approaches failed: {'; '.join(errors)}")


def split_text(text: str, max_chars: int = 450) -> list[str]:
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
    model, device, speakers = load_model()
    
    # model.speakers returns model names (v5_ru, v4_ru...), not voice names
    # Use voice directly — valid voices for v5_ru: xenia, baya, kseniya, natasha, aidar, irina, ruslan
    speaker = voice
    
    chunks = split_text(text)
    all_audio = []
    total_duration = 0.0
    
    for i, chunk in enumerate(chunks):
        print(f"[Silero] Chunk {i+1}/{len(chunks)}: {len(chunk)} chars", flush=True)
        try:
            audio = model.apply_tts(
                text=chunk,
                speaker=speaker,
                sample_rate=sample_rate,
            )
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
    
    combined = np.concatenate(all_audio)
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


def normalize_numbers(text: str) -> str:
    """Replace numeric expressions with Russian words (from laptop config)."""
    # 1. Exponents
    text = re.sub(r"10\u00b2\u00b2", "десять в двадцать второй степени", text)
    text = re.sub(r"(\d+)\s*\^\s*(\d+)", lambda m: f"{num2words(int(m.group(1)), lang='ru')} в степени {num2words(int(m.group(2)), lang='ru')}", text)

    months = r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"

    # 2. Date: 15 марта -> neuter genitive
    text = re.sub(r"(?<!\d)(\d{1,2})\s*(" + months + r")", lambda m: f"{num2words(int(m.group(1)), lang='ru', to='ordinal', gender='n', case='genitive')} {m.group(2)}", text)

    # 3. Number + "век" -> masculine nominative
    text = re.sub(r"(\d{1,2})\s*век\b", lambda m: f"{num2words(int(m.group(1)), lang='ru', to='ordinal', gender='m')} век", text)

    # 4. Year + "года" -> masculine genitive
    text = re.sub(r"(?<!\d)(\d{4})\s+года\b", lambda m: f"{num2words(int(m.group(1)), lang='ru', to='ordinal', gender='m', case='genitive')} года", text)
    text = re.sub(r"(?<!\d)(\d{1,3})\s+года\b", lambda m: f"{num2words(int(m.group(1)), lang='ru')} года", text)
    text = re.sub(r"(?<!\d)(\d{1,3})\s+лет\b", lambda m: f"{num2words(int(m.group(1)), lang='ru')} лет", text)

    # 5. Year + "году" -> masculine dative
    text = re.sub(r"(?<!\d)(\d{4})\s+году\b", lambda m: f"{num2words(int(m.group(1)), lang='ru', to='ordinal', gender='m', case='dative')} году", text)

    # 6. Year + "годом" -> masculine instrumental
    text = re.sub(r"(?<!\d)(\d{4})\s+годом\b", lambda m: f"{num2words(int(m.group(1)), lang='ru', to='ordinal', gender='m', case='instrumental')} годом", text)

    # 7. Year + "год"/"г." -> masculine nominative
    text = re.sub(r"(?<!\d)(\d{4})\s*г(?:од|\.)\b", lambda m: f"{num2words(int(m.group(1)), lang='ru', to='ordinal', gender='m')} год", text)
    text = re.sub(r"(?<!\d)(\d{4})\s*[\u2013\u2014\u2212-]\s*(\d{4})(?!\d)", lambda m: f"{num2words(int(m.group(1)), lang='ru', to='ordinal', gender='m')} — {num2words(int(m.group(2)), lang='ru', to='ordinal', gender='m')}", text)

    # 8. General number replacement
    def replace_number(m):
        raw = m.group(0).replace(" ", "")
        try:
            if "," in raw:
                parts = raw.split(",")
                integer = num2words(int(parts[0]), lang="ru")
                return f"{integer} целых {parts[1]}"
            num = int(raw)
            return num2words(num, lang="ru")
        except Exception:
            return raw

    text = re.sub(r"(?<![а-яёa-zA-Z\d])(\d{1,3}(?:[\s,]\d{3})*|\d+)(?![а-яёa-zA-Z\d])", replace_number, text)
    return text


def handler(job):
    job_input = job.get("input", {})
    text = job_input.get("text", "")
    if not text:
        return {"error": "No text provided"}
    voice = job_input.get("voice") or job_input.get("speaker") or DEFAULT_VOICE
    sample_rate = job_input.get("sample_rate", DEFAULT_SAMPLE_RATE)
    
    print(f"[Handler] voice={voice}, sr={sample_rate}, text_len={len(text)}", flush=True)
    try:
        text = normalize_numbers(text)
        print(f"[Handler] after normalization: {len(text)} chars", flush=True)
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

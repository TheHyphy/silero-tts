#!/usr/bin/env python3
"""
RunPod Serverless Handler — Silero TTS (Russian voice xenia)
============================================================
Fixed: torch.hub API without silero package dependency.
v2: Added Latin→Cyrillic transliteration + improved number normalization.
"""
import base64, io, json, os, re, struct, sys, time, traceback
from pathlib import Path

import torch
import numpy as np
from num2words import num2words
from g2p_en import G2p

DEFAULT_VOICE = "xenia"
DEFAULT_SAMPLE_RATE = 24000

_model = None
_model_device = None
_model_speakers = []
_g2p = None

# ── Latin→Cyrillic transliteration ───────────────────────────────────

ARPABET_TO_RU = {
    'AA': 'а', 'AE': 'э', 'AH': 'а', 'AO': 'о', 'AW': 'ау',
    'AY': 'ай', 'EH': 'э', 'ER': 'ер', 'EY': 'эй', 'IH': 'и',
    'IY': 'и', 'OW': 'оу', 'OY': 'ой', 'UH': 'у', 'UW': 'у',
    'B': 'б', 'CH': 'ч', 'D': 'д', 'DH': 'з', 'F': 'ф',
    'G': 'г', 'HH': 'х', 'JH': 'дж', 'K': 'к', 'L': 'л',
    'M': 'м', 'N': 'н', 'NG': 'нг', 'P': 'п', 'R': 'р',
    'S': 'с', 'SH': 'ш', 'T': 'т', 'TH': 'с', 'V': 'в',
    'W': 'в', 'Y': 'й', 'Z': 'з', 'ZH': 'ж',
}

LETTER_TO_RU = {
    'A': 'эй', 'B': 'би', 'C': 'си', 'D': 'ди', 'E': 'и',
    'F': 'эф', 'G': 'джи', 'H': 'эйч', 'I': 'ай', 'J': 'джей',
    'K': 'кей', 'L': 'эл', 'M': 'эм', 'N': 'эн', 'O': 'оу',
    'P': 'пи', 'Q': 'кью', 'R': 'ар', 'S': 'эс', 'T': 'ти',
    'U': 'ю', 'V': 'ви', 'W': 'дабл-ю', 'X': 'экс', 'Y': 'уай', 'Z': 'зэд',
}

CUSTOM_DICT = {
    'mg': 'миллиграмм', 'ml': 'миллилитр', 'mm': 'миллиметр',
    'cm': 'сантиметр', 'km': 'километр', 'kg': 'килограмм',
    'hz': 'герц', 'khz': 'килогерц', 'mhz': 'мегагерц',
    'ghz': 'гигагерц', 'w': 'ватт', 'kw': 'киловатт',
    'v': 'вольт', 'mv': 'милливольт', 'ma': 'миллиампер',
    'the': 'зе', 'and': 'энд', 'for': 'фор', 'with': 'виз',
    'that': 'зэт', 'this': 'зис',
    'a': 'а', 'i': 'ай', 'is': 'ис', 'in': 'ин', 'it': 'ит',
    'of': 'ов', 'on': 'он', 'to': 'ту', 'by': 'бай', 'as': 'эз',
    'are': 'ар', 'was': 'воз', 'were': 'вер',
    'be': 'би', 'been': 'бин', 'have': 'хэв', 'has': 'хэз',
    'not': 'нот', 'no': 'ноу', 'yes': 'йес',
    'we': 'ви', 'he': 'хи', 'she': 'ши', 'you': 'ю',
    # Madoka character names
    'madoka': 'мадока', 'kyubey': 'кьюбей', 'homura': 'хомура',
    'sayaka': 'саяка', 'mami': 'мами', 'kyoko': 'кёко',
    'akemi': 'акеми', 'tomoe': 'томоэ', 'miki': 'мики', 'sakura': 'сакура',
    'kaname': 'канамэ', 'puella magi': 'пуэлла маги',
    'mado': 'мадо', 'magi': 'маги', 'magica': 'магика',
    'anime': 'аниме', 'manga': 'манга',
}

def _get_g2p():
    global _g2p
    if _g2p is None:
        _g2p = G2p()
    return _g2p

def _is_acronym(word):
    if not word or len(word) < 2 or len(word) > 6:
        return False
    return all(c.isupper() for c in word if c.isalpha())

def _spell_acronym(word):
    letters = [c for c in word if c.isalpha()]
    ru_letters = [LETTER_TO_RU.get(c.upper(), c) for c in letters]
    return '-'.join(ru_letters)

def word_to_ru(word):
    """Transcribe a single English word to Russian phonetics."""
    if not word or not re.search(r'[a-zA-Z]', word):
        return word
    word_lower = word.lower()
    if word_lower in CUSTOM_DICT:
        return CUSTOM_DICT[word_lower]
    if _is_acronym(word):
        return _spell_acronym(word)
    g2p = _get_g2p()
    phonemes = g2p(word)
    result = []
    for p in phonemes:
        clean_p = re.sub(r'\d+$', '', p)
        if clean_p in ARPABET_TO_RU:
            result.append(ARPABET_TO_RU[clean_p])
        else:
            if p.strip():
                result.append(p)
    return ''.join(result)

LATIN_TOKEN_PATTERN = re.compile(
    r'(?<=\d)(?:mg|ml|mm|cm|km|kg|hz|khz|mhz|ghz|w|kw|v|mv|ma)(?!\w)'
    r'|'
    r'(?:[A-Z]\.)+[A-Z]?'
    r'|'
    r"[A-Za-z]+(?:['-][A-Za-z]+)*"
)

def transliterate_latin(text):
    """Replace Latin words/letters with Russian phonetic transcription."""
    def replace_match(m):
        word = m.group(0)
        if re.match(r'^[a-z]+$', word) and word.lower() in CUSTOM_DICT:
            return CUSTOM_DICT[word.lower()]
        return word_to_ru(word)
    return LATIN_TOKEN_PATTERN.sub(replace_match, text)

# ── Number normalization ─────────────────────────────────────────────

def normalize_numbers(text):
    """Replace numeric expressions with Russian words."""
    # 1. Section numbers: "1.1" → "один один", "3.5" → "три пять"
    text = re.sub(r'(?<!\d)(\d+)\.(\d+)(?!\d)', lambda m: (
        f"{num2words(int(m.group(1)), lang='ru')} {num2words(int(m.group(2)), lang='ru')}"
    ), text)
    
    # 2. Fractions: "7/7" → "семь из семи", "3/4" → "три четвертых"
    text = re.sub(r'(?<!\d)(\d+)/(\d+)(?!\d)', lambda m: (
        f"{num2words(int(m.group(1)), lang='ru')} из {num2words(int(m.group(2)), lang='ru')}"
    ), text)
    
    # 3. Exponents
    text = re.sub(r"10\u00b2\u00b2", "десять в двадцать второй степени", text)
    text = re.sub(r"(\d+)\s*\^\s*(\d+)", lambda m: (
        f"{num2words(int(m.group(1)), lang='ru')} в степени {num2words(int(m.group(2)), lang='ru')}"
    ), text)
    
    months = r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
    
    # 4. Date + month
    text = re.sub(r"(?<!\d)(\d{1,2})\s*(" + months + r")", lambda m: (
        f"{num2words(int(m.group(1)), lang='ru', to='ordinal', gender='n', case='genitive')} {m.group(2)}"
    ), text)
    
    # 5. Number + "век"
    text = re.sub(r"(\d{1,2})\s*век\b", lambda m: (
        f"{num2words(int(m.group(1)), lang='ru', to='ordinal', gender='m')} век"
    ), text)
    
    # 6. Year + case forms
    for pattern, ordinal in [
        (r"(?<!\d)(\d{4})\s+года\b", "genitive"),
        (r"(?<!\d)(\d{4})\s+году\b", "dative"),
        (r"(?<!\d)(\d{4})\s+годом\b", "instrumental"),
        (r"(?<!\d)(\d{4})\s*г(?:од|\.)\b", "nominative"),
    ]:
        text = re.sub(pattern, lambda m, case=ordinal: (
            f"{num2words(int(m.group(1)), lang='ru', to='ordinal', gender='m', case=case)} "
            f"{'год' if case in ('nominative', 'genitive') else 'году' if case == 'dative' else 'годом'}"
        ), text)
    
    # 7. Year ranges
    text = re.sub(r"(?<!\d)(\d{4})\s*[\u2013\u2014\u2212-]\s*(\d{4})(?!\d)", lambda m: (
        f"{num2words(int(m.group(1)), lang='ru', to='ordinal', gender='m')} — "
        f"{num2words(int(m.group(2)), lang='ru', to='ordinal', gender='m')}"
    ), text)
    
    # 8. "N лет/N года"
    text = re.sub(r"(?<!\d)(\d{1,3})\s+лет\b", lambda m: (
        f"{num2words(int(m.group(1)), lang='ru')} лет"
    ), text)
    text = re.sub(r"(?<!\d)(\d{1,3})\s+года\b", lambda m: (
        f"{num2words(int(m.group(1)), lang='ru')} года"
    ), text)
    
    # 9. General standalone numbers
    def replace_number(m):
        raw = m.group(0).replace(" ", "").replace("\u202f", "")
        try:
            if "," in raw:
                parts = raw.split(",")
                return f"{num2words(int(parts[0]), lang='ru')} целых {parts[1]}"
            num = int(raw)
            return num2words(num, lang="ru")
        except Exception:
            return raw
    
    text = re.sub(r"(?<![а-яёa-zA-Z\d])(\d{1,3}(?:[\s,\u202f]\d{3})*|\d+)(?![а-яёa-zA-Z\d])", replace_number, text)
    
    # 10. Percentages
    def replace_pct(m):
        try:
            n = int(m.group(1))
            word = num2words(n, lang='ru')
            last_two = n % 100
            last_digit = n % 10
            if last_two in (11, 12, 13, 14):
                pct_word = "процентов"
            elif last_digit == 1:
                pct_word = "процент"
            elif last_digit in (2, 3, 4):
                pct_word = "процента"
            else:
                pct_word = "процентов"
            return f"{word} {pct_word}"
        except ValueError:
            return m.group(0)
    
    text = re.sub(r"(\d+)\s*%", replace_pct, text)
    
    return text

# ── Silero model ─────────────────────────────────────────────────────

def load_model():
    global _model, _model_device, _model_speakers
    if _model is not None:
        return _model, _model_device, _model_speakers

    torch.backends.quantized.engine = "qnnpack"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    raise RuntimeError(f"All load approaches failed: {'; '.join(errors)}")

# ── Text splitting ───────────────────────────────────────────────────

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

# ── Synthesis ────────────────────────────────────────────────────────

def synthesize(text: str, voice: str = DEFAULT_VOICE,
               sample_rate: int = DEFAULT_SAMPLE_RATE,
               use_ssml: bool = False) -> tuple[bytes, float]:
    model, device, speakers = load_model()
    
    speaker = voice
    
    if use_ssml:
        # SSML: не разбивать на чанки, не нормализовать
        print(f"[Silero] SSML synthesis: {len(text)} chars", flush=True)
        try:
            audio = model.apply_tts(
                ssml_text=text,
                speaker=speaker,
                sample_rate=sample_rate,
            )
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy().flatten()
            else:
                audio_np = np.array(audio, dtype=np.float32).flatten()
            duration = len(audio_np) / sample_rate
            all_audio = [audio_np]
            total_duration = duration
        except Exception as e:
            print(f"[Silero] SSML synthesis failed: {e}", flush=True)
            raise RuntimeError(f"SSML synthesis failed: {e}")
    else:
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

# ── Handler ──────────────────────────────────────────────────────────

def handler(job):
    job_input = job.get("input", {})
    text = job_input.get("text", "")
    ssml_text = job_input.get("ssml_text", "")
    
    if not text and not ssml_text:
        return {"error": "No text provided"}
    
    voice = job_input.get("voice") or job_input.get("speaker") or DEFAULT_VOICE
    sample_rate = job_input.get("sample_rate", DEFAULT_SAMPLE_RATE)
    
    use_ssml = bool(ssml_text)
    if use_ssml:
        text_to_speak = ssml_text
        print(f"[Handler] SSML mode, len={len(ssml_text)}", flush=True)
    else:
        text_to_speak = text
        print(f"[Handler] voice={voice}, sr={sample_rate}, text_len={len(text)}", flush=True)
    
    try:
        if not use_ssml:
            text_to_speak = transliterate_latin(text_to_speak)
            print(f"[Handler] after transliteration: {len(text_to_speak)} chars", flush=True)
            text_to_speak = normalize_numbers(text_to_speak)
            print(f"[Handler] after normalization: {len(text_to_speak)} chars", flush=True)
        wav_bytes, duration = synthesize(
            text=text_to_speak,
            voice=voice,
            sample_rate=sample_rate,
            use_ssml=use_ssml
        )
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

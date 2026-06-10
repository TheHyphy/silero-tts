#!/usr/bin/env python3
"""
RunPod Silero TTS Client
========================
Отправляет текст на RunPod Serverless endpoint, получает WAV и сохраняет/воспроизводит.

Использование:
  python3 rp_silero_client.py --text "Привет, мир!" --output voice.wav
  python3 rp_silero_client.py --text "Привет" --play     # сразу проиграть
  echo "Текст" | python3 rp_silero_client.py --output -   # из stdin в stdout

Параметры:
  --text       текст для озвучки
  --voice      голос (default: xenia)
  --sr         sample rate (default: 24000)
  --output     путь сохранения .wav (default: silero_output.wav)
  --play       проиграть через termux-media-player
  --endpoint   RunPod endpoint ID
"""
import base64
import json
import os
import sys
import urllib.request
import argparse
from pathlib import Path


ENV_PATH = os.path.expanduser("~/.hermes/.env")
# TODO: заменить на реальный endpoint ID после создания
DEFAULT_ENDPOINT = "PUT_YOUR_ENDPOINT_ID_HERE"


def get_api_key():
    with open(ENV_PATH) as f:
        for line in f:
            if "RUNPOD_API_KEY" in line and "=" in line:
                return line.split("=", 1)[1].strip().strip("'\"").strip()
    raise ValueError("RUNPOD_API_KEY not found")


def synthesize(text: str, voice: str = "xenia",
               sample_rate: int = 24000,
               endpoint_id: str = None) -> dict:
    """Отправить запрос на RunPod и получить результат."""
    key = get_api_key()
    endpoint = endpoint_id or DEFAULT_ENDPOINT

    payload = {
        "input": {
            "text": text,
            "voice": voice,
            "sample_rate": sample_rate,
            "put_intonation": True,
        }
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    url = f"https://api.runpod.ai/v2/{endpoint}/runsync"

    print(f"[Client] Sending {len(text)} chars to {endpoint}...", file=sys.stderr)
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    status = result.get("status", "")
    if status == "FAILED":
        raise RuntimeError(f"RunPod error: {result.get('output', result.get('error', 'unknown'))}")
    if status != "COMPLETED":
        raise RuntimeError(f"Unexpected status: {status}")

    output = result.get("output", {})
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            raise RuntimeError(f"Unexpected output format: {output[:200]}")

    if "error" in output:
        raise RuntimeError(f"Handler error: {output['error']}")

    return output


def main():
    parser = argparse.ArgumentParser(description="RunPod Silero TTS Client")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--voice", type=str, default="xenia", help="Voice name")
    parser.add_argument("--sr", type=int, default=24000, help="Sample rate")
    parser.add_argument("--output", type=str, default="silero_output.wav",
                        help="Output WAV path")
    parser.add_argument("--play", action="store_true", help="Play via termux-media-player")
    parser.add_argument("--endpoint", type=str, default=None, help="RunPod endpoint ID")
    args = parser.parse_args()

    # Чтение текста
    text = args.text
    if not text:
        text = sys.stdin.read().strip()
    if not text:
        parser.print_help()
        sys.exit(1)

    # Синтез
    result = synthesize(
        text=text,
        voice=args.voice,
        sample_rate=args.sr,
        endpoint_id=args.endpoint,
    )

    audio_b64 = result.get("audio", "")
    if not audio_b64:
        print("No audio in response", file=sys.stderr)
        print(json.dumps(result, indent=2))
        sys.exit(1)

    wav_bytes = base64.b64decode(audio_b64)
    duration = result.get("duration_sec", 0)

    print(f"[Client] Received {len(wav_bytes)} bytes ({duration:.1f}s)", file=sys.stderr)

    # Сохранение или stdout
    if args.output == "-":
        sys.stdout.buffer.write(wav_bytes)
    else:
        output_path = os.path.abspath(args.output)
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        print(f"[Client] Saved to {output_path}", file=sys.stderr)

    # Воспроизведение
    if args.play:
        try:
            import subprocess
            subprocess.run(["termux-media-player", "play", output_path],
                         timeout=30, capture_output=True)
        except Exception as e:
            print(f"[Client] Playback failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

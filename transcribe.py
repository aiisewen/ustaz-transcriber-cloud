# -*- coding: utf-8 -*-
"""
Ustaz Transcriber — конвейер транскрибации кыргызской речи.

Структура папок:
    1_входящие/     — сюда кладёшь исходники (mp4, mp3, m4a, wav)
    2_аудио/        — извлечённое аудио (mp3), создаётся автоматически
    3_транскрипты/  — готовые кыргызские тексты
    4_переводы/     — сюда складываются русские переводы (вручную/из Claude)
    5_архив/        — обработанные исходники, разложены по месяцам
    история.csv     — журнал всех обработок

Использование:
    python transcribe.py                     — обработать ВСЁ из папки 1_входящие
    python transcribe.py "путь/к/файлу.mp4"  — обработать конкретный файл
    python transcribe.py --model medium      — точная (медленная) модель
    python transcribe.py --history           — показать последние обработки
"""

import argparse
import csv
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).parent
DIR_INBOX = BASE / "1_входящие"
DIR_AUDIO = BASE / "2_аудио"
DIR_TRANSCRIPTS = BASE / "3_транскрипты"
DIR_TRANSLATIONS = BASE / "4_переводы"
DIR_ARCHIVE = BASE / "5_архив"
HISTORY = BASE / "история.csv"

MEDIA_EXT = {".mp4", ".mov", ".mkv", ".webm", ".mp3", ".m4a", ".wav", ".ogg", ".aac"}

MODELS = {
    "small": "UlutSoftLLC/whisper-small-kyrgyz",   # быстрее
    "medium": "nineninesix/kyrgyz-whisper-medium",  # точнее, медленнее
}

# у medium не прошит язык — без принудительного языка она галлюцинирует по-русски
MODEL_GEN_KWARGS = {
    "small": {},
    "medium": {"language": "kk", "task": "transcribe"},
}

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
HISTORY_FIELDS = [
    "дата", "исходник", "длительность_сек", "модель",
    "обработка_мин", "транскрипт", "статус",
]


def ensure_dirs():
    for d in (DIR_INBOX, DIR_AUDIO, DIR_TRANSCRIPTS, DIR_TRANSLATIONS, DIR_ARCHIVE):
        d.mkdir(exist_ok=True)
    if not HISTORY.exists():
        with open(HISTORY, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writeheader()


def log_history(row: dict):
    with open(HISTORY, "a", newline="", encoding="utf-8-sig") as f:
        csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writerow(row)


def show_history(n=15):
    if not HISTORY.exists():
        print("История пуста.")
        return
    with open(HISTORY, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("История пуста.")
        return
    print(f"Последние {min(n, len(rows))} обработок:\n")
    for r in rows[-n:]:
        print(f"  {r['дата']}  |  {r['исходник']}  |  {r['длительность_сек']} сек"
              f"  |  {r['обработка_мин']} мин  |  {r['статус']}")
    print(f"\nПолный журнал: {HISTORY}")


def unique_path(p: Path) -> Path:
    """Если файл с таким именем есть — добавить суффикс, ничего не перезаписываем."""
    if not p.exists():
        return p
    i = 2
    while True:
        cand = p.with_stem(f"{p.stem}_{i}")
        if not cand.exists():
            return cand
        i += 1


def extract_audio(src: Path) -> Path:
    """Исходник (видео/аудио) -> чистый mp3 в 2_аудио/."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    dst = unique_path(DIR_AUDIO / f"{src.stem}.mp3")
    res = subprocess.run(
        [ff, "-y", "-i", str(src), "-vn", "-ar", str(SAMPLE_RATE), "-ac", "1", str(dst)],
        capture_output=True,
    )
    if res.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg не смог обработать {src.name}")
    return dst


def load_audio(mp3_path: Path):
    """mp3 -> массив сэмплов (через временный wav)."""
    import imageio_ffmpeg
    import numpy as np
    import wave
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = mp3_path.with_suffix(".tmp.wav")
    subprocess.run(
        [ff, "-y", "-i", str(mp3_path), "-ar", str(SAMPLE_RATE), "-ac", "1", str(tmp)],
        capture_output=True,
    )
    with wave.open(str(tmp)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    tmp.unlink(missing_ok=True)
    return audio.astype("float32") / 32768.0


_model_cache = {}

def get_model(model_key: str):
    """Модель грузим один раз, даже если файлов много."""
    if model_key not in _model_cache:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        name = MODELS[model_key]
        print(f"Загружаю модель {name} (первый раз — скачивание, потом кэш)…")
        processor = WhisperProcessor.from_pretrained(name)
        model = WhisperForConditionalGeneration.from_pretrained(name)
        model.eval()
        _model_cache[model_key] = (processor, model)
    return _model_cache[model_key]


FAST_MODELS = {
    "small": BASE / "models" / "kyrgyz-small-ct2",
    "medium": BASE / "models" / "kyrgyz-medium-ct2",
}
_fast_cache = {}


def transcribe_segments(audio, model_key: str):
    """Вернуть список (секунда_начала, текст). Быстрый движок, если модель сконвертирована."""
    fast_dir = FAST_MODELS.get(model_key)
    if fast_dir and fast_dir.exists():
        return _transcribe_fast(audio, model_key, fast_dir)
    return _transcribe_slow(audio, model_key)


def _transcribe_fast(audio, model_key: str, model_dir):
    """faster-whisper (CTranslate2 int8): в ~3.5 раза быстрее + VAD против галлюцинаций."""
    from faster_whisper import WhisperModel
    if model_key not in _fast_cache:
        print(f"Загружаю быструю модель {model_dir.name}…")
        _fast_cache[model_key] = WhisperModel(str(model_dir), device="cpu", compute_type="int8")
    model = _fast_cache[model_key]
    segments, _info = model.transcribe(audio, language="kk", vad_filter=True)
    out = []
    for s in segments:
        txt = s.text.strip()
        if txt:
            out.append((int(s.start), txt))
    return out


def _transcribe_slow(audio, model_key: str):
    """Старый движок (transformers) — запасной путь."""
    import torch
    processor, model = get_model(model_key)
    chunk = CHUNK_SECONDS * SAMPLE_RATE
    n_chunks = (len(audio) + chunk - 1) // chunk
    segments = []
    for i in range(0, len(audio), chunk):
        t0 = time.time()
        seg = audio[i: i + chunk]
        inputs = processor(seg, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(
                inputs.input_features,
                max_new_tokens=200,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
                **MODEL_GEN_KWARGS.get(model_key, {}),
            )
        txt = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if txt:
            segments.append((i // SAMPLE_RATE, txt))
        print(f"    кусок {i // chunk + 1}/{n_chunks} готов ({time.time() - t0:.0f} сек)")
    return segments


def fmt_time(sec: int) -> str:
    return f"{sec // 60}:{sec % 60:02d}"


def segments_to_text(segments) -> str:
    """Транскрипт с метками времени: [0:00] … [0:30] …"""
    return "\n\n".join(f"[{fmt_time(s)}] {t}" for s, t in segments)


def transcribe_audio(audio, model_key: str) -> str:
    return segments_to_text(transcribe_segments(audio, model_key))


def process_file(src: Path, model_key: str):
    print(f"\n=== {src.name} ===")
    t_start = time.time()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    status = "ошибка"
    duration = 0
    out_txt = ""
    try:
        audio_file = extract_audio(src)
        audio = load_audio(audio_file)
        duration = int(len(audio) / SAMPLE_RATE)
        print(f"  Длительность: {duration} сек")

        text = transcribe_audio(audio, model_key)

        out_path = unique_path(DIR_TRANSCRIPTS / f"{stamp}_{src.stem}.txt")
        header = (
            f"# Исходник: {src.name}\n"
            f"# Дата обработки: {stamp}\n"
            f"# Модель: {MODELS[model_key]}\n"
            f"# Длительность: {duration} сек\n"
            f"# Статус: черновик, нужна вычитка носителем\n\n"
        )
        out_path.write_text(header + text, encoding="utf-8")
        out_txt = out_path.name

        # исходник -> архив по месяцам
        month_dir = DIR_ARCHIVE / datetime.now().strftime("%Y-%m")
        month_dir.mkdir(exist_ok=True)
        if src.parent == DIR_INBOX:
            shutil.move(str(src), str(unique_path(month_dir / src.name)))

        status = "готово"
        print(f"\n  --- ТРАНСКРИПТ ---\n{text}\n")
        print(f"  Сохранено: {out_path}")
    except Exception as e:
        print(f"  ОШИБКА: {e}")
    finally:
        log_history({
            "дата": stamp,
            "исходник": src.name,
            "длительность_сек": duration,
            "модель": model_key,
            "обработка_мин": f"{(time.time() - t_start) / 60:.1f}",
            "транскрипт": out_txt,
            "статус": status,
        })


def main():
    ap = argparse.ArgumentParser(description="Транскрибация кыргызской речи")
    ap.add_argument("input", nargs="?", default=None,
                    help="Файл для обработки. Без аргумента — вся папка 1_входящие")
    ap.add_argument("--model", choices=list(MODELS), default="small")
    ap.add_argument("--history", action="store_true", help="Показать журнал")
    args = ap.parse_args()

    ensure_dirs()

    if args.history:
        show_history()
        return

    if args.input:
        src = Path(args.input)
        if not src.exists():
            sys.exit(f"Файл не найден: {src}")
        files = [src]
    else:
        files = sorted(p for p in DIR_INBOX.iterdir()
                       if p.suffix.lower() in MEDIA_EXT)
        if not files:
            print(f"Папка «{DIR_INBOX.name}» пуста. Положи туда видео/аудио и запусти снова.")
            return
        print(f"Найдено файлов во входящих: {len(files)}")

    for f in files:
        process_file(f, args.model)

    print("\nГотово. Журнал: python transcribe.py --history")


if __name__ == "__main__":
    main()

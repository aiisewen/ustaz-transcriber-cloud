# -*- coding: utf-8 -*-
"""
Ustaz Transcriber — веб-интерфейс (Gradio).

Запуск:  python app.py   →  открой http://127.0.0.1:7860

Вкладки:
  1. Перевод  — файл/ссылка → кыргызский транскрипт + русский перевод рядом
  2. Словарь  — глоссарий терминов (используется в каждом переводе)
  3. История  — журнал обработок
"""

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import anthropic
import gradio as gr

# берём готовый движок из transcribe.py
from transcribe import (
    DIR_TRANSCRIPTS,
    DIR_TRANSLATIONS,
    MODELS,
    ensure_dirs,
    extract_audio,
    load_audio,
    log_history,
    transcribe_audio,
    unique_path,
)

BASE = Path(__file__).parent
GLOSSARY = BASE / "словарь.csv"
HISTORY = BASE / "история.csv"
COOKIES = BASE / "cookies.txt"

# куки из переменной окружения (Render) -> файл при старте
if os.environ.get("COOKIES_TXT"):
    try:
        COOKIES.write_text(os.environ["COOKIES_TXT"], encoding="utf-8")
    except Exception:
        pass

claude = anthropic.Anthropic()


def download_url(url: str) -> Path:
    """Скачать аудио по ссылке (yt-dlp). Для Instagram нужен cookies.txt рядом с app.py."""
    import subprocess
    dl_dir = BASE / "downloads"
    dl_dir.mkdir(exist_ok=True)
    out_tpl = str(dl_dir / "%(id)s.%(ext)s")
    cmd = [sys.executable, "-m", "yt_dlp", "--no-update",
           "-f", "ba/b", "-x", "--audio-format", "mp3", "-o", out_tpl,
           "--print", "after_move:filepath", "--no-simulate", url.strip()]
    if COOKIES.exists():
        cmd += ["--cookies", str(COOKIES)]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        tail = (res.stderr or "")[-400:]
        if "login" in tail.lower() or "cookies" in tail.lower() or "empty media response" in tail.lower():
            raise RuntimeError(
                "Instagram требует логин. Положи файл cookies.txt в папку программы "
                "(инструкция в README) или сохрани видео вручную и загрузи файлом."
            )
        if "429" in tail or "Too Many Requests" in tail:
            raise RuntimeError(
                "Instagram не отдаёт видео серверам (блокировка облачных адресов). "
                "Сохрани видео на телефон (Поделиться → Скачать) и загрузи его файлом — "
                "это займёт 10 секунд и работает всегда."
            )
        raise RuntimeError(f"Не удалось скачать: {tail}")

    # 1) точный путь, который напечатал yt-dlp
    for line in reversed((res.stdout or "").strip().splitlines()):
        p = Path(line.strip())
        if p.suffix.lower() == ".mp3" and p.exists():
            return p
    # 2) запасной путь: ищем файл по ID ролика из ссылки
    import re
    m = re.search(r"/(?:p|reel|reels|tv)/([A-Za-z0-9_-]+)", url)
    if m:
        cand = dl_dir / f"{m.group(1)}.mp3"
        if cand.exists():
            return cand
    raise RuntimeError("Не смогла надёжно определить скачанный файл — попробуй загрузить файлом.")


# ---------- словарь ----------

def load_glossary_rows():
    if not GLOSSARY.exists():
        return []
    with open(GLOSSARY, encoding="utf-8") as f:
        return [[r.get("термин", ""), r.get("перевод", ""), r.get("пояснение", "")]
                for r in csv.DictReader(f)]


def save_glossary_rows(rows):
    with open(GLOSSARY, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["термин", "перевод", "пояснение"])
        for row in rows:
            cells = [str(c or "").strip() for c in row]
            if any(cells):
                w.writerow(cells[:3] + [""] * (3 - len(cells)))
    return "Словарь сохранён ✅"


def glossary_as_text() -> str:
    lines = []
    for term, tr, note in load_glossary_rows():
        line = f"- {term} → {tr}"
        if note:
            line += f" ({note})"
        lines.append(line)
    return "\n".join(lines)


# ---------- перевод ----------

TRANSLATE_SYSTEM = """Ты переводчик религиозных выступлений с кыргызского на русский.
Переводишь речи исламского учёного (устаза) для русскоязычной аудитории.

ГЛАВНЫЙ ПРИНЦИП: переводи СМЫСЛ, а не слова. Результат должен звучать так,
будто устаз изначально говорил по-русски — естественно и живо. Дословная калька
с кыргызского порядка слов = плохой перевод.

Правила:
1. СТРОГО следуй глоссарию терминов ниже — переводи термины только так, как указано.
2. Перестраивай предложения по законам русского языка: меняй порядок слов, дроби
   длинные фразы, объединяй обрывки — лишь бы мысль звучала по-русски естественно.
3. Это устная речь: убирай оговорки, самоповторы, слова-паразиты и фальстарты,
   которые не несут смысла. Но сохраняй живые обращения к слушателям, риторические
   вопросы, юмор и эмоцию — это голос устаза, его нельзя стерилизовать.
4. Пословицы и идиомы переводи по смыслу. Если кыргызская пословица понятна
   дословно и колоритна — оставь её образ, красиво оформив по-русски.
5. СОХРАНЯЙ метки времени: перевод идёт теми же блоками, что и транскрипт —
   под каждой меткой [0:00], [0:30], [1:00] перевод этого фрагмента.
6. Ошибки распознавания восстанавливай по контексту, но не выдумывай содержание.
   Совсем неразборчивое место помечай временем ближайшей метки: [неразборчиво ~1:30].
7. Сокращать словесный мусор устной речи — можно и нужно. Сокращать или добавлять
   СМЫСЛ — нельзя. Каждая мысль устаза должна остаться в переводе.
8. НЕ УСИЛИВАЙ ЭМОЦИЮ. Свобода тебе дана в порядке слов и грамматике — не в тоне:
   - восклицательные знаки только там, где восклицает сам устаз;
   - не добавляй иронии, сарказма и оценочных частиц («видите ли», «мол», «дескать»),
     если их нет в оригинале;
   - выбирай слово той же силы: «жаман көрүп калуу» = «плохо подумать/невзлюбить»,
     а не «возненавидеть». Не драматизируй и не смягчай.
9. Реакции зала и звуковые ремарки ([журт күлөт], (laughter), [смех] и т.п.)
   в перевод НЕ включай — опускай полностью, переводи только речь устаза.
10. Культурные слова без точного русского аналога (төр, той и т.п.) оставляй
    с кратким пояснением при первом упоминании: «төр (почётное место в доме)».
11. Пророка упоминай полной формой: «Пророк Саллаллаху алейхи ва саллям»
    (в коротких роликах — именно так, не «(сав)» и не «ﷺ»).
12. Верни ТОЛЬКО перевод, без вступлений и комментариев.

Образец стиля (фрагмент эталонного перевода ролика):
---
[0:00] В наше время мы утратили понимание ценности трёх важнейших вещей. Именно поэтому ислам постепенно исчезает из нашей повседневной жизни. Первое — мы перестали ценить мусульманина. Иными словами, мы утратили уважение друг к другу. Второе — мы забыли о взаимопомощи. Раньше человек стремился поддержать своего единоверца, совершить покупку именно у него, чтобы его дело процветало. Третье — мы с самого начала

[0:30] стали ошибаться в вопросе создания семьи. Одной из важнейших основ человеческой жизни является никах. Также передаётся, что Пророк Саллаллаху алейхи ва саллям сказал: «Есть четыре признака несчастья: сухость глаз, жестокость сердца, чрезмерно долгие надежды, ненасытная любовь к богатству».
---

Глоссарий:
{glossary}"""


def translate_text(kyrgyz_text: str) -> str:
    try:
        resp = claude.messages.create(
            model="claude-opus-5",
            max_tokens=4000,
            system=TRANSLATE_SYSTEM.format(glossary=glossary_as_text()),
            messages=[{"role": "user", "content": kyrgyz_text}],
        )
    except anthropic.AuthenticationError:
        return ("⚠️ Ключ Claude не принят (неверный или отозван). "
                "Проверь ANTHROPIC_API_KEY в файле .env")
    except anthropic.RateLimitError:
        return "⚠️ Claude перегружен запросами — подожди минуту и нажми «Перевести заново»."
    except anthropic.APIStatusError as e:
        msg = str(getattr(e, "message", "") or e).lower()
        if "credit" in msg or "billing" in msg or "balance" in msg:
            return ("⚠️ КРЕДИТЫ CLAUDE ЗАКОНЧИЛИСЬ. Пополни баланс на "
                    "console.anthropic.com → Billing, потом нажми «Перевести заново».")
        return f"⚠️ Ошибка Claude API: {e}"
    except anthropic.APIConnectionError:
        return "⚠️ Нет связи с Claude — проверь интернет и нажми «Перевести заново»."
    if resp.stop_reason == "refusal":
        return "[Перевод отклонён моделью — переведи вручную]"
    return "".join(b.text for b in resp.content if b.type == "text")


# ---------- облачное распознавание (ElevenLabs Scribe) ----------

def transcribe_cloud(audio_path: Path) -> str:
    """ElevenLabs Scribe: секунды вместо минут, пунктуация, высокая точность."""
    import urllib.request
    import json as _json
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("Нет ELEVENLABS_API_KEY в .env — облачный режим недоступен.")
    boundary = "----ustazboundary"
    data = audio_path.read_bytes()

    def part(name, value):
        return (f"--{boundary}\r\nContent-Disposition: form-data; "
                f"name=\"{name}\"\r\n\r\n{value}\r\n").encode()

    body = part("model_id", "scribe_v1") + part("language_code", "ky") + part("tag_audio_events", "false")
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"{audio_path.name}\"\r\nContent-Type: application/octet-stream"
             f"\r\n\r\n").encode() + data + b"\r\n" + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/speech-to-text", data=body,
        headers={"xi-api-key": key,
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        resp = _json.loads(urllib.request.urlopen(req, timeout=600).read())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        low = detail.lower()
        if e.code in (401, 403) and "quota" not in low:
            raise RuntimeError("⚠️ Ключ ElevenLabs не принят (неверный или отозван). "
                               "Проверь ELEVENLABS_API_KEY в файле .env")
        if e.code == 429 or "quota" in low or "credit" in low or "limit" in low:
            raise RuntimeError("⚠️ КРЕДИТЫ ELEVENLABS ЗАКОНЧИЛИСЬ (или лимит запросов). "
                               "Пополни на elevenlabs.io → Billing, либо переключи модель "
                               "на small/medium — они работают без облака.")
        raise RuntimeError(f"⚠️ Ошибка ElevenLabs ({e.code}): {detail}")

    # вставить метки времени [м:сс] каждые ~30 сек по словам
    words = resp.get("words") or []
    if not words:
        return resp.get("text", "")
    out, next_mark = [], 0
    for w in words:
        start = w.get("start") or 0
        if start >= next_mark:
            m, s = int(start) // 60, int(start) % 60
            out.append(f"\n\n[{m}:{s:02d}]")
            next_mark += 30
        if w.get("type") == "word":
            out.append(" " + (w.get("text") or ""))
        else:
            out.append(w.get("text") or "")
    import re as _re
    return _re.sub(r"[ 	]{2,}", " ", "".join(out)).strip()


# ---------- основной конвейер ----------

def clean_vocals(audio_path: Path, progress=None) -> Path:
    """Убрать фоновую музыку/шум: Demucs выделяет только голос."""
    import subprocess
    out_dir = BASE / "downloads" / "demucs"
    cmd = [sys.executable, "-m", "demucs.separate",
           "--two-stems=vocals", "-n", "htdemucs",
           "-o", str(out_dir), str(audio_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    vocals = out_dir / "htdemucs" / audio_path.stem / "vocals.wav"
    if res.returncode != 0 or not vocals.exists():
        raise RuntimeError(f"Очистка фона не удалась: {(res.stderr or '')[-300:]}")
    return vocals


def process(file_path, url, model_key, do_translate, do_clean=False, progress=gr.Progress()):
    ensure_dirs()
    t0 = time.time()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # источник
    if file_path:
        src = Path(file_path)
    elif url and url.strip().startswith("http"):
        try:
            progress(0.02, desc="Скачиваю по ссылке…")
            src = download_url(url)
        except Exception as e:
            return "", "", f"❌ {e}"
    else:
        return "", "", "Загрузи файл или вставь ссылку."

    try:
        progress(0.05, desc="Извлекаю аудио…")
        audio_file = extract_audio(src)
        if do_clean:
            progress(0.08, desc="Убираю фоновую музыку/шум (Demucs, несколько минут)…")
            audio_file = clean_vocals(audio_file)
        audio = load_audio(audio_file)
        dur = int(len(audio) / 16000)

        if model_key == "облако":
            progress(0.15, desc="Распознаю в облаке (ElevenLabs, секунды)…")
            ky_text = transcribe_cloud(audio_file)
        else:
            progress(0.15, desc=f"Распознаю речь ({dur} сек аудио, несколько минут)…")
            ky_text = transcribe_audio(audio, model_key)

        # сохранить транскрипт
        t_path = unique_path(DIR_TRANSCRIPTS / f"{stamp}_{src.stem}.txt")
        t_path.write_text(ky_text, encoding="utf-8")

        ru_text = ""
        if do_translate:
            progress(0.8, desc="Перевожу на русский (Claude)…")
            ru_text = translate_text(ky_text)
            r_path = unique_path(DIR_TRANSLATIONS / f"{stamp}_{src.stem}_ru.txt")
            r_path.write_text(ru_text, encoding="utf-8")

        log_history({
            "дата": stamp, "исходник": src.name, "длительность_сек": dur,
            "модель": model_key, "обработка_мин": f"{(time.time()-t0)/60:.1f}",
            "транскрипт": t_path.name, "статус": "готово",
        })
        mins = (time.time() - t0) / 60
        status = (f"✅ Готово за {mins:.1f} мин. Транскрипт: 3_транскрипты/{t_path.name}"
                  + (f", перевод: 4_переводы/{stamp}_{src.stem}_ru.txt" if ru_text else ""))
        return ky_text, ru_text, status
    except Exception as e:
        return "", "", f"❌ Ошибка: {e}"


def retranslate(ky_text):
    """Перевести заново (например, после ручной правки транскрипта)."""
    if not ky_text.strip():
        return "", "Сначала нужен кыргызский текст."
    ru = translate_text(ky_text)
    return ru, "Перевод обновлён ✅"


def process_store(file_path, url, model_key, do_translate, do_clean=False, progress=gr.Progress()):
    ky, ru, status = process(file_path, url, model_key, do_translate, do_clean, progress)
    return ky, ru, status, {"ky": ky, "ru": ru}


def retranslate_store(ky_text):
    ru, status = retranslate(ky_text)
    return ru, status, {"ky": ky_text, "ru": ru}


def restore_short(store):
    """Результат живёт в браузере: после обновления — на месте, у нового посетителя пусто."""
    store = store or {}
    ky, ru = store.get("ky", ""), store.get("ru", "")
    status = "↺ Восстановлено после обновления страницы" if (ky or ru) else ""
    return ky, ru, status


def load_history_rows():
    if not HISTORY.exists():
        return []
    with open(HISTORY, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [[r.get(k, "") for k in
             ("дата", "исходник", "длительность_сек", "модель", "обработка_мин",
              "статус", "транскрипт")]
            for r in reversed(rows)]


def show_history_item(evt: gr.SelectData):
    """Клик по строке истории → показать транскрипт и перевод."""
    rows = load_history_rows()
    if evt.index is None or evt.index[0] >= len(rows):
        return "", "", "Запись не найдена."
    t_name = rows[evt.index[0]][6]
    if not t_name:
        return "", "", "У этой записи нет сохранённого транскрипта."
    t_path = DIR_TRANSCRIPTS / t_name
    if not t_path.exists():
        return "", "", f"Файл не найден: {t_name}"
    ky = t_path.read_text(encoding="utf-8")
    if ky.startswith("#"):
        ky = "\n".join(l for l in ky.splitlines() if not l.startswith("#")).strip()
    ru_path = DIR_TRANSLATIONS / f"{t_path.stem}_ru.txt"
    ru = ru_path.read_text(encoding="utf-8") if ru_path.exists() else "(перевода не было)"
    return ky, ru, f"Показан: {t_name}"


# ---------- интерфейс ----------

MOBILE_CSS = """
/* мобильная адаптация */
@media (max-width: 768px) {
    .stack-mobile { flex-direction: column !important; }
    .stack-mobile > * { min-width: 100% !important; }
    button { min-height: 48px; font-size: 16px !important; }
    .gradio-container { padding: 8px !important; }
    h1 { font-size: 1.3em !important; }
}
/* 16px в полях ввода = айфон не зумит при тапе */
textarea, input[type="text"] { font-size: 16px !important; }
"""

with gr.Blocks(title="Ustaz Transcriber", css=MOBILE_CSS) as app:
    gr.Markdown("# 🎙 Ustaz Transcriber <span style='font-size:0.5em; color:#888;'>made by Aisezim ✨</span>\nКыргызская речь → транскрипт → русский перевод")

    with gr.Tab("Перевод"):
        with gr.Row(elem_classes="stack-mobile"):
            file_in = gr.File(label="Видео/аудио файл (mp4, mp3, m4a…)", type="filepath")
            with gr.Column():
                url_in = gr.Textbox(label="…или ссылка (Instagram — лучше файлом)",
                                    placeholder="https://www.instagram.com/p/…")
                model_in = gr.Radio(["облако"], value="облако", label="Модель распознавания",
                                    info="ElevenLabs Scribe — секунды, с пунктуацией")
                translate_in = gr.Checkbox(value=True, label="Сразу перевести на русский (Claude)")
                clean_in = gr.Checkbox(value=False, visible=False, label="очистка недоступна в облачной версии")
        go_btn = gr.Button("▶ Поехали", variant="primary")
        status_out = gr.Markdown()
        with gr.Row(elem_classes="stack-mobile"):
            ky_out = gr.Textbox(label="Кыргызский транскрипт (можно править)", lines=16,
                                interactive=True, buttons=["copy"])
            ru_out = gr.Textbox(label="Русский перевод (можно править)", lines=16,
                                interactive=True, buttons=["copy"])
        retr_btn = gr.Button("↻ Перевести заново (после правок слева)")

        short_store = gr.BrowserState({"ky": "", "ru": ""})
        go_btn.click(process_store, [file_in, url_in, model_in, translate_in, clean_in],
                     [ky_out, ru_out, status_out, short_store])
        retr_btn.click(retranslate_store, [ky_out], [ru_out, status_out, short_store])
        # после обновления страницы вернуть СВОИ результаты (из этого браузера)
        app.load(restore_short, [short_store], [ky_out, ru_out, status_out])

    with gr.Tab("Словарь терминов"):
        gr.Markdown("Глоссарий подставляется в каждый перевод. Добавляй новые термины — "
                    "перевод будет единообразным из недели в неделю.")
        gloss_df = gr.Dataframe(headers=["термин", "перевод", "пояснение"],
                                value=load_glossary_rows(), interactive=True,
                                col_count=(3, "fixed"))
        with gr.Row():
            gloss_save = gr.Button("💾 Сохранить словарь", variant="primary")
            gloss_reload = gr.Button("↻ Перечитать с диска")
        gloss_status = gr.Markdown()
        gloss_save.click(save_glossary_rows, [gloss_df], [gloss_status])
        gloss_reload.click(lambda: load_glossary_rows(), None, [gloss_df])

    with gr.Tab("История"):
        gr.Markdown("Кликни по строке — внизу откроются транскрипт и перевод этого ролика.")
        hist_df = gr.Dataframe(headers=["дата", "исходник", "сек", "модель", "мин",
                                        "статус", "файл"],
                               value=load_history_rows(), interactive=False)
        gr.Button("↻ Обновить").click(lambda: load_history_rows(), None, [hist_df])
        hist_status = gr.Markdown()
        with gr.Row(elem_classes="stack-mobile"):
            hist_ky = gr.Textbox(label="Кыргызский транскрипт", lines=12, buttons=["copy"])
            hist_ru = gr.Textbox(label="Русский перевод", lines=12, buttons=["copy"])
        hist_df.select(show_history_item, None, [hist_ky, hist_ru, hist_status])


if __name__ == "__main__":
    ensure_dirs()
    auth = None
    if os.environ.get("AUTH_USER") and os.environ.get("AUTH_PASS"):
        auth = (os.environ["AUTH_USER"], os.environ["AUTH_PASS"])
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port, auth=auth, pwa=True)

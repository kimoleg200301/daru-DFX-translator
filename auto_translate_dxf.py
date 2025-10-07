import argparse
import csv
import sys
import threading
import time
from contextlib import nullcontext
from itertools import cycle
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple

import ezdxf

import apply_dxf_translation as applier
import extract_dxf_texts as extractor
from auto_translation import TranslationEngine


class Spinner:
    def __init__(self, message: str = "", enabled: bool = True, interval: float = 0.1) -> None:
        self.message = message
        self.enabled = enabled
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join()
            sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")
            sys.stdout.flush()

    def _spin(self) -> None:
        spinner = cycle("|/-\\")
        while not self._stop_event.is_set():
            sys.stdout.write(f"\r{next(spinner)} {self.message}")
            sys.stdout.flush()
            time.sleep(self.interval)


def colorize(text: str, color_code: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{color_code}m{text}\033[0m"
    return text


def ensure_parent(path: Path) -> None:
    if path and path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def write_translated_txt(path: Path, items: List[Tuple[str, int]], translations: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for (text, count), translated in zip(items, translations):
            f.write(f"[{count}] {translated}\n")


def write_map_csv(path: Path, english: List[str], russian: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text_en", "text_ru"])
        for en, ru in zip(english, russian):
            writer.writerow([en, ru])


def apply_translations(doc, pairs: List[Tuple[str, str]], style_font: Optional[str] = None) -> None:
    if style_font:
        applier.STYLE_FONT = style_font
    ru_style = applier.ensure_ru_style(doc)
    for layout in [doc.modelspace(), *[l for l in doc.layouts if l.name != "Model"]]:
        applier.walk_layout(layout, pairs, ru_style)
    applier.walk_blocks(doc, pairs, ru_style)


def translate_dxf(
    *,
    input_path: Path,
    output_path: Path,
    translator_name: str = "google",
    source_lang: str = "en",
    target_lang: str = "ru",
    style_font: Optional[str] = None,
    deepl_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_temperature: float = 0.2,
    map_path: Optional[Path] = None,
    save_map: bool = True,
    extracted_txt_path: Optional[Path] = None,
    translated_txt_path: Optional[Path] = None,
    save_txt: bool = True,
    log: Optional[Callable[[str], None]] = None,
    translation_context_factory: Optional[Callable[[str], ContextManager[Any]]] = None,
) -> Dict[str, Any]:
    logger = log or (lambda message: None)

    logger(f"Загружаем DXF: {input_path}")
    doc = ezdxf.readfile(str(input_path))

    freq = extractor.extract_text_counts(doc)
    sorted_items = extractor.sort_frequency(freq)
    english_texts = [text for text, _ in sorted_items]
    logger(f"Найдено {len(english_texts)} текстовых элементов для перевода")

    translator = TranslationEngine(
        provider=translator_name,
        source_lang=source_lang,
        target_lang=target_lang,
        deepl_auth_key=deepl_key,
        openai_api_key=openai_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        openai_temperature=openai_temperature,
    )
    logger(f"Инициализирован движок перевода: {translator.backend_name()}")

    context_factory = translation_context_factory or (lambda _msg: nullcontext())
    logger("Начинаем перевод... [0%]")
    total_items = len(english_texts)
    russian_texts: List[str] = []
    if total_items == 0:
        logger("Начинаем перевод... [100%]")
    else:
        batch_size = max(1, total_items // 20)  # целимся максимум в ~20 шагов
        processed = 0
        with context_factory("Переводим..."):
            for start in range(0, total_items, batch_size):
                chunk = english_texts[start : start + batch_size]
                translated_chunk = translator.translate_many(chunk)
                russian_texts.extend(translated_chunk)
                processed += len(translated_chunk)
                percent = min(100, int(processed / total_items * 100))
                logger(f"Начинаем перевод... [{percent}%]")
        if processed < total_items:
            # На случай, если последний перевод не дожал до 100%
            logger("Начинаем перевод... [100%]")
    logger("Перевод завершён")

    if save_txt and extracted_txt_path:
        ensure_parent(extracted_txt_path)
        extractor.write_txt(freq, extracted_txt_path)
        logger(f"Сохранён TXT с исходными текстами: {extracted_txt_path}")
    if save_txt and translated_txt_path:
        ensure_parent(translated_txt_path)
        write_translated_txt(translated_txt_path, sorted_items, russian_texts)
        logger(f"Сохранён TXT с переводами: {translated_txt_path}")

    if save_map and map_path:
        ensure_parent(map_path)
        write_map_csv(map_path, english_texts, russian_texts)
        logger(f"Сохранена карта переводов CSV: {map_path}")

    pairs = list(zip(english_texts, russian_texts))
    pairs.sort(key=lambda x: -len(x[0]))
    logger("Применяем переводы к DXF")
    apply_translations(doc, pairs, style_font=style_font)

    ensure_parent(output_path)
    doc.saveas(str(output_path))
    logger(f"DXF сохранён: {output_path}")

    return {
        "output_path": output_path,
        "backend": translator.backend_name(),
        "map_saved": bool(save_map and map_path),
        "extracted_txt_saved": bool(save_txt and extracted_txt_path),
        "translated_txt_saved": bool(save_txt and translated_txt_path),
        "items_translated": len(english_texts),
    }


def run_pipeline(args: argparse.Namespace) -> None:
    input_path = Path(args.input_dxf)
    output_path = Path(args.output_dxf)
    map_path = Path(args.map_csv) if args.map_csv else None
    extracted_txt_path = Path(args.extracted_txt) if args.extracted_txt else None
    translated_txt_path = Path(args.translated_txt) if args.translated_txt else None

    def cli_log(message: str) -> None:
        print(colorize(message, "36")) if message else None

    try:
        result = translate_dxf(
            input_path=input_path,
            output_path=output_path,
            translator_name=args.translator,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            style_font=args.style_font,
            deepl_key=getattr(args, "deepl_key", None),
            openai_key=getattr(args, "openai_key", None),
            openai_model=getattr(args, "openai_model", None),
            openai_base_url=getattr(args, "openai_base_url", None),
            openai_temperature=getattr(args, "openai_temperature", 0.2),
            map_path=map_path,
            save_map=not args.no_map,
            extracted_txt_path=extracted_txt_path,
            translated_txt_path=translated_txt_path,
            save_txt=not args.skip_txt,
            log=cli_log,
            translation_context_factory=(
                (lambda msg: Spinner(colorize(msg, 30), enabled=sys.stdout.isatty()))
                if sys.stdout.isatty()
                else None
            ),
        )
    except RuntimeError as exc:
        print(colorize(f"Ошибка: {exc}", "31"))
        raise SystemExit(3)

    print(colorize("Saved:", "32"), colorize(str(result["output_path"]), "36"))
    print(colorize("Translator backend:", "32"), colorize(result["backend"], "36"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract, auto-translate and apply translations to DXF.")
    parser.add_argument("input_dxf", help="Входной DXF файл")
    parser.add_argument("output_dxf", help="Выходной DXF файл")
    parser.add_argument("--map-csv", default="map_auto.csv", help="Путь для сохранения карты переводов CSV")
    parser.add_argument("--no-map", action="store_true", help="Не сохранять CSV карту переводов")
    parser.add_argument("--extracted-txt", default="extracted_texts.txt", help="TXT с исходным текстом")
    parser.add_argument("--translated-txt", default="extracted_texts_ru.txt", help="TXT с переводом")
    parser.add_argument("--skip-txt", action="store_true", help="Не сохранять TXT-файлы")
    parser.add_argument(
        "--translator",
        default="google",
        help="Движок перевода (google/googletrans/deep_google/deepl/chatgpt/noop)",
    )
    parser.add_argument("--source-lang", default="en", help="Язык оригинала (default: en)")
    parser.add_argument("--target-lang", default="ru", help="Язык перевода (default: ru)")
    parser.add_argument("--style-font", help="Имя TTF-файла для стиля шрифта кириллицы")
    parser.add_argument("--deepl-key", help="Ключ DeepL (можно также задать через переменную DEEPL_AUTH_KEY)")
    parser.add_argument("--openai-key", help="Ключ OpenAI совместимого API (или переменная OPENAI_API_KEY)")
    parser.add_argument("--openai-model", help="Модель OpenAI (default: gpt-4o-mini)")
    parser.add_argument("--openai-base-url", help="Базовый URL для OpenAI-совместимого API")
    parser.add_argument(
        "--openai-temperature",
        type=float,
        default=0.2,
        help="Температура генерации ChatGPT (default: 0.2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

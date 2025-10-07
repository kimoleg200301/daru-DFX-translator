# Daru DXF Translator

Инструмент для автоматического извлечения текста из DXF-чертежей, перевода на выбранный язык и обратного применения перевода. Проект включает графический интерфейс на PySide6 и CLI-пайплайн, поддерживает Google/DeepL/OpenAI и может формировать вспомогательные CSV/TXT файлы для контроля качества.

## Возможности

- Извлечение всех текстовых сущностей из DXF и отображение частот.
- Перевод через `google`, `deep_google`, `googletrans`, `deepl`, `chatgpt` (OpenAI) или `noop`.
- Сохранение карты соответствий в CSV и текстовых файлов с оригиналом/переводом.
- Автоматическая замена текста в копии исходного чертежа.
- Графический интерфейс с полосатым логом, поддержкой drag & drop и настройками API.

## Требования

- Python 3.9+
- Системные зависимости из `requirements-gui.txt`:
  - PySide6, pyinstaller, ezdxf, deep-translator, googletrans==4.0.0-rc1, deepl, openai

Создайте виртуальное окружение и установите зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-gui.txt
```

## Настройка ключей API

- **OpenAI**: установите `OPENAI_API_KEY` (и при необходимости `OPENAI_BASE_URL`) через настройки GUI или переменные окружения. Клиент автоматически подхватит ключ для `chatgpt`.
- **DeepL**: задайте `DEEPL_AUTH_KEY`/`DEEPL_API_KEY` в окружении или в GUI.

## Запуск GUI

```bash
python3 daru_gui.py
```

В интерфейсе выберите входной DXF, при необходимости укажите пути для выходных файлов и выберите движок перевода. В разделе «Настройки API» сохраните ключи before запуска. Лог отображает ход работы, чередуя белый/серый фон строк.

## CLI-пайплайн

Для пакетной обработки без GUI используйте `auto_translate_dxf.py`:

```bash
python3 auto_translate_dxf.py INPUT.dxf OUTPUT_ru.dxf \
    --translator chatgpt \
    --map-csv OUTPUT_map.csv \
    --extracted-txt OUTPUT_texts.txt \
    --translated-txt OUTPUT_texts_ru.txt
```

- Используйте `--translator` для выбора движка.
- Опции `--no-map` и `--skip-txt` отменяют сохранение CSV/TXT.
- Для OpenAI/DeepL ключи должны быть заданы, как описано выше.

## Сборка standalone

PyInstaller-спека уже настроена. После установки зависимостей выполните:

```bash
pyinstaller daru_gui.spec
```

Готовый артефакт появится в директории `dist/`.

## Полезные файлы

- `auto_translation.py` — ядро движков перевода и нормализация текста.
- `apply_dxf_translation.py` — применение перевода к DXF.
- `extract_dxf_texts.py` — извлечение и сортировка текста.
- `build_translations.py` — билд-скрипт для пакетной обработки.

## Обратная связь

В логах CLI/GUI приводятся расшифровки ошибок. При проблемах с API убедитесь, что ключи заданы и сеть доступна.

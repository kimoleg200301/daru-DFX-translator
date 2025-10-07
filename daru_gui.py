#!/usr/bin/env python3
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QAbstractItemView,
        QListWidget,
        QListWidgetItem,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - GUI dependency guard
    print("PySide6 не установлен. Установите его командой: pip install PySide6", file=sys.stderr)
    raise SystemExit(1) from exc

from auto_translate_dxf import translate_dxf

SETTINGS_PATH = Path.home() / ".daru_gui_settings.json"

LANGUAGE_CHOICES = [
    "auto",
    "en",
    "ru",
    "de",
    "fr",
    "es",
    "it",
    "pl",
    "uk",
    "zh",
    "ja",
    "ko",
]

STYLE_FONT_CHOICES = [
    "Arial.ttf",
    "ArialUnicode.ttf",
    "Roboto-Regular.ttf",
    "NotoSans-Regular.ttf",
    "DejaVuSans.ttf",
]

OPENAI_MODEL_CHOICES = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-3.5-turbo",
]

OPENAI_BASE_URL_CHOICES = [
    "https://api.openai.com/v1",
    "https://api.groq.com/openai/v1",
    "https://openrouter.ai/api/v1",
    "https://api.perplexity.ai",
]


def populate_combo(combo: QComboBox, options: Iterable[str], current: str = "", allow_empty: bool = False) -> None:
    combo.blockSignals(True)
    combo.clear()
    entries = list(options)
    seen = set()
    if allow_empty:
        combo.addItem("")
        seen.add("")
    for option in entries:
        if option and option not in seen:
            combo.addItem(option)
            seen.add(option)
    if current and current not in seen:
        combo.addItem(current)
        seen.add(current)
    combo.setEditable(True)
    default_text = current or ("" if allow_empty else (entries[0] if entries else ""))
    combo.setCurrentText(default_text)
    combo.blockSignals(False)


@dataclass
class AppSettings:
    translator_name: str = "google"
    source_lang: str = "en"
    target_lang: str = "ru"
    style_font: str = ""
    deepl_key: str = ""
    openai_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""
    openai_temperature: float = 0.2
    save_map: bool = True
    save_txt: bool = True
    last_directory: str = str(Path.home())

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
        defaults = cls()
        defaults.__dict__.update(data)
        return defaults


class SettingsManager:
    def __init__(self, path: Path = SETTINGS_PATH) -> None:
        self.path = path
        self._settings = AppSettings()
        self.load()

    @property
    def data(self) -> AppSettings:
        return self._settings

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return
        self._settings = AppSettings.from_dict(raw)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump(self._settings.to_dict(), fh, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
        self.save()


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки API")
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.deepl_edit = QLineEdit(settings.deepl_key)
        self.openai_key_edit = QLineEdit(settings.openai_key)
        self.openai_model_combo = QComboBox()
        populate_combo(self.openai_model_combo, OPENAI_MODEL_CHOICES, settings.openai_model)
        self.openai_url_combo = QComboBox()
        populate_combo(self.openai_url_combo, OPENAI_BASE_URL_CHOICES, settings.openai_base_url, allow_empty=True)
        self.openai_temp_spin = QSpinBox()
        self.openai_temp_spin.setRange(0, 100)
        self.openai_temp_spin.setValue(int(settings.openai_temperature * 100))

        self.openai_key_edit.setEchoMode(QLineEdit.Password)
        self.deepl_edit.setEchoMode(QLineEdit.Password)

        form.addRow("DeepL API Key", self.deepl_edit)
        form.addRow("OpenAI API Key", self.openai_key_edit)
        form.addRow("OpenAI Model", self.openai_model_combo)
        form.addRow("OpenAI Base URL", self.openai_url_combo)
        form.addRow("OpenAI Temperature (x100)", self.openai_temp_spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_values(self) -> Dict[str, Any]:
        return {
            "deepl_key": self.deepl_edit.text().strip(),
            "openai_key": self.openai_key_edit.text().strip(),
            "openai_model": self.openai_model_combo.currentText().strip() or "gpt-4o-mini",
            "openai_base_url": self.openai_url_combo.currentText().strip(),
            "openai_temperature": self.openai_temp_spin.value() / 100.0,
        }


class TranslateWorker(QThread):
    log_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal(dict)

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params

    def run(self) -> None:
        try:
            result = translate_dxf(log=self.log_signal.emit, **self._params)
            self.finished_signal.emit(result)
        except Exception as exc:  # pragma: no cover - background thread errors
            self.error_signal.emit(str(exc))


class MainWindow(QWidget):
    def __init__(self, settings_manager: SettingsManager) -> None:
        super().__init__()
        self.settings_manager = settings_manager
        self.worker: Optional[TranslateWorker] = None
        self.setWindowTitle("Daru DXF Translator")
        self.resize(860, 640)
        self.setAcceptDrops(True)

        self.status_label = QLabel("Перетащите DXF файл или выберите его через обозреватель.")
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.input_browse = QPushButton("Обзор...")
        self.input_browse.clicked.connect(self.select_input_file)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Входной DXF"))
        input_row.addWidget(self.input_edit)
        input_row.addWidget(self.input_browse)

        self.output_edit = QLineEdit()
        self.output_browse = QPushButton("Сохранить как...")
        self.output_browse.clicked.connect(self.select_output_file)
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Выходной DXF"))
        output_row.addWidget(self.output_edit)
        output_row.addWidget(self.output_browse)

        self.translator_combo = QComboBox()
        self.translator_combo.addItems(["google", "deep_google", "googletrans", "deepl", "chatgpt", "noop"])
        self.source_lang_combo = QComboBox()
        populate_combo(self.source_lang_combo, LANGUAGE_CHOICES, settings_manager.data.source_lang)
        self.target_lang_combo = QComboBox()
        populate_combo(self.target_lang_combo, LANGUAGE_CHOICES, settings_manager.data.target_lang)
        self.style_font_combo = QComboBox()
        populate_combo(self.style_font_combo, STYLE_FONT_CHOICES, settings_manager.data.style_font, allow_empty=True)

        self.map_checkbox = QCheckBox("Сохранить CSV карту переводов")
        self.map_path_edit = QLineEdit()
        self.map_browse = QPushButton("...")
        self.map_browse.clicked.connect(lambda: self.browse_aux_file(self.map_path_edit))
        self.map_checkbox.toggled.connect(self.update_aux_controls)

        self.txt_checkbox = QCheckBox("Сохранять TXT промежуточные файлы")
        self.extracted_path_edit = QLineEdit()
        self.extracted_browse = QPushButton("...")
        self.extracted_browse.clicked.connect(lambda: self.browse_aux_file(self.extracted_path_edit))
        self.translated_path_edit = QLineEdit()
        self.translated_browse = QPushButton("...")
        self.translated_browse.clicked.connect(lambda: self.browse_aux_file(self.translated_path_edit))
        self.txt_checkbox.toggled.connect(self.update_aux_controls)

        self.start_button = QPushButton("Запустить перевод")
        self.start_button.clicked.connect(self.start_translation)
        self.settings_button = QPushButton("Настройки API")
        self.settings_button.clicked.connect(self.open_settings)
        self.clear_log_button = QPushButton("Очистить лог")
        self.clear_log_button.clicked.connect(self.clear_log)

        self.log_view = QListWidget()
        self.log_view.setAlternatingRowColors(True)
        self.log_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.log_view.setStyleSheet(
            "QListWidget { background: #ffffff; border: 1px solid #d0d0d0; }"
            "QListWidget::item { padding: 4px; background: #ffffff; }"
            "QListWidget::item:alternate { background: #f5f5f5; }"
        )

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(input_row)
        main_layout.addLayout(output_row)

        options_layout = QFormLayout()
        options_layout.addRow("Переводчик", self.translator_combo)
        options_layout.addRow("Исходный язык", self.source_lang_combo)
        options_layout.addRow("Целевой язык", self.target_lang_combo)
        options_layout.addRow("Шрифт стиля", self.style_font_combo)
        main_layout.addLayout(options_layout)

        map_layout = QHBoxLayout()
        map_layout.addWidget(self.map_checkbox)
        map_layout.addWidget(self.map_path_edit)
        map_layout.addWidget(self.map_browse)
        main_layout.addLayout(map_layout)

        txt_layout1 = QHBoxLayout()
        txt_layout1.addWidget(self.txt_checkbox)
        txt_layout1.addStretch()
        main_layout.addLayout(txt_layout1)

        txt_layout2 = QHBoxLayout()
        txt_layout2.addWidget(QLabel("TXT исходных"))
        txt_layout2.addWidget(self.extracted_path_edit)
        txt_layout2.addWidget(self.extracted_browse)
        main_layout.addLayout(txt_layout2)

        txt_layout3 = QHBoxLayout()
        txt_layout3.addWidget(QLabel("TXT переводов"))
        txt_layout3.addWidget(self.translated_path_edit)
        txt_layout3.addWidget(self.translated_browse)
        main_layout.addLayout(txt_layout3)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.start_button)
        buttons_row.addWidget(self.settings_button)
        buttons_row.addWidget(self.clear_log_button)
        buttons_row.addStretch()
        main_layout.addLayout(buttons_row)

        main_layout.addWidget(QLabel("Лог"))
        main_layout.addWidget(self.log_view, stretch=1)

        self.restore_from_settings()
        self.update_aux_controls()

    def restore_from_settings(self) -> None:
        data = self.settings_manager.data
        self.translator_combo.setCurrentText(data.translator_name)
        self.source_lang_combo.setCurrentText(data.source_lang)
        self.target_lang_combo.setCurrentText(data.target_lang)
        self.style_font_combo.setCurrentText(data.style_font)
        self.map_checkbox.setChecked(data.save_map)
        self.txt_checkbox.setChecked(data.save_txt)

    def browse_aux_file(self, target: QLineEdit) -> None:
        start_dir = Path(target.text()).parent if target.text() else Path(self.settings_manager.data.last_directory)
        filename, _ = QFileDialog.getSaveFileName(self, "Выберите файл", str(start_dir))
        if filename:
            target.setText(filename)

    def select_input_file(self) -> None:
        start_dir = self.settings_manager.data.last_directory
        filename, _ = QFileDialog.getOpenFileName(self, "Выберите DXF", start_dir, "DXF files (*.dxf)")
        if filename:
            self.set_input_file(Path(filename))

    def select_output_file(self) -> None:
        start_dir = Path(self.output_edit.text()).parent if self.output_edit.text() else self.settings_manager.data.last_directory
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить DXF", str(start_dir), "DXF files (*.dxf)")
        if filename:
            self.output_edit.setText(filename)

    def update_aux_controls(self) -> None:
        self.map_path_edit.setEnabled(self.map_checkbox.isChecked())
        self.map_browse.setEnabled(self.map_checkbox.isChecked())
        for widget in (self.extracted_path_edit, self.extracted_browse, self.translated_path_edit, self.translated_browse):
            widget.setEnabled(self.txt_checkbox.isChecked())

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(".dxf"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        for url in event.mimeData().urls():
            if url.isLocalFile() and url.toLocalFile().lower().endswith(".dxf"):
                self.set_input_file(Path(url.toLocalFile()))
                event.acceptProposedAction()
                return
        event.ignore()

    def set_input_file(self, path: Path) -> None:
        self.input_edit.setText(str(path))
        self.status_label.setText(f"Выбран файл: {path.name}")
        defaults = self.derive_default_paths(path)
        self.output_edit.setText(str(defaults["output"]))
        self.map_path_edit.setText(str(defaults["map"]))
        self.extracted_path_edit.setText(str(defaults["extracted"]))
        self.translated_path_edit.setText(str(defaults["translated"]))
        self.settings_manager.update(last_directory=str(path.parent))
        self.update_aux_controls()

    def derive_default_paths(self, input_path: Path) -> Dict[str, Path]:
        stem = input_path.stem
        parent = input_path.parent
        return {
            "output": parent / f"{stem}_ru{input_path.suffix}",
            "map": parent / f"{stem}_map.csv",
            "extracted": parent / f"{stem}_texts.txt",
            "translated": parent / f"{stem}_texts_ru.txt",
        }

    def append_log(self, message: str) -> None:
        item = QListWidgetItem(message)
        if message.startswith("Готово. Результат"):
            item.setForeground(QColor("#1E8F40"))
        self.log_view.addItem(item)
        self.log_view.scrollToBottom()

    def clear_log(self) -> None:
        self.log_view.clear()

    def start_translation(self) -> None:
        input_path = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        if not input_path:
            QMessageBox.warning(self, "Ошибка", "Выберите входной DXF файл")
            return
        if not output_path:
            QMessageBox.warning(self, "Ошибка", "Укажите путь для сохранения DXF")
            return
        params: Dict[str, Any] = {
            "input_path": Path(input_path),
            "output_path": Path(output_path),
            "translator_name": self.translator_combo.currentText(),
            "source_lang": self.source_lang_combo.currentText().strip() or "en",
            "target_lang": self.target_lang_combo.currentText().strip() or "ru",
            "style_font": self.style_font_combo.currentText().strip() or None,
            "deepl_key": self.settings_manager.data.deepl_key or None,
            "openai_key": self.settings_manager.data.openai_key or None,
            "openai_model": self.settings_manager.data.openai_model or None,
            "openai_base_url": self.settings_manager.data.openai_base_url or None,
            "openai_temperature": self.settings_manager.data.openai_temperature,
            "map_path": Path(self.map_path_edit.text()) if self.map_checkbox.isChecked() and self.map_path_edit.text() else None,
            "save_map": self.map_checkbox.isChecked(),
            "extracted_txt_path": Path(self.extracted_path_edit.text()) if self.txt_checkbox.isChecked() and self.extracted_path_edit.text() else None,
            "translated_txt_path": Path(self.translated_path_edit.text()) if self.txt_checkbox.isChecked() and self.translated_path_edit.text() else None,
            "save_txt": self.txt_checkbox.isChecked(),
        }
        self.append_log("Запуск процесса перевода...")
        self.start_button.setEnabled(False)
        self.worker = TranslateWorker(params)
        self.worker.log_signal.connect(self.append_log)
        self.worker.error_signal.connect(self.handle_error)
        self.worker.finished_signal.connect(self.handle_finished)
        self.worker.finished.connect(self.reset_worker)
        self.worker.start()
        self.status_label.setText("Перевод выполняется...")

    def handle_error(self, message: str) -> None:
        self.append_log(f"Ошибка: {message}")
        QMessageBox.critical(self, "Ошибка", message)
        self.start_button.setEnabled(True)
        self.status_label.setText("Ошибка при переводе")

    def handle_finished(self, payload: Dict[str, Any]) -> None:
        self.append_log(f"Готово. Результат: {payload['output_path']}")
        self.append_log(f"Движок перевода: {payload['backend']}")
        QMessageBox.information(self, "Готово", f"DXF сохранён: {payload['output_path']}")
        self.start_button.setEnabled(True)
        self.status_label.setText("Перевод завершён")
        self.settings_manager.update(
            translator_name=self.translator_combo.currentText(),
            source_lang=self.source_lang_combo.currentText().strip() or "en",
            target_lang=self.target_lang_combo.currentText().strip() or "ru",
            style_font=self.style_font_combo.currentText().strip(),
            save_map=self.map_checkbox.isChecked(),
            save_txt=self.txt_checkbox.isChecked(),
        )

    def reset_worker(self) -> None:
        self.worker = None

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings_manager.data, self)
        if dialog.exec() == QDialog.Accepted:
            values = dialog.get_values()
            self.settings_manager.update(**values)
            self.append_log("Настройки API сохранены")


def main() -> None:
    app = QApplication(sys.argv)
    manager = SettingsManager()
    window = MainWindow(manager)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

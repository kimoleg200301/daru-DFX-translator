import json
import os
import re
from itertools import zip_longest
from typing import Callable, Iterable, List, Optional, Sequence

DIM_PLACEHOLDER = "__DXF_DIM__"


def chunked(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def restore_edge_whitespace(original: str, translated: str) -> str:
    if not translated:
        return translated
    leading = len(original) - len(original.lstrip())
    trailing = len(original) - len(original.rstrip())
    prefix = original[:leading] if leading else ""
    suffix = original[len(original) - trailing :] if trailing else ""
    core = translated.strip() if (leading or trailing) else translated.strip() or translated
    return f"{prefix}{core}{suffix}"


def prepare_for_translation(text: str) -> str:
    if not text:
        return text
    prepared = text.replace("\\P", "\n").replace("<>", DIM_PLACEHOLDER)
    return prepared


def recover_after_translation(original: str, translated: str) -> str:
    if not translated:
        return translated
    restored = translated.replace(DIM_PLACEHOLDER, "<>")
    if "\\P" in original:
        restored = restored.replace("\r\n", "\n").replace("\n", "\\P")
    return restore_edge_whitespace(original, restored)


class TranslationEngine:
    def __init__(
        self,
        provider: str = "google",
        source_lang: str = "auto",
        target_lang: str = "ru",
        deepl_auth_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_temperature: float = 0.2,
    ):
        self.provider = (provider or "google").lower()
        self.source_lang = source_lang or "auto"
        self.target_lang = target_lang or "ru"
        self.deepl_auth_key = deepl_auth_key
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.openai_model = openai_model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.openai_base_url = openai_base_url or os.environ.get("OPENAI_BASE_URL")
        self.openai_temperature = openai_temperature
        self._translator = None
        self._backend = None
        self._translate_batch = None
        self._last_originals: Optional[Sequence[str]] = None
        self._init_translator()

    def _init_translator(self) -> None:
        tried = []

        if self.provider in ("google", "auto", "deep_google"):
            try:
                from deep_translator import GoogleTranslator  # type: ignore

                self._translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
                self._backend = "deep-google"
                self._translate_batch = self._deep_translate_batch
                return
            except ImportError:
                tried.append("pip install deep-translator")
            except Exception as exc:  # pragma: no cover - network-related
                tried.append(f"deep-translator error: {exc}")
                if self.provider != "auto":
                    raise

        if self.provider in ("google", "auto", "googletrans"):
            try:
                from googletrans import Translator  # type: ignore

                self._translator = Translator()
                self._backend = "googletrans"
                self._translate_batch = self._googletrans_translate_batch
                return
            except ImportError:
                tried.append("pip install googletrans==4.0.0-rc1")
            except Exception as exc:  # pragma: no cover - network-related
                tried.append(f"googletrans error: {exc}")
                if self.provider != "auto":
                    # try fallback implementation below
                    pass

        if self.provider in ("google", "auto", "googletrans", "google_free", "google-free"):
            self._backend = "google-free"
            self._translate_batch = self._google_free_translate_batch
            return

        if self.provider in ("deepl", "auto"):
            auth_key = self.deepl_auth_key or os.environ.get("DEEPL_AUTH_KEY") or os.environ.get("DEEPL_API_KEY")
            if auth_key:
                try:
                    import deepl  # type: ignore

                    self._translator = deepl.Translator(auth_key)
                    self._backend = "deepl"
                    self._translate_batch = self._deepl_translate_batch
                    return
                except ImportError:
                    tried.append("pip install deepl")
                except Exception as exc:  # pragma: no cover - network-related
                    tried.append(f"deepl error: {exc}")
                    if self.provider != "auto":
                        raise
            elif self.provider == "deepl":
                raise RuntimeError("DEEPL_AUTH_KEY не задан")

        if self.provider in ("chatgpt", "gpt", "openai"):
            if not self.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY не задан")

            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            if self.openai_base_url:
                os.environ["OPENAI_BASE_URL"] = self.openai_base_url
            try:
                import openai  # type: ignore

                client = None
                mode = "new"
                try:
                    from openai import OpenAI  # type: ignore

                    client_kwargs = {"api_key": self.openai_api_key}
                    if self.openai_base_url:
                        client_kwargs["base_url"] = self.openai_base_url
                    client = OpenAI(**client_kwargs)
                except (ImportError, AttributeError, TypeError):
                    mode = "legacy"
                except Exception as exc:  # pragma: no cover - network-related
                    tried.append(f"openai error: {exc}")
                    if self.provider != "auto":
                        raise
                    mode = None

                if mode == "legacy":
                    try:
                        openai.api_key = self.openai_api_key
                        if self.openai_base_url:
                            openai.api_base = self.openai_base_url
                    except Exception as exc:  # pragma: no cover - config errors
                        tried.append(f"openai error: {exc}")
                        if self.provider != "auto":
                            raise
                        mode = None

                if mode:
                    self._translator = {
                        "client": client,
                        "module": openai,
                        "model": self.openai_model,
                        "batch_size": 1 if mode == "legacy" else 16,
                        "mode": mode,
                    }
                    self._backend = "chatgpt"
                    self._translate_batch = self._chatgpt_translate_batch
                    return
            except ImportError:
                tried.append("pip install openai")

        if self.provider in ("noop", "identity"):
            self._backend = "identity"
            self._translate_batch = self._identity_translate_batch
            return

        if tried:
            raise RuntimeError("Переводчик недоступен: " + "; ".join(tried))
        raise RuntimeError("Не удалось инициализировать переводчик")

    def backend_name(self) -> str:
        return self._backend or "unknown"

    def translate_many(self, texts: Sequence[str]) -> List[str]:
        if not texts:
            return []
        prepared = [prepare_for_translation(t) for t in texts]
        reset_required = False
        if self._backend == "chatgpt":
            self._last_originals = list(texts)
            reset_required = True
        try:
            translated = self._translate_batch(prepared)
        finally:
            if reset_required:
                self._last_originals = None
        result: List[str] = []
        for original, prepared_text, translated_text in zip(texts, prepared, translated):
            if not translated_text:
                translated_text = original
            result.append(recover_after_translation(original, translated_text))
        return result

    # Backends -------------------------------------------------------------

    def _deep_translate_batch(self, texts: Sequence[str]) -> List[str]:
        try:
            return list(self._translator.translate_batch(list(texts)))  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover
            return [self._translator.translate(t) for t in texts]  # type: ignore[attr-defined]

    def _googletrans_translate_batch(self, texts: Sequence[str]) -> List[str]:  # pragma: no cover - network
        results: List[str] = []
        translator = self._translator  # type: ignore

        def _extract_texts(translation_result) -> List[str]:
            if isinstance(translation_result, list):
                items = translation_result
            else:
                items = [translation_result]
            texts_out: List[str] = []
            for item in items:
                text = getattr(item, "text", None)
                if text is None:
                    return []
                texts_out.append(text)
            return texts_out

        def _fallback_chunk(chunk: Sequence[str]) -> List[str]:
            chunk_results: List[str] = []
            for piece in chunk:
                try:
                    single = translator.translate(piece, src=self.source_lang, dest=self.target_lang)
                    text = getattr(single, "text", None)
                    chunk_results.append(text if text is not None else piece)
                except Exception:  # pragma: no cover - network
                    chunk_results.append(piece)
            return chunk_results

        for chunk in chunked(texts, 30):
            try:
                translated = translator.translate(list(chunk), src=self.source_lang, dest=self.target_lang)
                chunk_results = _extract_texts(translated)
                if len(chunk_results) != len(chunk):
                    raise ValueError("unexpected googletrans response length")
            except Exception:  # pragma: no cover - network
                chunk_results = _fallback_chunk(chunk)
            results.extend(chunk_results)
        return results

    def _deepl_translate_batch(self, texts: Sequence[str]) -> List[str]:  # pragma: no cover - network
        translator = self._translator  # type: ignore
        results: List[str] = []
        source_lang = (self.source_lang or "").strip()
        source = None if not source_lang or source_lang.lower() == "auto" else source_lang.upper()
        target_lang = (self.target_lang or "ru").strip() or "RU"
        target = target_lang.upper()

        for chunk in chunked(texts, 40):
            try:
                translated = translator.translate_text(list(chunk), target_lang=target, source_lang=source)
            except Exception:  # pragma: no cover - network
                chunk_results: List[str] = []
                for piece in chunk:
                    try:
                        single = translator.translate_text(piece, target_lang=target, source_lang=source)
                        text = getattr(single, "text", None)
                        chunk_results.append(text if text is not None else piece)
                    except Exception:
                        chunk_results.append(piece)
                results.extend(chunk_results)
                continue

            if not isinstance(translated, list):
                translated = [translated]

            chunk_results = []
            for original, item in zip_longest(chunk, translated):
                if item is None:
                    chunk_results.append(original)
                    continue
                text = getattr(item, "text", None)
                chunk_results.append(text if text is not None else original)
            results.extend(chunk_results)

        return results

    def _chatgpt_translate_batch(self, texts: Sequence[str]) -> List[str]:  # pragma: no cover - network
        originals = list(self._last_originals or [])
        translator = self._translator or {}
        mode = translator.get("mode", "new")
        client = translator.get("client")
        module = translator.get("module")
        model = translator.get("model")
        batch_size = translator.get("batch_size", 16)

        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.openai_base_url:
            os.environ["OPENAI_BASE_URL"] = self.openai_base_url

        chat_completion_create: Optional[Callable[..., object]] = None
        if module is not None:
            chat_namespace = getattr(module, "chat", None)
            completions_namespace = getattr(chat_namespace, "completions", None) if chat_namespace else None
            create_attr = getattr(completions_namespace, "create", None) if completions_namespace else None
            if callable(create_attr):
                chat_completion_create = create_attr

        openai_major_version: Optional[int] = None
        if module:
            version_value = getattr(module, "__version__", None)
            if isinstance(version_value, str):
                major_part = version_value.split(".", 1)[0]
                if major_part.isdigit():
                    openai_major_version = int(major_part)

        fallback_client = None
        if mode == "legacy":
            fallback_client = client
            if fallback_client is None and module and hasattr(module, "OpenAI"):
                client_kwargs = {"api_key": self.openai_api_key}
                if self.openai_base_url:
                    client_kwargs["base_url"] = self.openai_base_url
                try:
                    fallback_client = module.OpenAI(**client_kwargs)  # type: ignore[attr-defined]
                except TypeError:
                    fallback_client = None
                    try:
                        client_kwargs.pop("base_url", None)
                        fallback_client = module.OpenAI(**client_kwargs)  # type: ignore[attr-defined]
                        if self.openai_base_url and fallback_client is not None:
                            with_options = getattr(fallback_client, "with_options", None)
                            if callable(with_options):
                                fallback_client = with_options(base_url=self.openai_base_url)
                            else:
                                try:
                                    setattr(fallback_client, "base_url", self.openai_base_url)
                                except Exception:
                                    pass
                    except Exception:
                        fallback_client = None
                except Exception:
                    fallback_client = None

            if fallback_client is not None:
                client = fallback_client
                mode = "new"
            elif not chat_completion_create and openai_major_version and openai_major_version >= 1:
                raise RuntimeError(
                    "Установлен openai>=1.0.0, но не удалось инициализировать новый клиент. "
                    "Переустановите пакет 'openai' или закрепите версию <1.0."
                )

        results: List[str] = list(texts)
        if mode == "legacy":
            if (not chat_completion_create) and (not module or not model):
                return results
        else:
            if not client or not model:
                return results

        def should_use_ai(src: str) -> bool:
            stripped = (src or "").strip()
            if len(stripped) < 3:
                return False
            if not re.search(r"[A-Za-z]", stripped):
                return False
            if re.fullmatch(r"[A-Za-z]\.?", stripped):
                return False
            if re.fullmatch(r"[A-Za-z]\d+", stripped):
                return False
            if stripped.isupper() and len(stripped) <= 3 and " " not in stripped:
                return False
            if re.fullmatch(r"[-+]?\d+[\d\s./-]*", stripped):
                return False
            return True

        def extract_message(completion_obj) -> str:
            if completion_obj is None:
                return ""

            choices_obj = getattr(completion_obj, "choices", None)
            if choices_obj:
                first_choice = choices_obj[0]
                message_obj = getattr(first_choice, "message", None)
                if message_obj is not None:
                    content_attr = getattr(message_obj, "content", None)
                    if isinstance(content_attr, str) and content_attr:
                        return content_attr
                    if isinstance(message_obj, dict):
                        content = message_obj.get("content")
                        if isinstance(content, str) and content:
                            return content
                text_attr = getattr(first_choice, "text", None)
                if isinstance(text_attr, str) and text_attr:
                    return text_attr
                if isinstance(first_choice, dict):
                    message_dict = first_choice.get("message", {})
                    if isinstance(message_dict, dict):
                        content = message_dict.get("content")
                        if isinstance(content, str) and content:
                            return content
                    text_val = first_choice.get("text")
                    if isinstance(text_val, str) and text_val:
                        return text_val

            if isinstance(completion_obj, dict):
                choices_dict = completion_obj.get("choices", [])
                if isinstance(choices_dict, list) and choices_dict:
                    first_choice = choices_dict[0]
                    if isinstance(first_choice, dict):
                        message_dict = first_choice.get("message", {})
                        if isinstance(message_dict, dict):
                            content = message_dict.get("content")
                            if isinstance(content, str) and content:
                                return content
                        text_val = first_choice.get("text")
                        if isinstance(text_val, str) and text_val:
                            return text_val

            model_dump = getattr(completion_obj, "model_dump", None)
            if callable(model_dump):
                try:
                    dumped = model_dump()
                except Exception:  # pragma: no cover - best effort
                    dumped = None
                if isinstance(dumped, dict):
                    return extract_message(dumped)

            return ""

        candidate_indices = []
        for idx, prepared in enumerate(texts):
            original = originals[idx] if idx < len(originals) else prepared
            if should_use_ai(original):
                candidate_indices.append(idx)
            else:
                results[idx] = prepared

        if not candidate_indices:
            return results

        system_content = (
            "You are a professional technical translator. Translate the provided values from "
            f"{self.source_lang or 'auto-detected'} to {self.target_lang}. Preserve numbers, "
            "placeholders like '__DXF_DIM__', and DXF control sequences such as '\n'. Respond "
            "with strict JSON: {\"translations\": [{\"id\": \"<id>\", \"text\": \"<translated>\"}, ...]}"
        )

        for chunk in chunked(candidate_indices, batch_size):
            payload = [
                {"id": str(idx), "text": texts[idx]}
                for idx in chunk
            ]
            user_content = json.dumps({"items": payload}, ensure_ascii=False)
            messages_payload = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            if mode == "legacy":
                for idx in chunk:
                    single_payload = json.dumps(
                        {"items": [{"id": str(idx), "text": texts[idx]}]},
                        ensure_ascii=False,
                    )
                    single_messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": single_payload},
                    ]
                    try:
                        if chat_completion_create:
                            completion_kwargs = {
                                "model": model,
                                "temperature": self.openai_temperature,
                                "messages": single_messages,
                                "response_format": {"type": "json_object"},
                            }
                            completion = chat_completion_create(**completion_kwargs)  # type: ignore[misc]
                        elif module is not None:
                            completion = module.ChatCompletion.create(  # type: ignore[attr-defined]
                                model=model,
                                temperature=self.openai_temperature,
                                messages=single_messages,
                            )
                        else:  # pragma: no cover - defensive
                            raise RuntimeError("OpenAI legacy интерфейс недоступен")
                    except Exception as exc:  # pragma: no cover - network
                        raise RuntimeError(f"ChatGPT translation failed: {exc}") from exc

                    message = extract_message(completion) or "{}"

                    translated_text = texts[idx]
                    try:
                        data = json.loads(message or "{}")
                        for item in data.get("translations", []):
                            if isinstance(item, dict) and item.get("id") == str(idx):
                                candidate = item.get("text")
                                if isinstance(candidate, str) and candidate:
                                    translated_text = candidate
                                    break
                    except json.JSONDecodeError:
                        cleaned = message.strip()
                        if cleaned:
                            translated_text = cleaned
                    results[idx] = translated_text
                continue

            try:
                completion = client.chat.completions.create(  # type: ignore[attr-defined]
                    model=model,
                    temperature=self.openai_temperature,
                    response_format={"type": "json_object"},
                    messages=messages_payload,
                )
            except Exception as exc:  # pragma: no cover - network
                raise RuntimeError(f"ChatGPT translation failed: {exc}") from exc

            message = extract_message(completion) or "{}"
            try:
                data = json.loads(message or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("ChatGPT вернул неожиданный ответ (не JSON)") from exc

            mapping = {item.get("id"): item.get("text") for item in data.get("translations", []) if isinstance(item, dict)}

            for idx in chunk:
                translated = mapping.get(str(idx))
                results[idx] = translated if translated else texts[idx]

        return results

    def _google_free_translate_batch(self, texts: Sequence[str]) -> List[str]:  # pragma: no cover - network
        import json as _json
        import urllib.parse
        import urllib.request

        results: List[str] = []
        source = (self.source_lang or "auto").lower()
        target = (self.target_lang or "ru").lower()

        base_params = {
            "client": "gtx",
            "sl": source,
            "tl": target,
            "dt": "t",
        }
        base_query = urllib.parse.urlencode(base_params)

        for text in texts:
            q = urllib.parse.urlencode({"q": text})
            url = f"https://translate.googleapis.com/translate_a/single?{base_query}&{q}"

            try:
                with urllib.request.urlopen(url) as resp:
                    payload = resp.read().decode("utf-8")
                data = _json.loads(payload)
                translation = data[0][0][0] if isinstance(data, list) and data and isinstance(data[0], list) else None
            except Exception:
                translation = None

            results.append(translation if translation is not None else text)

        return results

    def _identity_translate_batch(self, texts: Sequence[str]) -> List[str]:
        return list(texts)


def auto_translate(
    texts: Sequence[str],
    provider: str = "google",
    source_lang: str = "auto",
    target_lang: str = "ru",
    deepl_auth_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_temperature: float = 0.2,
) -> List[str]:
    engine = TranslationEngine(
        provider=provider,
        source_lang=source_lang,
        target_lang=target_lang,
        deepl_auth_key=deepl_auth_key,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        openai_temperature=openai_temperature,
    )
    return engine.translate_many(texts)

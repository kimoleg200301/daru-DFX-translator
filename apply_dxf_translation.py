# apply_dxf_translation.py
# pip install ezdxf
import ezdxf, sys, csv, re, os, argparse
from typing import List, Tuple, Dict

# ===== Настройки стиля шрифта для кириллицы =====
STYLE_NAME = "RU"
STYLE_FONT = "Arial.ttf"  # при желании поменяй на Roboto/Noto/DejaVu

# ===== Загрузка словаря EN->RU из разных форматов =====

def _strip_count_prefix(s: str) -> str:
    # "[12] The text" -> "The text"
    m = re.match(r"^\s*\[\d+\]\s*(.*)$", s.strip())
    return m.group(1) if m else s.strip()

def load_map_from_csv(path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = [c.lower() for c in r.fieldnames or []]
        # допускаем разные заголовки
        c_en = "text_en" if "text_en" in cols else (r.fieldnames[0] if r.fieldnames else None)
        c_ru = "text_ru" if "text_ru" in cols else (r.fieldnames[1] if r.fieldnames and len(r.fieldnames) > 1 else None)
        if not c_en or not c_ru:
            raise ValueError("CSV должен содержать две колонки: text_en,text_ru")
        for row in r:
            en = (row.get(c_en) or "").strip()
            ru = (row.get(c_ru) or "").strip()
            if en:
                pairs.append((en, ru))
    # длинные ключи сначала (важно для фраз)
    pairs.sort(key=lambda x: -len(x[0]))
    return pairs

def load_map_from_txt_pair(en_txt: str, ru_txt: str) -> List[Tuple[str, str]]:
    with open(en_txt, "r", encoding="utf-8", errors="ignore") as f:
        en_lines = [ln.rstrip("\r\n") for ln in f]
    with open(ru_txt, "r", encoding="utf-8", errors="ignore") as f:
        ru_lines = [ln.rstrip("\r\n") for ln in f]

    # Удаляем префиксы [count] и хвостовые пробелы, выравниваем по длине меньшего
    en_clean = [_strip_count_prefix(x) for x in en_lines if x.strip()]
    ru_clean = [_strip_count_prefix(x) for x in ru_lines if x.strip()]

    n = min(len(en_clean), len(ru_clean))
    pairs = list(zip(en_clean[:n], ru_clean[:n]))
    # уникализуем по первому попаданию (DXF-замены точные, лучше избегать дубликатов-ключей)
    seen: Dict[str, str] = {}
    for en, ru in pairs:
        if en not in seen:
            seen[en] = ru
    pairs = [(k, v) for k, v in seen.items()]
    # длинные ключи сначала
    pairs.sort(key=lambda x: -len(x[0]))
    return pairs

# ===== DXF утилиты =====

def ensure_ru_style(doc) -> str:
    styles = doc.styles
    if STYLE_NAME not in styles:
        styles.new(STYLE_NAME, dxfattribs={"font": STYLE_FONT})
    else:
        try:
            styles.get(STYLE_NAME).dxf.font = STYLE_FONT
        except Exception:
            pass
    return STYLE_NAME

def protect_dim(text: str):
    return text.split("<>"), "<>"

def replace_all_exact(text: str, pairs: List[Tuple[str, str]]) -> str:
    out = text
    for en, ru in pairs:
        if en:
            out = out.replace(en, ru)
    return out

def translate_text_keep_dim_and_mtext_controls(s: str, pairs: List[Tuple[str, str]]) -> str:
    if not s:
        return s
    chunks, sep = protect_dim(s)
    out_chunks = []
    for ch in chunks:
        # не ломаем \P, \A1; и т.д.: заменяем только в «обычных» сегментах
        segments = re.split(r"(\\[A-Za-z][^\\]*)", ch)
        for i, seg in enumerate(segments):
            if not seg or re.fullmatch(r"\\[A-Za-z][^\\]*", seg):
                continue
            segments[i] = replace_all_exact(seg, pairs)
        out_chunks.append("".join(segments))
    return sep.join(out_chunks)

TARGETS = {"TEXT","MTEXT","ATTRIB","ATTDEF","DIMENSION","MULTILEADER","MLEADER","LEADER","TABLE"}

def safe_get_mtext(e) -> str:
    for attr in ("plain_text", "text"):
        try:
            v = getattr(e, attr)
            v = v() if callable(v) else v
            if isinstance(v, str):
                return v
        except Exception:
            pass
    try:
        return e.text
    except Exception:
        return ""

def safe_set_mtext(e, s: str):
    try:
        e.text = s
    except Exception:
        pass

def get_dim_override_text(dim) -> str:
    try:
        t = dim.get_text()
        return t if t and t.strip() != "<>" else ""
    except Exception:
        return ""

def set_dim_override_text(dim, s: str):
    try:
        if s and s.strip():
            dim.set_text(s)
    except Exception:
        pass

def safe_get_mleader_text(e) -> str:
    for meth in ("get_mtext", "mtext"):
        try:
            v = getattr(e, meth)
            v = v() if callable(v) else v
            if isinstance(v, str) and v.strip():
                return v
        except Exception:
            pass
    return ""

def safe_set_mleader_text(e, s: str):
    for meth in ("set_mtext", "set_text", "set_mleader_text"):
        try:
            fn = getattr(e, meth)
            fn(s)
            return
        except Exception:
            pass

def safe_table_cell_text(tbl, r, c) -> str:
    for meth in ("text_cell_content", "get_cell_text", "get_text"):
        try:
            fn = getattr(tbl, meth)
            v = fn(r, c)
            if isinstance(v, str):
                return v
        except Exception:
            pass
    return ""

def safe_table_set_text(tbl, r, c, s: str):
    for meth in ("set_text_cell_content", "set_cell_text", "set_text"):
        try:
            fn = getattr(tbl, meth)
            fn(r, c, s)
            return
        except Exception:
            pass

def process_entity(e, pairs: List[Tuple[str, str]], ru_style: str):
    t = None
    dxft = e.dxftype()
    if dxft == "TEXT":
        t = e.dxf.text or ""
        e.dxf.text = translate_text_keep_dim_and_mtext_controls(t, pairs)
        if ru_style:
            e.dxf.style = ru_style

    elif dxft == "MTEXT":
        t = safe_get_mtext(e)
        new_t = translate_text_keep_dim_and_mtext_controls(t, pairs)
        safe_set_mtext(e, new_t)
        if ru_style:
            try: e.dxf.style = ru_style
            except Exception: pass

    elif dxft in ("ATTRIB","ATTDEF"):
        t = e.dxf.text or ""
        e.dxf.text = translate_text_keep_dim_and_mtext_controls(t, pairs)
        if ru_style:
            try: e.dxf.style = ru_style
            except Exception: pass

    elif dxft == "DIMENSION":
        t = get_dim_override_text(e)
        if t:
            set_dim_override_text(e, translate_text_keep_dim_and_mtext_controls(t, pairs))

    elif dxft in ("MULTILEADER","MLEADER","LEADER"):
        t = safe_get_mleader_text(e)
        if t:
            safe_set_mleader_text(e, translate_text_keep_dim_and_mtext_controls(t, pairs))

    elif dxft == "TABLE":
        try:
            rows, cols = e.nrows, e.ncols
            for r in range(rows):
                for c in range(cols):
                    val = safe_table_cell_text(e, r, c)
                    if isinstance(val, str) and val.strip():
                        safe_table_set_text(e, r, c, translate_text_keep_dim_and_mtext_controls(val, pairs))
        except Exception:
            pass

def walk_layout(layout, pairs: List[Tuple[str, str]], ru_style: str):
    for e in layout:
        if e.dxftype() in TARGETS:
            process_entity(e, pairs, ru_style)

def walk_blocks(doc, pairs: List[Tuple[str, str]], ru_style: str):
    for block in doc.blocks:  # ВАЖНО: нет .items()
        for e in block:
            if e.dxftype() in TARGETS:
                process_entity(e, pairs, ru_style)

# ====== CLI ======

def parse_args():
    p = argparse.ArgumentParser(description="Apply RU translation to DXF using CSV or TXT mapping.")
    p.add_argument("input_dxf", help="Входной DXF")
    p.add_argument("mapping", help="map.csv (text_en,text_ru) ИЛИ translated.txt (русский TXT)")
    p.add_argument("output_dxf", help="Выходной DXF (русский)")
    p.add_argument("--source-en", help="Исходный EN TXT ([count] text). Обязателен, если mapping=*.txt без EN.")
    p.add_argument("--style-font", help=f"TTF шрифт для стиля {STYLE_NAME} (по умолчанию {STYLE_FONT})")
    return p.parse_args()

def main():
    args = parse_args()
    global STYLE_FONT
    if args.style_font:
        STYLE_FONT = args.style_font

    mapping_path = args.mapping.lower()
    if mapping_path.endswith(".csv"):
        pairs = load_map_from_csv(args.mapping)
    elif mapping_path.endswith(".txt"):
        if not args.source_en:
            print("Ошибка: для TXT-перевода нужен исходный английский TXT через --source-en <path>")
            sys.exit(2)
        pairs = load_map_from_txt_pair(args.source_en, args.mapping)
    else:
        print("mapping должен быть .csv или .txt")
        sys.exit(2)

    if not pairs:
        print("Не удалось загрузить пары переводов.")
        sys.exit(3)

    # Открываем DXF, применяем переводы
    doc = ezdxf.readfile(args.input_dxf)
    ru_style = ensure_ru_style(doc)

    # Model space
    walk_layout(doc.modelspace(), pairs, ru_style)
    # Paper layouts
    for layout in doc.layouts:
        if layout.name != "Model":
            walk_layout(layout, pairs, ru_style)
    # Blocks
    walk_blocks(doc, pairs, ru_style)

    doc.saveas(args.output_dxf)
    print("Saved:", args.output_dxf)

if __name__ == "__main__":
    main()

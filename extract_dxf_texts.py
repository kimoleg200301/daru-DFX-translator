# extract_dxf_texts.py
# pip install ezdxf
import ezdxf, sys, json, csv, re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

def is_meaningful(s: str) -> bool:
    return bool((s or "").strip())

def get_dim_override_text(dim):
    try:
        t = dim.get_text()
        return t if t and t.strip() != "<>" else ""
    except Exception:
        return ""

def safe_mtext_text(e):
    # Пытаемся взять видимый текст максимально совместимым способом
    for attr in ("plain_text", "text"):
        try:
            v = getattr(e, attr)
            v = v() if callable(v) else v
            if isinstance(v, str) and v.strip():
                return v
        except Exception:
            pass
    # ezdxf старых версий: e.text
    try:
        v = e.text
        if isinstance(v, str) and v.strip():
            return v
    except Exception:
        pass
    return ""

def safe_table_cell_text(tbl, r, c):
    # В разных версиях ezdxf разные API
    for meth in ("text_cell_content", "get_cell_text", "get_text"):
        try:
            fn = getattr(tbl, meth)
            v = fn(r, c)
            if isinstance(v, str) and v.strip():
                return v
        except Exception:
            pass
    return ""

def record_text(bag: List[Tuple[str, str, str]], kind: str, detail: str, text: str) -> None:
    if not isinstance(text, str):
        return
    normalized = text.strip()
    if not normalized:
        return
    bag.append((kind, detail, text))


def process_entity(e, bag: List[Tuple[str, str, str]], prefix: str = "") -> None:
    dxft = e.dxftype()
    kind = prefix or dxft

    if dxft == "TEXT":
        record_text(bag, kind, "", e.dxf.text or "")
        return

    if dxft == "MTEXT":
        record_text(bag, kind, "", safe_mtext_text(e))
        return

    if dxft in ("ATTRIB", "ATTDEF"):
        record_text(bag, kind, getattr(e.dxf, "tag", ""), e.dxf.text or "")
        return

    if dxft == "DIMENSION":
        record_text(bag, kind, "", get_dim_override_text(e))
        return

    if dxft in ("MULTILEADER", "MLEADER", "LEADER"):
        for meth in ("get_mtext", "mtext"):
            try:
                v = getattr(e, meth)
                v = v() if callable(v) else v
                if isinstance(v, str) and v.strip():
                    record_text(bag, kind, "", v)
                    break
            except Exception:
                continue
        return

    if dxft == "TABLE":
        try:
            rows, cols = e.nrows, e.ncols
            for r in range(rows):
                for c in range(cols):
                    val = safe_table_cell_text(e, r, c)
                    if val:
                        table_kind = f"{prefix}:TABLE" if prefix else "TABLE"
                        record_text(bag, table_kind, f"r{r}c{c}", val)
        except Exception:
            pass
        return

    if dxft in ("INSERT", "MINSERT"):
        insert_kind = prefix or dxft
        try:
            for attrib in getattr(e, "attribs", []):
                record_text(bag, f"{insert_kind}:ATTRIB", getattr(attrib.dxf, "tag", ""), getattr(attrib.dxf, "text", ""))
        except Exception:
            pass
        return


def walk_layout(layout, bag):
    for e in layout:
        process_entity(e, bag)


def walk_blocks(doc, bag):
    # ВАЖНО: у BlocksSection НЕТ .items() — итерируемся напрямую
    for block in doc.blocks:
        name = block.name
        prefix = f"BLOCK:{name}" if name else "BLOCK"
        for e in block:
            process_entity(e, bag, prefix)

def collect_text_bag(doc) -> List[Tuple[str, str, str]]:
    bag: List[Tuple[str, str, str]] = []

    # Model space
    walk_layout(doc.modelspace(), bag)
    # Paper layouts
    for layout in doc.layouts:
        if layout.name != "Model":
            walk_layout(layout, bag)
    # Blocks
    walk_blocks(doc, bag)
    return bag


def build_frequency(bag: Iterable[Tuple[str, str, str]]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for _, _, txt in bag:
        key = (txt or "").strip()
        if key:
            freq[key] = freq.get(key, 0) + 1
    return freq


def sort_frequency(freq: Dict[str, int]) -> List[Tuple[str, int]]:
    return sorted(freq.items(), key=lambda x: (-x[1], x[0]))


def write_csv(freq: Dict[str, int], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["count", "text_en"])
        for text, count in sort_frequency(freq):
            w.writerow([count, text])


def write_json(freq: Dict[str, int], path: Path) -> None:
    items = [{"text_en": text, "count": count} for text, count in sort_frequency(freq)]
    with path.open("w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)


def write_txt(freq: Dict[str, int], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for text, count in sort_frequency(freq):
            f.write(f"[{count}] {text}\n")


def extract_text_counts(doc) -> Dict[str, int]:
    return build_frequency(collect_text_bag(doc))


def extract_texts(inp: str) -> Dict[str, int]:
    doc = ezdxf.readfile(inp)
    return extract_text_counts(doc)


def main(inp, out_csv="extracted_texts.csv", out_json="extracted_texts.json", out_txt="extracted_texts.txt"):
    doc = ezdxf.readfile(inp)
    freq = extract_text_counts(doc)

    write_csv(freq, Path(out_csv))
    write_json(freq, Path(out_json))
    write_txt(freq, Path(out_txt))

    print(f"OK: {out_csv}, {out_json}, {out_txt}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_dxf_texts.py input.dxf [out_csv] [out_json] [out_txt]")
        sys.exit(1)

    inp = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "extracted_texts.csv"
    out_json = sys.argv[3] if len(sys.argv) > 3 else "extracted_texts.json"
    out_txt = sys.argv[4] if len(sys.argv) > 4 else "extracted_texts.txt"
    main(inp, out_csv, out_json, out_txt)

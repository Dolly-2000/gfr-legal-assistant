"""
GFR PDF Parser with Docling + Post-Processing
===============================================
Parses the 2025 GFR (General Financial Rules) document using Docling,
then post-processes the output to fix two-column layout issues.

Issues fixed:
  1. Rule labels (Rule 3, 4, 5, 6...) separated from their content
     due to two-column layout
  2. Two-column text misdetected as tables (Tables 0-11 in Docling output)
  3. Proper reading order: left column top→bottom, then right column top→bottom
  4. Clean structured markdown output suitable for RAG chunking
"""

import json
import re
import time
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ─── PATH CONFIGURATION ───
BASE_DIR = Path(__file__).parent.parent
INPUT_PDF_PATH = BASE_DIR / "data" / "raw_pdfs" / "2025_GFR.pdf"
OUTPUT_DIR = BASE_DIR / "data" / "parsed"
JSON_PATH = OUTPUT_DIR / "2025_GFR.json"
OUTPUT_MD_PATH = OUTPUT_DIR / "2025_GFR_clean.md"


# ─── DATA CLASSES ───
@dataclass
class TextElement:
    """A text element from the Docling JSON with position info."""
    text: str
    label: str  # text, list_item, section_header, caption, page_header, page_footer
    page: int
    left: float
    top: float
    right: float
    bottom: float
    idx: int = 0  # original index in texts[]

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2


# ─── STEP 1: RUN DOCLING (if JSON doesn't exist) ───
def run_docling_conversion() -> None:
    """Run Docling to convert the PDF to JSON + Markdown."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        AcceleratorOptions,
        TableFormerMode,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Docling] Processing device: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Docling] GPU: {gpu_name}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    # GPU acceleration
    accel_device = "cuda" if device.type == "cuda" else "cpu"
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=accel_device,
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"[Docling] Converting: {INPUT_PDF_PATH.name} ...")
    start = time.time()
    conv_result = doc_converter.convert(INPUT_PDF_PATH)
    elapsed = time.time() - start
    print(f"[Docling] Conversion done in {elapsed:.1f}s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw Markdown
    stem = INPUT_PDF_PATH.stem
    md_path = OUTPUT_DIR / f"{stem}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(conv_result.document.export_to_markdown())

    # Save JSON (with positional data)
    json_path = OUTPUT_DIR / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(conv_result.document.model_dump_json())

    print(f"[Docling] Saved: {md_path}")
    print(f"[Docling] Saved: {json_path}")


# ─── STEP 2: LOAD & PARSE DOCLING JSON ───
def load_docling_json(json_path: Path) -> dict:
    """Load the Docling JSON output."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_elements(doc: dict) -> list[TextElement]:
    """Extract all text elements with their positions from the JSON."""
    elements = []
    for i, t in enumerate(doc.get("texts", [])):
        prov = t.get("prov", [])
        if not prov:
            continue
        bbox = prov[0].get("bbox", {})
        elem = TextElement(
            text=t.get("text", "").strip(),
            label=t.get("label", "text"),
            page=prov[0].get("page_no", 0),
            left=bbox.get("l", 0),
            top=bbox.get("t", 0),
            right=bbox.get("r", 0),
            bottom=bbox.get("b", 0),
            idx=i,
        )
        if elem.text:  # skip empty
            elements.append(elem)
    return elements


def extract_table_cells(doc: dict) -> dict[int, list[dict]]:
    """
    Extract table cells with positions, grouped by table index.
    Returns dict: table_index -> list of cell dicts with text and position.
    """
    tables_data = {}
    for i, tbl in enumerate(doc.get("tables", [])):
        prov = tbl.get("prov", [])
        page = prov[0].get("page_no", 0) if prov else 0
        bbox = prov[0].get("bbox", {}) if prov else {}
        data = tbl.get("data", {})
        num_rows = data.get("num_rows", 0)
        num_cols = data.get("num_cols", 0)
        cells = data.get("table_cells", [])

        tables_data[i] = {
            "page": page,
            "bbox": bbox,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "cells": cells,
        }
    return tables_data


# ─── STEP 3: IDENTIFY FAKE vs REAL TABLES ───

def is_fake_table(table_info: dict) -> bool:
    """
    Determine if a Docling table is actually a two-column text layout
    (fake table) vs a real data table.

    Heuristics:
    - Fake tables have cells starting with "Rule X" pattern
    - Fake tables typically have 2 or 4 columns (left-col rule + right-col rule)
    - Real tables have structured headers (SI. No., Date, etc.)
    """
    cells = table_info["cells"]
    num_cols = table_info["num_cols"]

    if not cells:
        return False

    # Check if any cells start with "Rule" pattern
    rule_pattern = re.compile(r"^Rule\s+\d+")
    cells_with_rule = sum(1 for c in cells if rule_pattern.match(c.get("text", "")))

    # If >20% of cells start with "Rule X", it's likely a fake table
    if cells_with_rule > 0 and cells_with_rule / len(cells) > 0.1:
        return True

    # Check for structured table headers
    header_keywords = {
        "SI. No.", "Sl. No.", "Date", "Description", "Amount",
        "Object Head", "Budget", "Particulars", "Serial",
        "Period", "Month", "Year", "DSCR", "Category",
        "Grant", "Item No.", "Case No.", "ACCOUNTS",
    }
    first_row_texts = [
        c.get("text", "") for c in cells
        if c.get("start_row_offset_idx", 0) == 0
    ]
    for txt in first_row_texts:
        for kw in header_keywords:
            if kw.lower() in txt.lower():
                return False  # Has real table headers

    # Two-column or four-column with long text cells → likely fake
    if num_cols in (2, 4):
        avg_text_len = sum(len(c.get("text", "")) for c in cells) / max(len(cells), 1)
        if avg_text_len > 80:
            return True

    return False


# ─── STEP 4: EXTRACT TEXT FROM FAKE TABLES ───

def extract_fake_table_as_text(table_info: dict) -> list[TextElement]:
    """
    Convert a fake table (misdetected two-column layout) into text elements.
    Recombines split sentences intelligently.
    """
    cells = table_info["cells"]
    page = table_info["page"]
    num_cols = table_info["num_cols"]
    num_rows = table_info.get("num_rows", max((c.get("start_row_offset_idx", 0) for c in cells), default=0) + 1)
    page_height = 792.0

    def process_column_pair(label_col, text_col):
        labels = {}
        texts = {}
        left_b, right_b, top_b, bot_b = None, None, None, None
        
        for c in cells:
            r = c.get("start_row_offset_idx", 0)
            col = c.get("start_col_offset_idx", 0)
            if col not in (label_col, text_col): continue
                
            cb = c.get("bbox", {})
            t_val, b_val = cb.get("t", 0), cb.get("b", 0)
            l_val, r_val = cb.get("l", 0), cb.get("r", 0)
            
            if left_b is None or l_val < left_b: left_b = l_val
            if right_b is None or r_val > right_b: right_b = r_val
            if top_b is None or t_val < top_b: top_b = t_val
            if bot_b is None or b_val > bot_b: bot_b = b_val
                
            if col == label_col:
                if t := c.get("text", "").strip(): labels[r] = t
            elif col == text_col:
                if t := c.get("text", "").strip(): texts[r] = t
                
        merged = []
        import re
        for r in range(num_rows):
            lbl = labels.get(r, "")
            txt = texts.get(r, "")
            if lbl:
                if not merged:
                    merged.append(lbl + " " + txt)
                else:
                    last_txt = merged[-1]
                    if re.search(r'[\.\?\!]\s*$', last_txt) or re.search(r'
', last_txt):
                        merged.append("
" + lbl + " " + txt)
                    else:
                        match = re.search(r'[\.\?\!]\s+(?=[A-Z0-9(])', txt)
                        if match:
                            split_idx = match.end()
                            before = txt[:split_idx].strip()
                            after = txt[split_idx:].strip()
                            merged[-1] = merged[-1] + " " + before
                            merged.append("
" + lbl + " " + after)
                        else:
                            merged.append("
" + lbl + " " + txt)
            else:
                if txt: merged.append(txt)
                    
        content = " ".join(merged).replace(" 
", "
").replace("
 ", "
").strip()
        if not content: return None

        return TextElement(
            title="",
            content=content,
            page_numbers=[page],
            bboxes=[{"l": left_b, "r": right_b, "t": top_b, "b": bot_b}]
        )

    elements = []
    if num_cols >= 2:
        if el := process_column_pair(0, 1): elements.append(el)
    if num_cols >= 4:
        if el := process_column_pair(2, 3): elements.append(el)
        
    return elements
# ─── STEP 5: FORMAT REAL TABLES AS MARKDOWN ───

def format_real_table_as_markdown(table_info: dict) -> str:
    """Convert a real table to clean markdown table format."""
    cells = table_info["cells"]
    num_rows = table_info["num_rows"]
    num_cols = table_info["num_cols"]

    if not cells or num_rows == 0 or num_cols == 0:
        return ""

    # Build grid
    grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for c in cells:
        row = c.get("start_row_offset_idx", 0)
        col = c.get("start_col_offset_idx", 0)
        text = c.get("text", "").strip().replace("
", " ")
        if 0 <= row < num_rows and 0 <= col < num_cols:
            grid[row][col] = text

    # Build markdown table
    lines = []
    for r_idx, row in enumerate(grid):
        line = "| " + " | ".join(cell if cell else " " for cell in row) + " |"
        lines.append(line)
        if r_idx == 0:
            # Header separator
            sep = "| " + " | ".join("---" for _ in row) + " |"
            lines.append(sep)

    return "
".join(lines)


# ─── STEP 6: COLUMN-AWARE READING ORDER ───

RULE_LABEL_PATTERN = re.compile(r"^(?:Rule\s*\d+\s*)+(?:\(\d+\))?$", re.IGNORECASE)

Y_TOLERANCE = 15  # Points tolerance for "same vertical position" matching


def reorder_page_elements(
    elements: list[TextElement], page_width: float = 612.0
) -> list[TextElement]:
    """
    Reorder page elements using column-aware reading:
    1. Split into left/right columns by x-position
    2. Within EACH column, match rule labels with content at same y-position
    3. Also match rule labels across columns at same y-position
    4. Output: left column top→bottom, then right column top→bottom
    """
    mid_x = page_width / 2.0  # ~306

    # Filter out headers/footers
    content = [e for e in elements if e.label not in ("page_header", "page_footer")]
    if not content:
        return []

    left_col = [e for e in content if e.center_x < mid_x]
    right_col = [e for e in content if e.center_x >= mid_x]

    # Sort each column top→bottom (higher top value = higher on page in PDF coords)
    left_col.sort(key=lambda e: -e.top)
    right_col.sort(key=lambda e: -e.top)

    # --- Merge rule labels with content at same y-position ---
    # This works within a column (e.g., both Rule label and content in right col)
    # and across columns (Rule label in left col, content in right col)
    def merge_by_y_position(elems: list[TextElement]) -> list[TextElement]:
        """Within a list of elements, merge Rule labels with same-y content.
        Process rule labels FIRST to claim matches before non-labels consume them."""
        result = []
        used = set()

        # Phase 1: Process all rule labels first to claim their y-matches
        for i, e in enumerate(elems):
            if not RULE_LABEL_PATTERN.match(e.text):
                continue
            best_match = None
            best_idx = None
            for j, other in enumerate(elems):
                if j == i or j in used:
                    continue
                if RULE_LABEL_PATTERN.match(other.text):
                    continue  # skip other rule labels
                if abs(e.top - other.top) <= Y_TOLERANCE:
                    # Prefer element to the right of the label
                    if other.left > e.left:
                        best_match = other
                        best_idx = j
                        break
                    elif best_match is None:
                        best_match = other
                        best_idx = j

            if best_match is not None:
                merged = TextElement(
                    text=f"{e.text} {best_match.text}",
                    label="text",
                    page=e.page,
                    left=e.left,
                    top=e.top,
                    right=best_match.right,
                    bottom=min(e.bottom, best_match.bottom),
                )
                result.append(merged)
                used.add(i)
                used.add(best_idx)
            else:
                result.append(e)
                used.add(i)

        # Phase 2: Add remaining non-rule-label elements
        for i, e in enumerate(elems):
            if i in used:
                continue
            result.append(e)
            used.add(i)

        # Re-sort after merging
        result.sort(key=lambda e: -e.top)
        return result

    # Also try cross-column matching: rule labels in left col with content in right col
    all_page = left_col + right_col
    all_page.sort(key=lambda e: -e.top)
    merged_all = merge_by_y_position(all_page)

    # Now re-split into columns and output left-first, right-second
    left_final = [e for e in merged_all if e.center_x < mid_x]
    right_final = [e for e in merged_all if e.center_x >= mid_x]
    left_final.sort(key=lambda e: -e.top)
    right_final.sort(key=lambda e: -e.top)

    return left_final + right_final


def merge_rule_labels(elements: list[TextElement]) -> list[TextElement]:
    """
    Secondary pass: merge any remaining standalone rule labels
    (e.g., 'Rule 3') with their immediately following content element.
    This handles cases not caught by the y-position matching.
    """
    if not elements:
        return elements

    merged = []
    i = 0
    while i < len(elements):
        elem = elements[i]

        if RULE_LABEL_PATTERN.match(elem.text):
            # Look for the next non-label element to merge with
            if i + 1 < len(elements):
                next_elem = elements[i + 1]
                if not RULE_LABEL_PATTERN.match(next_elem.text):
                    merged_elem = TextElement(
                        text=f"{elem.text} {next_elem.text}",
                        label=next_elem.label,
                        page=elem.page,
                        left=elem.left,
                        top=elem.top,
                        right=next_elem.right,
                        bottom=next_elem.bottom,
                    )
                    merged.append(merged_elem)
                    i += 2
                    continue
            merged.append(elem)
            i += 1
        else:
            merged.append(elem)
            i += 1

    return merged


# ─── STEP 8: BUILD CLEAN MARKDOWN ───

def clean_text(text: str) -> str:
    """Clean up common artifacts in extracted text."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Normalize 'Rule46' -> 'Rule 46', 'Rule 15 Rule 15' -> 'Rule 15' (at the start of a string)
    text = re.sub(r"^(?:Rule\s*(\d+)\s*)+(?:\(\d+\))?\s*", r"Rule \1 ", text, flags=re.IGNORECASE)
    text = text.strip()
    # Fix common OCR artifacts
    text = text.replace("<!-- image -->", "")
    return text


def build_clean_markdown(
    doc: dict,
    text_elements: list[TextElement],
    tables_data: dict[int, dict],
    page_sizes: dict[int, tuple[float, float]],
) -> str:
    """
    Build a clean, well-structured markdown from all elements.
    Processes page by page with proper column reading order.
    """
    # Group text elements by page
    pages_text: dict[int, list[TextElement]] = {}
    for elem in text_elements:
        if elem.page not in pages_text:
            pages_text[elem.page] = []
        pages_text[elem.page].append(elem)

    # Group tables by page
    pages_tables: dict[int, list[tuple[int, dict]]] = {}
    for tbl_idx, tbl_info in tables_data.items():
        pg = tbl_info["page"]
        if pg not in pages_tables:
            pages_tables[pg] = []
        pages_tables[pg].append((tbl_idx, tbl_info))

    # Process fake tables → extract as text elements
    for pg, tbl_list in pages_tables.items():
        for tbl_idx, tbl_info in tbl_list:
            if is_fake_table(tbl_info):
                fake_elements = extract_fake_table_as_text(tbl_info)
                if pg not in pages_text:
                    pages_text[pg] = []
                pages_text[pg].extend(fake_elements)

    # Determine all pages
    all_pages = sorted(set(list(pages_text.keys()) + list(pages_tables.keys())))

    output_lines = []
    output_lines.append("# GENERAL FINANCIAL RULES 2017")
    output_lines.append("**Updated up to 31.07.2025**
")

    for pg in all_pages:
        page_elems = pages_text.get(pg, [])
        page_width = page_sizes.get(pg, (612.0, 792.0))[0]

        # Column-aware reordering with y-position rule-label matching
        ordered = reorder_page_elements(page_elems, page_width)

        # Secondary pass: merge any remaining standalone rule labels
        ordered = merge_rule_labels(ordered)

        # Emit text
        for elem in ordered:
            text = clean_text(elem.text)
            if not text:
                continue

            if elem.label == "section_header":
                # Determine heading level
                if text.startswith("Ch."):
                    output_lines.append(f"
## {text}
")
                elif text.startswith("APPENDIX"):
                    output_lines.append(f"
## {text}
")
                elif text.startswith("FORM GFR"):
                    output_lines.append(f"
## {text}
")
                else:
                    output_lines.append(f"
### {text}
")
            elif elem.label == "list_item":
                output_lines.append(f"- {text}")
            elif elem.label == "caption":
                output_lines.append(f"*{text}*
")
            elif elem.label in ("page_header", "page_footer"):
                continue  # Skip headers/footers
            else:
                output_lines.append(f"{text}
")

        # Emit real tables for this page
        if pg in pages_tables:
            for tbl_idx, tbl_info in pages_tables[pg]:
                if not is_fake_table(tbl_info):
                    md_table = format_real_table_as_markdown(tbl_info)
                    if md_table:
                        output_lines.append(f"
{md_table}
")

    return "
".join(output_lines)


# ─── STEP 9: ADDITIONAL POST-PROCESSING ───

def post_process_markdown(md_text: str) -> str:
    """Final cleanup pass on the generated markdown."""

    # Fix broken "Rule X" lines that got separated
    # Pattern: line ends with "Rule N" and next line has content
    md_text = re.sub(
        r"(Rule\s+\d+(?:\s*\(\d+\))?)

([A-Z])",
        r"\1 \2",
        md_text,
    )

    # Remove excessive blank lines (more than 2 consecutive)
    md_text = re.sub(r"
{4,}", "


", md_text)

    # Fix "life;" artifact from page-break issues
    md_text = md_text.replace("
### life;
", "
")

    # Clean up double spaces
    md_text = re.sub(r"  +", " ", md_text)

    return md_text


# ─── MAIN ───

def main():
    print("=" * 60)
    print("GFR PDF Parser (Docling + Post-Processing)")
    print("=" * 60)

    # Step 1: Run Docling if JSON doesn't exist
    if not JSON_PATH.exists():
        if not INPUT_PDF_PATH.exists():
            print(f"Error: PDF not found at {INPUT_PDF_PATH}")
            return
        print("
[Step 1] Running Docling conversion...")
        run_docling_conversion()
    else:
        print(f"
[Step 1] Using existing Docling JSON: {JSON_PATH}")

    # Step 2: Load JSON
    print("[Step 2] Loading Docling JSON...")
    doc = load_docling_json(JSON_PATH)
    print(f"  - {len(doc.get('texts', []))} text elements")
    print(f"  - {len(doc.get('tables', []))} tables")
    print(f"  - {len(doc.get('pages', {}))} pages")

    # Step 3: Extract all text elements with positions
    print("[Step 3] Extracting text elements...")
    text_elements = extract_text_elements(doc)
    print(f"  - {len(text_elements)} text elements with positions")

    # Step 4: Extract tables
    print("[Step 4] Analyzing tables...")
    tables_data = extract_table_cells(doc)
    fake_count = sum(1 for t in tables_data.values() if is_fake_table(t))
    real_count = len(tables_data) - fake_count
    print(f"  - {fake_count} fake tables (two-column text)")
    print(f"  - {real_count} real tables (data/forms)")

    # Step 5: Get page sizes
    page_sizes = {}
    for pg_str, pg_data in doc.get("pages", {}).items():
        sz = pg_data.get("size", {})
        page_sizes[int(pg_str)] = (sz.get("width", 612.0), sz.get("height", 792.0))

    # Step 6: Build clean markdown
    print("[Step 5] Building clean markdown...")
    md_text = build_clean_markdown(doc, text_elements, tables_data, page_sizes)

    # Step 7: Post-process
    print("[Step 6] Post-processing...")
    md_text = post_process_markdown(md_text)

    # Step 8: Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"
[Done] Clean markdown saved to: {OUTPUT_MD_PATH}")
    print(f"  - Total characters: {len(md_text):,}")
    print(f"  - Total lines: {md_text.count(chr(10)):,}")

    # Quick validation: check that Rules 3-6 have content
    print("
[Validation] Checking Rules 3-6...")
    for rule_num in [3, 4, 5, 6]:
        pattern = re.compile(rf"Rule\s+{rule_num}\b(.{{0,200}})", re.DOTALL)
        match = pattern.search(md_text)
        if match:
            snippet = match.group(0)[:120].replace("
", " ")
            has_content = len(match.group(1).strip()) > 5
            status = "OK" if has_content else "EMPTY"
            print(f"  Rule {rule_num}: [{status}] {snippet}")
        else:
            print(f"  Rule {rule_num}: [MISSING]")

    # Show first few rules as preview
    print("
[Preview] First 3000 characters of clean output:")
    print("-" * 60)
    print(md_text[:3000])


if __name__ == "__main__":
    main()

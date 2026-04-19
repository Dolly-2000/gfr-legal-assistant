"""
GFR Chunker v2 - Two-Zone Parsing
==================================
Zone 1 (Rule Definitions): Lines before the first ## APPENDIX header
Zone 2 (Appendices & Forms): Lines from ## APPENDIX onwards

Zone 1 uses rule-by-rule extraction with smart cross-reference detection.
Zone 2 uses section-based parsing (## APPENDIX / ## FORM GFR boundaries)
with rule attribution via [See Rule N] references.
"""

import re
import json
from pathlib import Path
from collections import defaultdict

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("Please install langchain-text-splitters")
    exit(1)

MD_PATH = Path("data/parsed/2025_GFR_clean.md")
OUTPUT_PATH = Path("data/parsed/2025_GFR_chunks.json")


# ─── CROSS-REFERENCE DETECTION ───

def is_cross_reference(title_text, line):
    """Determine if a 'Rule N ...' line is a cross-reference vs a new rule definition."""
    stripped = line.strip()
    
    if re.match(r'^\[?\(?\s*[Ss]ee\s', stripped):
        return True
    # Lines like "Rule 52]" or "Rule 286(1)]" - closing bracket references
    # Only match when the ] or ) is at/near the END of the line (not "Rule 50 (1) Content...")
    if re.match(r'^-?\s*Rule\s+\d+[A-Z]?\s*[\(\[]?\d*[\)\]]?\s*[\]\)]\s*\d*\s*$', stripped):
        return True
    if len(stripped) < 40 and (stripped.endswith(']') or stripped.endswith(')')):
        return True
    # Comma right after rule number: "Rule 30, a sanction..."
    if title_text and title_text.lstrip().startswith(','):
        return True
    # Parenthetical sub-ref followed by short text or conjunction: "Rule 133(2) or", "Rule 228(a): -"
    paren_match = re.match(r'^\s*\(\w+\)\s*(.*)', title_text) if title_text else None
    if paren_match:
        after_paren = paren_match.group(1).strip()
        if not after_paren or len(after_paren) < 20:
            return True
        if after_paren[0] in ',;:' or re.match(r'^(or|and|of|to|above|below|the|shall|for|in|with)\b', after_paren, re.IGNORECASE):
            return True
    if title_text and re.match(r'^(above|to|and|or|of|shall|for|the|in|with|has|below|may|would)\b', title_text):
        return True
    if title_text and re.match(r'^and\s+\d+', title_text):
        return True
    if title_text and len(title_text) < 60 and re.search(r'[\]\)]\s*$', title_text):
        return True
    if title_text and re.match(r'^contained\s+in', title_text):
        return True
    if title_text and title_text[0].islower() and len(title_text) < 80:
        return True
    return False


# ─── ZONE 1: RULE DEFINITIONS ───

def parse_rule_zone(lines):
    """Extract rule definitions from the main body (before appendices)."""
    current_chapter = "Preamble/Introduction"
    current_rule_num = None
    current_rule_title = ""
    current_text = []
    
    rules_dict = defaultdict(lambda: {"chapter": "", "title": "", "content_parts": []})
    
    chapter_pattern = re.compile(r'^##\s*(Ch\..*)')
    rule_pattern = re.compile(r'^-?\s*(?:Ru1e|Rule)\s+([0-9]+[A-Z]?)(.*)')

    def save_current_rule():
        nonlocal current_rule_num, current_rule_title
        if current_rule_num and current_text:
            content = "\n".join(current_text).strip()
            if content:
                entry = rules_dict[current_rule_num]
                if not entry["chapter"]:
                    entry["chapter"] = current_chapter
                if len(current_rule_title) > len(entry["title"]):
                    entry["title"] = current_rule_title.strip()
                entry["content_parts"].append(content)
            current_text.clear()

    for line in lines:
        ch_match = chapter_pattern.match(line)
        if ch_match:
            current_chapter = ch_match.group(1).strip()
            continue
        
        rule_match = rule_pattern.match(line)
        if rule_match:
            rule_num = rule_match.group(1).strip()
            title_text = rule_match.group(2).lstrip()
            
            if is_cross_reference(title_text, line):
                if current_rule_num is not None:
                    current_text.append(line)
                continue
                
            save_current_rule()
            current_rule_num = rule_num
            
            t_text = rule_match.group(2).strip()
            if t_text.startswith("-"):
                t_text = t_text[1:].strip()
            t_text = re.sub(r'^\(\d+\)\s*', '', t_text)
            current_rule_title = t_text
            current_text.append(line)
        elif current_rule_num is not None:
            current_text.append(line)

    save_current_rule()
    return rules_dict


# ─── ZONE 2: APPENDIX & FORM SECTIONS ───

def find_rule_reference(text_block):
    """
    Find a Rule N reference in a text block.
    Handles patterns like:
      [See Rule 52]
      [See Rule 61 and Rule 69]
      ## FORM GFR 16 [see Rule 286 (1)]
      [Rule 66]
      [(See Rule 239)]
      [See Note below Rule 52]
    Returns the last rule number found (most specific).
    """
    # Find all Rule N patterns in the block
    refs = re.findall(r'Rule\s*(\d+)', text_block)
    if refs:
        # Filter to valid rule numbers (1-324)
        valid = [r for r in refs if 1 <= int(r) <= 324]
        if valid:
            return valid[-1]  # Return last (most specific) reference
    return None


def parse_appendix_zone(lines):
    """
    Extract appendix/form sections using ## headers as boundaries.
    Attribute content to rules via [See Rule N] references.
    """
    sections = []
    
    # Split into sections by ## headers (APPENDIX / FORM GFR)
    section_starts = []
    for i, line in enumerate(lines):
        if line.startswith('## '):
            section_starts.append(i)
    
    # Also split on [See / (See lines that precede ### headers
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'^\[?\(?\s*[Ss]ee\b', stripped) or re.match(r'^\[?\s*Rule\s+\d+', stripped):
            # Check if this starts a new appendix section (has ### within next 5 lines)
            for j in range(i+1, min(i+6, len(lines))):
                if lines[j].strip().startswith('###'):
                    if i not in section_starts:
                        section_starts.append(i)
                    break
    
    section_starts.sort()
    
    if not section_starts:
        return []
    
    # Build sections
    for idx, start in enumerate(section_starts):
        end = section_starts[idx + 1] if idx + 1 < len(section_starts) else len(lines)
        section_lines = lines[start:end]
        section_text = "\n".join(section_lines).strip()
        
        if len(section_text) < 50:
            continue
        
        # Find the rule reference in the first few lines (header area)
        header_text = "\n".join(section_lines[:6])
        rule_ref = find_rule_reference(header_text)
        
        # Extract a title from ### headers
        title = None
        for sl in section_lines:
            sl_stripped = sl.strip()
            if sl_stripped.startswith('###'):
                title = sl_stripped.lstrip('#').strip()
                break
            if sl_stripped.startswith('## FORM'):
                title = sl_stripped.lstrip('#').strip()
                break
            if sl_stripped.startswith('## APPENDIX'):
                title = sl_stripped.lstrip('#').strip()
                break
        
        if not title:
            title = section_lines[0].strip()[:80]
        
        sections.append({
            "rule_number": rule_ref,
            "title": title,
            "content": section_text,
        })
    
    return sections


# ─── MAIN ───

def main():
    if not MD_PATH.exists():
        print(f"File not found: {MD_PATH}")
        return
        
    text = MD_PATH.read_text(encoding='utf-8')
    lines = text.split('\n')
    
    # Find the boundary between Zone 1 (rules) and Zone 2 (appendices)
    appendix_start = len(lines)
    for i, line in enumerate(lines):
        if re.match(r'^##\s+APPENDIX', line):
            appendix_start = i
            break
    
    print(f"[Zone Split] Rule zone: lines 0-{appendix_start-1} | Appendix zone: lines {appendix_start}-{len(lines)-1}")
    
    zone1_lines = lines[:appendix_start]
    zone2_lines = lines[appendix_start:]
    
    # ─── Pass 1: Rule Definitions ───
    print("[Pass 1] Extracting rule definitions from main body...")
    rules_dict = parse_rule_zone(zone1_lines)
    
    # Merge content parts for multi-part rules
    rules = []
    for rule_num in sorted(rules_dict.keys(), key=lambda x: int(x)):
        entry = rules_dict[rule_num]
        merged_content = "\n\n".join(entry["content_parts"])
        merged_content = re.sub(r'\n{3,}', '\n\n', merged_content).strip()
        rules.append({
            "rule_number": rule_num,
            "chapter": entry["chapter"],
            "title": entry["title"] if entry["title"] else f"Rule {rule_num}",
            "content": merged_content
        })
    
    rule_nums = sorted(set(int(r["rule_number"]) for r in rules))
    missing = sorted(set(range(1, 325)) - set(rule_nums))
    print(f"  Found {len(rules)} unique rules | Missing: {missing if missing else 'NONE'}")
    
    # ─── Pass 2: Appendix & Form Content ───
    print("[Pass 2] Extracting appendix and form content...")
    appendix_sections = parse_appendix_zone(zone2_lines)
    print(f"  Found {len(appendix_sections)} appendix/form sections")
    
    # Merge appendix content into existing rules or create standalone chunks
    rule_map = {r["rule_number"]: r for r in rules}
    merged_count = 0
    standalone = []
    
    for section in appendix_sections:
        rn = section["rule_number"]
        if rn and rn in rule_map:
            rule_map[rn]["content"] += f"\n\n--- {section['title']} ---\n{section['content']}"
            merged_count += 1
        else:
            standalone.append(section)
    
    print(f"  Merged {merged_count} sections into existing rules")
    print(f"  Standalone sections (no rule match): {len(standalone)}")
    
    # ─── Build Final Chunks ───
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    final_chunks = []
    
    # Chunks from rules
    for rule in rules:
        content = rule['content']
        token_est = len(content) // 4
        
        if token_est > 800:
            splits = text_splitter.split_text(content)
            for i, split_content in enumerate(splits):
                final_chunks.append({
                    "id": f"rule_{rule['rule_number']}_part{i+1}",
                    "rule_number": rule["rule_number"],
                    "chapter": rule["chapter"],
                    "title": rule["title"],
                    "content": split_content.strip(),
                    "token_estimate": len(split_content) // 4,
                    "metadata": {
                        "source": "2025_GFR_clean.md",
                        "type": "rule",
                        "is_sub_chunk": True,
                        "sub_chunk_index": i,
                        "total_sub_chunks": len(splits)
                    }
                })
        else:
            final_chunks.append({
                "id": f"rule_{rule['rule_number']}",
                "rule_number": rule["rule_number"],
                "chapter": rule["chapter"],
                "title": rule["title"],
                "content": content,
                "token_estimate": token_est,
                "metadata": {
                    "source": "2025_GFR_clean.md",
                    "type": "rule",
                    "is_sub_chunk": False,
                    "sub_chunk_index": None,
                    "total_sub_chunks": None
                }
            })
    
    # Chunks from standalone appendix/form sections
    for idx, section in enumerate(standalone):
        content = section['content']
        token_est = len(content) // 4
        rn = section.get('rule_number', 'unknown') or 'appendix'
        chunk_id = f"appendix_{idx+1}_{rn}"
        
        if token_est > 800:
            splits = text_splitter.split_text(content)
            for i, split_content in enumerate(splits):
                final_chunks.append({
                    "id": f"{chunk_id}_part{i+1}",
                    "rule_number": str(rn),
                    "chapter": "Appendix/Forms",
                    "title": section["title"],
                    "content": split_content.strip(),
                    "token_estimate": len(split_content) // 4,
                    "metadata": {
                        "source": "2025_GFR_clean.md",
                        "type": "appendix",
                        "is_sub_chunk": True,
                        "sub_chunk_index": i,
                        "total_sub_chunks": len(splits)
                    }
                })
        else:
            final_chunks.append({
                "id": chunk_id,
                "rule_number": str(rn),
                "chapter": "Appendix/Forms",
                "title": section["title"],
                "content": content,
                "token_estimate": token_est,
                "metadata": {
                    "source": "2025_GFR_clean.md",
                    "type": "appendix",
                    "is_sub_chunk": False,
                    "sub_chunk_index": None,
                    "total_sub_chunks": None
                }
            })
    
    OUTPUT_PATH.write_text(json.dumps(final_chunks, indent=2))
    
    # ─── Summary ───
    total_content = sum(len(c['content']) for c in final_chunks)
    rule_nums_final = sorted(set(int(c['rule_number']) for c in final_chunks if c['rule_number'].isdigit()))
    missing_final = sorted(set(range(1, 325)) - set(rule_nums_final))
    
    print(f"\n{'='*60}")
    print(f"CHUNKING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total chunks:       {len(final_chunks)}")
    print(f"  Unique rules:       {len(rule_nums_final)}/324")
    print(f"  Missing rules:      {missing_final if missing_final else 'NONE (100%)'}")
    print(f"  Total content:      {total_content:,} chars")
    print(f"  MD source:          {len(text):,} chars")
    print(f"  Capture ratio:      {total_content/len(text):.1%}")
    print(f"  Output:             {OUTPUT_PATH}")
    
    # Show top 10 largest rules
    rule_sizes = defaultdict(int)
    rule_chunk_counts = defaultdict(int)
    for c in final_chunks:
        rule_sizes[c['rule_number']] += len(c['content'])
        rule_chunk_counts[c['rule_number']] += 1
    
    print(f"\n  Top 10 largest rules by content:")
    for rn, size in sorted(rule_sizes.items(), key=lambda x: -x[1])[:10]:
        print(f"    Rule {rn:>4}: {size:>8,} chars ({rule_chunk_counts[rn]} chunks)")


if __name__ == '__main__':
    main()

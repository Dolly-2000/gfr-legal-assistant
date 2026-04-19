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


def is_cross_reference(rule_num_str, title_text, line, current_rule_num):
    """
    Determine if a 'Rule N ...' line is a cross-reference inside another rule's
    text vs an actual new rule definition.
    
    Cross-reference indicators:
      - Appears inside brackets: [See Rule 52], (Rule 140 above)
      - Followed by connectors: 'Rule 154 and 155', 'Rule 140 above by'
      - Very short trailing text ending with ], ), period
      - Text starts with lowercase AND is very short (< 80 chars)
    """
    stripped = line.strip()
    
    # Lines like "[See Rule 52]" or "(See Rule 69)" are references
    if re.match(r'^\[?\(?\s*[Ss]ee\s', stripped):
        return True
    
    # Lines like "Rule 52]" or "Rule 286(1)]" - closing bracket references
    if re.match(r'^-?\s*Rule\s+\d+[A-Z]?\s*[\(\[]?\d*[\)\]]?\s*\]', stripped):
        return True
    
    # Very short lines ending with ] or ) are references
    if len(stripped) < 40 and (stripped.endswith(']') or stripped.endswith(')')):
        return True
    
    # "Rule N above", "Rule N to", "Rule N and" are cross-references
    if title_text and re.match(r'^(above|to|and|or|of|shall|for|the|in|with|has|below)\b', title_text):
        return True
    
    # Lines like "Rule 154 and 155, Ministries..." - referencing multiple rules
    if title_text and re.match(r'^and\s+\d+', title_text):
        return True
    
    # Short text ending with cross-ref punctuation
    if title_text and len(title_text) < 60 and re.search(r'[\]\)]\s*$', title_text):
        return True
    
    # "Rule N contained in this Chapter" pattern
    if title_text and re.match(r'^contained\s+in', title_text):
        return True
    
    # If it starts with lowercase and total content is very short, likely a reference
    if title_text and title_text[0].islower() and len(title_text) < 80:
        return True
    
    return False


def extract_rules_from_markdown(lines):
    """
    Pass 1: Extract all rule definitions from the main body of the document.
    Merges multi-part rules (e.g. Rule 52(1), Rule 52(2)) into a single rule.
    """
    current_chapter = "Preamble/Introduction"
    current_rule_num = None
    current_rule_title = ""
    current_text = []
    
    # Use a dict to merge multi-part rules (Rule 52(1) + Rule 52(2) -> Rule 52)
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
                # Keep the longest/most descriptive title
                if len(current_rule_title) > len(entry["title"]):
                    entry["title"] = current_rule_title.strip()
                entry["content_parts"].append(content)
            current_text.clear()

    # Pattern to detect "[See" or "(See" on preceding line -> "Rule N]" on current line
    see_prefix_pattern = re.compile(r'^\[?\(?\s*[Ss]ee\s*$|^\[?\(?\s*[Ss]ee\s+Note', re.IGNORECASE)
    # Pattern for "[See Rule N]" or "(See Rule N)" all on one line
    see_rule_inline = re.compile(r'^\[?\(?\s*[Ss]ee\s+(?:Note\s+(?:below\s+)?)?Rule[s]?\s+(\d+)', re.IGNORECASE)
    # Pattern for "FORM GFR N [see Rule M]"
    form_rule_pattern = re.compile(r'FORM\s+GFR\s+\d+\s*\[?\s*[Ss]ee\s*$|FORM\s+GFR\s+\d+\s*\[?\s*[Ss]ee\s+Rule\s+(\d+)', re.IGNORECASE)
    
    prev_line = ""
    
    for line in lines:
        ch_match = chapter_pattern.match(line)
        if ch_match:
            current_chapter = ch_match.group(1).strip()
            prev_line = line
            continue
        
        rule_match = rule_pattern.match(line)
        if rule_match:
            rule_num = rule_match.group(1).strip()
            title_text = rule_match.group(2).lstrip()
            
            # Check if this is a "[See\nRule N]" appendix reference pattern
            # If the previous non-empty line was "[See" or "(See", this is an appendix
            # section reference -> redirect content to the referenced rule
            prev_stripped = prev_line.strip()
            is_see_ref = see_prefix_pattern.match(prev_stripped)
            is_bracket_close = re.match(r'^-?\s*Rule\s+\d+[A-Z]?\s*[\(\[]?\d*[\)\]]?\s*[\]\)]', line.strip())
            
            if is_see_ref and is_bracket_close:
                # This is an appendix section for Rule N
                # Save current rule, switch collection to the referenced rule
                save_current_rule()
                current_rule_num = rule_num
                current_rule_title = f"Appendix for Rule {rule_num}"
                current_text.append(line)
                prev_line = line
                continue
            
            # Determine if this is a cross-reference or a real rule definition
            if is_cross_reference(rule_num, title_text, line, current_rule_num):
                # It's a cross-reference: append to current rule's content
                if current_rule_num is not None:
                    current_text.append(line)
                prev_line = line
                continue
                
            # It's a real rule definition
            save_current_rule()
            current_rule_num = rule_num
            
            t_text = rule_match.group(2).strip()
            if t_text.startswith("-"):
                t_text = t_text[1:].strip()
            # Remove sub-section markers like (1), (2) from title
            t_text = re.sub(r'^\(\d+\)\s*', '', t_text)
            
            current_rule_title = t_text
            current_text.append(line)
        else:
            # Check for inline [See Rule N] on this line (full reference on one line)
            see_inline_match = see_rule_inline.match(line.strip())
            form_match = form_rule_pattern.search(line.strip())
            
            redirect_rule = None
            if see_inline_match:
                redirect_rule = see_inline_match.group(1)
            elif form_match and form_match.group(1):
                redirect_rule = form_match.group(1)
            
            if redirect_rule and int(redirect_rule) >= 1 and int(redirect_rule) <= 324:
                # Redirect subsequent content to this rule
                save_current_rule()
                current_rule_num = redirect_rule
                current_rule_title = f"Appendix for Rule {redirect_rule}"
                current_text.append(line)
            elif current_rule_num is not None:
                # Include ALL lines (even blank) to preserve paragraph structure
                current_text.append(line)
        
        prev_line = line if line.strip() else prev_line
               
    save_current_rule()
    return rules_dict


def extract_appendix_content(lines):
    """
    Pass 2: Extract appendix/form sections at the end of the document.
    These sections are referenced by [See Rule N] and contain supplementary
    content (procedures, forms, templates, etc.).
    """
    appendix_chunks = []
    
    # Find all [See Rule N] or [see Rule N] references followed by content
    see_rule_pattern = re.compile(r'\[?\(?\s*[Ss]ee\s+(?:Note\s+(?:below\s+)?)?Rule[s]?\s+(\d+)', re.IGNORECASE)
    form_rule_pattern = re.compile(r'FORM\s+GFR\s+\d+\s*\[?\s*[Ss]ee\s+Rule\s+(\d+)', re.IGNORECASE)
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for [See Rule N] reference patterns
        see_match = see_rule_pattern.search(line)
        form_match = form_rule_pattern.search(line)
        
        ref_rule = None
        if form_match:
            ref_rule = form_match.group(1)
        elif see_match and line.strip().startswith(('[', '(', '#')):
            ref_rule = see_match.group(1)
        
        if ref_rule:
            # Collect all content after this reference until the next [See Rule] or ## header
            section_lines = [line]
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                # Stop at the next [See Rule] reference or ## chapter header
                if see_rule_pattern.search(next_line) and next_line.strip().startswith(('[', '(', '#')):
                    break
                if form_rule_pattern.search(next_line):
                    break
                if next_line.startswith('## ') and not next_line.startswith('###'):
                    break
                section_lines.append(next_line)
                j += 1
            
            content = "\n".join(section_lines).strip()
            if len(content) > 100:  # Only keep meaningful appendix sections
                # Extract a title from ### headers in the section
                title = f"Appendix for Rule {ref_rule}"
                for sl in section_lines:
                    if sl.strip().startswith('###'):
                        title = sl.strip().lstrip('#').strip()
                        break
                
                appendix_chunks.append({
                    "rule_number": ref_rule,
                    "title": title,
                    "content": content,
                    "type": "appendix"
                })
            i = j
        else:
            i += 1
    
    return appendix_chunks


def main():
    if not MD_PATH.exists():
        print(f"File not found: {MD_PATH}")
        return
        
    text = MD_PATH.read_text(encoding='utf-8')
    lines = text.split('\n')
    
    # Pass 1: Extract rule definitions
    print("[Pass 1] Extracting rule definitions from main body...")
    rules_dict = extract_rules_from_markdown(lines)
    
    # Merge content parts for multi-part rules
    rules = []
    for rule_num in sorted(rules_dict.keys(), key=lambda x: int(x)):
        entry = rules_dict[rule_num]
        merged_content = "\n\n".join(entry["content_parts"])
        # Clean up excessive blank lines
        merged_content = re.sub(r'\n{3,}', '\n\n', merged_content).strip()
        rules.append({
            "rule_number": rule_num,
            "chapter": entry["chapter"],
            "title": entry["title"] if entry["title"] else f"Rule {rule_num}",
            "content": merged_content
        })
    
    print(f"  Found {len(rules)} unique rules")
    
    # Pass 2: Extract appendix/form content
    print("[Pass 2] Extracting appendix and form content...")
    appendix_sections = extract_appendix_content(lines)
    print(f"  Found {len(appendix_sections)} appendix sections")
    
    # Merge appendix content into rules where possible
    rule_map = {r["rule_number"]: r for r in rules}
    appended_count = 0
    standalone_appendices = []
    
    for appendix in appendix_sections:
        rn = appendix["rule_number"]
        if rn in rule_map:
            # Append to existing rule
            rule_map[rn]["content"] += f"\n\n--- Appendix: {appendix['title']} ---\n{appendix['content']}"
            appended_count += 1
        else:
            # Create standalone appendix chunk
            standalone_appendices.append(appendix)
    
    print(f"  Merged {appended_count} appendix sections into existing rules")
    if standalone_appendices:
        print(f"  {len(standalone_appendices)} standalone appendix sections (no matching rule)")
    
    # Build final chunks using text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    final_chunks = []
    
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

    # Add standalone appendix sections as their own chunks
    for appendix in standalone_appendices:
        content = appendix['content']
        token_est = len(content) // 4
        chunk_id = f"appendix_rule_{appendix['rule_number']}"
        
        if token_est > 800:
            splits = text_splitter.split_text(content)
            for i, split_content in enumerate(splits):
                final_chunks.append({
                    "id": f"{chunk_id}_part{i+1}",
                    "rule_number": appendix["rule_number"],
                    "chapter": "Appendix",
                    "title": appendix["title"],
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
                "rule_number": appendix["rule_number"],
                "chapter": "Appendix",
                "title": appendix["title"],
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
    
    # Summary
    total_content = sum(len(c['content']) for c in final_chunks)
    rule_nums = sorted(set(int(c['rule_number']) for c in final_chunks))
    missing = sorted(set(range(1, 325)) - set(rule_nums))
    
    print(f"\n{'='*60}")
    print(f"CHUNKING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total chunks:       {len(final_chunks)}")
    print(f"  Unique rules:       {len(rule_nums)}/324")
    print(f"  Missing rules:      {missing if missing else 'NONE (100%)'}")
    print(f"  Total content:      {total_content:,} chars")
    print(f"  MD source:          {len(text):,} chars")
    print(f"  Capture ratio:      {total_content/len(text):.1%}")
    print(f"  Output:             {OUTPUT_PATH}")
    
if __name__ == '__main__':
    main()

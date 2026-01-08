"""
Ultimate RAG Parser - "The Fusion Architecture"

Combines:
1. Object-Oriented Extraction (Table/Code separate extraction) -> High Precision
2. Text Stream Anchoring (Linking text to tables) -> Context Preservation
3. Multi-Granularity Chunking (Window + Fixed Sizes) -> High Recall

Logic Flow:
1. Extract Tables/Images per page -> Create Structured Chunks
2. Extract Raw Text -> Insert Anchors to extracted objects (e.g., [See Table: table_1])
3. Extract Code Blocks from text -> Create Code Chunks
4. Chunk the remaining Text Stream using Multi-Granularity strategy
"""

import json
import re
import hashlib
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import pdfplumber
import camelot


# ==========================================
# Data Structures
# ==========================================

@dataclass
class ChunkMetadata:
    page_numbers: List[int]
    source_type: str  # table, code, text_window, text_fixed
    related_objects: List[str] = None  # IDs of related tables/images
    
@dataclass
class ParsedChunk:
    chunk_id: str
    content: str
    metadata: ChunkMetadata

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": asdict(self.metadata)
        }

# ==========================================
# Specialized Extractors
# ==========================================

class TableExtractor:
    """Uses Camelot to extract tables with high precision"""
    def extract(self, pdf_path: str, page_num: int) -> List[Dict]:
        tables_found = []
        try:
            # 1. Try Lattice (lines)
            tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='lattice')
            if not tables:
                # 2. Fallback to Stream (whitespace)
                tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='stream')
            
            for i, table in enumerate(tables):
                # Filter noise (tiny tables)
                if table.df.shape[0] < 2 or table.df.shape[1] < 2:
                    continue
                
                # Convert to Markdown for LLM readability
                content = table.df.to_markdown(index=False)
                chunk_id = f"table_{page_num}_{i}"
                
                tables_found.append({
                    "chunk_id": chunk_id,
                    "content": content,
                    "rows": table.df.shape[0]
                })
        except Exception as e:
            # print(f"Table extraction silent error on page {page_num}: {e}")
            pass
        return tables_found

class CodeExtractor:
    """Extracts code blocks from text using Heuristics"""
    def __init__(self):
        # Improved Regex for common code patterns
        self.code_start_patterns = [
            r'^\s*(def|class|if|for|while|import|from|try|except)\s+', # Python
            r'^\s*(function|const|let|var|console\.)', # JS
            r'^\s*(public|private|protected|void|int|float|string)\s+', # Java/C++
            r'^\s*(SELECT|INSERT|UPDATE|DELETE|FROM)\s+', # SQL
        ]

    def extract_from_text(self, text: str) -> List[Dict]:
        """Returns list of code blocks and the CLEANED text (code removed)"""
        lines = text.split('\n')
        code_blocks = []
        clean_lines = []
        
        in_code_block = False
        current_block = []
        start_line_idx = 0
        
        for idx, line in enumerate(lines):
            # Detection logic: Is this line code?
            is_code_line = any(re.search(p, line) for p in self.code_start_patterns) or \
                           (in_code_block and (line.strip() == '' or line.startswith('    ') or line.startswith('\t'))) or \
                           (line.strip().endswith('{') or line.strip().endswith(';'))

            if is_code_line:
                if not in_code_block:
                    in_code_block = True
                    start_line_idx = idx
                current_block.append(line)
            else:
                if in_code_block:
                    # Block ended
                    if len(current_block) > 1: # Ignore single lines
                        code_content = "\n".join(current_block)
                        block_id = hashlib.md5(code_content.encode()).hexdigest()[:8]
                        code_blocks.append({
                            "content": code_content,
                            "id": f"code_{block_id}"
                        })
                        # Add placeholder in text
                        clean_lines.append(f"\n[Code Block: {block_id}]\n")
                    else:
                        # False alarm, append back to text
                        clean_lines.extend(current_block)
                    
                    current_block = []
                    in_code_block = False
                
                clean_lines.append(line)
        
        # Flush remaining
        if in_code_block and len(current_block) > 1:
            code_content = "\n".join(current_block)
            block_id = hashlib.md5(code_content.encode()).hexdigest()[:8]
            code_blocks.append({
                "content": code_content,
                "id": f"code_{block_id}"
            })
            clean_lines.append(f"\n[Code Block: {block_id}]\n")
        elif current_block:
            clean_lines.extend(current_block)

        return code_blocks, "\n".join(clean_lines)

# ==========================================
# Main Logic: Multi-Granularity Chunker
# ==========================================

class UltimateParser:
    def __init__(self):
        self.table_extractor = TableExtractor()
        self.code_extractor = CodeExtractor()

    def _chunk_text_multi_granularity(self, text: str, page_map: List[Dict]) -> List[ParsedChunk]:
        """
        Multi-Granularity Chunking Strategy:
        1. Sliding Window (800) - for narrative/context
        2. Fixed Sizes (256, 512, 1024) - for facts/definitions
        3. Cross-page windows - for concepts spanning multiple pages
        """
        chunks = []

        # Strategy 1: Sliding Window (Best for Context/Narrative)
        window_size = 800
        overlap = 200

        for i in range(0, len(text), window_size - overlap):
            chunk_text = text[i : i + window_size]
            if len(chunk_text) < 50: continue

            chunk_mid = i + len(chunk_text) // 2
            idx = min(chunk_mid, len(page_map)-1)
            meta_info = page_map[idx]

            # Get page range for this chunk
            start_idx = max(0, i)
            end_idx = min(len(page_map)-1, i + len(chunk_text))
            pages = sorted(set([page_map[j]['page'] for j in range(start_idx, end_idx+1)]))
            related = sorted(set([obj for j in range(start_idx, end_idx+1) for obj in page_map[j]['related_objects']]))

            chunks.append(ParsedChunk(
                chunk_id=f"text_window_800_{i}",
                content=chunk_text,
                metadata=ChunkMetadata(
                    page_numbers=pages,
                    source_type="text_sliding_window_800",
                    related_objects=related
                )
            ))

        # Strategy 2: Fixed Size Chunks (256, 512, 1024)
        for size in [256, 512, 1024]:
            for i in range(0, len(text), size):
                chunk_text = text[i : i + size]
                if len(chunk_text) < 30: continue

                idx = min(i + size // 2, len(page_map)-1)
                meta_info = page_map[idx]

                # Get page range
                start_idx = max(0, i)
                end_idx = min(len(page_map)-1, i + len(chunk_text))
                pages = sorted(set([page_map[j]['page'] for j in range(start_idx, end_idx+1)]))
                related = sorted(set([obj for j in range(start_idx, end_idx+1) for obj in page_map[j]['related_objects']]))

                chunks.append(ParsedChunk(
                    chunk_id=f"text_fixed_{size}_{i}",
                    content=chunk_text,
                    metadata=ChunkMetadata(
                        page_numbers=pages,
                        source_type=f"text_fixed_{size}",
                        related_objects=related
                    )
                ))

        return chunks

    def parse(self, pdf_path: str) -> List[Dict]:
        print(f"Starting Ultimate Parse for: {pdf_path}")
        final_chunks = []
        
        # --- Phase 1: Object Extraction (Tables) ---
        print("Phase 1: Extracting Structured Objects...")
        page_objects_map = defaultdict(list) # page_num -> list of object IDs
        
        # We need total pages first
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
        for p_num in range(1, total_pages + 1):
            # Extract Tables
            tables = self.table_extractor.extract(pdf_path, p_num)
            for t in tables:
                # Add Table Chunk
                final_chunks.append(ParsedChunk(
                    chunk_id=t['chunk_id'],
                    content=t['content'],
                    metadata=ChunkMetadata(
                        page_numbers=[p_num],
                        source_type="table",
                        related_objects=[]
                    )
                ))
                page_objects_map[p_num].append(t['chunk_id'])
                print(f"  - Found Table on Page {p_num}")

        # --- Phase 2: Text Stream Construction & Code Extraction ---
        print("Phase 2: Building Text Stream & Extracting Code...")
        full_text_stream = ""
        text_map = [] # Maps character index to metadata
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                raw_text = page.extract_text() or ""
                
                # A. Extract Code from this page's text
                code_blocks, clean_text = self.code_extractor.extract_from_text(raw_text)
                
                # Add Code Chunks
                for cb in code_blocks:
                    final_chunks.append(ParsedChunk(
                        chunk_id=cb['id'],
                        content=cb['content'],
                        metadata=ChunkMetadata(
                            page_numbers=[page_num],
                            source_type="code_block",
                            related_objects=[]
                        )
                    ))
                
                # B. Append Anchors for Tables found in Phase 1
                related_objs = page_objects_map[page_num]
                if related_objs:
                    # Add a visible marker in text for the LLM
                    anchor_text = f"\n\n[Reference: This section contains structured data in: {', '.join(related_objs)}]\n\n"
                    clean_text += anchor_text
                
                # C. Build Stream Mapping
                # We map every character in this page's text to this page number
                start_idx = len(full_text_stream)
                full_text_stream += clean_text + "\n\n"
                end_idx = len(full_text_stream)
                
                # Fill map
                for _ in range(start_idx, end_idx):
                    text_map.append({
                        "page": page_num,
                        "related_objects": related_objs
                    })

        # --- Phase 3: Multi-Granularity Chunking ---
        print("Phase 3: Chunking Text Stream...")
        text_chunks = self._chunk_text_multi_granularity(full_text_stream, text_map)
        
        # Merge all chunks
        all_parsed_chunks = [c.to_dict() for c in final_chunks] + [c.to_dict() for c in text_chunks]
        
        print(f"\nParse Complete!")
        print(f"Total Chunks: {len(all_parsed_chunks)}")
        
        # Print Stats
        stats = defaultdict(int)
        for c in all_parsed_chunks:
            stats[c['metadata']['source_type']] += 1
        print("Breakdown:", dict(stats))
        
        return all_parsed_chunks

# ==========================================
# Execution
# ==========================================

if __name__ == "__main__":
    parser = UltimateParser()
    chunks = parser.parse("slides/7215_slides.pdf")

    # Save to JSON
    output_path = "data/parsed/enhanced_chunks.json"
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_path}")
"""
Multi-Modal PDF Parser for Academic Slides

Combines three parsing strategies:
1. Camelot: Extract tables with structure preserved
2. pdfplumber: Extract text with visual layout information
3. ParseBlock: Sliding window for cross-page concepts

Target: Academic course slides with semi-structured content
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import PyPDF2
import pdfplumber
from tqdm import tqdm


@dataclass
class ParsedChunk:
    """A parsed chunk with metadata"""
    content: str
    chunk_id: str
    source_type: str  # 'table', 'text_block', 'sliding_window'
    page_numbers: List[int] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_document_string(self) -> str:
        """Convert to langchain Document format"""
        pages_str = ",".join(map(str, self.page_numbers))
        return f"[Pages {pages_str}] {self.content}"


class TableExtractor:
    """
    Extract tables using Camelot

    Camelot is chosen because:
    - Preserves table structure (rows/columns)
    - Handles complex tables better than pdfplumber
    - Outputs Markdown format for downstream processing
    """

    def __init__(self):
        try:
            import camelot
            self.camelot = camelot
            self.available = True
        except ImportError:
            print("Warning: camelot not installed, table extraction disabled")
            self.available = False

    def extract_tables(self, pdf_path: str, pages: str = "all") -> List[Dict]:
        """
        Extract tables from PDF

        Args:
            pdf_path: Path to PDF file
            pages: Pages to extract (e.g., "1-5,10" or "all")

        Returns:
            List of dict with 'content', 'page', 'table_data'
        """
        if not self.available:
            return []

        tables = []

        try:
            # Extract tables with lattice mode (for lines)
            lattice_tables = self.camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor="lattice",
                suppress_stdout=True
            )

            # Extract tables with stream mode (for no lines)
            stream_tables = self.camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor="stream",
                suppress_stdout=True
            )

            all_tables = list(lattice_tables) + list(stream_tables)

            for i, table in enumerate(all_tables):
                if table.df.empty:
                    continue

                # Get table page number
                page = table.page

                # Convert to Markdown format
                markdown = table.df.to_markdown(index=False)

                # Calculate accuracy (Camelot's built-in metric)
                accuracy = table.accuracy if hasattr(table, 'accuracy') else 0

                if accuracy > 50:  # Filter low-quality extractions
                    tables.append({
                        "content": markdown,
                        "page": page,
                        "table_data": table.df.to_dict(),
                        "accuracy": accuracy,
                        "extractor": "camelot"
                    })

        except Exception as e:
            print(f"Warning: Table extraction failed: {e}")

        return tables


class LayoutAwareParser:
    """
    Parse text using pdfplumber with visual layout awareness

    pdfplumber is chosen because:
    - Preserves spatial information (x, y coordinates)
    - Handles multi-column layouts
    - Better text extraction than PyPDF2
    """

    def __init__(self):
        pass

    def extract_text_blocks(self, pdf_path: str, min_text_length: int = 50) -> List[Dict]:
        """
        Extract text blocks with layout information

        Args:
            pdf_path: Path to PDF file
            min_text_length: Minimum text length to include

        Returns:
            List of dict with 'content', 'page', 'bbox', etc.
        """
        blocks = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc="Extracting text")):
                # Extract text with character-level detail
                text = page.extract_text()

                if not text or len(text.strip()) < min_text_length:
                    continue

                # Get page bounding box
                bbox = page.bbox

                blocks.append({
                    "content": text.strip(),
                    "page": page_num + 1,
                    "bbox": bbox,
                    "width": page.width,
                    "height": page.height,
                    "extractor": "pdfplumber"
                })

        return blocks


class ParseBlock:
    """
    Sliding window parser for cross-page concepts

    Strategy:
    1. Extract individual pages
    2. Detect content similarity between adjacent pages
    3. Create overlapping windows for related content
    4. Merge pages within windows into semantic chunks
    """

    def __init__(self, window_size: int = 3, overlap: int = 1):
        self.window_size = window_size
        self.overlap = overlap

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts using keyword overlap"""
        # Simple word overlap similarity
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0

    def detect_section_boundaries(self, pages: List[Dict]) -> List[int]:
        """
        Detect section boundaries based on content similarity

        A section boundary is where content similarity drops significantly
        """
        if len(pages) < 2:
            return [0]

        boundaries = [0]
        similarities = []

        # Compute pairwise similarities
        for i in range(len(pages) - 1):
            sim = self._compute_similarity(
                pages[i]["content"],
                pages[i + 1]["content"]
            )
            similarities.append(sim)

        # Find significant drops (more than 2 std below mean)
        if similarities:
            mean_sim = sum(similarities) / len(similarities)
            std_sim = (sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)) ** 0.5
            threshold = mean_sim - 1.5 * std_sim

            for i, sim in enumerate(similarities):
                if sim < threshold:
                    boundaries.append(i + 1)

        boundaries.append(len(pages))
        return sorted(set(boundaries))

    def create_sliding_windows(self, pages: List[Dict]) -> List[List[Dict]]:
        """
        Create overlapping sliding windows of pages

        Returns:
            List of page groups (each group is a list of page dicts)
        """
        if not pages:
            return []

        windows = []
        step = self.window_size - self.overlap

        for i in range(0, len(pages), step):
            window = pages[i:i + self.window_size]
            if window:
                windows.append(window)

        return windows

    def merge_window(self, pages: List[Dict], chunk_id: int) -> ParsedChunk:
        """Merge pages in a window into a single chunk"""
        content_parts = []
        page_numbers = []

        for page in pages:
            page_numbers.append(page["page"])
            content_parts.append(page["content"])

        # Merge content with page markers
        merged_content = "\n\n".join(content_parts)

        # Generate unique ID
        content_hash = hashlib.md5(merged_content.encode()).hexdigest()[:8]

        return ParsedChunk(
            content=merged_content,
            chunk_id=f"window_{chunk_id}_{content_hash}",
            source_type="sliding_window",
            page_numbers=page_numbers,
            metadata={
                "num_pages": len(pages),
                "avg_page_length": sum(len(p["content"]) for p in pages) // len(pages)
            }
        )

    def parse(self, text_blocks: List[Dict]) -> List[ParsedChunk]:
        """
        Parse text blocks using sliding window strategy

        Args:
            text_blocks: List of text blocks from pdfplumber

        Returns:
            List of ParsedChunk objects
        """
        if not text_blocks:
            return []

        chunks = []

        # Detect sections first
        boundaries = self.detect_section_boundaries(text_blocks)

        # Process each section
        chunk_idx = 0
        for i in range(len(boundaries) - 1):
            section_pages = text_blocks[boundaries[i]:boundaries[i + 1]]

            # If section is small, just merge it
            if len(section_pages) <= self.window_size:
                chunk = self.merge_window(section_pages, chunk_idx)
                chunks.append(chunk)
                chunk_idx += 1
            else:
                # Use sliding window for large sections
                windows = self.create_sliding_windows(section_pages)
                for window in windows:
                    chunk = self.merge_window(window, chunk_idx)
                    chunks.append(chunk)
                    chunk_idx += 1

        return chunks


class MultiModalPDFParser:
    """
    Main parser that combines all three strategies:
    1. Camelot for tables
    2. pdfplumber for text blocks
    3. ParseBlock for sliding windows
    """

    def __init__(self, window_size: int = 3, overlap: int = 1):
        self.table_extractor = TableExtractor()
        self.layout_parser = LayoutAwareParser()
        self.parse_block = ParseBlock(window_size, overlap)

    def parse(
        self,
        pdf_path: str,
        use_tables: bool = True,
        use_sliding_window: bool = True
    ) -> List[ParsedChunk]:
        """
        Parse PDF using multi-modal strategy

        Args:
            pdf_path: Path to PDF file
            use_tables: Whether to extract tables with Camelot
            use_sliding_window: Whether to use ParseBlock sliding window

        Returns:
            List of ParsedChunk objects
        """
        all_chunks = []

        print("=" * 50)
        print("MULTI-MODAL PDF PARSER")
        print("=" * 50)

        # 1. Extract text blocks with pdfplumber
        print("\n[1/3] Extracting text blocks with pdfplumber...")
        text_blocks = self.layout_parser.extract_text_blocks(pdf_path)
        print(f"  Extracted {len(text_blocks)} text blocks")

        # 2. Extract tables with Camelot
        if use_tables and self.table_extractor.available:
            print("\n[2/3] Extracting tables with Camelot...")
            tables = self.table_extractor.extract_tables(pdf_path)
            print(f"  Extracted {len(tables)} tables")

            # Convert tables to chunks
            for i, table in enumerate(tables):
                chunk = ParsedChunk(
                    content=table["content"],
                    chunk_id=f"table_{i}_{hashlib.md5(table['content'].encode()).hexdigest()[:8]}",
                    source_type="table",
                    page_numbers=[table["page"]],
                    metadata={
                        "accuracy": table.get("accuracy", 0),
                        "extractor": "camelot"
                    }
                )
                all_chunks.append(chunk)
        else:
            tables = []
            print("\n[2/3] Table extraction skipped (unavailable or disabled)")

        # 3. Create sliding window chunks
        if use_sliding_window and text_blocks:
            print("\n[3/3] Creating sliding window chunks...")
            window_chunks = self.parse_block.parse(text_blocks)
            print(f"  Created {len(window_chunks)} window chunks")
            all_chunks.extend(window_chunks)
        else:
            print("\n[3/3] Sliding window disabled or no text blocks")

        print(f"\nTotal chunks: {len(all_chunks)}")
        print("=" * 50)

        return all_chunks

    def save_chunks(self, chunks: List[ParsedChunk], output_path: str):
        """Save chunks to JSON file"""
        data = [chunk.to_dict() for chunk in chunks]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(chunks)} chunks to {output_path}")

    def load_chunks(self, input_path: str) -> List[ParsedChunk]:
        """Load chunks from JSON file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = []
        for item in data:
            chunks.append(ParsedChunk(**item))

        return chunks


# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse PDF using multi-modal strategy")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", default="data/parsed/chunks.json", help="Output JSON file")
    parser.add_argument("--window-size", type=int, default=3, help="Sliding window size")
    parser.add_argument("--overlap", type=int, default=1, help="Window overlap")
    parser.add_argument("--no-tables", action="store_true", help="Disable table extraction")
    parser.add_argument("--no-window", action="store_true", help="Disable sliding window")

    args = parser.parse_args()

    # Create parser
    pdf_parser = MultiModalPDFParser(
        window_size=args.window_size,
        overlap=args.overlap
    )

    # Parse PDF
    chunks = pdf_parser.parse(
        args.pdf_path,
        use_tables=not args.no_tables,
        use_sliding_window=not args.no_window
    )

    # Save results
    pdf_parser.save_chunks(chunks, args.output)

    # Print summary
    print("\n" + "=" * 50)
    print("CHUNK SUMMARY")
    print("=" * 50)
    print(f"Total chunks: {len(chunks)}")
    print(f"Tables: {sum(1 for c in chunks if c.source_type == 'table')}")
    print(f"Text blocks: {sum(1 for c in chunks if c.source_type == 'sliding_window')}")

    print("\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ({chunk.source_type}, pages {chunk.page_numbers}) ---")
        print(chunk.content[:200] + "...")

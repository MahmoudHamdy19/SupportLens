"""
Layer 0: Data Acquisition and Preprocessing
Component: Policy Preprocessing
=========================================
Performs:
1. Markdown Structure Analysis:
   - Identifies headings (#, ##, ###, underlined), tables, and lists.
   - Specifically extracts Version Support sections, Reporting sections, and Patch SLAs.
2. Verbatim Line Preservation & Traceable Segmentation:
   - Preserves original raw text and maps 1-indexed line numbers (line_start, line_end).
   - Cleans HTML comments and badges without destroying structural traceability.
"""

import re
from typing import Dict, List, Any, Optional, Tuple

class PolicyPreprocessor:
    """Parses, segments, and cleans raw SECURITY.md markdown text with line traceability."""

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Strip badges, raw HTML comments, and redundant blank lines (line endings become LF).

        This is the Layer 1 input (thesis 3.2.1/3.2.2): annotation, training and inference all
        use the text returned here.
        """
        if not text:
            return ""
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove HTML comments <!-- ... -->
        cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)
        # Remove markdown badge links [![...](...)](...)
        cleaned = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', cleaned)
        # Remove standalone badge images ![...](...)
        cleaned = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned)
        # Collapse excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    @classmethod
    def segment_with_line_numbers(cls, raw_text: str) -> List[Dict[str, Any]]:
        """
        Segments markdown text by headings while strictly preserving original line references.
        Returns a list of section dictionaries with exact line_start and line_end (1-indexed).
        """
        if not raw_text or len(raw_text.strip()) == 0:
            return []

        lines = raw_text.splitlines()
        segments: List[Dict[str, Any]] = []

        current_heading = "Header / Preamble"
        current_level = 0
        current_lines: List[str] = []
        current_start = 1

        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')

        for idx, line in enumerate(lines, start=1):
            match = heading_pattern.match(line.strip())
            if match:
                # Save previous segment if non-empty
                if current_lines:
                    seg_raw = "\n".join(current_lines)
                    segments.append({
                        "heading": current_heading,
                        "level": current_level,
                        "line_start": current_start,
                        "line_end": idx - 1,
                        "raw_content": seg_raw,
                        "cleaned_content": cls.clean_text(seg_raw),
                        "section_type": cls._classify_section(current_heading, seg_raw)
                    })
                current_heading = match.group(2).strip()
                current_level = len(match.group(1))
                current_lines = [line]
                current_start = idx
            else:
                current_lines.append(line)

        # Append final segment
        if current_lines:
            seg_raw = "\n".join(current_lines)
            segments.append({
                "heading": current_heading,
                "level": current_level,
                "line_start": current_start,
                "line_end": len(lines),
                "raw_content": seg_raw,
                "cleaned_content": cls.clean_text(seg_raw),
                "section_type": cls._classify_section(current_heading, seg_raw)
            })

        return segments

    @classmethod
    def _classify_section(cls, heading: str, content: str) -> str:
        """Classify markdown section into thematic policy categories."""
        h_lower = heading.lower()
        c_lower = content.lower()

        if any(k in h_lower for k in ["supported", "support", "versions", "compatibility", "maintenance", "releases"]):
            return "supported_versions"
        if any(k in h_lower for k in ["report", "disclosure", "vulnerability", "contact", "bounty", "reporting"]):
            return "vulnerability_reporting"

        # Check content cues if heading is generic
        if "|" in content and any(k in c_lower for k in ["version", "supported"]):
            return "supported_versions"

        return "general"

    @classmethod
    def extract_version_support_section(cls, raw_text: str) -> Dict[str, Any]:
        """
        Specifically analyzes and extracts the Version Support section:
        - Locates section heading and line range
        - Extracts raw markdown tables and bullet list items
        """
        segments = cls.segment_with_line_numbers(raw_text)
        version_segments = [s for s in segments if s["section_type"] == "supported_versions"]

        if not version_segments:
            # Fallback: search for any segment containing a markdown table with version headers
            for s in segments:
                if re.search(r'\|\s*version\s*\|\s*supported\s*\|', s["raw_content"], re.IGNORECASE):
                    version_segments.append(s)
                    break

        if not version_segments:
            return {
                "found": False,
                "heading": None,
                "line_start": None,
                "line_end": None,
                "raw_content": "",
                "cleaned_content": "",
                "tables": [],
                "list_items": []
            }

        primary = version_segments[0]
        # Extract markdown tables
        table_rows = []
        for line in primary["raw_content"].splitlines():
            if line.strip().startswith("|") and line.strip().endswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 2 and not all(c.startswith("-") for c in cells):
                    table_rows.append(cells)

        # Extract list items
        list_items = []
        for line in primary["raw_content"].splitlines():
            m = re.match(r'^\s*[\*\-]\s+(.+)$', line)
            if m:
                list_items.append(m.group(1).strip())

        return {
            "found": True,
            "heading": primary["heading"],
            "line_start": primary["line_start"],
            "line_end": primary["line_end"],
            "raw_content": primary["raw_content"],
            "cleaned_content": primary["cleaned_content"],
            "tables": table_rows,
            "list_items": list_items
        }

    @classmethod
    def extract_relevant_sections(cls, text: str) -> Dict[str, Any]:
        """
        Extracts structured sections preserving backward compatibility,
        while attaching line references and classified segments.
        """
        segments = cls.segment_with_line_numbers(text)
        cleaned_full = cls.clean_text(text)

        supported_versions_parts = []
        reporting_parts = []
        line_references = {}

        for seg in segments:
            stype = seg["section_type"]
            heading = seg["heading"]
            content = seg["raw_content"]
            line_ref = f"L{seg['line_start']}-L{seg['line_end']}"
            line_references[heading] = {
                "line_start": seg["line_start"],
                "line_end": seg["line_end"],
                "section_type": stype
            }

            if stype == "supported_versions":
                supported_versions_parts.append(f"<!-- {line_ref} -->\n{content}")
            elif stype == "vulnerability_reporting":
                reporting_parts.append(f"<!-- {line_ref} -->\n{content}")

        version_section_detail = cls.extract_version_support_section(text)

        return {
            "supported_versions": "\n\n".join(supported_versions_parts).strip(),
            "reporting": "\n\n".join(reporting_parts).strip(),
            "full_preprocessed_text": cleaned_full,
            "segments": segments,
            "line_references": line_references,
            "version_section_detail": version_section_detail
        }

    @classmethod
    def prioritize_version_support(cls, raw_text: str) -> str:
        """
        Constructs preprocessed text with the version support section prioritized first,
        followed by remaining policy content, ensuring neural extractor truncation does
        not omit critical version support tables or commitments.
        """
        if not raw_text:
            return ""
        sections = cls.extract_relevant_sections(raw_text)
        v_detail = sections.get("version_section_detail", {})
        supp_sec = sections.get("supported_versions", "")

        if v_detail.get("found") and v_detail.get("raw_content"):
            supp_content = v_detail["raw_content"].strip()
            remaining = raw_text.replace(v_detail["raw_content"], "").strip()
            prioritized = f"### [Prioritized Version Support Scope]\n{supp_content}\n\n### [General Security Policy Context]\n{remaining}"
            return cls.clean_text(prioritized)
        elif supp_sec:
            remaining = raw_text
            for s in sections.get("segments", []):
                if s.get("section_type") == "supported_versions":
                    remaining = remaining.replace(s.get("raw_content", ""), "")
            prioritized = f"### [Prioritized Version Support Scope]\n{supp_sec}\n\n### [General Security Policy Context]\n{remaining.strip()}"
            return cls.clean_text(prioritized)

        return cls.clean_text(raw_text)


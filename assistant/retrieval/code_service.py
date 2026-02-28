"""CodeIndexService: manages code index snapshotting, rebuild detection, snippet retrieval,
version hashing, and technical response context assembly.

This isolates code-specific evolution from the core RagEngine.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import hashlib

logger = logging.getLogger(__name__)

class CodeIndexService:
    def __init__(self, settings=None, code_index=None):
        self.settings = settings
        self.code_index = code_index  # instance of CodeIndex or None
        self._snapshot: Dict[str, float] = {}
        self.snapshot_sources()

    # --- Snapshot & Versioning --------------------------------------------------
    def snapshot_sources(self):
        self._snapshot = {}
        base = Path('.')
        for py in base.rglob('src/**/*.py'):
            try:
                stat = py.stat()
                self._snapshot[str(py)] = stat.st_mtime
            except Exception:
                continue

    def version(self) -> str:
        if not self._snapshot:
            return "none"
        items = sorted(self._snapshot.items())
        concat = ''.join(f"{k}:{v}" for k, v in items)
        return hashlib.sha256(concat.encode('utf-8')).hexdigest()[:12]

    # --- Rebuild Detection ------------------------------------------------------
    def ensure_current(self):
        # Respect optional flag
        try:
            if getattr(self.settings, 'disable_auto_rebuild', False):
                return
        except Exception:
            pass
        base = Path('.')
        changed = False
        for py in base.rglob('src/**/*.py'):
            try:
                mtime = py.stat().st_mtime
                old = self._snapshot.get(str(py))
                if old is None or mtime > old:
                    changed = True
                    break
            except Exception:
                continue
        if changed:
            # Always update snapshot first so version changes even if rebuild fails
            self.snapshot_sources()
            if self.code_index is not None:
                try:
                    from assistant.retrieval.code_index import CodeIndex
                    self.code_index = CodeIndex(getattr(self.settings, 'code_index_path', 'vector_stores/code_index'))
                    logger.info("Code index rebuilt after source changes detected")
                except Exception as e:
                    logger.warning(f"Failed rebuilding code index: {e}")

    # --- Snippet Retrieval ------------------------------------------------------
    def retrieve_snippets(self, query: str, role: Optional[str], max_results: int = 3) -> List[Dict[str, Any]]:
        if role not in ["Hiring Manager (technical)", "Software Developer"]:
            return []
        if not self.code_index:
            return []
        try:
            keywords = [w.strip().lower() for w in query.split() if len(w) > 3 and w.lower() not in {'what','how','the','this','that'}]
            results = self.code_index.search_code(query, max_results=max_results)
            if not results and keywords:
                results = self.code_index.search_by_keywords(keywords, max_results=max_results)
            snippets = []
            for r in results:
                snippets.append({
                    "file": r["file"],
                    "citation": r["citation"],
                    "content": r["content"],
                    "type": r["type"],
                    "name": r["name"],
                    "github_url": r.get("github_url"),
                    "line_start": r.get("line_start"),
                    "line_end": r.get("line_end"),
                })
            return snippets
        except Exception as e:
            logger.warning(f"Code snippet retrieval failed: {e}")
            return []

    # --- Enrichment -------------------------------------------------------------
    def enrich_with_code(self, career_results: Dict[str, Any], query: str, role: Optional[str]) -> Dict[str, Any]:
        self.ensure_current()
        snippets = self.retrieve_snippets(query, role)
        return {
            **career_results,
            "code_snippets": snippets,
            "has_code": len(snippets) > 0,
            "code_index_version": self.version()
        }

    # --- Prompt Assembly --------------------------------------------------------
    def build_technical_context(self, career_matches: List[str], snippets: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        if career_matches:
            parts.append("Career Knowledge:")
            for m in career_matches[:3]:
                parts.append(f"- {m}")
        if snippets:
            parts.append("\nCode Examples:")
            for s in snippets:
                parts.append(f"- {s['name']} in {s['citation']}")
                code_preview = s['content'][:300] + "..." if len(s['content']) > 300 else s['content']
                parts.append(f"```python\n{code_preview}\n```")
        return "\n".join(parts)

    def technical_prompt(self, query: str, context: str) -> str:
        return f"""
Based on the following context about Noah's work, provide a technical response:

{context}

User Question: {query}

Provide a detailed technical response with:
1. Engineer Detail section with code examples and citations
2. Plain-English Summary section
3. Include specific file:line references where relevant

Be thorough and reference the specific code examples provided.
""".strip()

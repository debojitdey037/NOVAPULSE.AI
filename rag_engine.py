"""
NovaPulse X1 - Intelligent Python RAG (Retrieval-Augmented Generation) Engine
Features:
- Smart semantic chunking with metadata extraction (source, section, word count, frame targets)
- Vectorization using TF-IDF sublinear n-gram space + Cosine Similarity ranking
- Dimensionality reduction (PCA / SVD) to generate live 2D Vector Space coordinates
- Dynamic query projection into the 2D vector space for the interactive animated radar
- Frame target mapping to sync RAG responses with 3D exploded view animation
- Dual synthesis mode: Google Gemini API (if key present) or Grounded Local RAG Synthesizer
- Dynamic document ingestion (TXT, MD, PDF) and full corpus indexing
"""

import os
import re
import json
import math
import glob
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Optional pypdf for PDF ingestion
try:
    import pypdf
except ImportError:
    pypdf = None

# Component to Frame Target Mapping
COMPONENT_FRAME_MAP = [
    {
        "name": "CNC Aluminum Chassis & Acoustic Cup",
        "frame": 35,
        "keywords": ["aluminum", "chassis", "cowl", "cnc", "shell", "cup", "enclosure", "ear cup", "casing"]
    },
    {
        "name": "Magnetic Memory Foam Cushion & Gimbals",
        "frame": 75,
        "keywords": ["cushion", "memory foam", "gimbal", "earpad", "ear pad", "protein leather", "comfort", "detachment"]
    },
    {
        "name": "50mm Beryllium Transducer & Neodymium Magnet",
        "frame": 115,
        "keywords": ["driver", "beryllium", "transducer", "diaphragm", "peek", "magnet", "neodymium", "tesla", "voice coil", "thd", "frequency", "sub-bass", "excursion"]
    },
    {
        "name": "NovaFlow Neural ANC Motherboard & DSP",
        "frame": 125,
        "keywords": ["anc", "motherboard", "chip", "dsp", "npu", "noise cancellation", "microphone", "mems", "transparency", "attenuation", "wind", "latency", "kalman"]
    },
    {
        "name": "Dual Split Lithium-Polymer Battery System",
        "frame": 142,
        "keywords": ["battery", "power", "charge", "playback", "hours", "mah", "warpcharge", "cells", "endurance", "thermal"]
    },
    {
        "name": "Spring-Steel Headband & Modular Yoke",
        "frame": 210,
        "keywords": ["headband", "teardown", "disassembly", "repair", "yoke", "steel", "screws", "torx", "ribbon", "modular"]
    }
]

class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        title: str,
        text: str,
        frame_target: Optional[int] = None,
        keywords: Optional[List[str]] = None
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.title = title
        self.text = text.strip()
        self.word_count = len(self.text.split())
        self.frame_target = frame_target
        self.keywords = keywords or []
        self.coords_2d = [0.0, 0.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "word_count": self.word_count,
            "frame_target": self.frame_target,
            "keywords": self.keywords,
            "coords_2d": [round(float(c), 3) for c in self.coords_2d]
        }


class RAGEngine:
    def __init__(self, corpus_dir: str = "corpus", uploads_dir: str = "static/uploads"):
        self.corpus_dir = corpus_dir
        self.uploads_dir = uploads_dir
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.svd_model = None
        self.coords_2d_matrix = None
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

        os.makedirs(self.corpus_dir, exist_ok=True)
        os.makedirs(self.uploads_dir, exist_ok=True)

        self.reload_and_index()

    def set_api_key(self, api_key: str):
        self.gemini_api_key = api_key.strip()

    def _extract_frame_target(self, text: str, default: int = 120) -> int:
        # Check explicit "Target Exploded Frame: \d+"
        m = re.search(r"Target Exploded Frame:\s*(\d+)", text, re.IGNORECASE)
        if m:
            return int(m.group(1))

        # Check keywords
        text_lower = text.lower()
        for comp in COMPONENT_FRAME_MAP:
            for kw in comp["keywords"]:
                if kw in text_lower:
                    return comp["frame"]

        return default

    def _extract_keywords(self, text: str, top_n: int = 4) -> List[str]:
        words = re.findall(r"\b[A-Za-z0-9_-]{4,}\b", text.lower())
        stop_words = {
            "with", "this", "that", "from", "have", "were", "which", "their", "about",
            "there", "these", "other", "using", "under", "after", "while", "where",
            "overall", "target", "phase", "frames", "features", "allows", "ensures"
        }
        filtered = [w for w in words if w not in stop_words]
        counts: Dict[str, int] = {}
        for w in filtered:
            counts[w] = counts.get(w, 0) + 1
        sorted_kws = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_kws[:top_n]]

    def chunk_markdown(self, doc_id: str, content: str) -> List[DocumentChunk]:
        """Splits markdown content into semantic chunks by H1/H2/H3 headers or double newlines."""
        chunks: List[DocumentChunk] = []
        sections = re.split(r"(?=(?:^|\n)##?#? )", content)
        doc_title = doc_id.replace(".md", "").replace("_", " ").title()

        chunk_counter = 1
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            # Determine section title
            lines = sec.split("\n")
            first_line = lines[0].strip()
            if first_line.startswith("#"):
                sec_title = first_line.lstrip("#").strip()
                body = "\n".join(lines[1:]).strip()
            else:
                sec_title = doc_title
                body = sec

            if not body and not first_line:
                continue

            full_text = f"{sec_title}\n{body}" if body else sec_title
            # If text is too long (> 300 words), split into smaller paragraphs
            paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
            cur_buffer = []
            cur_words = 0

            for p in paragraphs:
                p_words = len(p.split())
                if cur_words + p_words > 220 and cur_buffer:
                    chunk_text = "\n\n".join(cur_buffer)
                    target_frame = self._extract_frame_target(chunk_text)
                    kws = self._extract_keywords(chunk_text)
                    chunk_id = f"{doc_id}_{chunk_counter:03d}"
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        title=f"{doc_title}: {sec_title}",
                        text=chunk_text,
                        frame_target=target_frame,
                        keywords=kws
                    ))
                    chunk_counter += 1
                    cur_buffer = [p]
                    cur_words = p_words
                else:
                    cur_buffer.append(p)
                    cur_words += p_words

            if cur_buffer:
                chunk_text = "\n\n".join(cur_buffer)
                target_frame = self._extract_frame_target(chunk_text)
                kws = self._extract_keywords(chunk_text)
                chunk_id = f"{doc_id}_{chunk_counter:03d}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    title=f"{doc_title}: {sec_title}",
                    text=chunk_text,
                    frame_target=target_frame,
                    keywords=kws
                ))
                chunk_counter += 1

        return chunks

    def chunk_pdf(self, doc_id: str, file_path: str) -> List[DocumentChunk]:
        """Extracts text from PDF file and chunks by page or size."""
        if not pypdf:
            return []
        chunks = []
        try:
            reader = pypdf.PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text or not text.strip():
                    continue
                target_frame = self._extract_frame_target(text)
                kws = self._extract_keywords(text)
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_page_{idx+1}",
                    doc_id=doc_id,
                    title=f"{doc_id} - Page {idx+1}",
                    text=text.strip(),
                    frame_target=target_frame,
                    keywords=kws
                ))
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return chunks

    def reload_and_index(self):
        """Loads all documents from corpus_dir and uploads_dir, chunks them, and builds vector index."""
        all_chunks: List[DocumentChunk] = []

        # Load corpus files
        for fpath in sorted(glob.glob(os.path.join(self.corpus_dir, "*.md"))):
            fname = os.path.basename(fpath)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            all_chunks.extend(self.chunk_markdown(fname, content))

        # Load uploads
        for fpath in sorted(glob.glob(os.path.join(self.uploads_dir, "*"))):
            fname = os.path.basename(fpath)
            if fname.endswith((".md", ".txt")):
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                all_chunks.extend(self.chunk_markdown(fname, content))
            elif fname.endswith(".pdf") and pypdf:
                all_chunks.extend(self.chunk_pdf(fname, fpath))

        self.chunks = all_chunks

        if not self.chunks:
            return

        # Build TF-IDF Vectorizer
        corpus_texts = [f"{c.title} {c.text}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            max_df=0.95,
            min_df=1
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

        # 2D projection via SVD
        n_components = min(2, self.tfidf_matrix.shape[1], len(self.chunks))
        if n_components >= 2:
            self.svd_model = TruncatedSVD(n_components=2, random_state=42)
            coords = self.svd_model.fit_transform(self.tfidf_matrix)
            # Normalize coordinates to [-1, 1] range for visualizer
            max_val = np.max(np.abs(coords)) if np.max(np.abs(coords)) > 0 else 1.0
            norm_coords = coords / max_val
            for idx, c in enumerate(self.chunks):
                c.coords_2d = [float(norm_coords[idx, 0]), float(norm_coords[idx, 1])]
            self.coords_2d_matrix = norm_coords
        else:
            # Fallback circle distribution
            for idx, c in enumerate(self.chunks):
                angle = 2 * math.pi * idx / max(1, len(self.chunks))
                c.coords_2d = [math.cos(angle) * 0.7, math.sin(angle) * 0.7]

    def project_query_2d(self, query_vec) -> List[float]:
        """Projects a query TF-IDF vector into 2D space for the visual radar."""
        if self.svd_model is not None:
            try:
                coords = self.svd_model.transform(query_vec)
                max_val = np.max(np.abs(coords)) if np.max(np.abs(coords)) > 0 else 1.0
                norm_coords = coords[0] / max_val
                # clamp
                cx = max(-0.95, min(0.95, float(norm_coords[0])))
                cy = max(-0.95, min(0.95, float(norm_coords[1])))
                return [cx, cy]
            except Exception:
                pass
        return [0.0, 0.0]

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieves top_k chunks matching query with cosine similarity scores."""
        if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []

        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))

        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            score = float(sims[idx])

            # Find matching terms
            chunk_words = set(re.findall(r"\b\w{3,}\b", chunk.text.lower()))
            overlap = list(query_words.intersection(chunk_words))

            results.append({
                "rank": rank + 1,
                "score": round(score, 4),
                "relevance_pct": round(score * 100, 1),
                "chunk": chunk.to_dict(),
                "matching_terms": overlap
            })

        return results

    def determine_target_frame(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> int:
        """Determines the best exploded view animation frame for this query/retrieval."""
        # 1. Check direct query keywords
        q_lower = query.lower()
        for comp in COMPONENT_FRAME_MAP:
            for kw in comp["keywords"]:
                if kw in q_lower:
                    return comp["frame"]

        # 2. Check top retrieved chunk frame target
        if retrieved_chunks and retrieved_chunks[0]["chunk"]["frame_target"]:
            return retrieved_chunks[0]["chunk"]["frame_target"]

        return 120 # Default exploded view center

    def local_synthesize(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """High-fidelity local grounded RAG generation when external LLM is not active."""
        if not retrieved_chunks:
            return "No matching technical documentation was found in the indexed corpus. Try adjusting your search query or uploading additional documents."

        top_matches = [r for r in retrieved_chunks if r["score"] > 0.05]
        if not top_matches:
            top_matches = retrieved_chunks[:2]

        q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        key_facts = []

        for r in top_matches:
            chunk = r["chunk"]
            # Split into clean logical units (sentences or markdown list items)
            raw_lines = [l.strip() for l in chunk["text"].split("\n") if l.strip()]
            for line in raw_lines:
                # Remove markdown headers or bullet markers
                cleaned = re.sub(r"^[#\-\*\d\.\s]+", "", line).strip()
                if len(cleaned) < 20:
                    continue
                line_words = set(re.findall(r"\b\w{3,}\b", cleaned.lower()))
                common = q_words.intersection(line_words)
                if common or any(kw in cleaned.lower() for kw in chunk.get("keywords", [])):
                    citation = f"[{chunk['title']}]"
                    key_facts.append((len(common), cleaned, citation))

        # Sort facts by keyword match count descending
        key_facts.sort(key=lambda x: x[0], reverse=True)
        unique_facts = []
        seen = set()
        for _, fact, cite in key_facts:
            core = fact[:45].lower()
            if core not in seen:
                seen.add(core)
                unique_facts.append(f"• **{fact}** <sub>*{cite}*</sub>")
            if len(unique_facts) >= 4:
                break

        top_chunk = top_matches[0]["chunk"]
        response_lines = [
            f"**Retrieval Grounding** ({top_chunk['title']} — *Score: {top_matches[0]['relevance_pct']}%*):",
            ""
        ]

        if unique_facts:
            response_lines.extend(unique_facts)
        else:
            first_p = top_chunk["text"].split("\n\n")[0].replace("\n", " ")
            response_lines.append(f"> {first_p} <sub>*[{top_chunk['title']}]*</sub>")

        target_frame = self.determine_target_frame(query, retrieved_chunks)
        response_lines.append("")
        response_lines.append(f"🎯 *Hardware Animation synchronized to **Frame #{target_frame:03d}**.*")

        return "\n".join(response_lines)

    async def generate_response(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Full RAG Pipeline: Retrieval -> Prompt Augmentation -> Synthesis -> Frame Sync."""
        # 1. Retrieval
        retrieved = self.retrieve(query, top_k=top_k)
        target_frame = self.determine_target_frame(query, retrieved)

        # 2. Query 2D projection for radar
        query_vec = self.vectorizer.transform([query]) if self.vectorizer else None
        query_2d = self.project_query_2d(query_vec) if query_vec is not None else [0.0, 0.0]

        # 3. Augmented Prompt Builder
        context_blocks = []
        for idx, r in enumerate(retrieved):
            c = r["chunk"]
            context_blocks.append(f"--- [CHUNK {idx+1}: {c['title']} | Frame: {c['frame_target']}] ---\n{c['text']}")
        augmented_context = "\n\n".join(context_blocks)

        augmented_prompt = (
            f"SYSTEM: You are the NovaPulse Engineering AI Assistant. Answer the question precisely using ONLY the provided technical context.\n"
            f"Ground your answer with citations like [Chunk 1].\n\n"
            f"CONTEXT:\n{augmented_context}\n\n"
            f"USER QUERY: {query}\n"
            f"ANSWER:"
        )

        # 4. Synthesis: Gemini API or Local Engine
        answer = ""
        llm_engine_used = "Local High-Fidelity Synthesizer"

        if self.gemini_api_key:
            try:
                import urllib.request
                import urllib.error

                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
                payload = {
                    "contents": [{
                        "parts": [{"text": augmented_prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 600
                    }
                }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=8) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    cand = res_data.get("candidates", [{}])[0]
                    content_parts = cand.get("content", {}).get("parts", [{}])[0]
                    answer = content_parts.get("text", "")
                    llm_engine_used = "Google Gemini 1.5 Flash"
            except Exception as e:
                print(f"Gemini API call failed, falling back to local: {e}")
                answer = self.local_synthesize(query, retrieved)
        else:
            answer = self.local_synthesize(query, retrieved)

        return {
            "query": query,
            "answer": answer,
            "llm_engine": llm_engine_used,
            "target_frame": target_frame,
            "query_coords_2d": [round(float(q), 3) for q in query_2d],
            "retrieved_chunks": retrieved,
            "augmented_prompt": augmented_prompt,
            "corpus_stats": {
                "total_chunks": len(self.chunks),
                "vocabulary_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0
            }
        }

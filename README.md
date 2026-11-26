# NovaPulse X1 — Animated RAG Website (Python)

A state-of-the-art interactive **Animated Retrieval-Augmented Generation (RAG) Website** powered by Python, FastAPI, Scikit-Learn vector embeddings, and an Apple-style 60fps HTML5 Canvas exploded-view scrollytelling engine.

---

## 🌟 Key Features

### 1. 🎧 60fps Exploded View Canvas Scrollytelling
- **240-Frame High-Resolution Hardware Teardown**: Precision sequence showing the NovaPulse X1 headphones disassembling into its core acoustic, structural, and electronic sub-assemblies.
- **Scroll-Synchronized Progression**: Buttery-smooth lerp interpolation mapping scroll position directly to frame index.
- **Interactive Scrubber Controls**: Play, pause, forward, reverse, manual scrub slider, and instant hotspot quick-jump pills.
- **Hardware Component Hotspots**:
  - `Frame #035`: 7075-T6 Aerospace Aluminum Acoustic Enclosure
  - `Frame #075`: Magnetic Slow-Recovery Protein Leather Cushions
  - `Frame #115`: 50mm Pure PVD Beryllium Transducer & N52 Motor
  - `Frame #125`: NovaFlow-X Neural ANC & DSP Motherboard
  - `Frame #142`: Bifurcated 1200mAh Lithium-Polymer Power System
  - `Frame #210`: Austenitic Spring-Steel & Carbon Core (Right-to-Repair)

### 2. 🤖 Python-Powered RAG Engine (`rag_engine.py`)
- **Semantic Text Chunking & Metadata Extraction**: Automatically parses Markdown, plain text, and PDF documents into logical sections with token budgets, keywords, and hardware visual targets.
- **Vector Space & Cosine Similarity Ranking**: Sublinear TF-IDF n-gram vectorizer generating normalized high-dimensional representations with sub-10ms query retrieval.
- **Visual Hardware Grounding**: Queries regarding hardware components automatically return the relevant frame ID, triggering the canvas animation to scrub directly to that part!
- **Dual Synthesis Mode**:
  - **Local Grounded Synthesizer**: Built-in engine requiring zero external API keys, generating citations directly anchored in the retrieved chunks.
  - **Google Gemini 1.5 Flash**: Option to plug in a Gemini API key for generative intelligence.

### 3. 🛰️ 2D Semantic Vector Space Radar
- **Dimensionality Reduction**: Projects the high-dimensional document chunk vectors into 2D semantic coordinates using SVD/PCA.
- **Live Interactive Radar**: Renders concentric cosine similarity distance rings, rotating radar sweep, and color-coded document nodes.
- **Real-Time Query Sonar**: Animates the user's query vector landing in the semantic space with expanding sonar ripples and pulsing laser beams to top-retrieved chunks.
- **Interactive Tooltip HUD**: Hover over any chunk node to preview title, word count, and click to synchronize the exploded view.

### 4. 📚 Custom Document Ingestion
- **Drag-and-Drop Ingestion**: Upload custom `.txt`, `.md`, or `.pdf` technical manuals or research papers.
- **Automatic Re-indexing**: Live recalculation of vector embeddings and 2D coordinates.
- **Chunk Inspector Table**: Real-time view of all indexed documents and token statistics.
- **Factory Reset**: Instant rollback to original engineering documentation.

---

## 🚀 Quickstart & Running the App

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000
```
or run directly:
```bash
python3 main.py
```

### 3. Open in Browser
Visit: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📁 Project Architecture

```
Animated-RAG-Website/
├── main.py                  # FastAPI web server and REST API endpoints
├── rag_engine.py            # Python RAG engine (chunking, vectorization, cosine similarity, SVD/PCA, synthesis)
├── requirements.txt         # Python dependencies
├── corpus/                  # Default engineering knowledge base
│   ├── 01_system_overview.md
│   ├── 02_acoustic_drivers.md
│   ├── 03_neural_anc_dsp.md
│   ├── 04_battery_power.md
│   ├── 05_teardown_disassembly.md
│   └── 06_materials_spatial.md
└── static/
    ├── index.html           # Single-page web application
    ├── css/
    │   └── style.css        # Sleek cyber-audiophile dark mode design
    ├── js/
    │   ├── app.js           # Chat controller, telemetry, ingestion, audio waveform
    │   ├── scrolly.js       # 60fps Canvas scrollytelling & scrubber engine
    │   └── visualizer.js    # 2D Vector Space Radar & RAG pipeline visualizer
    └── frames/              # 240 exploded view frames (ezgif-frame-001.jpg .. 240.jpg)
```

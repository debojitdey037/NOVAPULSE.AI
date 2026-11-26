"""
NovaPulse X1 - Animated RAG Website Backend
FastAPI server serving the animated scrollytelling frontend, Python RAG pipeline,
vector space projections, file uploads, and component frame synchronization.
"""

import os
import time
import shutil
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine import RAGEngine, COMPONENT_FRAME_MAP

app = FastAPI(
    title="NovaPulse X1 - Animated RAG Website",
    description="Interactive 60fps Scrollytelling & 2D Vector Space RAG Assistant",
    version="1.0.0"
)

# Enable CORS for flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine
rag_engine = RAGEngine(corpus_dir="corpus", uploads_dir="static/uploads")

# Hotspots configuration
HOTSPOTS = [
    {
        "id": "chassis",
        "name": "7075-T6 Aerospace Aluminum Enclosure",
        "frame": 35,
        "tag": "Structural",
        "description": "5-axis CNC-milled chassis with micro-perforated acoustic pressure release vents.",
        "sample_questions": [
            "What material is used for the headphone ear cups?",
            "How does the acoustic chamber suppress standing waves?"
        ]
    },
    {
        "id": "cushions",
        "name": "Magnetic Protein Leather Cushions",
        "frame": 75,
        "tag": "Ergonomics",
        "description": "8-point neodymium magnetic latch with slow-recovery memory foam and acoustic silicone seal.",
        "sample_questions": [
            "How do the ear cushions attach to the headphones?",
            "What materials are used for the memory foam ear pads?"
        ]
    },
    {
        "id": "driver",
        "name": "50mm Beryllium Transducer & N52 Motor",
        "frame": 115,
        "tag": "Acoustics",
        "description": "18-micron PEEK substrate with 2.4-micron pure PVD Beryllium dome and 1.55 Tesla flux.",
        "sample_questions": [
            "What are the specs of the 50mm Beryllium driver?",
            "What is the frequency response range and harmonic distortion?"
        ]
    },
    {
        "id": "anc",
        "name": "NovaFlow-X Neural ANC & DSP Motherboard",
        "frame": 125,
        "tag": "Electronics",
        "description": "6-mic hybrid matrix powered by 1.2 TOPS neural NPU with -48dB peak attenuation.",
        "sample_questions": [
            "How does the NovaFlow Neural ANC filter wind noise?",
            "What is the processing latency of the transparency mode?"
        ]
    },
    {
        "id": "battery",
        "name": "Dual Split 1200mAh Li-Po System",
        "frame": 142,
        "tag": "Power",
        "description": "Bifurcated 600mAh cells for 50/50 lateral balance, offering up to 68 hours of playback.",
        "sample_questions": [
            "How long does the battery last with LDAC and ANC on?",
            "How fast does WarpCharge 3.0 recharge the headphones?"
        ]
    },
    {
        "id": "core",
        "name": "Austenitic Spring-Steel & Carbon Core",
        "frame": 210,
        "tag": "Teardown",
        "description": "Modular right-to-repair architecture with zero glue and tool-less battery latch.",
        "sample_questions": [
            "How do I disassemble the headphones for battery replacement?",
            "What is the repairability score of the NovaPulse X1?"
        ]
    }
]

class ChatRequest(BaseModel):
    query: str
    top_k: int = 4
    api_key: Optional[str] = None

@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "frames_available": 240,
        "indexed_chunks": len(rag_engine.chunks),
        "vocabulary_size": len(rag_engine.vectorizer.vocabulary_) if rag_engine.vectorizer else 0
    }

@app.get("/api/hotspots")
async def get_hotspots():
    return {"hotspots": HOTSPOTS}

@app.get("/api/chunks")
async def get_chunks():
    """Returns all indexed chunks, titles, and 2D coordinates for the vector radar."""
    return {
        "total_chunks": len(rag_engine.chunks),
        "chunks": [c.to_dict() for c in rag_engine.chunks]
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if request.api_key:
        rag_engine.set_api_key(request.api_key)

    start_time = time.time()
    result = await rag_engine.generate_response(request.query.strip(), top_k=request.top_k)
    duration_ms = round((time.time() - start_time) * 1000, 2)
    result["latency_ms"] = duration_ms

    return result

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    if not filename.lower().endswith((".txt", ".md", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt, .md, and .pdf files are supported.")

    dest_path = os.path.join(rag_engine.uploads_dir, filename)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Re-index
    rag_engine.reload_and_index()

    return {
        "message": f"Successfully ingested and indexed '{filename}'.",
        "filename": filename,
        "new_total_chunks": len(rag_engine.chunks)
    }

@app.post("/api/reset-corpus")
async def reset_corpus():
    """Clears user uploads and re-indexes the default NovaPulse corpus."""
    for f in os.listdir(rag_engine.uploads_dir):
        fp = os.path.join(rag_engine.uploads_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)

    rag_engine.reload_and_index()
    return {
        "message": "Corpus successfully reset to default factory specifications.",
        "total_chunks": len(rag_engine.chunks)
    }

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

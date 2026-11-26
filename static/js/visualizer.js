/**
 * NovaPulse X1 - Interactive 2D Vector Space Radar & RAG Visualizer
 * Plots high-dimensional document chunk embeddings projected into 2D semantic space,
 * renders cosine similarity radar grids, real-time query sonar waves, and laser connections.
 */

class VectorSpaceRadar {
  constructor(canvasId = "radar-canvas") {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.chunks = [];
    this.queryPoint = null; // { x, y, query, timestamp }
    this.topMatches = [];
    this.hoveredChunk = null;
    this.sonarRadius = 0;
    this.animationPhase = 0;

    this.init();
  }

  init() {
    if (!this.canvas || !this.ctx) return;
    this.resize();
    window.addEventListener("resize", () => this.resize());

    // Mouse hover detection
    this.canvas.addEventListener("mousemove", (e) => this.handleMouseMove(e));
    this.canvas.addEventListener("mouseleave", () => { this.hoveredChunk = null; });
    this.canvas.addEventListener("click", () => this.handleClick());

    this.render();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.width = rect.width;
    this.height = rect.height;
  }

  setChunks(chunks) {
    this.chunks = chunks || [];
  }

  setQuery(query, coords2D, retrievedChunks) {
    this.queryPoint = {
      x: coords2D[0] || 0,
      y: coords2D[1] || 0,
      query: query,
      timestamp: Date.now()
    };
    this.topMatches = retrievedChunks || [];
    this.sonarRadius = 0;
  }

  getChunkColor(docId) {
    if (!docId) return "#00f0ff";
    if (docId.includes("driver") || docId.includes("acoustic")) return "#00f0ff"; // Cyan
    if (docId.includes("anc") || docId.includes("dsp")) return "#b5179e";         // Violet
    if (docId.includes("battery") || docId.includes("power")) return "#00ff88";    // Emerald
    if (docId.includes("teardown") || docId.includes("disassembly")) return "#ffb703"; // Amber
    return "#4cc9f0"; // Blue
  }

  handleMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const centerX = this.width / 2;
    const centerY = this.height / 2;
    const scale = Math.min(this.width, this.height) * 0.42;

    let closest = null;
    let minDist = 18; // pixel threshold

    for (const c of this.chunks) {
      const px = centerX + c.coords_2d[0] * scale;
      const py = centerY - c.coords_2d[1] * scale;
      const dist = Math.hypot(mx - px, my - py);
      if (dist < minDist) {
        minDist = dist;
        closest = { chunk: c, px, py };
      }
    }

    this.hoveredChunk = closest;
    this.canvas.style.cursor = closest ? "pointer" : "default";
  }

  handleClick() {
    if (this.hoveredChunk && this.hoveredChunk.chunk.frame_target) {
      if (window.scrubToFrame) {
        window.scrubToFrame(this.hoveredChunk.chunk.frame_target);
      }
    }
  }

  render() {
    if (!this.ctx) return;
    this.ctx.clearRect(0, 0, this.width, this.height);

    const centerX = this.width / 2;
    const centerY = this.height / 2;
    const scale = Math.min(this.width, this.height) * 0.42;

    this.animationPhase += 0.025;

    // 1. Draw Radar Background Grid
    this.drawRadarGrid(centerX, centerY, scale);

    // 2. Draw Connection Lines from Query to Top Matches
    if (this.queryPoint && this.topMatches.length > 0) {
      const qx = centerX + this.queryPoint.x * scale;
      const qy = centerY - this.queryPoint.y * scale;

      this.topMatches.forEach((m, idx) => {
        const c = m.chunk;
        const tx = centerX + c.coords_2d[0] * scale;
        const ty = centerY - c.coords_2d[1] * scale;

        // Laser beam line
        this.ctx.save();
        this.ctx.beginPath();
        this.ctx.moveTo(qx, qy);
        this.ctx.lineTo(tx, ty);
        this.ctx.strokeStyle = `rgba(0, 240, 255, ${0.7 - idx * 0.12})`;
        this.ctx.lineWidth = 1.5;
        this.ctx.setLineDash([4, 4]);
        this.ctx.lineDashOffset = -this.animationPhase * 20;
        this.ctx.stroke();

        // Cosine Score pill along the line
        const midX = (qx + tx) / 2;
        const midY = (qy + ty) / 2;
        this.ctx.fillStyle = "rgba(7, 9, 14, 0.85)";
        this.ctx.fillRect(midX - 22, midY - 9, 44, 18);
        this.ctx.strokeStyle = "rgba(0, 240, 255, 0.4)";
        this.ctx.setLineDash([]);
        this.ctx.strokeRect(midX - 22, midY - 9, 44, 18);
        this.ctx.fillStyle = "#00f0ff";
        this.ctx.font = "10px monospace";
        this.ctx.textAlign = "center";
        this.ctx.textBaseline = "middle";
        this.ctx.fillText(`${m.relevance_pct}%`, midX, midY);
        this.ctx.restore();
      });
    }

    // 3. Draw Document Chunk Nodes
    this.chunks.forEach((c) => {
      const px = centerX + c.coords_2d[0] * scale;
      const py = centerY - c.coords_2d[1] * scale;

      // Subtle float motion
      const floatY = Math.sin(this.animationPhase + c.coords_2d[0] * 5) * 1.5;
      const color = this.getChunkColor(c.doc_id);

      // Check if this chunk is a top retrieved match
      const matchIdx = this.topMatches.findIndex(m => m.chunk.chunk_id === c.chunk_id);
      const isMatch = matchIdx !== -1;

      this.ctx.save();
      // Outer glow
      if (isMatch) {
        this.ctx.beginPath();
        this.ctx.arc(px, py + floatY, 9, 0, Math.PI * 2);
        this.ctx.fillStyle = "rgba(0, 240, 255, 0.35)";
        this.ctx.shadowColor = "#00f0ff";
        this.ctx.shadowBlur = 12;
        this.ctx.fill();
      }

      // Main node dot
      this.ctx.beginPath();
      this.ctx.arc(px, py + floatY, isMatch ? 5.5 : 4, 0, Math.PI * 2);
      this.ctx.fillStyle = isMatch ? "#ffffff" : color;
      this.ctx.fill();
      this.ctx.restore();
    });

    // 4. Draw Query Node with Expanding Sonar Ripple
    if (this.queryPoint) {
      const qx = centerX + this.queryPoint.x * scale;
      const qy = centerY - this.queryPoint.y * scale;

      this.sonarRadius += 1.2;
      if (this.sonarRadius > 65) this.sonarRadius = 0;

      this.ctx.save();
      // Sonar ripple
      this.ctx.beginPath();
      this.ctx.arc(qx, qy, this.sonarRadius, 0, Math.PI * 2);
      this.ctx.strokeStyle = `rgba(255, 183, 3, ${1 - this.sonarRadius / 65})`;
      this.ctx.lineWidth = 2;
      this.ctx.stroke();

      // Query center point
      this.ctx.beginPath();
      this.ctx.arc(qx, qy, 6.5, 0, Math.PI * 2);
      this.ctx.fillStyle = "#ffb703";
      this.ctx.shadowColor = "#ffb703";
      this.ctx.shadowBlur = 14;
      this.ctx.fill();

      // Label
      this.ctx.font = "bold 10px monospace";
      this.ctx.fillStyle = "#ffb703";
      this.ctx.textAlign = "center";
      this.ctx.fillText("QUERY", qx, qy - 12);
      this.ctx.restore();
    }

    // 5. Draw Hover Tooltip HUD
    if (this.hoveredChunk) {
      this.drawTooltip(this.hoveredChunk);
    }

    requestAnimationFrame(() => this.render());
  }

  drawRadarGrid(cx, cy, radius) {
    this.ctx.save();

    // Concentric rings
    const rings = [0.25, 0.5, 0.75, 1.0];
    rings.forEach((r) => {
      this.ctx.beginPath();
      this.ctx.arc(cx, cy, radius * r, 0, Math.PI * 2);
      this.ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
      this.ctx.lineWidth = 1;
      this.ctx.stroke();

      // Distance tag
      this.ctx.font = "9px monospace";
      this.ctx.fillStyle = "rgba(255, 255, 255, 0.25)";
      this.ctx.fillText(`cos(θ) ${(1 - r * 0.8).toFixed(1)}`, cx + radius * r + 4, cy - 2);
    });

    // Crosshairs
    this.ctx.beginPath();
    this.ctx.moveTo(cx - radius * 1.05, cy);
    this.ctx.lineTo(cx + radius * 1.05, cy);
    this.ctx.moveTo(cx, cy - radius * 1.05);
    this.ctx.lineTo(cx, cy + radius * 1.05);
    this.ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    this.ctx.stroke();

    // Rotating sweep ray
    this.ctx.beginPath();
    this.ctx.moveTo(cx, cy);
    const sweepAngle = this.animationPhase * 0.6;
    const sx = cx + Math.cos(sweepAngle) * radius;
    const sy = cy + Math.sin(sweepAngle) * radius;
    this.ctx.lineTo(sx, sy);
    this.ctx.strokeStyle = "rgba(0, 240, 255, 0.12)";
    this.ctx.lineWidth = 1.5;
    this.ctx.stroke();

    this.ctx.restore();
  }

  drawTooltip(hovered) {
    const { chunk, px, py } = hovered;
    const pad = 12;
    const w = 240;
    const h = 76;
    let x = px + 12;
    let y = py - h - 12;

    if (x + w > this.width) x = px - w - 12;
    if (y < 10) y = py + 12;

    this.ctx.save();
    // Glass card background
    this.ctx.fillStyle = "rgba(10, 15, 26, 0.94)";
    this.ctx.strokeStyle = "rgba(0, 240, 255, 0.35)";
    this.ctx.lineWidth = 1;
    this.ctx.shadowColor = "rgba(0,0,0,0.6)";
    this.ctx.shadowBlur = 10;
    this.ctx.fillRect(x, y, w, h);
    this.ctx.strokeRect(x, y, w, h);

    // Title
    this.ctx.font = "bold 11px sans-serif";
    this.ctx.fillStyle = "#ffffff";
    this.ctx.textAlign = "left";
    const title = chunk.title.length > 28 ? chunk.title.substring(0, 26) + "..." : chunk.title;
    this.ctx.fillText(title, x + pad, y + 20);

    // Target Frame Tag
    this.ctx.font = "10px monospace";
    this.ctx.fillStyle = "#00f0ff";
    this.ctx.fillText(`Target Frame: #${String(chunk.frame_target || 120).padStart(3, "0")}`, x + pad, y + 38);

    // Click hint
    this.ctx.fillStyle = "#8e9bb0";
    this.ctx.font = "9px sans-serif";
    this.ctx.fillText("Click to sync exploded view frame", x + pad, y + 58);

    this.ctx.restore();
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.vectorRadar = new VectorSpaceRadar("radar-canvas");
});

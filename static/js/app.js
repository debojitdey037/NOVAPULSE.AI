/**
 * NovaPulse X1 - Main Application Controller
 * Handles RAG chat, audio waveform simulation, pipeline telemetry,
 * document ingestion, and hotspot interaction.
 */

document.addEventListener("DOMContentLoaded", () => {
  const chatMessages = document.getElementById("chat-messages");
  const chatInput = document.getElementById("chat-input");
  const chatForm = document.getElementById("chat-form");
  const audioWaveCanvas = document.getElementById("audio-wave-canvas");
  const chunksTableBody = document.getElementById("chunks-tbody");
  const fileDropzone = document.getElementById("file-dropzone");
  const fileInput = document.getElementById("file-input");
  const btnReset = document.getElementById("btn-reset-corpus");

  // Pipeline step elements
  const stepToken = document.getElementById("p-step-1");
  const stepEmbed = document.getElementById("p-step-2");
  const stepRank = document.getElementById("p-step-3");
  const stepSynth = document.getElementById("p-step-4");

  // Load API key from localStorage
  let apiKey = localStorage.getItem("novapulse_gemini_key") || "";

  // 1. Audio Waveform Simulator
  initAudioWaveform(audioWaveCanvas);

  // 2. Fetch and render initial chunks
  loadCorpusChunks();

  // 3. Fetch and render hotspots
  loadHotspots();

  // 4. Bind chat form submit
  if (chatForm) {
    chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const query = chatInput.value.trim();
      if (!query) return;
      submitQuery(query);
      chatInput.value = "";
    });
  }

  // 5. Suggestion pills click
  document.querySelectorAll(".suggestion-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      const q = pill.getAttribute("data-query") || pill.textContent.trim();
      submitQuery(q);
    });
  });

  // 6. File Dropzone Upload
  if (fileDropzone && fileInput) {
    fileDropzone.addEventListener("click", () => fileInput.click());

    fileDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      fileDropzone.classList.add("dragover");
    });

    fileDropzone.addEventListener("dragleave", () => {
      fileDropzone.classList.remove("dragover");
    });

    fileDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      fileDropzone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        uploadFile(e.target.files[0]);
      }
    });
  }

  // 7. Reset Corpus
  if (btnReset) {
    btnReset.addEventListener("click", async () => {
      if (!confirm("Reset knowledge base to factory engineering specifications?")) return;
      try {
        const res = await fetch("/api/reset-corpus", { method: "POST" });
        const data = await res.json();
        alert(data.message);
        loadCorpusChunks();
      } catch (err) {
        alert("Failed to reset corpus: " + err);
      }
    });
  }

  // 8. API Key Modal bindings
  const keyModal = document.getElementById("key-modal");
  const btnOpenKey = document.getElementById("btn-api-key");
  const btnSaveKey = document.getElementById("btn-save-key");
  const btnCloseKey = document.getElementById("btn-close-key");
  const inputApiKey = document.getElementById("input-api-key");

  if (btnOpenKey && keyModal) {
    btnOpenKey.addEventListener("click", () => {
      if (inputApiKey) inputApiKey.value = apiKey;
      keyModal.classList.add("open");
    });
  }
  if (btnCloseKey && keyModal) {
    btnCloseKey.addEventListener("click", () => keyModal.classList.remove("open"));
  }
  if (btnSaveKey && keyModal) {
    btnSaveKey.addEventListener("click", () => {
      apiKey = inputApiKey.value.trim();
      localStorage.setItem("novapulse_gemini_key", apiKey);
      const badge = document.getElementById("chat-mode-badge");
      if (badge) {
        badge.textContent = apiKey ? "MODE: GEMINI 1.5 FLASH" : "MODE: LOCAL GROUNDED";
      }
      keyModal.classList.remove("open");
    });
  }

  // Initial badge setup
  const modeBadge = document.getElementById("chat-mode-badge");
  if (modeBadge && apiKey) {
    modeBadge.textContent = "MODE: GEMINI 1.5 FLASH";
  }

  // ================= FUNCTIONS ================= //

  async function loadCorpusChunks() {
    try {
      const res = await fetch("/api/chunks");
      const data = await res.json();
      if (window.vectorRadar) {
        window.vectorRadar.setChunks(data.chunks);
      }

      // Update stat counters
      const statChunks = document.getElementById("stat-chunks");
      if (statChunks) statChunks.textContent = `${data.total_chunks} Chunks`;

      // Render table
      if (chunksTableBody) {
        chunksTableBody.innerHTML = "";
        data.chunks.forEach((c) => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><code>${c.chunk_id}</code></td>
            <td><strong>${escapeHtml(c.title)}</strong></td>
            <td>${c.word_count} words</td>
            <td><button class="msg-sync-btn" onclick="window.scrubToFrame(${c.frame_target || 120})">Frame #${String(c.frame_target || 120).padStart(3, "0")}</button></td>
            <td><code>[${c.coords_2d[0]}, ${c.coords_2d[1]}]</code></td>
          `;
          chunksTableBody.appendChild(tr);
        });
      }
    } catch (err) {
      console.error("Failed to load chunks:", err);
    }
  }

  async function loadHotspots() {
    try {
      const res = await fetch("/api/hotspots");
      const data = await res.json();
      const container = document.getElementById("hotspot-pills-container");
      if (container && data.hotspots) {
        container.innerHTML = "";
        data.hotspots.forEach((h) => {
          const pill = document.createElement("button");
          pill.className = "hotspot-pill";
          pill.setAttribute("data-frame", h.frame);
          pill.innerHTML = `<strong>${h.tag}</strong> #${String(h.frame).padStart(3, "0")}`;
          pill.title = `${h.name}: ${h.description}`;
          pill.addEventListener("click", () => {
            if (window.scrubToFrame) window.scrubToFrame(h.frame);
            if (h.sample_questions && h.sample_questions.length > 0) {
              submitQuery(h.sample_questions[0]);
            }
          });
          container.appendChild(pill);
        });
      }
    } catch (err) {
      console.error("Failed to load hotspots:", err);
    }
  }

  async function submitQuery(query) {
    appendUserMessage(query);

    // Animate pipeline status
    setPipelineStage(1, "Tokenizing & Vectorizing query...");

    try {
      setPipelineStage(2, "Generating TF-IDF & Cosine Embeddings...");

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, top_k: 4, api_key: apiKey || null })
      });

      if (!res.ok) throw new Error("Server responded with error: " + res.status);
      const data = await res.json();

      setPipelineStage(3, `Retrieved ${data.retrieved_chunks.length} Chunks in ${data.latency_ms}ms`);

      // Update 2D Radar
      if (window.vectorRadar) {
        window.vectorRadar.setQuery(query, data.query_coords_2d, data.retrieved_chunks);
      }

      setPipelineStage(4, `Synthesized via ${data.llm_engine}`);

      // Scrub canvas animation smoothly to target frame!
      if (window.scrubToFrame && data.target_frame) {
        window.scrubToFrame(data.target_frame);
      }

      // Append assistant answer with typing animation
      appendAssistantMessage(data);

      setTimeout(() => resetPipelineStage(), 2500);

    } catch (err) {
      console.error("Query failed:", err);
      appendAssistantMessage({
        answer: "Error processing query: " + err.message,
        llm_engine: "Error",
        target_frame: 120,
        latency_ms: 0,
        retrieved_chunks: []
      });
      resetPipelineStage();
    }
  }

  function setPipelineStage(activeIdx, statusText) {
    const steps = [stepToken, stepEmbed, stepRank, stepSynth];
    steps.forEach((s, idx) => {
      if (!s) return;
      if (idx + 1 === activeIdx) {
        s.classList.add("active");
        const statusEl = s.querySelector(".p-step-status");
        if (statusEl) statusEl.textContent = statusText;
      } else if (idx + 1 < activeIdx) {
        s.classList.remove("active");
        const statusEl = s.querySelector(".p-step-status");
        if (statusEl) statusEl.textContent = "Done";
      } else {
        s.classList.remove("active");
      }
    });
  }

  function resetPipelineStage() {
    const steps = [stepToken, stepEmbed, stepRank, stepSynth];
    steps.forEach((s) => {
      if (s) {
        s.classList.remove("active");
        const statusEl = s.querySelector(".p-step-status");
        if (statusEl) statusEl.textContent = "Standby";
      }
    });
  }

  function appendUserMessage(text) {
    if (!chatMessages) return;
    const msg = document.createElement("div");
    msg.className = "msg user";
    msg.innerHTML = `
      <div class="msg-bubble">${escapeHtml(text)}</div>
      <div class="msg-meta">USER QUERY • ${new Date().toLocaleTimeString()}</div>
    `;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendAssistantMessage(data) {
    if (!chatMessages) return;
    const msg = document.createElement("div");
    msg.className = "msg assistant";

    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";

    // Format markdown-like bold and bullet tags
    let formattedText = escapeHtml(data.answer)
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/&lt;sub&gt;(.*?)&lt;\/sub&gt;/g, "<sub>$1</sub>")
      .replace(/• (.*?)(?=(?:\n|$))/g, "<div style='margin-bottom:6px;'>• $1</div>")
      .replace(/\n\n/g, "<br><br>");

    bubble.innerHTML = formattedText;

    const meta = document.createElement("div");
    meta.className = "msg-meta";
    meta.innerHTML = `
      <span>⚡ ${data.latency_ms}ms</span>
      <span>•</span>
      <span>${data.llm_engine}</span>
      <span>•</span>
      <span>Frame #${String(data.target_frame || 120).padStart(3, "0")}</span>
    `;

    const syncBtn = document.createElement("button");
    syncBtn.className = "msg-sync-btn";
    syncBtn.innerHTML = `🎯 Focus Exploded Frame #${String(data.target_frame || 120).padStart(3, "0")}`;
    syncBtn.onclick = () => {
      if (window.scrubToFrame) window.scrubToFrame(data.target_frame || 120);
    };

    msg.appendChild(bubble);
    msg.appendChild(syncBtn);
    msg.appendChild(meta);

    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    const dropzone = document.getElementById("file-dropzone");
    const origHtml = dropzone ? dropzone.innerHTML : "";
    if (dropzone) {
      dropzone.innerHTML = `<div class="dropzone-icon">⏳</div><div>Ingesting & Indexing <strong>${escapeHtml(file.name)}</strong>...</div>`;
    }

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload error");

      alert(data.message);
      loadCorpusChunks();
    } catch (err) {
      alert("Upload failed: " + err.message);
    } finally {
      if (dropzone) dropzone.innerHTML = origHtml;
    }
  }

  function initAudioWaveform(canvas) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let width = canvas.width = 120;
    let height = canvas.height = 24;
    const bars = 16;

    function renderWave() {
      ctx.clearRect(0, 0, width, height);
      const time = Date.now() * 0.005;

      for (let i = 0; i < bars; i++) {
        const barH = Math.sin(time + i * 0.5) * 8 + 10;
        const x = i * (width / bars);
        const y = (height - barH) / 2;

        ctx.fillStyle = "#00f0ff";
        ctx.shadowColor = "#00f0ff";
        ctx.shadowBlur = 4;
        ctx.fillRect(x + 1, y, (width / bars) - 3, barH);
      }
      requestAnimationFrame(renderWave);
    }
    renderWave();
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});

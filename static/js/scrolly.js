/**
 * NovaPulse X1 - Scrollytelling & 60fps Canvas Frame Engine
 * Preloads 240 frames, handles Apple-style scroll-scrubbing,
 * manual slider interaction, playback loop, and programmatic smooth scrub.
 */

class ScrollytellingEngine {
  constructor(totalFrames = 240) {
    this.totalFrames = totalFrames;
    this.currentFrame = 1;
    this.targetFrame = 1;
    this.images = [];
    this.loadedCount = 0;
    this.isPlaying = false;
    this.playDirection = 1; // 1 for forward, -1 for reverse
    this.playbackSpeed = 1;
    this.lastFrameTime = 0;

    this.canvas = document.getElementById("canvas-viewport");
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.slider = document.getElementById("frame-slider");
    this.frameIndicator = document.getElementById("frame-indicator");
    this.loadProgressBar = document.getElementById("load-progress");

    this.init();
  }

  init() {
    if (!this.canvas || !this.ctx) return;

    // Set internal render resolution
    this.canvas.width = 1280;
    this.canvas.height = 720;

    this.preloadFrames();
    this.bindEvents();
    this.renderLoop();
  }

  getFrameUrl(index) {
    const pad = String(index).padStart(3, "0");
    return `/static/frames/ezgif-frame-${pad}.jpg?v=clean2`;
  }

  preloadFrames() {
    // Priority load: First load frame 1, 60, 120, 180, 240 immediately, then the rest
    const priorityIndices = [1, 35, 75, 115, 120, 125, 142, 210, 240];
    const remainingIndices = [];
    for (let i = 1; i <= this.totalFrames; i++) {
      if (!priorityIndices.includes(i)) remainingIndices.push(i);
    }
    const loadOrder = [...priorityIndices, ...remainingIndices];

    loadOrder.forEach((idx) => {
      const img = new Image();
      img.src = this.getFrameUrl(idx);
      img.onload = () => {
        this.loadedCount++;
        this.images[idx] = img;
        if (this.loadProgressBar) {
          const pct = Math.round((this.loadedCount / this.totalFrames) * 100);
          this.loadProgressBar.style.width = `${pct}%`;
          if (pct >= 100) {
            setTimeout(() => {
              if (this.loadProgressBar.parentElement) {
                this.loadProgressBar.parentElement.style.opacity = "0";
              }
            }, 500);
          }
        }
        // If this is the initial frame, draw immediately
        if (idx === 1 && Math.round(this.currentFrame) === 1) {
          this.drawFrame(1);
        }
      };
      img.onerror = () => {
        // Fallback or retry
        this.loadedCount++;
      };
    });
  }

  bindEvents() {
    // Slider scrub
    if (this.slider) {
      this.slider.addEventListener("input", (e) => {
        this.targetFrame = parseInt(e.target.value, 10);
        this.pause();
      });
    }

    // Play / Pause button
    const playBtn = document.getElementById("btn-play");
    if (playBtn) {
      playBtn.addEventListener("click", () => {
        this.togglePlay();
      });
    }

    // Direction button
    const dirBtn = document.getElementById("btn-reverse");
    if (dirBtn) {
      dirBtn.addEventListener("click", () => {
        this.playDirection *= -1;
        dirBtn.textContent = this.playDirection === 1 ? "⏩ Forward" : "⏪ Reverse";
      });
    }

    // Scroll listener for Apple-style scrollytelling
    const scrollyContainer = document.querySelector(".scrolly-container");
    if (scrollyContainer) {
      window.addEventListener("scroll", () => {
        if (this.isPlaying) return; // Don't override manual playback

        const rect = scrollyContainer.getBoundingClientRect();
        const containerTop = window.scrollY + rect.top;
        const containerHeight = scrollyContainer.offsetHeight - window.innerHeight;
        
        if (window.scrollY >= containerTop && window.scrollY <= containerTop + containerHeight) {
          const scrollPct = (window.scrollY - containerTop) / Math.max(1, containerHeight);
          const mappedFrame = Math.max(1, Math.min(this.totalFrames, Math.round(scrollPct * (this.totalFrames - 1)) + 1));
          this.targetFrame = mappedFrame;
        }
      }, { passive: true });
    }
  }

  drawFrame(frameIdx) {
    const idx = Math.max(1, Math.min(this.totalFrames, Math.round(frameIdx)));
    const img = this.images[idx];

    if (img && img.complete && img.naturalWidth > 0) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
    } else {
      // Find nearest loaded frame if current isn't ready
      for (let offset = 1; offset < 20; offset++) {
        const prev = this.images[idx - offset];
        if (prev && prev.complete) {
          this.ctx.drawImage(prev, 0, 0, this.canvas.width, this.canvas.height);
          break;
        }
        const next = this.images[idx + offset];
        if (next && next.complete) {
          this.ctx.drawImage(next, 0, 0, this.canvas.width, this.canvas.height);
          break;
        }
      }
    }

    // Update UI Indicators
    if (this.frameIndicator) {
      this.frameIndicator.textContent = `FRAME: ${String(idx).padStart(3, "0")} / ${this.totalFrames}`;
    }
    if (this.slider && document.activeElement !== this.slider) {
      this.slider.value = idx;
    }
  }

  renderLoop(timestamp = 0) {
    // Smooth lerp towards targetFrame
    const diff = this.targetFrame - this.currentFrame;
    if (Math.abs(diff) > 0.05) {
      this.currentFrame += diff * 0.18; // Lerp factor for snappy, buttery 60fps response
      this.drawFrame(this.currentFrame);
    }

    // Handle auto-playback
    if (this.isPlaying) {
      const interval = 1000 / (30 * this.playbackSpeed);
      if (timestamp - this.lastFrameTime >= interval) {
        this.lastFrameTime = timestamp;
        let next = this.targetFrame + this.playDirection;
        if (next > this.totalFrames) next = 1;
        if (next < 1) next = this.totalFrames;
        this.targetFrame = next;
      }
    }

    requestAnimationFrame((ts) => this.renderLoop(ts));
  }

  scrubToFrame(frameNum, durationMs = 800) {
    this.pause();
    const clamped = Math.max(1, Math.min(this.totalFrames, Math.round(frameNum)));
    this.targetFrame = clamped;

    // Highlight active hotspot pill
    document.querySelectorAll(".hotspot-pill").forEach(p => {
      const target = parseInt(p.getAttribute("data-frame"), 10);
      if (Math.abs(target - clamped) < 8) {
        p.classList.add("active");
      } else {
        p.classList.remove("active");
      }
    });
  }

  togglePlay() {
    this.isPlaying = !this.isPlaying;
    const playBtn = document.getElementById("btn-play");
    if (playBtn) {
      playBtn.textContent = this.isPlaying ? "⏸ Pause" : "▶ Play";
    }
  }

  pause() {
    this.isPlaying = false;
    const playBtn = document.getElementById("btn-play");
    if (playBtn) {
      playBtn.textContent = "▶ Play";
    }
  }
}

// Global instance initialization
window.addEventListener("DOMContentLoaded", () => {
  window.scrollyEngine = new ScrollytellingEngine(240);
  window.scrubToFrame = (frame, dur) => window.scrollyEngine.scrubToFrame(frame, dur);
});

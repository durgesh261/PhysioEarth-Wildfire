/**
 * TopoCollapse Interactive Dashboard Controller
 */

let simulationData = [];
let currentGen = 0;
let isPlaying = false;
let playInterval = null;

// DOM Elements
const genSlider = document.getElementById('genSlider');
const activeGenBadge = document.getElementById('activeGenBadge');
const playBtn = document.getElementById('playBtn');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');

// Canvases
const manifoldCanvas = document.getElementById('manifoldCanvas');
const ctxManifold = manifoldCanvas.getContext('2d');
const persistenceCanvas = document.getElementById('persistenceCanvas');
const ctxPersistence = persistenceCanvas.getContext('2d');

// Load Data
async function loadSimulationData() {
  try {
    const response = await fetch('data.json');
    simulationData = await response.json();
    initApp();
  } catch (error) {
    console.error('Error loading simulation data:', error);
  }
}

function initApp() {
  resizeCanvases();
  window.addEventListener('resize', () => {
    resizeCanvases();
    renderCurrentState();
  });

  // Slider controls
  genSlider.max = simulationData.length - 1;
  genSlider.value = 0;

  genSlider.addEventListener('input', (e) => {
    setGeneration(parseInt(e.target.value));
  });

  playBtn.addEventListener('click', togglePlay);
  prevBtn.addEventListener('click', () => setGeneration(Math.max(0, currentGen - 1)));
  nextBtn.addEventListener('click', () => setGeneration(Math.min(simulationData.length - 1, currentGen + 1)));

  populateTable();
  setGeneration(0);
}

function resizeCanvases() {
  const dpr = window.devicePixelRatio || 1;
  const rectM = manifoldCanvas.parentElement.getBoundingClientRect();
  manifoldCanvas.width = rectM.width * dpr;
  manifoldCanvas.height = rectM.height * dpr;
  ctxManifold.scale(dpr, dpr);

  const rectP = persistenceCanvas.parentElement.getBoundingClientRect();
  persistenceCanvas.width = rectP.width * dpr;
  persistenceCanvas.height = rectP.height * dpr;
  ctxPersistence.scale(dpr, dpr);
}

function setGeneration(idx) {
  currentGen = idx;
  genSlider.value = idx;
  const item = simulationData[idx];
  activeGenBadge.textContent = `${item.label}`;

  updateMetrics(item.metrics);
  renderCurrentState();
  updateTableHighlight(idx);
}

function togglePlay() {
  isPlaying = !isPlaying;
  playBtn.textContent = isPlaying ? '⏸ Pause' : '▶ Play Autophagy';
  playBtn.classList.toggle('primary', !isPlaying);

  if (isPlaying) {
    playInterval = setInterval(() => {
      let next = currentGen + 1;
      if (next >= simulationData.length) next = 0;
      setGeneration(next);
    }, 1800);
  } else {
    clearInterval(playInterval);
  }
}

function updateMetrics(m) {
  document.getElementById('m-h1-max').textContent = m.h1_max_lifetime ? m.h1_max_lifetime.toFixed(4) : 'N/A';
  document.getElementById('m-h1-tot').textContent = m.h1_total_persistence ? m.h1_total_persistence.toFixed(4) : 'N/A';
  
  const bDist = m.h1_bottleneck_dist;
  document.getElementById('m-b-dist').textContent = bDist !== null ? bDist.toFixed(4) : 'Baseline (0)';
  
  const fd = m.frechet_distance;
  document.getElementById('m-fd').textContent = fd !== null ? fd.toFixed(4) : 'Baseline (0)';
  
  document.getElementById('m-vendi').textContent = m.vendi_score ? m.vendi_score.toFixed(4) : 'N/A';
  document.getElementById('m-cov').textContent = m.coverage ? (m.coverage * 100).toFixed(1) + '%' : '100.0%';

  // Update dynamic indicator badges
  const bBadge = document.getElementById('b-badge');
  if (currentGen === 0) {
    bBadge.textContent = 'Reference Ground Truth';
    bBadge.className = 'metric-delta delta-neutral';
  } else if (currentGen === 5) {
    bBadge.textContent = '▲ +177% Divergence';
    bBadge.className = 'metric-delta delta-bad';
  } else {
    bBadge.textContent = '▲ Monotonic Growth';
    bBadge.className = 'metric-delta delta-bad';
  }

  const fdBadge = document.getElementById('fd-badge');
  if (currentGen === 0) {
    fdBadge.textContent = 'Reference';
    fdBadge.className = 'metric-delta delta-neutral';
  } else {
    fdBadge.textContent = 'Flat (~0.003, Oblivious)';
    fdBadge.className = 'metric-delta delta-neutral';
  }
}

function renderCurrentState() {
  const item = simulationData[currentGen];
  renderManifold(item.points);
  renderPersistenceDiagram(item.h0_diagram, item.h1_diagram);
}

function renderManifold(points) {
  const w = manifoldCanvas.clientWidth;
  const h = manifoldCanvas.clientHeight;
  ctxManifold.clearRect(0, 0, w, h);

  // Draw grid
  ctxManifold.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctxManifold.lineWidth = 1;
  const step = 40;
  for (let x = 0; x < w; x += step) {
    ctxManifold.beginPath();
    ctxManifold.moveTo(x, 0);
    ctxManifold.lineTo(x, h);
    ctxManifold.stroke();
  }
  for (let y = 0; y < h; y += step) {
    ctxManifold.beginPath();
    ctxManifold.moveTo(0, y);
    ctxManifold.lineTo(w, y);
    ctxManifold.stroke();
  }

  // Find scale
  // Points roughly in range [-1.5, 3.0]
  const centerX = w * 0.42;
  const centerY = h * 0.5;
  const scale = Math.min(w, h) / 4.8;

  const isGen0 = currentGen === 0;
  const ptColor = isGen0 ? 'rgba(16, 185, 129, 0.65)' : 'rgba(168, 85, 247, 0.65)';
  const glowColor = isGen0 ? 'rgba(16, 185, 129, 0.2)' : 'rgba(168, 85, 247, 0.2)';

  points.forEach(pt => {
    const x = centerX + pt[0] * scale;
    const y = centerY - pt[1] * scale;

    ctxManifold.beginPath();
    ctxManifold.arc(x, y, 2.5, 0, 2 * Math.PI);
    ctxManifold.fillStyle = ptColor;
    ctxManifold.fill();
  });
}

function renderPersistenceDiagram(h0, h1) {
  const w = persistenceCanvas.clientWidth;
  const h = persistenceCanvas.clientHeight;
  ctxPersistence.clearRect(0, 0, w, h);

  const margin = 40;
  const plotW = w - 2 * margin;
  const plotH = h - 2 * margin;

  // Find max value
  const maxVal = 1.0;
  const toScreenX = (v) => margin + (v / maxVal) * plotW;
  const toScreenY = (v) => (h - margin) - (v / maxVal) * plotH;

  // Diagonal line y = x
  ctxPersistence.strokeStyle = 'rgba(255, 255, 255, 0.25)';
  ctxPersistence.setLineDash([4, 4]);
  ctxPersistence.beginPath();
  ctxPersistence.moveTo(toScreenX(0), toScreenY(0));
  ctxPersistence.lineTo(toScreenX(maxVal), toScreenY(maxVal));
  ctxPersistence.stroke();
  ctxPersistence.setLineDash([]);

  // Axes
  ctxPersistence.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctxPersistence.lineWidth = 1;
  ctxPersistence.beginPath();
  ctxPersistence.moveTo(margin, h - margin);
  ctxPersistence.lineTo(w - margin, h - margin);
  ctxPersistence.moveTo(margin, h - margin);
  ctxPersistence.lineTo(margin, margin);
  ctxPersistence.stroke();

  // Axis Labels
  ctxPersistence.fillStyle = '#64748b';
  ctxPersistence.font = '11px JetBrains Mono';
  ctxPersistence.fillText('Birth α_b →', w - margin - 60, h - margin + 24);
  ctxPersistence.save();
  ctxPersistence.translate(margin - 16, margin + 60);
  ctxPersistence.rotate(-Math.PI / 2);
  ctxPersistence.fillText('Death α_d →', 0, 0);
  ctxPersistence.restore();

  // Draw H0 points
  h0.forEach(pair => {
    const b = pair[0];
    const d = pair[1];
    if (b <= maxVal && d <= maxVal) {
      ctxPersistence.beginPath();
      ctxPersistence.arc(toScreenX(b), toScreenY(d), 3.0, 0, 2 * Math.PI);
      ctxPersistence.fillStyle = 'rgba(56, 189, 248, 0.6)';
      ctxPersistence.fill();
    }
  });

  // Draw H1 points (crimson triangles)
  h1.forEach(pair => {
    const b = pair[0];
    const d = pair[1];
    if (b <= maxVal && d <= maxVal) {
      const sx = toScreenX(b);
      const sy = toScreenY(d);

      ctxPersistence.beginPath();
      ctxPersistence.moveTo(sx, sy - 5);
      ctxPersistence.lineTo(sx - 4.5, sy + 4);
      ctxPersistence.lineTo(sx + 4.5, sy + 4);
      ctxPersistence.closePath();
      ctxPersistence.fillStyle = '#f43f5e';
      ctxPersistence.fill();
      ctxPersistence.strokeStyle = 'rgba(255, 255, 255, 0.6)';
      ctxPersistence.lineWidth = 1;
      ctxPersistence.stroke();
    }
  });
}

function populateTable() {
  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = '';

  simulationData.forEach((item, idx) => {
    const m = item.metrics;
    const tr = document.createElement('tr');
    tr.id = `row-gen-${idx}`;

    const bDistVal = m.h1_bottleneck_dist !== null ? m.h1_bottleneck_dist.toFixed(4) : '—';
    const fdVal = m.frechet_distance !== null ? m.frechet_distance.toFixed(4) : '—';
    const covVal = m.coverage !== null ? (m.coverage * 100).toFixed(1) + '%' : '100.0%';
    const h1TotVal = m.h1_total_persistence !== null ? m.h1_total_persistence.toFixed(4) : '—';

    // Bar percentage based on max persistence
    const barPct = Math.min(100, ((m.h1_total_persistence || 0) / 2.2) * 100);

    tr.innerHTML = `
      <td><strong>${item.label}</strong></td>
      <td>${m.h1_max_lifetime ? m.h1_max_lifetime.toFixed(4) : '—'}</td>
      <td>
        <div style="display:flex; align-items:center; gap:8px;">
          <span>${h1TotVal}</span>
          <div class="mini-bar-bg" style="width:60px;">
            <div class="mini-bar-fill" style="width:${barPct}%; background:${idx === 0 ? '#10b981' : '#f43f5e'};"></div>
          </div>
        </div>
      </td>
      <td style="color:${m.h1_bottleneck_dist > 0.1 ? '#f43f5e' : 'inherit'}">${bDistVal}</td>
      <td style="color:#94a3b8">${fdVal}</td>
      <td>${covVal}</td>
    `;

    tr.style.cursor = 'pointer';
    tr.addEventListener('click', () => setGeneration(idx));
    tbody.appendChild(tr);
  });
}

function updateTableHighlight(idx) {
  for (let i = 0; i < simulationData.length; i++) {
    const r = document.getElementById(`row-gen-${i}`);
    if (r) {
      r.classList.toggle('highlighted-row', i === idx);
    }
  }
}

// Start
window.addEventListener('DOMContentLoaded', loadSimulationData);

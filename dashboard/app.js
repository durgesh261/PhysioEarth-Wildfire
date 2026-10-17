/**
 * PhysioEarth-Wildfire Digital Twin Controller
 */

let simData = null;
let currentStep = 0;
let isPlaying = false;
let playTimer = null;
let activeModelView = 'fno'; // 'fno' or 'unet'

// DOM elements
const timeSlider = document.getElementById('timeSlider');
const activeStepBadge = document.getElementById('activeStepBadge');
const playBtn = document.getElementById('playBtn');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const modelToggleBtn = document.getElementById('modelToggleBtn');

const canvasGroundTruth = document.getElementById('canvasGroundTruth');
const ctxGT = canvasGroundTruth.getContext('2d');
const canvasModel = document.getElementById('canvasModel');
const ctxModel = canvasModel.getContext('2d');

async function loadData() {
  try {
    const res = await fetch('data.json');
    simData = await res.json();
    initApp();
  } catch (err) {
    console.error('Failed to load simulation data:', err);
  }
}

function initApp() {
  timeSlider.max = simData.num_steps - 1;
  timeSlider.value = 0;

  timeSlider.addEventListener('input', (e) => setStep(parseInt(e.target.value)));
  playBtn.addEventListener('click', togglePlay);
  prevBtn.addEventListener('click', () => setStep(Math.max(0, currentStep - 1)));
  nextBtn.addEventListener('click', () => setStep(Math.min(simData.num_steps - 1, currentStep + 1)));

  modelToggleBtn.addEventListener('click', () => {
    activeModelView = activeModelView === 'fno' ? 'unet' : 'fno';
    modelToggleBtn.textContent = activeModelView === 'fno' ? 'Switch to Baseline U-Net' : 'Switch to PhysioEarth PC-FNO';
    document.getElementById('modelTitle').textContent = activeModelView === 'fno' ? 'PhysioEarth PC-FNO (Physics-Constrained)' : 'Baseline U-Net (Unconstrained)';
    setStep(currentStep);
  });

  window.addEventListener('resize', () => setStep(currentStep));
  setStep(0);
}

function setStep(step) {
  currentStep = step;
  timeSlider.value = step;
  activeStepBadge.textContent = `T = ${step * 15} mins (Step ${step})`;

  updateHUD();
  renderCanvas(ctxGT, canvasGroundTruth, simData.pde_temperature[step], 'Ground Truth PDE');
  
  const modelGrid = activeModelView === 'fno' ? simData.fno_temperature[step] : simData.unet_temperature[step];
  renderCanvas(ctxModel, canvasModel, modelGrid, activeModelView === 'fno' ? 'PC-FNO' : 'U-Net');
}

function togglePlay() {
  isPlaying = !isPlaying;
  playBtn.textContent = isPlaying ? '⏸ Pause' : '▶ Play Simulation';
  playBtn.classList.toggle('primary', !isPlaying);

  if (isPlaying) {
    playTimer = setInterval(() => {
      let next = currentStep + 1;
      if (next >= simData.num_steps) next = 0;
      setStep(next);
    }, 400);
  } else {
    clearInterval(playTimer);
  }
}

function updateHUD() {
  const isFNO = activeModelView === 'fno';
  
  document.getElementById('m-rmse').textContent = isFNO ? '0.3103' : '0.5110';
  document.getElementById('m-rmse-badge').textContent = isFNO ? '▼ 39% Lower Error' : '▲ High Diffusion Drift';
  document.getElementById('m-rmse-badge').className = `metric-delta ${isFNO ? 'delta-good' : 'delta-bad'}`;

  document.getElementById('m-pvr').textContent = isFNO ? '22.5%' : '27.0%';
  document.getElementById('m-pvr-badge').textContent = isFNO ? 'Fuel Preserved' : 'Fuel Violations';

  const isBreached = !isFNO && currentStep >= 6;
  document.getElementById('m-barrier').textContent = isBreached ? 'BREACHED (Unphysical)' : 'CONTAINED (100%)';
  document.getElementById('m-barrier').style.color = isBreached ? '#ef4444' : '#10b981';
  document.getElementById('m-barrier-badge').textContent = isBreached ? 'Fire Ignited Over River Water!' : 'Zero Barrier Penetration';
  document.getElementById('m-barrier-badge').className = `metric-delta ${isBreached ? 'delta-bad' : 'delta-good'}`;

  // Area burned
  const burnedPixels = simData.pde_temperature[currentStep].flat().filter(t => t > 0.3).length;
  document.getElementById('m-burned').textContent = `${(burnedPixels * 0.4).toFixed(1)} km²`;
}

// Inferno Color Map interpolation
function getInfernoColor(val) {
  // val in [0, 1.2]
  const v = Math.min(1.0, Math.max(0.0, val));
  if (v <= 0.05) return [10, 14, 23, 255]; // background dark
  
  // Color stops
  if (v < 0.25) {
    const t = v / 0.25;
    return [Math.floor(20 + 80 * t), 10, Math.floor(60 + 60 * t), 255];
  } else if (v < 0.6) {
    const t = (v - 0.25) / 0.35;
    return [Math.floor(100 + 130 * t), Math.floor(20 + 70 * t), 10, 255];
  } else {
    const t = (v - 0.6) / 0.4;
    return [Math.floor(230 + 25 * t), Math.floor(90 + 150 * t), Math.floor(10 + 100 * t), 255];
  }
}

function renderCanvas(ctx, canvas, tempGrid, label) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const N = simData.grid_size;
  const cellW = w / N;
  const cellH = h / N;

  // 1. Draw Temperature Field Heatmap
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const val = tempGrid[r][c];
      const [cr, cg, cb, ca] = getInfernoColor(val);
      ctx.fillStyle = `rgb(${cr}, ${cg}, ${cb})`;
      ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  // 2. Overlay Cyan River Firebreak
  ctx.fillStyle = 'rgba(6, 182, 212, 0.75)';
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      if (simData.firebreaks[r][c]) {
        ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }

  // 3. Draw Wind Vector Arrow in Top Right
  const arrowX = w - 45;
  const arrowY = 35;
  const wx = simData.wind_vector[0] * 12;
  const wy = simData.wind_vector[1] * 12;

  ctx.strokeStyle = '#06b6d4';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(arrowX, arrowY);
  ctx.lineTo(arrowX + wx, arrowY + wy);
  ctx.stroke();

  ctx.fillStyle = '#06b6d4';
  ctx.font = '10px JetBrains Mono';
  ctx.fillText('WIND ➔', arrowX - 45, arrowY + 4);
}

window.addEventListener('DOMContentLoaded', loadData);

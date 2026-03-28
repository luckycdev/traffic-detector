let currentCamera = null;
let statsTimerId = null;
let consecutiveStatsFailures = 0;
const STATS_SUCCESS_INTERVAL_MS = 200;
const STATS_MAX_BACKOFF_MS = 5000;

function updateVideoSource(cameraName) {
  const streamImage = document.getElementById('video_feed');
  if (!streamImage || !cameraName) return;
  streamImage.src = `/video_feed?camera=${encodeURIComponent(cameraName)}&t=${Date.now()}`;
}

function getCameraFromUrl() {
  const params = new URLSearchParams(window.location.search);
  return params.get('camera');
}

function setCameraInUrl(cameraName) {
  const params = new URLSearchParams(window.location.search);
  params.set('camera', cameraName);
  const nextUrl = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState({}, '', nextUrl);
}

async function loadCameras() {
  try {
    const response = await fetch('/cameras', { cache: 'no-store' });
    if (!response.ok) return;
    const payload = await response.json();
    const cameraSelect = document.getElementById('camera_select');
    cameraSelect.innerHTML = '';

    for (const cameraName of payload.cameras || []) {
      const option = document.createElement('option');
      option.value = cameraName;
      option.textContent = cameraName;
      cameraSelect.appendChild(option);
    }

    const cameraFromUrl = getCameraFromUrl();
    const initialCamera = (payload.cameras || []).includes(cameraFromUrl)
      ? cameraFromUrl
      : payload.selected_camera;

    if (initialCamera) {
      currentCamera = initialCamera;
      cameraSelect.value = initialCamera;
      setCameraInUrl(initialCamera);
      updateVideoSource(initialCamera);
      document.getElementById('camera_status').textContent = `Current: ${initialCamera}`;
    }
  } catch (error) {
    console.error('Failed to load cameras:', error);
  }
}

async function switchCamera() {
  const cameraSelect = document.getElementById('camera_select');
  const selected = cameraSelect.value;
  if (!selected) return;

  currentCamera = selected;
  setCameraInUrl(selected);
  updateVideoSource(selected);
  document.getElementById('camera_status').textContent = `Current: ${selected}`;
}

async function refreshStats() {
  if (!currentCamera) return false;

  try {
    const response = await fetch(`/stats?camera=${encodeURIComponent(currentCamera)}&t=${Date.now()}`, { cache: 'no-store' });
    if (!response.ok) return false;
    const data = await response.json();

    document.getElementById('vehicle_count').textContent = data.vehicle_count;
    document.getElementById('fps').textContent = Number(data.fps || 0).toFixed(2);
    document.getElementById('resolution').textContent = data.resolution || '-';
    document.getElementById('traffic_score').textContent = data.traffic_score.toFixed(2);
    document.getElementById('traffic_label').textContent = data.traffic_label || '-';
    document.getElementById('coverage').textContent = `${data.coverage.toFixed(2)}%`;
    document.getElementById('raw_coverage').textContent = `${data.raw_coverage.toFixed(2)}%`;
    //document.getElementById('road_learning_ready').textContent = data.road_learning_ready ? 'Ready' : 'Not ready';
    document.getElementById('road_learned_percent').textContent = `${data.road_learned_percent.toFixed(2)}%`;
    document.getElementById('last_updated').textContent = data.last_updated || '-';

    const classList = document.getElementById('class_counts');
    classList.innerHTML = '';
    const entries = Object.entries(data.class_counts || {});
    if (!entries.length) {
      const li = document.createElement('li');
      li.className = 'muted';
      li.textContent = 'No vehicles detected in current frame';
      classList.appendChild(li);
    } else {
      entries.sort((a, b) => b[1] - a[1]);
      for (const [name, count] of entries) {
        const li = document.createElement('li');
        li.textContent = `${name}: ${count}`;
        classList.appendChild(li);
      }
    }
    return true;
  } catch (error) {
    return false;
  }
}

function getNextStatsDelay(success) {
  if (success) {
    consecutiveStatsFailures = 0;
    return STATS_SUCCESS_INTERVAL_MS;
  }

  consecutiveStatsFailures += 1;
  const backoff = STATS_SUCCESS_INTERVAL_MS * (2 ** consecutiveStatsFailures);
  return Math.min(backoff, STATS_MAX_BACKOFF_MS);
}

async function scheduleStatsRefresh() {
  const success = await refreshStats();
  const nextDelay = getNextStatsDelay(success);
  statsTimerId = setTimeout(scheduleStatsRefresh, nextDelay);
}

document.getElementById('camera_apply').addEventListener('click', switchCamera);
loadCameras();
if (!statsTimerId) {
  scheduleStatsRefresh();
}

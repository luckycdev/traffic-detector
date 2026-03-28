let currentCamera = null;
let statsTimerId = null;
let consecutiveStatsFailures = 0;
const STATS_SUCCESS_INTERVAL_MS = 200;
const STATS_MAX_BACKOFF_MS = 5000;
let mapInstance = null;
let mapLayer = null;

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

function buildCameraUrl(cameraName) {
  const encoded = encodeURIComponent(cameraName || '').replace(/%20/g, '+');
  return `http://127.0.0.1:5000/?camera=${encoded}`;
}

async function loadMapCameras() {
  try {
    const response = await fetch('/map_cameras', { cache: 'no-store' });
    if (!response.ok) return;
    const payload = await response.json();
    const points = Array.isArray(payload.cameras) ? payload.cameras : [];
    if (!points.length) return;

    if (!mapInstance) {
      mapInstance = L.map('camera_map', { preferCanvas: true }).setView([38.5733, -92.6041], 7);
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
      }).addTo(mapInstance);
      mapLayer = L.layerGroup().addTo(mapInstance);
    }

    mapLayer.clearLayers();
    const bounds = [];
    const renderer = L.canvas({ padding: 0.5 });

    for (const camera of points) {
      if (typeof camera.x !== 'number' || typeof camera.y !== 'number') continue;

      const marker = L.circleMarker([camera.y, camera.x], {
        renderer,
        radius: 5,
        color: '#0f172a',
        weight: 1,
        fillColor: '#22c55e',
        fillOpacity: 0.9
      });

      marker.bindTooltip(camera.location || 'Camera', {
        direction: 'top',
        offset: [0, -6],
        opacity: 0.95
      });

      marker.on('click', function () {
        const target = buildCameraUrl(camera.location || '');
        window.location.href = target;
      });

      marker.addTo(mapLayer);
      bounds.push([camera.y, camera.x]);
    }

    if (bounds.length) {
      mapInstance.fitBounds(bounds, { padding: [20, 20] });
    }
  } catch (error) {
    console.error('Failed to load map cameras:', error);
  }
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
    //document.getElementById('raw_coverage').textContent = `${data.raw_coverage.toFixed(2)}%`;
    //document.getElementById('road_learning_ready').textContent = data.road_learning_ready ? 'Ready' : 'Not ready';
    document.getElementById('road_mask_percent').textContent = `${data.road_mask_percent.toFixed(2)}%`;
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
        if(count == 1) {
          li.textContent = `${count} ${name}`;
        }
        else if(name === 'bus') {
          li.textContent = `${count} ${name}es`;
        }
        else {
          li.textContent = `${count} ${name}s`;
        }
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
loadMapCameras();
if (!statsTimerId) {
  scheduleStatsRefresh();
}

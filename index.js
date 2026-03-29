let currentCamera = null;
let statsTimerId = null;
let consecutiveStatsFailures = 0;
const STATS_SUCCESS_INTERVAL_MS = 200;
const STATS_MAX_BACKOFF_MS = 5000;
let mapInstance = null;
let mapLayer = null;
let mapCameraPoints = [];
let mapCameraByName = {};
let activeVideoCamera = null;
let isMapDarkMode = false;

function setMapTheme(isDarkMode) {
  isMapDarkMode = Boolean(isDarkMode);
  document.body.classList.toggle('map-dark', isMapDarkMode);

  const mapThemeToggle = document.getElementById('map_theme_toggle');
  if (!mapThemeToggle) return;
  mapThemeToggle.textContent = isMapDarkMode ? '☾' : '☼';
  mapThemeToggle.setAttribute('aria-label', isMapDarkMode ? 'Switch map to bright mode' : 'Switch map to dark mode');
  mapThemeToggle.setAttribute('title', isMapDarkMode ? 'Switch map to bright mode' : 'Switch map to dark mode');
}

function initializeMapThemeToggle() {
  const mapThemeToggle = document.getElementById('map_theme_toggle');
  if (!mapThemeToggle) return;

  setMapTheme(false);
  mapThemeToggle.addEventListener('click', function () {
    setMapTheme(!isMapDarkMode);
  });
}

function updateVideoSource(cameraName) {
  const streamImage = document.getElementById('video_feed');
  if (!streamImage || !cameraName) return;
  activeVideoCamera = cameraName;
  streamImage.src = `/video_feed?camera=${encodeURIComponent(cameraName)}&t=${Date.now()}`;
}

function clearVideoSource() {
  const streamImage = document.getElementById('video_feed');
  if (!streamImage) return;
  activeVideoCamera = null;
  streamImage.removeAttribute('src');
}

function setCameraLoadingVisible(isVisible) {
  const cameraLoading = document.getElementById('camera_loading');
  if (!cameraLoading) return;
  cameraLoading.style.display = isVisible ? 'block' : 'none';
}

function getCameraFromUrl() {
  const params = new URLSearchParams(window.location.search);
  return params.get('camera');
}

function setCameraInUrl(cameraName, historyMode = 'replace') {
  const params = new URLSearchParams(window.location.search);
  params.set('camera', cameraName);
  const nextUrl = `${window.location.pathname}?${params.toString()}`;

  if (`${window.location.pathname}${window.location.search}` === nextUrl) {
    return;
  }

  if (historyMode === 'push') {
    window.history.pushState({}, '', nextUrl);
    return;
  }

  if (historyMode === 'replace') {
    window.history.replaceState({}, '', nextUrl);
  }
}

function haversineMiles(lat1, lon1, lat2, lon2) {
  const toRad = Math.PI / 180;
  const dLat = (lat2 - lat1) * toRad;
  const dLon = (lon2 - lon1) * toRad;
  const a = Math.sin(dLat / 2) ** 2
    + Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  const earthRadiusMiles = 3958.8;
  return earthRadiusMiles * c;
}

function getCompassDirection(lat1, lon1, lat2, lon2) {
  const toRad = Math.PI / 180;
  const toDeg = 180 / Math.PI;
  const lat1Rad = lat1 * toRad;
  const lat2Rad = lat2 * toRad;
  const dLon = (lon2 - lon1) * toRad;

  const y = Math.sin(dLon) * Math.cos(lat2Rad);
  const x = Math.cos(lat1Rad) * Math.sin(lat2Rad)
    - Math.sin(lat1Rad) * Math.cos(lat2Rad) * Math.cos(dLon);
  const bearing = (Math.atan2(y, x) * toDeg + 360) % 360;

  const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  const directionIndex = Math.round(bearing / 45) % directions.length;
  return directions[directionIndex];
}

function renderNearbyCameras() {
  const nearbyMeta = document.getElementById('nearby_meta');
  const nearbyList = document.getElementById('nearby_list');
  if (!nearbyMeta || !nearbyList) return;

  nearbyList.innerHTML = '';

  if (!currentCamera || !mapCameraPoints.length) {
    nearbyMeta.textContent = 'Nearby cameras are unavailable yet.';
    return;
  }

  const origin = mapCameraByName[currentCamera];
  if (!origin) {
    nearbyMeta.textContent = 'No map location found for current camera.';
    return;
  }

  const nearest = mapCameraPoints
    .filter((camera) => camera.location !== currentCamera)
    .map((camera) => ({
      camera,
      miles: haversineMiles(origin.y, origin.x, camera.y, camera.x),
      direction: getCompassDirection(origin.y, origin.x, camera.y, camera.x)
    }))
    .sort((a, b) => a.miles - b.miles)
    .slice(0, 5);

  if (!nearest.length) {
    nearbyMeta.textContent = 'No nearby cameras found.';
    return;
  }

  nearbyMeta.textContent = `Nearest to ${currentCamera}`;

  for (const item of nearest) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'nearby-item';
    button.textContent = `${item.camera.location} (${item.direction}, ${item.miles.toFixed(1)} mi)`;
    button.addEventListener('click', function () {
      applyCameraSelection(item.camera.location, { historyMode: 'push' });
    });
    nearbyList.appendChild(button);
  }
}

function applyCameraSelection(cameraName, options = {}) {
  const historyMode = options.historyMode || 'replace';
  if (!cameraName) return;
  currentCamera = cameraName;
  setCameraInUrl(cameraName, historyMode);
  clearVideoSource();

  const cameraSelect = document.getElementById('camera_select');
  if (cameraSelect) {
    cameraSelect.value = cameraName;
  }

  const resolution = document.getElementById('resolution');
  if (resolution) {
    resolution.textContent = '-';
  }
  setCameraLoadingVisible(true);

  const cameraStatus = document.getElementById('camera_status');
  if (cameraStatus) {
    cameraStatus.textContent = `Current: ${cameraName}`;
  }

  renderNearbyCameras();
}

async function loadMapCameras() {
  try {
    const response = await fetch('/map_cameras', { cache: 'no-store' });
    if (!response.ok) return;
    const payload = await response.json();
    const points = Array.isArray(payload.cameras) ? payload.cameras : [];
    if (!points.length) return;
    mapCameraPoints = points;
    mapCameraByName = {};
    for (const point of points) {
      if (point && typeof point.location === 'string') {
        mapCameraByName[point.location] = point;
      }
    }

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
        applyCameraSelection(camera.location || '', { historyMode: 'push' });
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });

      marker.addTo(mapLayer);
      bounds.push([camera.y, camera.x]);
    }

    if (bounds.length) {
      mapInstance.fitBounds(bounds, { padding: [20, 20] });
    }

    renderNearbyCameras();
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
      applyCameraSelection(initialCamera, { historyMode: 'replace' });
    }
  } catch (error) {
    console.error('Failed to load cameras:', error);
  }
}

async function switchCamera() {
  const cameraSelect = document.getElementById('camera_select');
  const selected = cameraSelect.value;
  if (!selected) return;

  applyCameraSelection(selected, { historyMode: 'push' });
}

async function refreshStats() {
  if (!currentCamera) return false;

  try {
    const response = await fetch(`/stats?camera=${encodeURIComponent(currentCamera)}&t=${Date.now()}`, { cache: 'no-store' });
    if (!response.ok) return false;
    const data = await response.json();

    document.getElementById('vehicle_count').textContent = data.vehicle_count;
    document.getElementById('fps').textContent = Number(data.fps || 0).toFixed(2);
    const isResolutionLoading = data.resolution === null || data.resolution === '-';
    const resolutionText = isResolutionLoading ? '-' : (data.resolution || '-');
    document.getElementById('resolution').textContent = resolutionText;
    setCameraLoadingVisible(isResolutionLoading);

    if (!isResolutionLoading && currentCamera && activeVideoCamera !== currentCamera) {
      updateVideoSource(currentCamera);
    }

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

window.addEventListener('popstate', function () {
  const cameraFromUrl = getCameraFromUrl();
  const cameraSelect = document.getElementById('camera_select');
  if (!cameraFromUrl || !cameraSelect) return;

  const exists = Array.from(cameraSelect.options).some(function (option) {
    return option.value === cameraFromUrl;
  });
  if (!exists) return;

  applyCameraSelection(cameraFromUrl, { historyMode: 'none' });
});

document.getElementById('camera_apply').addEventListener('click', switchCamera);
initializeMapThemeToggle();
loadCameras();
loadMapCameras();
if (!statsTimerId) {
  scheduleStatsRefresh();
}

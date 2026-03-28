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

    if (payload.selected_camera) {
      cameraSelect.value = payload.selected_camera;
      document.getElementById('camera_status').textContent = `Current: ${payload.selected_camera}`;
    }
  } catch (error) {
    console.error('Failed to load cameras:', error);
  }
}

async function switchCamera() {
  const cameraSelect = document.getElementById('camera_select');
  const selected = cameraSelect.value;
  if (!selected) return;

  try {
    const response = await fetch('/select_camera', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera: selected }),
      cache: 'no-store'
    });

    const payload = await response.json();
    if (!response.ok) {
      document.getElementById('camera_status').textContent = payload.error || 'Failed to switch camera';
      return;
    }

    document.getElementById('camera_status').textContent = `Current: ${payload.selected_camera}`;
  } catch (error) {
    console.error('Failed to switch camera:', error);
  }
}

async function refreshStats() {
  try {
    const response = await fetch(`/stats?t=${Date.now()}`, { cache: 'no-store' });
    if (!response.ok) return;
    const data = await response.json();

    document.getElementById('vehicle_count').textContent = data.vehicle_count;
    document.getElementById('coverage').textContent = `${data.coverage.toFixed(2)}%`;
    document.getElementById('raw_coverage').textContent = `${data.raw_coverage.toFixed(2)}%`;
    document.getElementById('traffic_score').textContent = data.traffic_score.toFixed(2);
    document.getElementById('traffic_label').textContent = data.traffic_label || '-';
    document.getElementById('road_learning_ready').textContent = data.road_learning_ready ? 'Ready' : 'Not ready';
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
  } catch (error) {
    console.error('Failed to fetch stats:', error);
  }
}

document.getElementById('camera_apply').addEventListener('click', switchCamera);
loadCameras();
document.getElementById('video_feed').src = '/video_feed';
setInterval(refreshStats, 200);
refreshStats();

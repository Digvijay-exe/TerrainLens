from flask import Flask, render_template_string
from datetime import datetime, timezone

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TerrainLens Mobile</title>
<style>
body {
  background: #101820;
  color: #eafaff;
  font-family: Arial, sans-serif;
  max-width: 600px;
  margin: auto;
  padding: 20px;
}
h1 { color: #51e5f2; }
.card {
  background: #1b2a36;
  padding: 16px;
  border-radius: 12px;
  margin: 14px 0;
  overflow-wrap: anywhere;
}
button {
  padding: 12px;
  margin: 5px 3px;
  border: 0;
  border-radius: 8px;
  font-weight: bold;
}
.start { background: #51e5f2; }
.stop { background: #ff8c8c; }
small { color: #a9bdc8; }
</style>
</head>
<body>
<h1>TERRAINLENS</h1>
<p>Mobile Field Sensor Dashboard</p>

<div class="card">
<h3>GPS telemetry</h3>
<p id="status">Ready to start</p>
<p>Latitude: <span id="lat">--</span></p>
<p>Longitude: <span id="lon">--</span></p>
<p>Accuracy: <span id="acc">--</span></p>
<p>Last reading: <span id="time">--</span></p>
</div>

<div class="card">
<button class="start" onclick="startGPS()">Start GPS</button>
<button class="stop" onclick="stopGPS()">Stop GPS</button>
<button onclick="downloadCSV()">Export CSV</button>
<p><small>Readings stay in this browser until exported.
No cloud upload is performed.</small></p>
</div>

<script>
let watchId = null;
let readings = [];

function startGPS() {
  if (!navigator.geolocation) {
    document.getElementById("status").textContent =
      "Geolocation is unavailable in this browser.";
    return;
  }

  if (!window.isSecureContext) {
    document.getElementById("status").textContent =
      "GPS requires HTTPS or a trusted local context. " +
      "We will configure a private connection next.";
    return;
  }

  document.getElementById("status").textContent =
    "Requesting location permission...";

  watchId = navigator.geolocation.watchPosition(
    position => {
      const c = position.coords;
      const row = {
        timestamp: new Date().toISOString(),
        latitude: c.latitude,
        longitude: c.longitude,
        accuracy_m: c.accuracy
      };

      readings.push(row);
      document.getElementById("lat").textContent =
        c.latitude.toFixed(6);
      document.getElementById("lon").textContent =
        c.longitude.toFixed(6);
      document.getElementById("acc").textContent =
        c.accuracy.toFixed(1) + " m";
      document.getElementById("time").textContent =
        row.timestamp;
      document.getElementById("status").textContent =
        "Recording location";
    },
    error => {
      document.getElementById("status").textContent =
        "Location error: " + error.message;
    },
    {enableHighAccuracy: true, maximumAge: 1000, timeout: 15000}
  );
}

function stopGPS() {
  if (watchId !== null) {
    navigator.geolocation.clearWatch(watchId);
    watchId = null;
  }
  document.getElementById("status").textContent = "GPS stopped";
}

function downloadCSV() {
  if (!readings.length) {
    alert("No readings recorded yet.");
    return;
  }

  const header = "timestamp,latitude,longitude,accuracy_m";
  const rows = readings.map(r =>
    [r.timestamp, r.latitude, r.longitude, r.accuracy_m].join(",")
  );

  const blob = new Blob(
    [[header, ...rows].join("\\n")],
    {type: "text/csv"}
  );
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "terrainlens_gps.csv";
  a.click();
  URL.revokeObjectURL(url);
}
</script>
</body>
</html>
"""

@app.get("/")
def dashboard():
    return render_template_string(PAGE)

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8765, debug=False)

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const output = document.getElementById("output");

// Priority color map
const PRIORITY_COLORS = {
  HIGH:   "#ef4444",   // strong red
  MEDIUM: "#f59e0b",   // amber
  LOW:    "#6b7280",   // dim gray-green
};

// One id per tab, so the server keeps separate motion history and alert
// cooldowns for every viewer.
const STREAM_ID = "tab-" + Math.random().toString(36).slice(2, 12);

// Start live camera
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(err => alert("Camera access denied: " + err));

// The server decides WHAT and WHEN to speak (core/alerts.py) and attaches the
// sentence to at most one detection per frame. A HIGH warning interrupts
// whatever is being said; anything else waits for silence.
function announce(detections) {
  const top = detections.find(det => det.announcement);
  if (!top || !("speechSynthesis" in window)) return;

  if (window.speechSynthesis.speaking) {
    if (top.priority !== "HIGH") return;
    window.speechSynthesis.cancel();
  }
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(top.announcement));
}

function drawDetections(detections) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  detections.forEach(det => {
    const [x1, y1, x2, y2] = det.bbox;
    const color = PRIORITY_COLORS[det.priority] || PRIORITY_COLORS.LOW;
    const priorityTag = det.priority || "";
    const rankTag = det.rank === 1 ? " [Primary]" : "";
    const distance = det.smoothed_distance_m ?? det.distance_m;
    const movementTag = det.movement === "approaching" ? " ↓" : "";

    // Box (the primary hazard gets a thicker outline)
    ctx.strokeStyle = color;
    ctx.lineWidth = det.rank === 1 ? 4 : 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Label background for readability
    const label = `${det.object} | ${det.direction} | ${distance != null ? distance + "m" : "?"}${movementTag} | ${priorityTag} ${det.risk_score ?? ""}${rankTag}`;
    ctx.font = "bold 14px Arial";
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fillRect(x1, y1 - 20, textWidth + 8, 20);

    // Label text
    ctx.fillStyle = color;
    ctx.fillText(label, x1 + 4, y1 - 5);
  });
}

// Send frame to backend every 300ms
setInterval(() => {
  if (video.readyState !== 4) return;

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;
  tempCanvas.getContext("2d").drawImage(video, 0, 0);

  tempCanvas.toBlob(blob => {
    const formData = new FormData();
    formData.append("frame", blob);
    formData.append("stream_id", STREAM_ID);

    fetch("/detect", { method: "POST", body: formData })
      .then(res => res.json())
      .then(data => {
        if (!Array.isArray(data)) return;
        drawDetections(data);
        announce(data);
        output.textContent = JSON.stringify(data, null, 2);
      })
      .catch(err => console.error(err));
  }, "image/jpeg");
}, 300);

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const output = document.getElementById("output");

// Start live camera
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream)
  .catch(err => alert("Camera access denied: " + err));

function drawDetections(detections) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  detections.forEach(det => {
    const [x1, y1, x2, y2] = det.bbox;
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.fillStyle = "lime";
    ctx.font = "16px Arial";
    ctx.fillText(`${det.object} | ${det.direction} | ${det.distance_m}m`, x1, y1 - 5);
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

    fetch("/detect", { method: "POST", body: formData })
      .then(res => res.json())
      .then(data => {
        drawDetections(data);
        output.textContent = JSON.stringify(data, null, 2);
      })
      .catch(err => console.error(err));
  }, "image/jpeg");
}, 300);

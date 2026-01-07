const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  });

setInterval(() => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  sendFrame();
}, 700);

function sendFrame() {
  const imageData = canvas.toDataURL("image/jpeg");

  fetch("http://127.0.0.1:5501/detect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageData })
  })
    .then(res => res.json())
    .then(drawResults);
}

function drawResults(objects) {
  ctx.strokeStyle = "red";
  ctx.fillStyle = "yellow";
  ctx.font = "16px Arial";

  objects.forEach(obj => {
    const [x, y, w, h] = obj.bbox;

    ctx.strokeRect(x, y, w, h);
    ctx.fillText(
      `${obj.label} | ${obj.distance}m | ${obj.direction}`,
      x, y - 10
    );

    speak(`${obj.label} ${obj.distance} meters ${obj.direction}`);
  });
}

function speak(text) {
  const msg = new SpeechSynthesisUtterance(text);
  window.speechSynthesis.speak(msg);
}

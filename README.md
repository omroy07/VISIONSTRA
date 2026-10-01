# VisionSTRA 🚦

**VisionSTRA** is a browser-based, privacy-first AI road safety assistant designed to help visually impaired users navigate streets independently and safely using real-time computer vision and multimodal feedback.

---

## 🌟 Key Features
- **Real-time Detection**: Vehicles, pedestrians, obstacles, and traffic signals  
- **Browser-Based**: Runs fully in the browser—no app installation  
- **Privacy-First**: Camera data processed locally; no video uploads  
- **Multimodal Alerts**: Voice, visual, and vibration feedback  
- **Lightweight & Fast**: Optimized for low-end smartphones  
- **Accessibility-Focused**: Inclusive UI aligned with WCAG principles  

---

## 🎯 Problem We Solve
Visually impaired individuals often face unsafe and dependent road navigation due to limited real-time awareness. VisionSTRA transforms a device camera into an intelligent safety companion.

---

## 👥 Target Users
**Primary**
- Visually impaired individuals (partial or complete blindness)  
- Age: 15–60 years  
- Urban & semi-urban smartphone users  

**Secondary**
- Elderly users with age-related vision loss  
- NGOs & rehabilitation centers  
- Families and caregivers  

---

## 🧠 How It Works
1. Open VisionSTRA in a supported browser  
2. Allow camera access  
3. AI analyzes the live feed locally  
4. Hazards are detected in real time  
5. Alerts are delivered via voice/visual cues  

---

## 🚨 Intelligent Hazard Prioritization
When several objects are in view, VisionSTRA ranks them instead of announcing all of them:
- Every object gets a **risk score (0–100)** from its **type, distance, position, movement and detection confidence**, and a **LOW / MEDIUM / HIGH** priority with a plain-English reason
- Movement and **time-to-contact** come from tracking each object across frames, so a fast-approaching car is HIGH even when it is still far away
- Only the **single most important** hazard is spoken, with cooldowns so the voice stays calm
- Every weight and threshold is configurable (`backend/core/hazard_config.py`)

Methodology, weights, assumptions and limitations: [docs/HAZARD_SCORING.md](docs/HAZARD_SCORING.md). Run the tests with:
```bash
python -m unittest discover -s backend/tests -v
```

---

## 🛠️ Tech Stack
- **Frontend**: HTML, Tailwind CSS, JavaScript  
- **AI/ML**: Computer Vision (object detection)  
- **APIs**: Browser Camera APIs, Web Audio  
- **Hosting**: Vercel  

---

## 🌍 Diversity & Inclusion
Built with inclusive design principles to support diverse visual abilities, ages, devices, and environments—empowering independence with dignity through multimodal feedback.

---

## 🚀 Getting Started
1. Open the website in a modern browser  
2. Allow camera permissions  
3. Start real-time detection  

---

## 📈 Roadmap
- Directional & distance-based alerts  
- Offline mode optimization  
- NGO & city-level pilots  
- Wearable device integration  

---

---

## 🔬 Model Evaluation & Benchmarks
Toolkit to benchmark object detection (CV) and LLM responses (`evalkit/`, `compare.py`, `run_cv.py`):
- Run CV benchmarks: `python run_cv.py --name yolov8n-nms0.7`
- Compare runs: `python compare.py results/cv_yolov8n-nms0.7.json results/cv_yolov8n-nms0.5.json`
- Methodology: [docs/methodology.md](docs/methodology.md)

---

## 👨‍💻 Team
- **Om Roy** – Founder, Team Lead & ML Engineer  
- **Anshika** – ML & Frontend Engineer  
- **Shubhangi** – Pitch & Visual Storytelling Lead  
- **Anushika** – Research & Growth Lead  

---

## 📄 License
For educational, research, and social impact purposes.

---

**VisionSTRA — Where Vision Meets Intelligence.**


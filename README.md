# FlowState | Bio-Canvas  
A biometric-driven creative workspace that visualizes flow states in real time.

---

## 📖 Overview
**FlowState** is a hackathon project that blends biometric data with interactive creative tools. Using your webcam and a Python bridge, the app estimates heart rate and breathing rate, streams them into a React/Tailwind frontend, and visualizes your “flow state” as a dynamic orb. The dashboard provides timers, notes, and AI-powered insights to help users optimize focus and creativity.

---

## ✨ Features
- **Real-time Flow Visualization**  
  Dynamic orb that changes color based on biometric data (heart rate, breathing, flow score).

- **Biometric Bridge**  
  Python script (`sensor_bridge.py`) captures webcam signals and writes live JSON (`received_live.json`).

- **Interactive Canvas**  
  Drag-and-drop notes with color-coded organization for brainstorming and planning.

- **Timer System**  
  Stopwatch and countdown modes, integrated with biometric tracking.

- **Flow Analytics**  
  Session history, “flow DNA” visualization, and metrics cards for heart rate, breathing rate, and flow score.

- **AI Insights (Optional)**  
  Hooks for integrating AI suggestions (e.g., breathing techniques, productivity nudges).

---

## 🚀 Getting Started

### 1. Clone the repo
```
bash
git clone <your-repo-url>
cd FlowState/frontend
```

### 2. Set Python environment
```
python -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell
pip install opencv-python scipy flask
```

### 3. Run the biometric bridge

### 4. Serve the frontend

### 5. (Optional) Control bridge via Flask

## Tech Stack
Frontend: React 18, Tailwind CSS, JavaScript/HTML/CSS

Backend Bridge: Python (OpenCV, SciPy, Flask)

Data Flow: JSON served via local HTTP server

AI (Optional): Gemini API or other integrations for insights

## Project Structure

FlowState/
├── frontend/
│   ├── index.html            # Main dashboard
│   ├── DashboardView.jsx     # React component for metrics & orb
│   ├── sensor_bridge.py      # Webcam → biometric JSON bridge
│   ├── bridge_server.py      # Flask controller for start/stop
│   ├── received_live.json    # Live biometric data
│   └── venv/                 # Virtual environment
└── docs/                     # Notes, design drafts

## Hackathon Vision
FlowState was built to demonstrate how biometric feedback can enhance creative work. By visualizing heart rate and breathing in real time, users gain awareness of their focus levels and can adjust their workflow to stay in the “flow zone.”


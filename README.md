# 👁️ Smart Vision — Face Detection System

A real-time, multi-model face detection pipeline built with Python and modern
computer vision libraries. Detects human faces from live webcam feeds, uploaded
videos, and static images — with automated attendance logging, snapshot capture,
and an interactive Streamlit dashboard.

---

## 🚀 Features

- **3 detection models** — Haar Cascade · HOG+SVM · CNN (dlib ResNet)
- **Multi-source input** — Live webcam · Video files · Static images
- **Auto attendance logging** — CSV with timestamps & de-duplication
- **Annotated output** — Bounding boxes · Confidence scores · Face count
- **Streamlit dashboard** — Live feed · Stats · Download reports
- **Matplotlib reports** — Bar & pie charts for detection analytics
- **Modular architecture** — Each stage independently replaceable

---

## 🛠️ Tech Stack

| Layer        | Tools                                         |
|--------------|-----------------------------------------------|
| Language     | Python 3.9+                                   |
| CV Engine    | OpenCV 4.8+                                   |
| Detection    | face_recognition · dlib · MediaPipe           |
| Data         | NumPy · Pandas                                |
| Visualization| Matplotlib                                    |
| Dashboard    | Streamlit                                     |

---

## 📦 Detection Models

| Model         | Speed (CPU)  | Accuracy | Best For                  |
|---------------|-------------|----------|---------------------------|
| Haar Cascade  | 30–60 FPS   | ~85%     | IoT / Edge devices        |
| HOG + SVM     | 15–25 FPS   | ~92%     | General deployment        |
| CNN (dlib)    | 5–12 FPS    | ~97%     | High-security systems     |

---

## 📁 Project Structure

```
smart_vision/
├── app.py                  # Streamlit dashboard entry point
├── requirements.txt
├── modules/
│   ├── detector.py         # Face detection engine (Haar/HOG/CNN)
│   ├── preprocessor.py     # Frame preprocessing pipeline
│   ├── annotator.py        # Bounding box drawing & snapshot saver
│   ├── attendance.py       # CSV attendance logger with de-duplication
│   └── reporter.py         # Matplotlib chart & CSV report generator
├── utils/
│   └── helpers.py          # Shared utility functions
├── logs/                   # attendance.csv stored here
├── snapshots/              # Saved face crop images
└── reports/                # Generated charts and report CSVs
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/smart-vision-face-detection.git
cd smart-vision-face-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

> **Note:** Installing `dlib` requires CMake. On Ubuntu: `sudo apt install cmake`.
> On Windows, use a pre-built wheel from https://github.com/jloh02/dlib.

---

## 🖥️ Usage

1. Open the Streamlit dashboard in your browser (default: `http://localhost:8501`)
2. Select a **Detection Model** from the sidebar
3. Choose an **Input Source**: Webcam / Video / Image
4. Click **▶️ Start Detection**
5. View live annotations, face count, and attendance log in real time
6. Click **📥 Generate Report** to download a CSV + charts

---

## 📂 Dataset

| Dataset              | Images      | Faces       | Source                              |
|----------------------|-------------|-------------|-------------------------------------|
| LFW                  | 13,233      | 13,233      | http://vis-www.cs.umass.edu/lfw/    |
| WIDER FACE           | 32,203      | 393,703     | http://shuoyang1213.me/WIDERFACE/   |
| Custom (Webcam)      | ~500        | ~500        | Self-collected                      |

---

## 🔮 Future Scope

- Face recognition (named identification)
- Emotion detection (happy / sad / angry / neutral)
- Age & gender estimation
- Edge deployment (Jetson Nano, Raspberry Pi)
- Anti-spoofing / liveness detection
- Multi-camera cloud scaling

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

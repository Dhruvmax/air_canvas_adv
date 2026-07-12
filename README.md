# 🎨 Air Canvas Pro

Air Canvas Pro is a browser-native, real-time computer vision drawing application built with **Streamlit**, **MediaPipe**, and **OpenCV**. It transitions legacy desktop hand-tracking paint apps into a high-performance web experience, allowing users to draw in the air using hand gestures, write text interactively, or draw using their mouse.

---

## ✨ Features

- **🎥 Real-Time WebRTC Streaming:** Zero-latency camera capture processed directly inside the browser using `streamlit-webrtc`.
- **🖐️ Hand Gesture Drawing:** 
  - **Draw Mode:** Pinch your Index Finger and Thumb together to draw lines in the air.
  - **Hover Mode:** Raise your Index and Middle fingers to move the cursor without drawing, letting you select colors or brush sizes.
- **🖱️ Interactive Mouse Drawing:** Seamlessly switch to mouse-based freehand drawing on the canvas with auto-merge features.
- **📝 Click-to-Place Text:** Click anywhere on the canvas preview to instantly place custom text. Customize font size, color, and stroke thickness.
- **🎨 Creative Toolkit:** Adjust brush size, toggle the eraser, pick custom colors (Blue, Green, Red, Yellow), or clear the canvas with a single click.
- **💎 Premium Design System:** Dark-mode glassmorphic interface, smooth hover animations, and custom typography for a modern user experience.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python **3.10** or **3.11** (Recommended)
- Webcam (internal or external)

### Local installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dhruvmax/air_canvas_adv.git
   cd air_canvas_adv
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment on Streamlit Cloud

To deploy this app on **Streamlit Community Cloud**, make sure your environment configuration is set up correctly:

1. **Python Version:** Make sure to select **Python 3.11** (or **3.10**) in the **Advanced Settings** of your Streamlit deployment panel. Version 3.14+ is unsupported by MediaPipe and will cause dependency conflicts.
2. **System Dependencies:** The project requires graphical libraries at the OS level. The included `packages.txt` ensures that `libgl1` and `libglib2.0-0t64` are installed on the Debian-based cloud container.
3. **Compatibility Adapters:** A custom shim is included inside `app.py` to adapt the legacy `image_to_url` API used by `streamlit-drawable-canvas` to the modern Streamlit layout engine.

---

## 📂 Project Structure

```
├── app.py                     # Main Streamlit web application & WebRTC handlers
├── requirements.txt           # Python dependency declarations with version pins
├── packages.txt               # Debian-level system packages (for OpenCV/MediaPipe)
├── runtime.txt                # Python environment runtime override (Python 3.11)
└── README.md                  # Project documentation
```

---

## 🔒 Security & Performance

- **Thread Safety:** Uses mutex locking (`threading.Lock()`) for synchronizing canvas state mutations between WebRTC background worker threads and the main Streamlit UI process.
- **Resource Optimization:** Pinned to `opencv-python-headless` to eliminate bloated GUI library overlays, keeping memory footprints low during cloud hosting.

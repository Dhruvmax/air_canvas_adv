import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
import threading
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Air Canvas Pro", page_icon="🎨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
}
h1,h2,h3,label,.stMarkdown p { color: white !important; }
.stButton>button {
    width:100%; border-radius:10px;
    background: rgba(102,126,234,0.25);
    border: 1px solid rgba(102,126,234,0.5);
    color: white; transition: all 0.3s;
}
.stButton>button:hover { background: rgba(102,126,234,0.5); transform: translateY(-2px); }
.info-box {
    background: rgba(102,126,234,0.15);
    border-left: 3px solid #667eea;
    border-radius: 0 10px 10px 0;
    padding: 10px 14px; margin: 8px 0;
    color: rgba(255,255,255,0.9) !important;
    font-size: 0.85rem;
}
.stRadio label, .stSelectbox label, .stSlider label { color: white !important; }
div[data-testid="stRadio"] div label p { color: white !important; }
</style>
""", unsafe_allow_html=True)

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

COLORS = {
    "🔴 Red":    (0,   0,   255),
    "🟢 Green":  (0,   200, 0  ),
    "🔵 Blue":   (255, 50,  0  ),
    "🟡 Yellow": (0,   230, 255),
    "⚫ Black":  (10,  10,  10 ),
    "🟣 Purple": (200, 0,   200),
    "🟠 Orange": (0,   140, 255),
    "🩷 Pink":   (180, 105, 255),
}

MODE_INFO = {
    "Hand Gesture": "✋ Raise your index finger to draw. Move it around the frame.",
    "Eye Tracking":  "👁️ Your right eye position controls the cursor.",
    "Pen Mode":      "✏️ Pinch thumb & index finger together to draw. Release to stop.",
}


class AppState:
    """Thread-safe shared state between Streamlit UI and WebRTC callback."""

    def __init__(self):
        self.canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        self.prev_x = None
        self.prev_y = None
        self.mode = "Hand Gesture"
        self.color = (0, 0, 255)
        self.brush_size = 5
        self.eraser = False
        self.lock = threading.Lock()
        self._init_mp()

    def _init_mp(self):
        mp_face = mp.solutions.face_mesh
        self.face_mesh = mp_face.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_h = mp.solutions.hands
        self.mp_hands = mp_h
        self.hands = mp_h.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def clear(self):
        with self.lock:
            self.canvas[:] = 255
            self.prev_x = self.prev_y = None

    def get_canvas(self):
        with self.lock:
            return self.canvas.copy()

    def process(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(cv2.resize(img, (640, 480)), 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with self.lock:
            mode = self.mode
            color = (255, 255, 255) if self.eraser else self.color
            bsize = 20 if self.eraser else self.brush_size

        x, y, should_draw = -1, -1, False

        if mode == "Eye Tracking":
            try:
                res = self.face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark[133]
                    x, y = int(lm.x * 640), int(lm.y * 480)
                    should_draw = True
            except Exception:
                pass

        elif mode == "Hand Gesture":
            res = self.hands.process(rgb)
            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hl, self.mp_hands.HAND_CONNECTIONS)
                    tip = hl.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x, y = int(tip.x * 640), int(tip.y * 480)
                    should_draw = True

        elif mode == "Pen Mode":
            res = self.hands.process(rgb)
            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hl, self.mp_hands.HAND_CONNECTIONS)
                    th = hl.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                    ix_lm = hl.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    tx, ty = int(th.x * 640), int(th.y * 480)
                    ix2, iy2 = int(ix_lm.x * 640), int(ix_lm.y * 480)
                    dist = np.hypot(tx - ix2, ty - iy2)
                    x, y = ix2, iy2
                    should_draw = dist < 50

        with self.lock:
            if x != -1 and y != -1:
                cv2.circle(img, (x, y), 10, (0, 255, 255), -1)
                if mode == "Pen Mode":
                    if should_draw:
                        cv2.circle(self.canvas, (x, y), bsize, color, -1)
                    self.prev_x, self.prev_y = x, y
                else:
                    if should_draw and self.prev_x is not None:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), color, bsize)
                    self.prev_x = x if should_draw else None
                    self.prev_y = y if should_draw else None
            else:
                self.prev_x = self.prev_y = None

            # Overlay drawings on webcam
            gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            bg = cv2.bitwise_and(img, img, mask=mask_inv)
            fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
            out = cv2.add(bg, fg)

        return av.VideoFrame.from_ndarray(out, format="bgr24")


# ── Session state ─────────────────────────────────────────────────────────────
if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()
state: AppState = st.session_state.app_state

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎨 Air Canvas Pro")
    st.markdown("---")

    st.markdown("### 🖱️ Drawing Mode")
    mode = st.radio("", list(MODE_INFO.keys()), key="mode_radio")
    state.mode = mode
    st.markdown(f'<div class="info-box">{MODE_INFO[mode]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎨 Color")
    color_name = st.selectbox("Pick a color", list(COLORS.keys()), key="color_sel")
    state.color = COLORS[color_name]
    st.markdown(f'<div style="width:100%;height:30px;border-radius:8px;'
                f'background:rgb{tuple(reversed(COLORS[color_name]))};'
                f'border:1px solid rgba(255,255,255,0.2);"></div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ✏️ Brush Size")
    state.brush_size = st.slider("", 2, 25, 5, key="brush_sl")

    st.markdown("---")
    eraser = st.toggle("🧹 Eraser Mode", key="eraser_tog")
    state.eraser = eraser

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear"):
            state.clear()
    with col2:
        canvas_img = state.get_canvas()
        pil_img = Image.fromarray(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        st.download_button("💾 Save", buf.getvalue(), "canvas.png", "image/png")

    st.markdown("---")
    st.markdown('<p style="font-size:0.75rem;opacity:0.5;text-align:center;">Air Canvas Pro • Streamlit Edition</p>',
                unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<h1 style="text-align:center;background:linear-gradient(90deg,#667eea,#764ba2);'
            '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
            'font-size:2.2rem;margin-bottom:5px;">🎨 Air Canvas Pro</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;opacity:0.6;margin-bottom:20px;">'
            'Draw in the air using your hands or eyes — powered by MediaPipe</p>',
            unsafe_allow_html=True)

col_cam, col_canvas = st.columns([3, 2])

with col_cam:
    st.markdown("#### 📷 Live Camera")
    # Pre-initialize key to avoid KeyError on rerun (streamlit-webrtc bug)
    if "air-canvas" not in st.session_state:
        st.session_state["air-canvas"] = None
    webrtc_ctx = webrtc_streamer(
        key="air-canvas",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=state.process,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_canvas:
    st.markdown("#### 🖼️ Canvas Preview")
    canvas_placeholder = st.empty()
    canvas_img = state.get_canvas()
    pil_canvas = Image.fromarray(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB))
    canvas_placeholder.image(pil_canvas, use_container_width=True)

# Instructions
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="info-box">
    <b>✋ Hand Gesture Mode</b><br>
    Raise your index finger and move it to draw continuously on the canvas.
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="info-box">
    <b>👁️ Eye Tracking Mode</b><br>
    Your right eye landmark position controls the cursor and draws on the canvas.
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="info-box">
    <b>✏️ Pen Mode</b><br>
    Pinch your thumb and index finger together to activate drawing. Release to lift the pen.
    </div>""", unsafe_allow_html=True)

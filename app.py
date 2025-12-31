import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import time

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MODEL_PATH = "asl_model.keras"
LABEL_PATH = "class_labels.json"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(LABEL_PATH, "r") as f:
        class_indices = json.load(f)

    labels = {v: k for k, v in class_indices.items()}
    return model, labels

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="ASL Recognition", layout="wide")
st.title("🤟 ASL Alphabet Recognition")
st.caption("Automatic hand detection • Real-time prediction")

col1, col2 = st.columns([3, 1])

with col1:
    frame_placeholder = st.empty()

with col2:
    letter_box = st.metric("Predicted Letter", "—")
    confidence_box = st.metric("Confidence", "—")

start = st.button("▶ Start Camera")
stop = st.button("⏹ Stop Camera")

if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

# ---------------- CAMERA LOOP ----------------
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        predicted_letter = "—"
        confidence = 0.0

        if result.multi_hand_landmarks:
            h, w, _ = frame.shape
            lm = result.multi_hand_landmarks[0]

            xs = [int(p.x * w) for p in lm.landmark]
            ys = [int(p.y * h) for p in lm.landmark]

            x1, x2 = max(0, min(xs)-20), min(w, max(xs)+20)
            y1, y2 = max(0, min(ys)-20), min(h, max(ys)+20)

            hand = frame[y1:y2, x1:x2]

            if hand.size > 0:
                hand = cv2.resize(hand, (IMG_SIZE, IMG_SIZE))
                hand = hand.astype("float32") / 255.0
                hand = np.expand_dims(hand, axis=0)

                preds = model.predict(hand, verbose=0)
                idx = np.argmax(preds)
                predicted_letter = labels[idx]
                confidence = float(preds[0][idx])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{predicted_letter} ({confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

        frame_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

        letter_box.metric("Predicted Letter", predicted_letter)
        confidence_box.metric("Confidence", f"{confidence:.1%}")

        time.sleep(0.03)

    cap.release()

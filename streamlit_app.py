import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import os
from PIL import Image
import time

class ASLDetector:
    def __init__(self, model_path='asl_model.h5', labels_path='class_labels.json'):
        if os.path.exists(model_path) and os.path.exists(labels_path):
            self.model = tf.keras.models.load_model(model_path)
            with open(labels_path, 'r') as f:
                class_indices = json.load(f)
            self.labels = {v: k for k, v in class_indices.items()}
        else:
            self.model = None
            self.labels = None
        
    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (64, 64))
        frame = frame.astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=0)
        return frame
    
    def predict(self, frame):
        if self.model is None:
            return "No Model", 0.0
        processed_frame = self.preprocess_frame(frame)
        predictions = self.model.predict(processed_frame, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        return self.labels[predicted_class], confidence

def main():
    st.set_page_config(page_title="ASL Recognition", layout="wide")
    
    st.title("🤟 ASL Alphabet Recognition")
    
    if not os.path.exists('asl_model.h5'):
        st.error("Model not found! Please train the model first: `python train_model.py`")
        return
    
    detector = ASLDetector()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Camera Feed")
        camera_placeholder = st.empty()
        
    with col2:
        st.subheader("Detection")
        letter_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
    start_btn = st.button("Start Camera")
    stop_btn = st.button("Stop Camera")
    
    if start_btn:
        st.session_state.camera_running = True
        
    if stop_btn:
        st.session_state.camera_running = False
        
    if st.session_state.get('camera_running', False):
        cap = cv2.VideoCapture(0)
        
        while st.session_state.get('camera_running', False):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            roi = frame[100:400, 100:400]
            cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
            
            predicted_letter, confidence = detector.predict(roi)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, use_column_width=True)
            
            with letter_placeholder.container():
                st.metric("Letter", predicted_letter, delta=None)
            with confidence_placeholder.container():
                st.metric("Confidence", f"{confidence:.1%}")
            
            time.sleep(0.1)
        
        cap.release()

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

class ASLDetector:
    def __init__(self, model_path='asl_model.h5', labels_path='class_labels.json'):
        self.model = keras.models.load_model(model_path)
        
        with open(labels_path, 'r') as f:
            class_indices = json.load(f)
        
        # Reverse the dictionary to get index -> label mapping
        self.labels = {v: k for k, v in class_indices.items()}
        
    def preprocess_frame(self, frame):
        # Resize and normalize
        frame = cv2.resize(frame, (64, 64))
        frame = frame.astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=0)
        return frame
    
    def predict(self, frame):
        processed_frame = self.preprocess_frame(frame)
        predictions = self.model.predict(processed_frame, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        return self.labels[predicted_class], confidence

def main():
    detector = ASLDetector()
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Define ROI for hand detection
        roi = frame[100:400, 100:400]
        
        # Draw rectangle around ROI
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        
        # Make prediction
        try:
            predicted_letter, confidence = detector.predict(roi)
            
            # Display prediction
            text = f"{predicted_letter}: {confidence:.2f}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        except Exception as e:
            cv2.putText(frame, "Loading...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('ASL Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
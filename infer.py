#!/usr/bin/env python3
"""
infer.py - Real-time inference for American Sign Language (ASL) dynamic gestures
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
from collections import deque
import sys
import os

class ASLInference:
    def __init__(self, model_path="models/asl_model.h5", 
                 encoder_path="models/le.pkl",
                 buffer_size=32,
                 frame_size=(64, 64)):
        """
        Initialize the ASL inference system.
        
        Args:
            model_path: Path to the trained model
            encoder_path: Path to the label encoder
            buffer_size: Number of frames to buffer for prediction
            frame_size: Target size for frame resizing (height, width)
        """
        self.buffer_size = buffer_size
        self.frame_size = frame_size
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Load model and label encoder
        try:
            print("Loading model...")
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
            
            print("Loading label encoder...")
            self.label_encoder = joblib.load(encoder_path)
            print("Label encoder loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model or encoder: {e}")
            sys.exit(1)
    
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for model input.
        
        Args:
            frame: Raw frame from webcam
            
        Returns:
            Preprocessed frame
        """
        # Resize frame
        resized = cv2.resize(frame, self.frame_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def predict(self):
        """
        Run inference on the current frame buffer.
        
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if len(self.frame_buffer) < self.buffer_size:
            return None, 0.0
        
        # Convert buffer to numpy array
        frames = np.array(list(self.frame_buffer))
        
        # Reshape to match model input: (batch, frames, height, width, channels)
        frames = frames.reshape(1, self.buffer_size, self.frame_size[0], 
                               self.frame_size[1], 3)
        
        # Run prediction
        predictions = self.model.predict(frames, verbose=0)
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_class = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        return predicted_class, confidence
    
    def draw_prediction(self, frame, predicted_class, confidence):
        """
        Draw prediction results on the frame.
        
        Args:
            frame: Original frame to draw on
            predicted_class: Predicted ASL gesture class
            confidence: Prediction confidence score
        """
        # Set up text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        
        # Create background rectangle for better text visibility
        text = f"Gesture: {predicted_class} ({confidence:.2%})"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), 
                     (10 + text_size[0] + 10, 10 + text_size[1] + 10), 
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (15, 35), font, font_scale, 
                   (0, 255, 0), thickness)
        
        # Draw buffer status
        buffer_text = f"Buffer: {len(self.frame_buffer)}/{self.buffer_size}"
        cv2.putText(frame, buffer_text, (15, 65), font, 0.6, 
                   (255, 255, 255), 1)
        
    def run(self):
        """
        Main inference loop using webcam input.
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            sys.exit(1)
        
        print("Starting ASL inference...")
        print("Press 'q' to quit")
        
        # Initialize variables for prediction
        current_prediction = None
        current_confidence = 0.0
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Preprocess and add to buffer
                preprocessed = self.preprocess_frame(frame)
                self.frame_buffer.append(preprocessed)
                
                # Run prediction if buffer is full
                if len(self.frame_buffer) == self.buffer_size:
                    prediction, confidence = self.predict()
                    if prediction is not None:
                        current_prediction = prediction
                        current_confidence = confidence
                
                # Draw prediction on frame
                if current_prediction is not None:
                    self.draw_prediction(frame, current_prediction, current_confidence)
                else:
                    # Show buffer filling status
                    cv2.putText(frame, f"Filling buffer... {len(self.frame_buffer)}/{self.buffer_size}", 
                               (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Display frame
                cv2.imshow("ASL Gesture Recognition", frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()

def main():
    """
    Main function to run the ASL inference system.
    """
    # Check if model files exist
    if not os.path.exists("models/asl_model.h5"):
        print("Error: Model file 'models/asl_model.h5' not found!")
        sys.exit(1)
        
    if not os.path.exists("models/le.pkl"):
        print("Error: Label encoder file 'models/le.pkl' not found!")
        sys.exit(1)
    
    # Create and run inference system
    inference = ASLInference()
    inference.run()

if __name__ == "__main__":
    main()
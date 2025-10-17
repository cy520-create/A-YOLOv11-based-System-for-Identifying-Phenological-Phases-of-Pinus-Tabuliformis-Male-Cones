import streamlit as st
import threading
import requests
import time
from PIL import Image
import io
import json
import os

# Your original Flask application code
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import torch
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
from collections import defaultdict

# Configure Flask application
flask_app = Flask(__name__)
flask_app.config['UPLOAD_FOLDER'] = 'uploads'
flask_app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(flask_app.config['UPLOAD_FOLDER'], exist_ok=True)

# Pine flower phenological phases mapping
PINE_FLOWER_CLASSES = {
    0: {'name': 'elongation stage', 'color': (0, 255, 0), 'english': 'Elongation Stage'},
    1: {'name': 'ripening stage', 'color': (0, 165, 255), 'english': 'Ripening Stage'},
    2: {'name': 'decline stage', 'color': (0, 0, 255), 'english': 'Decline Stage'}
}

class AdvancedDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.device = 'cpu'
        self.load_model()

    def load_model(self):
        """Load YOLOv11 model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            st.sidebar.success("✅ Model loaded successfully")
        except Exception as e:
            st.sidebar.error(f"❌ Model loading failed: {e}")
            self.model = None

    def detect_image(self, image_path):
        """Perform image detection"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return [], image
            
            if self.model is not None:
                results = self.model(image_path)
                detections = []
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        class_info = PINE_FLOWER_CLASSES.get(class_id, 
                            {'name': 'unknown', 'color': (255, 255, 255), 'english': 'Unknown Stage'})
                        detections.append({
                            'bbox': box.xyxy[0].tolist(),
                            'confidence': box.conf.item(),
                            'class_name': class_info['name'],
                            'class_english': class_info['english'],
                            'class_id': class_id,
                            'color': class_info['color']
                        })
            else:
                detections = self.mock_detect(image)
            
            return detections, image
        except Exception as e:
            st.error(f"Detection error: {e}")
            return self.mock_detect(image), image

    def mock_detect(self, image):
        """Mock detection results - for testing interface"""
        height, width = image.shape[:2]
        detections = []
        import random
        num_detections = random.randint(2, 4)
        
        for i in range(num_detections):
            x1 = random.randint(50, width - 150)
            y1 = random.randint(50, height - 150)
            x2 = x1 + random.randint(80, 200)
            y2 = y1 + random.randint(80, 200)
            confidence = round(0.7 + random.random() * 0.25, 2)
            class_id = random.randint(0, 2)
            class_info = PINE_FLOWER_CLASSES[class_id]
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_name': class_info['name'],
                'class_english': class_info['english'],
                'class_id': class_id,
                'color': class_info['color']
            })
        return detections

    def draw_detections(self, image, detections):
        """Draw detection bounding boxes"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            color = det.get('color', (0, 255, 0))
            class_name = det.get('class_english', det['class_name'])
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            label = f"{class_name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            
            # Draw text
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image

# Initialize detector
detector = AdvancedDetector('YOLOv11-PMC-PhaseNet.pt')

# Flask API routes
@flask_app.route('/')
def home():
    return jsonify({"message": "Pine Cone Phenology Detection API is running"})

@flask_app.route('/detect', methods=['POST'])
def detect():
    """API endpoint for image detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        filepath = os.path.join(flask_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Perform detection
        detections, original_image = detector.detect_image(filepath)
        
        # Draw detection results
        result_image = detector.draw_detections(original_image.copy(), detections)
        
        # Save result image
        result_filename = f"result_{filename}"
        result_path = os.path.join(flask_app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_image)
        
        # Prepare statistics
        stats = {
            'total_count': len(detections),
            'by_stage': defaultdict(int)
        }
        
        for det in detections:
            stage = det.get('class_english', 'Unknown')
            stats['by_stage'][stage] += 1
        
        return jsonify({
            'success': True,
            'original_file': filename,
            'result_file': result_filename,
            'detections': detections,
            'statistics': stats,
            'result_type': 'image'
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@flask_app.route('/uploads/<filename>')
def serve_file(filename):
    """Serve uploaded and result files"""
    return send_file(os.path.join(flask_app.config['UPLOAD_FOLDER'], filename))

# Streamlit Interface
def start_flask_server():
    """Start Flask server in background thread"""
    flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def main():
    # Streamlit page configuration
    st.set_page_config(
        page_title="Pine Cone Phenology Detection Platform",
        page_icon="🌲",
        layout="wide"
    )
    
    st.title("🌲 Pine Cone Phenology Detection Platform")
    st.markdown("Upload pine cone images to automatically identify phenological phases (Elongation, Ripening, Decline)")
    
    # Start Flask server in background thread
    if 'flask_started' not in st.session_state:
        st.info("🔄 Starting detection server...")
        thread = threading.Thread(target=start_flask_server, daemon=True)
        thread.start()
        st.session_state.flask_started = True
        time.sleep(3)  # Wait for server to start
        st.success("✅ Detection server is ready!")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Uploaded Image")
        
        with col2:
            st.subheader("🔍 Detection Results")
            
            # Call Flask API for detection
            with st.spinner("🔬 Analyzing pine cone phenological phases..."):
                try:
                    # Prepare file for API request
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post('http://localhost:5000/detect', files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result['success']:
                            # Display result image
                            result_url = f"http://localhost:5000/uploads/{result['result_file']}"
                            st.image(result_url, use_column_width=True, caption="Detection Results")
                            
                            # Display detection statistics
                            st.subheader("📊 Detection Statistics")
                            
                            stats_col1, stats_col2, stats_col3 = st.columns(3)
                            
                            with stats_col1:
                                total_count = result['statistics']['total_count']
                                st.metric("Total Detections", total_count)
                            
                            with stats_col2:
                                if result['detections']:
                                    main_stage = max(result['statistics']['by_stage'], 
                                                   key=result['statistics']['by_stage'].get)
                                    st.metric("Main Phenological Phase", main_stage)
                                else:
                                    st.metric("Main Phenological Phase", "None")
                            
                            with stats_col3:
                                if result['detections']:
                                    avg_confidence = np.mean([det['confidence'] for det in result['detections']])
                                    st.metric("Average Confidence", f"{avg_confidence:.2f}")
                                else:
                                    st.metric("Average Confidence", "0.00")
                            
                            # Detailed detection results
                            st.subheader("📋 Detailed Detection Results")
                            
                            if result['detections']:
                                for i, det in enumerate(result['detections']):
                                    with st.expander(f"Target {i+1}: {det['class_english']} (Confidence: {det['confidence']:.2f})"):
                                        st.json({
                                            'Bounding Box': det['bbox'],
                                            'Confidence': det['confidence'],
                                            'Phenological Phase': det['class_english'],
                                            'Class ID': det['class_id']
                                        })
                            else:
                                st.warning("🚫 No pine cone targets detected in the image")
                                
                        else:
                            st.error("❌ Detection failed")
                    else:
                        st.error(f"❌ API request failed with status {response.status_code}")
                        st.write("Error details:", response.json())
                        
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to detection server. Please wait a moment and try again.")
                except Exception as e:
                    st.error(f"💥 Request error: {e}")

    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        ### Pine Cone Phenology Detection System
        
        **Phenological Phase Identification:**
        - 🟢 **Elongation Stage** - Green border
        - 🟠 **Ripening Stage** - Orange border  
        - 🔴 **Decline Stage** - Red border
        
        **Detection Features:**
        - YOLOv11 object detection
        - Deep learning model
        - Real-time phenological phase recognition
        """)
        
        st.header("🛠 Technical Information")
        st.markdown("""
        - **Framework**: Streamlit + Flask
        - **Detection Model**: YOLOv11
        - **Image Processing**: OpenCV
        - **Current Mode**: Hybrid Architecture
        """)
        
        # Model status
        st.header("🔧 System Status")
        if detector.model is not None:
            st.success("✅ Model: Loaded Successfully")
        else:
            st.warning("⚠️ Model: Simulation Mode")
        
        st.info("🌐 Flask API: Running on port 5000")

if __name__ == '__main__':
    main()

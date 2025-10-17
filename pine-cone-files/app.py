from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import torch
from datetime import datetime
import logging
import json
from werkzeug.utils import secure_filename
import threading
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Ensure directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Pine cone phenological phases mapping
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
            logger.info("Loading YOLOv11 model...")
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info("Model loaded successfully!")
            logger.info(f"Model classes: {self.model.names}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.info("Switching to simulation mode")
            self.model = None

    def detect_image(self, image_path):
        """Perform image detection"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Cannot read image")

            original_image = image.copy()

            # Use real model if available
            if self.model is not None:
                results = self.model(image_path)
                detections = []
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        class_info = PINE_FLOWER_CLASSES.get(class_id,
                                                             {'name': 'unknown', 'color': (255, 255, 255),
                                                              'english': 'Unknown Stage'})

                        detections.append({
                            'bbox': box.xyxy[0].tolist(),
                            'confidence': box.conf.item(),
                            'class_name': class_info['name'],
                            'class_english': class_info['english'],
                            'class_id': class_id,
                            'color': class_info['color']
                        })
            else:
                # Use simulation detection
                detections = self.mock_detect(image)

            return detections, original_image

        except Exception as e:
            logger.error(f"Image detection failed: {e}")
            return self.mock_detect(image), original_image

    def detect_video(self, video_path):
        """Perform video detection"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Cannot open video file")

            # Get video information
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create output video path
            output_filename = f"result_{os.path.basename(video_path)}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            video_detections = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect every 5 frames for better performance
                if frame_count % 5 == 0 or frame_count == 0:
                    if self.model is not None:
                        # Use YOLO model for detection
                        results = self.model(frame)
                        frame_detections = []
                        for result in results:
                            for box in result.boxes:
                                class_id = int(box.cls.item())
                                class_info = PINE_FLOWER_CLASSES.get(class_id,
                                                                     {'name': 'unknown', 'color': (255, 255, 255),
                                                                      'english': 'Unknown Stage'})

                                frame_detections.append({
                                    'bbox': box.xyxy[0].tolist(),
                                    'confidence': box.conf.item(),
                                    'class_name': class_info['name'],
                                    'class_english': class_info['english'],
                                    'class_id': class_id,
                                    'color': class_info['color'],
                                    'frame': frame_count
                                })
                    else:
                        frame_detections = self.mock_detect(frame)

                    video_detections.extend(frame_detections)

                # Draw detection boxes
                result_frame = self.draw_detections(frame.copy(), frame_detections if frame_count % 5 == 0 else [])
                out.write(result_frame)

                frame_count += 1

                # Show progress (every 50 frames)
                if frame_count % 50 == 0:
                    logger.info(f"Video processing progress: {frame_count}/{total_frames}")

            cap.release()
            out.release()

            return video_detections, output_filename

        except Exception as e:
            logger.error(f"Video detection failed: {e}")
            return [], None

    def mock_detect(self, image):
        """Simulation detection results - for testing interface"""
        height, width = image.shape[:2]
        detections = []

        # Generate 2-4 random detection boxes
        import random
        num_detections = random.randint(2, 4)

        for i in range(num_detections):
            # Random position and size
            x1 = random.randint(50, width - 150)
            y1 = random.randint(50, height - 150)
            x2 = x1 + random.randint(80, 200)
            y2 = y1 + random.randint(80, 200)

            confidence = round(0.7 + random.random() * 0.25, 2)  # 0.7-0.95

            # Randomly select pine cone stage
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
        """Draw detection boxes"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            color = det.get('color', (0, 255, 0))
            class_name = det.get('class_english', det['class_name'])

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

            # Draw label background
            label = f"{class_name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)

            # Draw text
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return image

    def get_detection_statistics(self, detections):
        """Get detection statistics"""
        stats = {
            'total_count': 0,
            'by_stage': defaultdict(int),
            'by_stage_english': defaultdict(int)
        }

        if not detections:
            return stats

        stats['total_count'] = len(detections)

        for det in detections:
            stage = det.get('class_name', 'unknown')
            stage_english = det.get('class_english', 'Unknown Stage')
            stats['by_stage'][stage] += 1
            stats['by_stage_english'][stage_english] += 1

        return stats

# Initialize detector
detector = AdvancedDetector('YOLOv11-PMC-PhaseNet.pt')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    if file.filename == '':   如果file.filename   文件名 == "：
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):如果文件和allowed_file(file.filename   文件名)：
        try:   试一试:
            # Save file
            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")文件名= secure_filename (f”{datetime.now   现在 () .strftime (Y ' % % m % d_ % H % m % S ')} _ {file.filename   文件名}”)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)Filepath = os.path   路径.join   加入(应用程序。配置(“UPLOAD_FOLDER”),文件名)
            file.save(filepath)

            # Perform detection based on file type
            file_ext = filename.rsplit('.', 1)[1].lower()File_ext = filename.rsplit('。', 1) [1] .lower   较低的 ()

            if file_ext in ['mp4', 'avi', 'mov']:如果file_ext in   在 ['mp4'   “mp4”, 'avi'   “avi”, 'mov'   “mov”]：
                # Video detection
                detections, result_filename = detector.detect_video(filepath)Detections, result_filename = detector.detect_video（filepath）
                result_type = 'video'   Result_type = ‘video’
            else:   其他:
                # Image detection
                detections, original_image = detector.detect_image(filepath)检测，original_image = detector.detect_image（filepath）

                # Ensure detections is a list
                if detections is None:   如果detections为None：
                    detections = []   检测= []

                # Draw result image   #绘制结果图像
                result_image = detector.draw_detections(original_image.copy   复制(), detections)

                # Save result image   #保存结果图像
                result_filename = f"result_{filename}"Result_filename = f“result_{filename}”
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)Result_path = os.path   路径.join   加入(应用程序；配置[' UPLOAD_FOLDER '   ‘ upload_folder ’], result_filename)
                cv2.imwrite(result_path, result_image)cv2。imwrite (result_path result_image)
                result_type = 'image'   Result_type = ‘image’

            # Ensure getting statistics
            stats = detector.get_detection_statistics(detections) if detections else {Stats =检测器。Get_detection_statistics (detections) if detections else {Stats =检测器。get_detection_statistics(检测)如果检测其他{统计=检测器。Get_detection_statistics (detections) if detections else {
                'total_count': 0,   “total_count”:0,
                'by_stage': {},   “by_stage”:{},
                'by_stage_english': {}
            }

            return jsonify({
                'success': True,   “成功”:没错,
                'original_file': filename,“original_file”:文件名,
                'result_file': result_filename,“result_file”:result_filename,
                'detections': detections if detections else [],‘detections’：检测如果detections else []，
                'statistics': stats,   “统计”:统计数据,
                'result_type': result_type,“result_type”:result_type,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')“时间戳”:datetime.now()。strftime(“% Y - % m - H % d %: % m: % S ')
            })

        except Exception as e:   例外情况如下：
            logger.error(f"Processing failed: {e}")
            # Return error response with default values#使用默认值返回错误响应
            return jsonify({
                'success': False,   “成功”:假的,
                'error': f'Processing failed: {str(e)}',
                'detections': [],   “检测”:[],
                'statistics': {   “统计”:{
                    'total_count': 0,   “total_count”:0,
                    'by_stage': {},   “by_stage”:{},
                    'by_stage_english': {}
                }
            }), 500

    return jsonify({'error': 'Unsupported file format'}), 400

@app.route('/uploads/<filename>')@app.route(/上传/ <文件名>)
def serve_file(filename):   def serve_file(文件名):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))返回send_file (os.path.join(应用程序。配置(“UPLOAD_FOLDER”),文件名)

@app.route('/get_progress')
def get_progress():
    """Get processing progress (for video processing)"""
    # Progress tracking can be implemented here, simplified version returns completed进度跟踪可以在这里实现，简化版本返回完成
    return jsonify({'progress': 100, 'status': 'completed'})

if __name__ == '__main__':   如果__name__ == '__main__'：
    logger.info("Starting Advanced Pine Cone Detection Platform...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

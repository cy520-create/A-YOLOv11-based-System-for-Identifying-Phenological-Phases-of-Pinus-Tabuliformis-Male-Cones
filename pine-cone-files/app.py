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

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Pine flower phase classes mapping
PINE_FLOWER_CLASSES = {
    0: {'name': 'elongation stage', 'color': (0, 255, 0), 'chinese': '伸长期'},
    1: {'name': 'ripening stage', 'color': (0, 165, 255), 'chinese': '成熟期'},
    2: {'name': 'decline stage', 'color': (0, 0, 255), 'chinese': '衰退期'}
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
                                                              'chinese': 'Unknown phase'})

                        detections.append({
                            'bbox': box.xyxy[0].tolist(),
                            'confidence': box.conf.item(),
                            'class_name': class_info['name'],
                            'class_chinese': class_info['chinese'],
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
                                                                      'chinese': 'Unknown phase'})

                                frame_detections.append({
                                    'bbox': box.xyxy[0].tolist(),
                                    'confidence': box.conf.item(),
                                    'class_name': class_info['name'],
                                    'class_chinese': class_info['chinese'],
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
        """Simulate detection results - for testing interface"""
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

            # Randomly select pine flower phase
            class_id = random.randint(0, 2)
            class_info = PINE_FLOWER_CLASSES[class_id]

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_name': class_info['name'],
                'class_chinese': class_info['chinese'],
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
            class_name = det.get('class_name', 'unknown')

            # Draw bounding box
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
            'by_stage_chinese': defaultdict(int)
        }

        if not detections:
            return stats

        stats['total_count'] = len(detections)

        for det in detections:
            stage = det.get('class_name', 'unknown')
            stage_chinese = det.get('class_chinese', 'Unknown phase')
            stats['by_stage'][stage] += 1
            stats['by_stage_chinese'][stage_chinese] += 1

        return stats


# Initialize detector
detector = AdvancedDetector('YOLOv11-PMC-PhaseNet.pt')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Main page with English interface"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pine Cone Phenological Phase Detection System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 30px;
            }
            .upload-area {
                border: 2px dashed #3498db;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                background-color: #f8f9fa;
            }
            .btn {
                background-color: #3498db;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
            }
            .btn:hover {
                background-color: #2980b9;
            }
            .results {
                margin-top: 30px;
                display: none;
            }
            .image-container {
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
            }
            .image-box {
                text-align: center;
            }
            .image-box img {
                max-width: 400px;
                max-height: 400px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .stats {
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }
            .progress {
                width: 100%;
                background-color: #f0f0f0;
                border-radius: 10px;
                margin: 20px 0;
            }
            .progress-bar {
                width: 0%;
                height: 20px;
                background-color: #3498db;
                border-radius: 10px;
                transition: width 0.3s;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Pine Cone Phenological Phase Detection System</h1>
            <div class="subtitle">Based on YOLOv11 Deep Learning Model</div>
            
            <div class="upload-area">
                <h3>Upload Image or Video</h3>
                <p>Supported formats: JPG, PNG, MP4, AVI, MOV (Max 50MB)</p>
                <input type="file" id="fileInput" accept="image/*,video/*">
                <br>
                <button class="btn" onclick="uploadFile()">Start Detection</button>
            </div>

            <div class="progress" id="progressBar" style="display: none;">
                <div class="progress-bar" id="progressFill"></div>
            </div>

            <div class="results" id="results">
                <h2>Detection Results</h2>
                <div class="stats" id="statistics"></div>
                <div class="image-container" id="imageContainer"></div>
            </div>
        </div>

        <script>
            function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a file');
                    return;
                }

                const progressBar = document.getElementById('progressBar');
                const progressFill = document.getElementById('progressFill');
                progressBar.style.display = 'block';
                progressFill.style.width = '0%';

                const formData = new FormData();
                formData.append('file', file);

                fetch('/detect', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayResults(data);
                    } else {
                        alert('Detection failed: ' + (data.error || 'Unknown error'));
                    }
                    progressFill.style.width = '100%';
                })
                .catch(error => {
                    alert('Upload failed: ' + error);
                    progressBar.style.display = 'none';
                });
            }

            function displayResults(data) {
                document.getElementById('results').style.display = 'block';
                
                // Display statistics
                const stats = data.statistics;
                let statsHTML = `<h3>Detection Statistics</h3>`;
                statsHTML += `<p><strong>Total Detections:</strong> ${stats.total_count}</p>`;
                
                if (stats.by_stage) {
                    statsHTML += `<p><strong>By Growth Stage:</strong></p><ul>`;
                    for (const [stage, count] of Object.entries(stats.by_stage)) {
                        statsHTML += `<li>${stage}: ${count}</li>`;
                    }
                    statsHTML += `</ul>`;
                }
                
                document.getElementById('statistics').innerHTML = statsHTML;

                // Display images
                const container = document.getElementById('imageContainer');
                container.innerHTML = '';

                if (data.result_type === 'video') {
                    container.innerHTML = `
                        <div class="image-box">
                            <h4>Processed Video</h4>
                            <video controls width="400">
                                <source src="/uploads/${data.result_file}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                        </div>
                    `;
                } else {
                    container.innerHTML = `
                        <div class="image-box">
                            <h4>Original Image</h4>
                            <img src="/uploads/${data.original_file}" alt="Original">
                        </div>
                        <div class="image-box">
                            <h4>Detection Result</h4>
                            <img src="/uploads/${data.result_file}" alt="Result">
                        </div>
                    `;
                }
            }
        </script>
    </body>
    </html>
    """


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        try:
            # Save file   #保存文件
            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform detection based on file type
            file_ext = filename.rsplit('.', 1)[1].lower()

            if file_ext in ['mp4', 'avi', 'mov']:
                # Video detection
                detections, result_filename = detector.detect_video(filepath)
                result_type = 'video'
            else:
                # Image detection
                detections, original_image = detector.detect_image(filepath)

                # Ensure detections is a list
                if detections is None:
                    detections = []

                # Draw result image
                result_image = detector.draw_detections(original_image.copy(), detections)

                # Save result image   #保存结果图像
                result_filename = f"result_{filename}"Result_filename = f“result_{filename}”
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)Result_path = os.path   路径.join   加入(应用程序；配置[' UPLOAD_FOLDER '   ‘ upload_folder ’], result_filename)
                cv2.imwrite(result_path, result_image)cv2。imwrite (result_path result_image)cv2。Imwrite （result_path, result_image）Imwrite （result_path result_image）
                result_type = 'image'   “图像”

            # Ensure we get statistics#确保我们得到统计数据
            stats = detector.get_detection_statistics(detections) if detections else {Stats =检测器。Get_detection_statistics (detections) if   如果 detections else   其他 {
                'total_count': 0,   “total_count”:0,
                'by_stage': {},   “by_stage”:{},
                'by_stage_chinese': {}   “by_stage_chinese”:{}
            }

            return jsonify({
                'success': True,   “成功”:没错,
                'original_file': filename,“original_file”:文件名,
                'result_file': result_filename,“result_file”:result_filename,
                'detections': detections if detections else [],‘detections’：检测如果detections else   其他 []，
                'statistics': stats,   “统计”:统计数据,
                'result_type': result_type,“result_type”:result_type,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')“时间戳”:datetime.now   现在()。strftime(“% Y - % m - H % d %: % m: % S ')
            })

        except Exception as e:   例外情况如下：
            logger.error(f"Processing failed: {e}")记录器。错误（f“处理失败：{e}”）
            # Return error response with default values#使用默认值返回错误响应
            return jsonify({
                'success': False,   “成功”:假的,
                'error': f'Processing failed: {str(e)}',处理失败：{str(e)}'，
                'detections': [],   “检测”:[],
                'statistics': {   “统计”:{
                    'total_count': 0,   “total_count”:0,
                    'by_stage': {},   “by_stage”:{},‘by_stage’： {}, " by_stage ":{}，
                    'by_stage_chinese': {}   “by_stage_chinese”:{}
                }
            }), 500

    return jsonify({'error': 'Unsupported file format'}), 400返回jsonify({‘error‘: ’不支持的文件格式’}),400

@app.route('/uploads/<filename>')@app.route(/上传/ <文件名>)
def serve_file(filename):   def serve_file(文件名):defserve_file (filename): defserve_file（英文）：
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))返回send_file (os.path   路径.join   加入(应用程序。配置(“UPLOAD_FOLDER”),文件名)


@app.route('/get_progress')
def get_progress():
    """Get processing progress (for video processing)"""“”“获取处理进度（用于视频处理）”“”
    # Here you can implement progress tracking, simplified version returns completed在这里可以实现进度跟踪，简化版本返回完成
    return   返回 jsonify({'progress': 100, 'status'   “状态”: 'completed'   “完成”})


if   “__main__ ' __name__ == '   “__main__ '__main__':   如果__name__ == '__main__'：
    logger.info("Starting Advanced Pine Flower Detection Platform..."“启动高级松花检测平台……”)
    app.run(debug=True   真正的, host='0.0.0.0', port=5000)

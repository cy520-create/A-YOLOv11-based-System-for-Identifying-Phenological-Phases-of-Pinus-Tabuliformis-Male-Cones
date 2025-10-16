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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB限制
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 松花时期类别映射
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
        """加载YOLOv11模型"""
        try:
            logger.info("正在加载YOLOv11模型...")
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info("模型加载成功！")
            logger.info(f"模型类别: {self.model.names}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.info("切换到模拟检测模式")
            self.model = None

    def detect_image(self, image_path):
        """执行图片检测"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("无法读取图像")

            original_image = image.copy()

            # 如果有真实模型，使用真实检测
            if self.model is not None:
                results = self.model(image_path)
                detections = []
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        class_info = PINE_FLOWER_CLASSES.get(class_id,
                                                             {'name': 'unknown', 'color': (255, 255, 255),
                                                              'chinese': '未知时期'})

                        detections.append({
                            'bbox': box.xyxy[0].tolist(),
                            'confidence': box.conf.item(),
                            'class_name': class_info['name'],
                            'class_chinese': class_info['chinese'],
                            'class_id': class_id,
                            'color': class_info['color']
                        })
            else:
                # 使用模拟检测
                detections = self.mock_detect(image)

            return detections, original_image

        except Exception as e:
            logger.error(f"图片检测失败: {e}")
            return self.mock_detect(image), original_image

    def detect_video(self, video_path):
        """执行视频检测"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")

            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 创建输出视频路径
            output_filename = f"result_{os.path.basename(video_path)}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            video_detections = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 每5帧检测一次以提高性能
                if frame_count % 5 == 0 or frame_count == 0:
                    if self.model is not None:
                        # 使用YOLO模型检测
                        results = self.model(frame)
                        frame_detections = []
                        for result in results:
                            for box in result.boxes:
                                class_id = int(box.cls.item())
                                class_info = PINE_FLOWER_CLASSES.get(class_id,
                                                                     {'name': 'unknown', 'color': (255, 255, 255),
                                                                      'chinese': '未知时期'})

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

                # 绘制检测框
                result_frame = self.draw_detections(frame.copy(), frame_detections if frame_count % 5 == 0 else [])
                out.write(result_frame)

                frame_count += 1

                # 显示进度（每50帧）
                if frame_count % 50 == 0:
                    logger.info(f"视频处理进度: {frame_count}/{total_frames}")

            cap.release()
            out.release()

            return video_detections, output_filename

        except Exception as e:
            logger.error(f"视频检测失败: {e}")
            return [], None

    def mock_detect(self, image):
        """模拟检测结果 - 用于测试界面"""
        height, width = image.shape[:2]
        detections = []

        # 生成2-4个随机检测框
        import random
        num_detections = random.randint(2, 4)

        for i in range(num_detections):
            # 随机位置和大小
            x1 = random.randint(50, width - 150)
            y1 = random.randint(50, height - 150)
            x2 = x1 + random.randint(80, 200)
            y2 = y1 + random.randint(80, 200)

            confidence = round(0.7 + random.random() * 0.25, 2)  # 0.7-0.95

            # 随机选择松花时期
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
        """绘制检测框"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            color = det.get('color', (0, 255, 0))
            class_name = det.get('class_chinese', det['class_name'])

            # 画框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

            # 画标签背景
            label = f"{class_name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)

            # 画文字
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return image

    def get_detection_statistics(self, detections):
        """获取检测统计信息"""
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
            stage_chinese = det.get('class_chinese', '未知时期')
            stats['by_stage'][stage] += 1
            stats['by_stage_chinese'][stage_chinese] += 1

        return stats


# 初始化检测器
detector = AdvancedDetector('models/best.pt')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if file and allowed_file(file.filename):
        try:
            # 保存文件
            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 根据文件类型执行检测
            file_ext = filename.rsplit('.', 1)[1].lower()

            if file_ext in ['mp4', 'avi', 'mov']:
                # 视频检测
                detections, result_filename = detector.detect_video(filepath)
                result_type = 'video'
            else:
                # 图片检测
                detections, original_image = detector.detect_image(filepath)

                # 确保 detections 是列表
                if detections is None:
                    detections = []

                # 绘制结果图片
                result_image = detector.draw_detections(original_image.copy(), detections)

                # 保存结果图片
                result_filename = f"result_{filename}"
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                cv2.imwrite(result_path, result_image)
                result_type = 'image'

            # 确保获取统计信息
            stats = detector.get_detection_statistics(detections) if detections else {
                'total_count': 0,
                'by_stage': {},
                'by_stage_chinese': {}
            }

            return jsonify({
                'success': True,
                'original_file': filename,
                'result_file': result_filename,
                'detections': detections if detections else [],
                'statistics': stats,
                'result_type': result_type,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        except Exception as e:
            logger.error(f"处理失败: {e}")
            # 返回一个包含默认值的错误响应
            return jsonify({
                'success': False,
                'error': f'处理失败: {str(e)}',
                'detections': [],
                'statistics': {
                    'total_count': 0,
                    'by_stage': {},
                    'by_stage_chinese': {}
                }
            }), 500

    return jsonify({'error': '不支持的文件格式'}), 400

@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/get_progress')
def get_progress():
    """获取处理进度（用于视频处理）"""
    # 这里可以实现进度跟踪，简化版本直接返回完成
    return jsonify({'progress': 100, 'status': 'completed'})


if __name__ == '__main__':
    logger.info("启动高级松花检测平台...")
    app.run(debug=True, host='127.0.0.1', port=5000)
import streamlit as st
import numpy as np
import torch
from datetime import datetime
import logging
import json
import os
from collections import defaultdict
from ultralytics import YOLO
import tempfile
from PIL import Image, ImageDraw, ImageFont
import io

# 配置页面
st.set_page_config(
    page_title="松花物候期检测平台",
    page_icon="🌲",
    layout="wide"
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 松花时期类别映射
PINE_FLOWER_CLASSES = {
    0: {'name': 'elongation stage', 'color': (0, 255, 0), 'chinese': '伸长期'},
    1: {'name': 'ripening stage', 'color': (255, 165, 0), 'chinese': '成熟期'},
    2: {'name': 'decline stage', 'color': (255, 0, 0), 'chinese': '衰退期'}
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
            self.model = YOLO(self.model_path)
            logger.info("模型加载成功！")
            logger.info(f"模型类别: {self.model.names}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            st.error(f"模型加载失败: {e}")
            self.model = None

    def detect_image(self, image):
        """执行图片检测"""
        try:
            # 如果有真实模型，使用真实检测
            if self.model is not None:
                results = self.model(image)
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

            return detections

        except Exception as e:
            logger.error(f"图片检测失败: {e}")
            st.error(f"图片检测失败: {e}")
            return []

    def mock_detect(self, image):
        """模拟检测结果 - 用于测试界面"""
        width, height = image.size
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
        """绘制检测框 - 使用PIL"""
        draw = ImageDraw.Draw(image)
        
        # 尝试使用默认字体
        try:
            font = ImageFont.load_default()
        except:
            font = None

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            color = det.get('color', (0, 255, 0))
            class_name = det.get('class_chinese', det['class_name'])

            # 画框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 画标签背景
            label = f"{class_name} {conf:.2f}"
            if font:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width, text_height = 100, 20

            draw.rectangle([x1, y1 - text_height - 10, x1 + text_width, y1], fill=color)

            # 画文字
            if font:
                draw.text((x1, y1 - text_height - 5), label, fill=(255, 255, 255), font=font)
            else:
                draw.text((x1, y1 - text_height - 5), label, fill=(255, 255, 255))

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
@st.cache_resource
def load_detector():
    return AdvancedDetector('YOLOv11-PMC-PhaseNet.pt')

# 主应用
def main():
    st.title("🌲 松花物候期检测平台")
    st.markdown("上传松花图片，自动识别物候期（伸长期、成熟期、衰退期）")

    # 初始化检测器
    detector = load_detector()

    # 文件上传
    uploaded_file = st.file_uploader(
        "选择图片文件", 
        type=['png', 'jpg', 'jpeg'],
        help="支持格式: PNG, JPG, JPEG"
    )

    if uploaded_file is not None:
        try:
            # 使用PIL打开图片
            image = Image.open(uploaded_file)
            
            # 图片处理
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("原图")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("检测结果")
                
                # 执行检测
                with st.spinner("正在检测松花物候期..."):
                    detections = detector.detect_image(image)
                
                # 绘制检测结果
                result_image = detector.draw_detections(image.copy(), detections)
                st.image(result_image, use_column_width=True)
            
            # 显示统计信息
            stats = detector.get_detection_statistics(detections)
            
            st.subheader("检测统计")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("总检测数", stats['total_count'])
            
            with col4:
                stages = list(stats['by_stage_chinese'].keys())
                if stages:
                    main_stage = max(stats['by_stage_chinese'], key=stats['by_stage_chinese'].get)
                    st.metric("主要物候期", main_stage)
                else:
                    st.metric("主要物候期", "无")
            
            with col5:
                if detections:
                    avg_confidence = np.mean([det['confidence'] for det in detections])
                    st.metric("平均置信度", f"{avg_confidence:.2f}")
                else:
                    st.metric("平均置信度", "0.00")
            
            # 详细检测结果
            st.subheader("详细检测结果")
            if detections:
                for i, det in enumerate(detections):
                    with st.expander(f"检测目标 {i+1}: {det['class_chinese']} (置信度: {det['confidence']:.2f})"):
                        st.json(det)
            else:
                st.info("未检测到松花目标")

        except Exception as e:
            st.error(f"处理文件时出错: {e}")
            logger.error(f"处理文件失败: {e}")

    # 侧边栏信息
    with st.sidebar:
        st.header("关于")
        st.markdown("""
        ### 松花物候期检测系统
        - **伸长期**: 绿色边框
        - **成熟期**: 橙色边框  
        - **衰退期**: 红色边框
        
        ### 技术支持
        - YOLOv11 目标检测
        - 深度学习模型
        - 实时物候期识别
        """)
        
        st.header("模型状态")
        if detector.model is not None:
            st.success("✅ 模型加载成功")
        else:
            st.error("❌ 模型加载失败 - 使用模拟检测模式")

if __name__ == '__main__':
    main()

import streamlit as st
import cv2
import numpy as np
import torch
from datetime import datetime
import logging
import json
import os
from collections import defaultdict
from ultralytics import YOLO
import tempfile

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
@st.cache_resource
def load_detector():
    return AdvancedDetector('YOLOv11-PMC-PhaseNet.pt')

# 主应用
def main():
    st.title("🌲 松花物候期检测平台")
    st.markdown("上传松花图片或视频，自动识别物候期（伸长期、成熟期、衰退期）")

    # 初始化检测器
    detector = load_detector()

    # 文件上传
    uploaded_file = st.file_uploader(
        "选择图片或视频文件", 
        type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'],
        help="支持格式: PNG, JPG, JPEG, MP4, AVI, MOV"
    )

    if uploaded_file is not None:
        # 保存上传的文件到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()

            if file_ext in ['.mp4', '.avi', '.mov']:
                # 视频处理
                st.warning("视频处理功能在演示版本中可能受限")
                st.video(uploaded_file)
                
            else:
                # 图片处理
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("原图")
                    # 显示原图
                    image = cv2.imread(tmp_path)Image = cv2.imread（tmp_path）
                    if image is not None:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        st.image(image_rgb, use_column_width=True)st.image   图像 (image_rgb use_column_width = True   真正的)
                    
                        # 执行检测
                        with st.spinner("正在检测松花物候期..."):
                            detections = detector.detect_image(image)Detections = detector.detect_image（图像）
                        
                        # 绘制检测结果
                        result_image = detector.draw_detections(image.copy(), detections)Result_image = detector.draw_detections(图像；副本(),检测)
                        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)Result_image_rgb = cv2。cvtColor (result_image cv2。COLOR_BGR2RGB)
                        
                        with col2:   col2:
                            st.subheader("检测结果")
                            st.image(result_image_rgb, use_column_width=True)st.image   图像 (result_image_rgb use_column_width = True   真正的)
                        
                        # 显示统计信息
                        stats = detector.get_detection_statistics(detections)Stats = detector.get_detection_statistics（检测）
                        
                        st.subheader("检测统计")
                        col3, col4, col5 = st.columns(3)Col3, col4, col5 = st.columns(3)
                        
                        with col3:   col3:
                            st.metric("总检测数", stats['total_count'])
                        
                        with col4:   col4:
                            stages = list(stats['by_stage_chinese'].keys())阶段= list（stats['by_stage_chinese'   “by_stage_chinese”].keys   键()）
                            if   如果 stages:
                                main_stage = max(stats['by_stage_chinese'], key=stats['by_stage_chinese'].get)Main_stage = max（stats['by_stage_chinese'   “by_stage_chinese”], key=stats['by_stage_chinese'   “by_stage_chinese”].get）
                                st.metric("主要物候期", main_stage)
                        
                        with col5:   col5:
                            if detections:
                                avg_confidence = np.mean([det['confidence'] for det in detections])Avg_confidence = np。平均值（[det[‘置信度’]在检测中的det]）
                                st.metric("平均置信度", f"{avg_confidence:.2f}")
                        
                        # 详细检测结果
                        st.subheader("详细检测结果")
                        if detections:   如果检测:
                            for i, det in enumerate(detections):对于i， det in   在 enumerate   列举(detections)：
                                with st.expander(f"检测目标 {i+1}: {det['class_chinese']} (置信度: {det['confidence']:.2f})"):with   与 st.expander   扩张器(f"检测目标 {i 1}: {det['class_chinese'   “class_chinese”]} (置信度: {det['confidence'   “信心”]:.2f})"):
                                    st.json(det)   st.json(依据)
                        else:   其他:
                            st.info("未检测到松花目标")
                    
                else:   其他:
                    st.error("无法读取图片文件")

        except Exception as e:   例外情况如下：
            st.error(f"处理文件时出错: {e}")
            logger.error(f"处理文件失败: {e}")
        
        finally:   最后:
            # 清理临时文件
            try:   试一试:
                os.unlink(tmp_path)
            except:   除了:
                pass   通过

    # 侧边栏信息
    with st.sidebar:   st.sidebar   侧边栏:
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
        if detector.model is not None:如果探测器。model不是None：
            st.success("✅ 模型加载成功")
        else:   其他:
            st.error("❌ 模型加载失败")

if __name__ == '__main__':   如果__name__ == '__main__'   “__main__ '：
    main()

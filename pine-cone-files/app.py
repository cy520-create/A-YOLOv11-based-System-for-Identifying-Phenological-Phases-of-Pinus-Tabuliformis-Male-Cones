import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import json
import os
from collections import defaultdict
import random

# 配置页面
st.set_page_config(
    page_title="松花物候期检测平台",
    page_icon="🌲", 
    layout="wide"
)

# 松花时期类别映射
PINE_FLOWER_CLASSES = {
    0: {'name': 'elongation stage', 'color': (0, 255, 0), 'chinese': '伸长期'},
    1: {'name': 'ripening stage', 'color': (255, 165, 0), 'chinese': '成熟期'},
    2: {'name': 'decline stage', 'color': (255, 0, 0), 'chinese': '衰退期'}
}

class SimpleDetector:
    def __init__(self):
        self.model_loaded = False
        # 模拟模型加载
        st.sidebar.success("✅ 模拟检测模式已启动")
        
    def detect_image(self, image):
        """模拟图片检测"""
        try:
            width, height = image.size
            detections = []
            
            # 生成2-4个随机检测框
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

        except Exception as e:
            st.error(f"检测失败: {e}")
            return []

    def draw_detections(self, image, detections):
        """绘制检测框"""
        draw = ImageDraw.Draw(image)
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            color = det.get('color', (0, 255, 0))
            class_name = det.get('class_chinese', det['class_name'])

            # 画框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 画标签背景和文字
            label = f"{class_name} {conf:.2f}"
            # 简单估算文本大小
            text_width = len(label) * 10
            text_height = 20
            
            draw.rectangle([x1, y1 - text_height - 10, x1 + text_width, y1], fill=color)
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
    return SimpleDetector()

def main():
    st.title("🌲 松花物候期检测平台")
    st.markdown("上传松花图片，自动识别物候期（伸长期、成熟期、衰退期）")
    
    # 显示说明
    with st.expander("重要说明", expanded=True):
        st.info("""
        **当前运行在演示模式：**
        - 使用模拟检测算法展示界面功能
        - 检测结果为随机生成，用于演示界面
        - 实际部署时需要连接真实的YOLO模型
        """)

    # 初始化检测器
    detector = load_detector()

    # 文件上传
    uploaded_file = st.file_uploader(
        "选择松花图片文件", 
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
                st.subheader("📷 原图")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("🔍 检测结果")
                
                # 执行检测
                with st.spinner("正在分析松花物候期..."):
                    detections = detector.detect_image(image)
                
                # 绘制检测结果
                result_image = detector.draw_detections(image.copy(), detections)
                st.image(result_image, use_column_width=True)
            
            # 显示统计信息
            if detections:
                stats = detector.get_detection_statistics(detections)
                
                st.subheader("📊 检测统计")
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    st.metric("总检测数", stats['total_count'])
                
                with col4:
                    stages = list(stats['by_stage_chinese'].keys())
                    if stages:
                        main_stage = max(stats['by_stage_chinese'], key=stats['by_stage_chinese'].get)
                        st.metric("主要物候期", main_stage)
                
                with col5:
                    avg_confidence = np.mean([det['confidence'] for det in detections])
                    st.metric("平均置信度", f"{avg_confidence:.2f}")
                
                # 详细检测结果
                st.subheader("📋 详细检测结果")
                for i, det in enumerate(detections):
                    with st.expander(f"目标 {i+1}: {det['class_chinese']} (置信度: {det['confidence']:.2f})"):
                        st.json(det)
            else:
                st.warning("未检测到松花目标")

        except Exception as e:
            st.error(f"处理图片时出错: {e}")

    # 侧边栏信息
    with st.sidebar:
        st.header("ℹ️ 关于")
        st.markdown("""
        ### 松花物候期检测系统
        
        **物候期标识：**
        - 🟢 伸长期 - 绿色边框
        - 🟠 成熟期 - 橙色边框  
        - 🔴 衰退期 - 红色边框
        
        **当前模式：**
        - 演示版本
        - 模拟检测算法
        - 功能完整展示
        """)
        
        st.header("🛠 技术信息")
        st.markdown("""
        - 框架: Streamlit
        - 图像处理: Pillow
        - 检测模式: 模拟算法
        - 状态: **运行正常**
        """)

if __name__ == '__main__':
    main()

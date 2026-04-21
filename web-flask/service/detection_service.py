# MD5: bf16a6d048d4b3528c92c4d3200b8835


"""
植物病害检测服务模块
"""
import os
import base64
import io
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from config import MODEL_CONFIGS, AVAILABLE_MODELS, CLASS_NAME_MAPPING


# 全局模型缓存
_model_cache = {}


def load_model(model_name):
    """加载模型，使用缓存避免重复加载"""
    if model_name not in _model_cache:
        model_config = MODEL_CONFIGS.get(model_name)
        if not model_config:
            raise ValueError(f"不支持的模型: {model_name}")
        
        model_path = model_config['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}，请先执行训练脚本code/train.py来生成新模型的权重文件")
        
        print(f"加载模型: {model_name} from {model_path}")
        _model_cache[model_name] = YOLO(model_path)
    
    return _model_cache[model_name]


def get_models():
    """获取可用模型列表"""
    return {'code': 200, 'data': AVAILABLE_MODELS}


def decode_base64_image(image_data):
    """解码base64图像数据"""
    try:
        # 移除data:image/xxx;base64,前缀
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # 解码base64
        image_bytes = base64.b64decode(image_data)
        
        # 转换为PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # 转换为RGB格式（如果需要）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        raise ValueError(f"图像解码失败: {str(e)}")


def draw_detection_boxes(image, detections):
    """
    在图像上绘制检测框和标签
    
    Args:
        image: PIL Image对象
        detections: 检测结果列表
        
    Returns:
        str: 带检测框的图像的base64编码
    """
    try:
        # 转换PIL图像为OpenCV格式
        img_array = np.array(image)
        img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 定义颜色（BGR格式）- 为不同类别分配不同颜色
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色  
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 洋红色
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 紫色
            (255, 165, 0),  # 橙色
            (0, 128, 255),  # 深蓝色
            (128, 255, 0),  # 黄绿色
        ]
        
        for detection in detections:
            # 获取边界框坐标
            bbox = detection['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            # 获取类别信息
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # 选择颜色
            color = colors[class_id % len(colors)]
            
            # 绘制检测框
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签文本
            label = f"{class_name} {confidence:.2f}"
            
            # 计算文本尺寸
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 绘制标签背景
            cv2.rectangle(img_cv2, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), 
                         color, -1)
            
            # 绘制标签文本
            cv2.putText(img_cv2, label, 
                       (x1, y1 - 5), 
                       font, font_scale, 
                       (255, 255, 255), thickness)
        
        # 转换回PIL图像
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(img_rgb)
        
        # 转换为base64
        buffer = io.BytesIO()
        result_image.save(buffer, format='JPEG', quality=90)
        img_data = buffer.getvalue()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_base64}"
        
    except Exception as e:
        raise ValueError(f"绘制检测框失败: {str(e)}")


def detect_objects(model_name, image_data):
    """对图像进行植物病害检测"""
    try:
        # 检查模型是否支持
        if model_name not in MODEL_CONFIGS:
            return {'code': 400, 'message': f'不支持的模型: {model_name}'}
        
        # 加载模型
        model = load_model(model_name)
        
        # 解码图像
        image = decode_base64_image(image_data)
        
        # 创建临时文件保存图像
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image.save(temp_file.name, 'JPEG')
            temp_path = temp_file.name
        
        try:
            # 进行植物病害检测预测
            results = model(temp_path, verbose=False, imgsz=640)
            
            # 解析检测结果
            if results and len(results) > 0:
                result = results[0]
                
                # 获取检测结果
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    # 获取图像尺寸
                    img_height, img_width = image.size[1], image.size[0]
                    
                    detections = []
                    boxes = result.boxes
                    
                    # 遍历每个检测到的病害
                    for i in range(len(boxes)):
                        # 获取边界框坐标（xyxy格式）
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        # 获取置信度
                        confidence = float(boxes.conf[i].cpu().numpy())
                        
                        # 获取类别
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name_en = result.names[class_id]
                        class_name_zh = CLASS_NAME_MAPPING.get(class_name_en, class_name_en)
                        
                        # 计算中心点和宽高
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        # 归一化坐标
                        center_x_norm = center_x / img_width
                        center_y_norm = center_y / img_height
                        width_norm = width / img_width
                        height_norm = height / img_height
                        
                        detection_info = {
                            'detection_id': i + 1,
                            'class_name': class_name_en,
                            'class_name_zh': class_name_zh,
                            'class_id': class_id,
                            'confidence': confidence,
                            'percentage': f"{confidence * 100:.2f}%",
                            'bbox': {
                                'x1': float(x1),
                                'y1': float(y1),
                                'x2': float(x2),
                                'y2': float(y2),
                                'center_x': float(center_x),
                                'center_y': float(center_y),
                                'width': float(width),
                                'height': float(height)
                            },
                            'bbox_normalized': {
                                'center_x': float(center_x_norm),
                                'center_y': float(center_y_norm),
                                'width': float(width_norm),
                                'height': float(height_norm)
                            }
                        }
                        detections.append(detection_info)
                    
                    # 按置信度排序
                    detections.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    # 生成带检测框的图片
                    detection_image_base64 = None
                    if detections:  # 只有检测到病害时才生成带框的图片
                        try:
                            detection_image_base64 = draw_detection_boxes(image, detections)
                        except Exception as e:
                            print(f"Warning: Failed to draw detection boxes: {e}")
                    
                    return {
                        'code': 200,
                        'data': {
                            'model': model_name,
                            'image_size': {
                                'width': img_width,
                                'height': img_height
                            },
                            'detections': detections,
                            'total_detections': len(detections),
                            'highest_confidence': detections[0] if detections else None,
                            'detection_image': detection_image_base64  # 带检测框的图片
                        }
                    }
                else:
                    # 没有检测到任何目标
                    return {
                        'code': 200,
                        'data': {
                            'model': model_name,
                            'image_size': {
                                'width': image.size[0],
                                'height': image.size[1]
                            },
                            'detections': [],
                            'total_detections': 0,
                            'detection_image': None,  # 没有检测到病害时不返回处理后的图片
                            'message': '未检测到任何植物病害'
                        }
                    }
            else:
                return {'code': 500, 'message': '模型预测失败'}
                
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except FileNotFoundError as e:
        return {'code': 404, 'message': str(e)}
    except ValueError as e:
        return {'code': 400, 'message': str(e)}
    except Exception as e:
        return {'code': 500, 'message': f'植物病害检测过程中发生错误: {str(e)}'}
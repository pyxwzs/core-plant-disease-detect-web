"""
植物病害视频检测服务模块
"""
import os
import cv2
import time
import threading
from ultralytics import YOLO
from config import MODEL_CONFIGS, CLASS_NAME_MAPPING


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
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"加载模型: {model_name} from {model_path}")
        _model_cache[model_name] = YOLO(model_path)
    
    return _model_cache[model_name]


def draw_detection_boxes_on_frame(frame, detections):
    """
    在视频帧上绘制检测框和标签
    
    Args:
        frame: OpenCV图像帧
        detections: 检测结果列表
        
    Returns:
        frame: 带检测框的图像帧
    """
    if not detections:
        return frame
    
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        label = f"{class_name} {confidence:.2f}"
        
        # 计算文本尺寸
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 绘制标签背景
        cv2.rectangle(frame, 
                     (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), 
                     color, -1)
        
        # 绘制标签文本
        cv2.putText(frame, label, 
                   (x1, y1 - 5), 
                   font, font_scale, 
                   (255, 255, 255), thickness)
    
    return frame


def detect_objects_in_frame(model, frame):
    """
    对单帧图像进行病害检测
    
    Args:
        model: YOLO模型
        frame: OpenCV图像帧
        
    Returns:
        list: 检测结果列表
    """
    try:
        # 进行预测
        results = model(frame, verbose=False, imgsz=640)
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            return []
        
        # 获取图像尺寸
        img_height, img_width = frame.shape[:2]
        
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
        return detections
        
    except Exception as e:
        print(f"帧检测错误: {e}")
        return []


def process_video(model_name, video_path, progress_callback=None):
    """
    处理视频，对每一帧进行病害检测并绘制检测框
    
    Args:
        model_name: 模型名称
        video_path: 输入视频路径
        progress_callback: 进度回调函数
        
    Returns:
        dict: 处理结果
    """
    try:
        start_time = time.time()
        
        # 检查模型是否支持
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f'不支持的模型: {model_name}')
        
        # 加载模型
        if progress_callback:
            progress_callback(5, "正在加载模型...")
        model = load_model(model_name)
        
        # 打开视频文件
        if progress_callback:
            progress_callback(10, "正在读取视频文件...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if progress_callback:
            progress_callback(15, f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
        
        # 创建输出视频文件
        output_path = video_path.replace('.', '_processed.')
        if progress_callback:
            progress_callback(12, "正在初始化视频编码器...")
        
        # 尝试使用H.264编码以确保浏览器兼容性
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 如果H.264不可用，回退到XVID编码
        if not out.isOpened():
            print("avc1编码不可用，尝试XVID编码")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        # 最后回退选项：mp4v编码
        if not out.isOpened():
            print("XVID编码不可用，尝试mp4v编码")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        # 如果所有编码都失败，抛出错误
        if not out.isOpened():
            raise ValueError("无法创建视频输出文件，请检查系统是否支持视频编码")
        
        print(f"成功初始化视频编码器，输出路径: {output_path}")
        
        # 统计信息
        total_detections = 0
        frames_with_detections = 0
        current_frame = 0
        detection_stats = {}  # 各类别检测统计
        frame_detection_details = []  # 每帧检测详情
        
        if progress_callback:
            progress_callback(20, "开始处理视频帧...")
        
        # 逐帧处理
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame += 1
            
            # 更新进度
            frame_progress = 20 + (current_frame / total_frames) * 70  # 20-90%
            if progress_callback and current_frame % 10 == 0:  # 每10帧更新一次进度
                progress_callback(
                    frame_progress, 
                    f"处理第 {current_frame}/{total_frames} 帧"
                )
            
            # 对当前帧进行检测
            detections = detect_objects_in_frame(model, frame)
            
            # 统计检测结果
            frame_detection_info = {
                'frame_number': current_frame,
                'detections_count': len(detections),
                'detections': []
            }
            
            if detections:
                frames_with_detections += 1
                total_detections += len(detections)
                
                # 统计各类别检测数量
                for detection in detections:
                    class_name_en = detection['class_name']
                    class_name_zh = detection['class_name_zh']
                    confidence = detection['confidence']
                    
                    # 更新类别统计
                    if class_name_en not in detection_stats:
                        detection_stats[class_name_en] = {
                            'class_name_en': class_name_en,
                            'class_name_zh': class_name_zh,
                            'count': 0,
                            'total_confidence': 0,
                            'max_confidence': 0,
                            'min_confidence': 1.0,
                            'frames_appeared': set()
                        }
                    
                    stats = detection_stats[class_name_en]
                    stats['count'] += 1
                    stats['total_confidence'] += confidence
                    stats['max_confidence'] = max(stats['max_confidence'], confidence)
                    stats['min_confidence'] = min(stats['min_confidence'], confidence)
                    stats['frames_appeared'].add(current_frame)
                    
                    # 记录当前帧检测详情
                    frame_detection_info['detections'].append({
                        'class_name_en': class_name_en,
                        'class_name_zh': class_name_zh,
                        'confidence': confidence,
                        'bbox': detection['bbox']
                    })
            
            # 记录帧检测详情
            frame_detection_details.append(frame_detection_info)
            
            # 在帧上绘制检测框
            processed_frame = draw_detection_boxes_on_frame(frame, detections)
            
            # 写入输出视频
            out.write(processed_frame)
        
        # 释放资源
        cap.release()
        out.release()
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        if progress_callback:
            progress_callback(95, "正在生成结果...")
        
        # 处理类别统计数据（转换set为list以便JSON序列化）
        processed_detection_stats = []
        for class_name_en, stats in detection_stats.items():
            avg_confidence = stats['total_confidence'] / stats['count'] if stats['count'] > 0 else 0
            frames_appeared = len(stats['frames_appeared'])
            
            processed_detection_stats.append({
                'class_name_en': stats['class_name_en'],
                'class_name_zh': stats['class_name_zh'],
                'count': stats['count'],
                'frames_appeared': frames_appeared,
                'frame_appearance_rate': (frames_appeared / total_frames) * 100 if total_frames > 0 else 0,
                'avg_confidence': round(avg_confidence, 3),
                'max_confidence': round(stats['max_confidence'], 3),
                'min_confidence': round(stats['min_confidence'], 3),
                'detection_rate': (stats['count'] / total_detections) * 100 if total_detections > 0 else 0
            })
        
        # 按检测数量排序
        processed_detection_stats.sort(key=lambda x: x['count'], reverse=True)
        
        # 生成处理结果
        result = {
            'processed_video_path': output_path,
            'total_frames': total_frames,
            'frames_with_detections': frames_with_detections,
            'total_detections': total_detections,
            'processing_time': processing_time,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'duration': total_frames / fps if fps > 0 else 0
            },
            'detection_statistics': {
                'total_classes_detected': len(detection_stats),
                'detection_rate': (frames_with_detections / total_frames) * 100 if total_frames > 0 else 0,
                'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0,
                'class_statistics': processed_detection_stats
            },
            # 可选：包含详细的帧检测信息
            'frame_details': frame_detection_details  # 全量帧数据，供历史记录保存
        }
        
        if progress_callback:
            progress_callback(100, "处理完成！")
        
        return {'code': 200, 'data': result}
        
    except FileNotFoundError as e:
        return {'code': 404, 'message': str(e)}
    except ValueError as e:
        return {'code': 400, 'message': str(e)}
    except Exception as e:
        return {'code': 500, 'message': f'视频处理过程中发生错误: {str(e)}'}


class VideoProcessor:
    """视频处理器类，支持进度回调"""
    
    def __init__(self):
        self.progress = 0
        self.message = ""
        self.is_processing = False
        self.result = None
        self.error = None
    
    def update_progress(self, progress, message):
        """更新处理进度"""
        self.progress = progress
        self.message = message
        print(f"进度: {progress}% - {message}")
    
    def process_video_async(self, model_name, video_path):
        """异步处理视频"""
        self.is_processing = True
        self.progress = 0
        self.message = "开始处理..."
        
        try:
            result = process_video(model_name, video_path, self.update_progress)
            if result['code'] == 200:
                self.result = result['data']
            else:
                self.error = result['message']
        except Exception as e:
            self.error = str(e)
        finally:
            self.is_processing = False
    
    def get_status(self):
        """获取当前处理状态"""
        return {
            'is_processing': self.is_processing,
            'progress': self.progress,
            'message': self.message,
            'result': self.result,
            'error': self.error
        }


# 全局视频处理器实例
_video_processors = {}


def start_video_processing(session_id, model_name, video_path):
    """开始视频处理"""
    processor = VideoProcessor()
    _video_processors[session_id] = processor
    
    # 在后台线程中处理视频
    thread = threading.Thread(
        target=processor.process_video_async,
        args=(model_name, video_path)
    )
    thread.daemon = True
    thread.start()
    
    return processor


def get_processing_status(session_id):
    """获取视频处理状态"""
    processor = _video_processors.get(session_id)
    if not processor:
        return {'code': 404, 'message': '未找到处理任务'}
    
    status = processor.get_status()
    return {'code': 200, 'data': status}


def cleanup_processing_session(session_id):
    """清理处理会话"""
    if session_id in _video_processors:
        del _video_processors[session_id]
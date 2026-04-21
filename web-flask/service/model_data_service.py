# MD5: aabad31ba5e1bc855cb9c09edf4f6545


"""
模型数据服务
"""
import os
import csv
import base64
from config import MODEL_CONFIGS


def get_model_data(data_type, model_key):
    """
    获取模型数据
    
    Args:
        data_type: 数据类型 ('training' 或 'validation')
        model_key: 模型键名
    
    Returns:
        dict: 包含数据的响应
    """
    try:
        # 检查模型是否存在
        if model_key not in MODEL_CONFIGS:
            return {'code': 404, 'message': '模型不存在'}
        
        model_config = MODEL_CONFIGS[model_key]
        
        if data_type == 'training':
            return get_training_data(model_config)
        elif data_type == 'validation':
            return get_validation_data(model_config)
        else:
            return {'code': 400, 'message': '不支持的数据类型'}
            
    except Exception as e:
        return {'code': 500, 'message': f'获取数据失败: {str(e)}'}


def get_training_data(model_config):
    """
    获取目标检测训练数据
    
    Args:
        model_config: 模型配置
    
    Returns:
        dict: 训练数据响应
    """
    train_results_path = model_config.get('train_results_path')
    
    if not train_results_path or not os.path.exists(train_results_path):
        return {'code': 404, 'message': '训练结果文件不存在'}
    
    try:
        training_data = []
        
        with open(train_results_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # 解析目标检测CSV数据
                epoch = int(row.get('epoch', 0))
                
                # 训练损失
                train_box_loss = float(row.get('train/box_loss', 0))
                train_cls_loss = float(row.get('train/cls_loss', 0))
                train_dfl_loss = float(row.get('train/dfl_loss', 0))
                
                # 验证损失
                val_box_loss = float(row.get('val/box_loss', 0))
                val_cls_loss = float(row.get('val/cls_loss', 0))
                val_dfl_loss = float(row.get('val/dfl_loss', 0))
                
                # 检测指标
                precision = float(row.get('metrics/precision(B)', 0))
                recall = float(row.get('metrics/recall(B)', 0))
                map50 = float(row.get('metrics/mAP50(B)', 0))
                map50_95 = float(row.get('metrics/mAP50-95(B)', 0))
                
                # 总损失计算
                train_total_loss = train_box_loss + train_cls_loss + train_dfl_loss
                val_total_loss = val_box_loss + val_cls_loss + val_dfl_loss
                
                training_data.append({
                    'epoch': epoch,
                    'train_total_loss': train_total_loss,
                    'val_total_loss': val_total_loss,
                    'train_box_loss': train_box_loss,
                    'train_cls_loss': train_cls_loss,
                    'train_dfl_loss': train_dfl_loss,
                    'val_box_loss': val_box_loss,
                    'val_cls_loss': val_cls_loss,
                    'val_dfl_loss': val_dfl_loss,
                    'precision': precision,
                    'recall': recall,
                    'map50': map50,
                    'map50_95': map50_95
                })
        
        return {
            'code': 200,
            'message': '获取训练数据成功',
            'data': training_data
        }
        
    except Exception as e:
        return {'code': 500, 'message': f'解析训练数据失败: {str(e)}'}


def get_validation_data(model_config):
    """
    获取验证数据
    
    Args:
        model_config: 模型配置
    
    Returns:
        dict: 验证数据响应
    """
    val_data_path = model_config.get('val_data_path')
    val_accuracy_path = model_config.get('val_accuracy_path')
    
    if not val_data_path or not os.path.exists(val_data_path):
        return {'code': 404, 'message': '验证数据目录不存在'}
    
    try:
        validation_data = {
            'accuracy': None,
            'images': []
        }
        
        # 读取目标检测精度信息
        if val_accuracy_path and os.path.exists(val_accuracy_path):
            accuracy_data = {}
            class_results = []
            
            with open(val_accuracy_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('all'):
                        # 解析总体精度
                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                accuracy_data = {
                                    'images': int(parts[1]),
                                    'instances': int(parts[2]),
                                    'precision': float(parts[3]),
                                    'recall': float(parts[4]),
                                    'mAP50': float(parts[5]),
                                    'mAP50_95': float(parts[6])
                                }
                            except (ValueError, IndexError):
                                continue
                    elif len(line.split()) >= 7 and line.split()[0] not in ['Class', 'all']:
                        # 解析各类别精度
                        parts = line.split()
                        try:
                            class_result = {
                                'class_name': parts[0],
                                'images': int(parts[1]),
                                'instances': int(parts[2]),
                                'precision': float(parts[3]),
                                'recall': float(parts[4]),
                                'mAP50': float(parts[5]),
                                'mAP50_95': float(parts[6])
                            }
                            class_results.append(class_result)
                        except (ValueError, IndexError):
                            continue
            
            validation_data['accuracy'] = accuracy_data
            validation_data['class_results'] = class_results
        
        # 读取验证图片
        image_files = []
        for filename in os.listdir(val_data_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(filename)
        
        # 排序文件名
        image_files.sort()
        
        # 将图片转换为base64编码
        for filename in image_files:
            image_path = os.path.join(val_data_path, filename)
            try:
                with open(image_path, 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    
                    # 根据文件扩展名确定MIME类型
                    if filename.lower().endswith('.png'):
                        mime_type = 'image/png'
                    else:
                        mime_type = 'image/jpeg'
                    
                    validation_data['images'].append({
                        'name': filename,
                        'url': f'data:{mime_type};base64,{img_base64}'
                    })
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
                continue
        
        return {
            'code': 200,
            'message': '获取验证数据成功',
            'data': validation_data
        }
        
    except Exception as e:
        return {'code': 500, 'message': f'获取验证数据失败: {str(e)}'}
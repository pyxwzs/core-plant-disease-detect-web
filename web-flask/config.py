

"""
配置文件
"""
import os

# 用户数据：SQLite（单文件，路径相对于 web-flask）
DATABASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'users.db'
)

# 旧版 users.json（迁移用，迁移完成后可手动删除）
LEGACY_USERS_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'users.json'
)

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# web-flask 目录（本文件所在目录）
WEB_FLASK_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_BASE_PATH = os.path.join(BASE_DIR, 'other', 'model_train', 'detect', 'output')

# 检测历史记录（原图 MD5 + 检测结果 JSON，按次保存）
DETECTION_HISTORY_DIR = os.path.join(WEB_FLASK_DIR, 'data', 'detection_history')
DETECTION_RECORDS_DIR = os.path.join(DETECTION_HISTORY_DIR, 'records')
DETECTION_IMAGES_DIR = os.path.join(DETECTION_HISTORY_DIR, 'images')
DETECTION_VIDEOS_DIR = os.path.join(DETECTION_HISTORY_DIR, 'videos')
DETECTION_INDEX_FILE = os.path.join(DETECTION_HISTORY_DIR, 'index.json')
DETECTION_DB_PATH    = os.path.join(DETECTION_HISTORY_DIR, 'detection.db')

# 茶树病虫害知识库 PDF（书籍/资料）
PDF_KNOWLEDGE_DIR = os.path.join(WEB_FLASK_DIR, 'data', 'pdf')

# 结构化知识库：参考图片上传目录（仅存文件名，库内 JSON 引用）
KNOWLEDGE_UPLOAD_DIR = os.path.join(WEB_FLASK_DIR, 'data', 'knowledge_uploads')

# 植物病害检测模型配置
MODEL_CONFIGS = {
    'ready-model': {
        'name': 'YOLO 茶叶病害检测模型(已经训练好的模型)',
        'model_path': os.path.join(MODEL_BASE_PATH, '已经训练好的模型和测试结果', 'train', 'weights', 'best.pt'),
        'train_results_path': os.path.join(MODEL_BASE_PATH, '已经训练好的模型和测试结果', 'train', 'results.csv'),
        'val_data_path': os.path.join(MODEL_BASE_PATH, '已经训练好的模型和测试结果', 'val'),
        'val_accuracy_path': os.path.join(MODEL_BASE_PATH, '已经训练好的模型和测试结果', 'val', '测试集精度.txt'),
    },
    'training-model': {
        'name': 'YOLOv8 茶叶病害检测模型(新训练的模型)',
        'model_path': os.path.join(MODEL_BASE_PATH, 'train', 'weights', 'best.pt'),
        'train_results_path': os.path.join(MODEL_BASE_PATH, 'train', 'results.csv'),
        'val_data_path': os.path.join(MODEL_BASE_PATH, 'val'),
        'val_accuracy_path': os.path.join(MODEL_BASE_PATH, 'val', '测试集精度.txt'),
    }
}

# 植物病害检测类别中文映射
CLASS_NAME_MAPPING = {
    'Apple Scab Leaf': '黑星病叶片',
    'Apple leaf': '茶白星病',
    'Apple rust leaf': '锈病叶片',
    'Bell_pepper leaf': '甜椒叶片',
    'Bell_pepper leaf spot': '甜椒叶斑病',
    'Blueberry leaf': '茶炭疽病',
    'Cherry leaf': '茶云纹叶枯病',
    'Corn Gray leaf spot': '灰叶斑病',
    'Corn leaf blight': '叶枯病',
    'Corn rust leaf': '锈病叶片',
    'Peach leaf': '茶根腐病',
    'Potato leaf': '茶橙瘿螨',
    'Potato leaf early blight': '早疫病',
    'Potato leaf late blight': '晚疫病',
    'Raspberry leaf': '覆盆子叶片',
    'Soyabean leaf': '茶苗根结线虫病',
    'Soybean leaf': '茶树膏药病',
    'Squash Powdery mildew leaf': '南瓜白粉病叶片',
    'Strawberry leaf': '茶枝梢黑点病',
    'Tomato Early blight leaf': '早疫病叶片',
    'Tomato Septoria leaf spot': '叶斑病',
    'Tomato leaf': '叶片',
    'Tomato leaf bacterial spot': '细菌性斑点病',
    'Tomato leaf late blight': '晚疫病叶片',
    'Tomato leaf mosaic virus': '花叶病毒',
    'Tomato leaf yellow virus': '黄化病毒',
    'Tomato mold leaf': '霉病叶片',
    'Tomato two spotted spider mites leaf': '叶螨叶片',
    'grape leaf': '茶黑毒蛾',
    'grape leaf black rot': '黑腐病叶片'
}

# 可用模型列表
AVAILABLE_MODELS = [
    {
        'key': key,
        'name': config['name'],
        'model_path': config['model_path'],
        'train_results_path': config.get('train_results_path'),
        'val_data_path': config.get('val_data_path'),
        'val_accuracy_path': config.get('val_accuracy_path'),
        'num_classes': len(CLASS_NAME_MAPPING),
        'supported_classes': list(CLASS_NAME_MAPPING.values())
    }
    for key, config in MODEL_CONFIGS.items()
]

# JWT 配置
JWT_SECRET = os.environ.get(
    'JWT_SECRET', 'tea-detect-jwt-secret-2025-change-in-production'
)
JWT_EXPIRE_SECONDS = 24 * 3600  # 24 小时



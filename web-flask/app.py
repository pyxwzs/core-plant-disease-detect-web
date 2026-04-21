

"""
智慧茶叶病害检测系统 - Flask后端
"""
import base64
import hashlib
import hmac as _hmac
import json as _json
import os
import json
import uuid
import tempfile
import time
import re
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from config import (
    DETECTION_IMAGES_DIR,
    DETECTION_VIDEOS_DIR,
    PDF_KNOWLEDGE_DIR,
    KNOWLEDGE_UPLOAD_DIR,
    JWT_SECRET,
    JWT_EXPIRE_SECONDS,
)
from service.user_service import (
    login_user,
    register_user,
    init_default_users,
    verify_admin_credentials,
    list_users,
    set_user_status,
    reset_user_password,
    delete_user,
)
from service.knowledge_service import (
    init_knowledge_table,
    list_knowledge_items,
    get_knowledge_item,
    create_knowledge_item,
    update_knowledge_item,
    delete_knowledge_item,
    save_knowledge_image,
    safe_knowledge_image_name,
)
from service.log_service import (
    init_log_table,
    write_log,
    list_logs,
    get_log,
    export_logs_csv,
)
from service.detection_service import get_models, detect_objects
from service.model_data_service import get_model_data
from service.video_detection_service import start_video_processing, get_processing_status
from service.realtime_detection_service import detect_objects_realtime
from service.detection_history_service import (
    init_detection_history_table,
    save_detection_record,
    save_video_history_record,
    save_realtime_history_record,
    list_detection_history,
    get_detection_record,
    get_detection_frames,
    delete_detection_records,
    safe_image_filename,
    safe_video_filename,
)

# 模板目录（基于 app.py 位置，避免启动时工作目录不在 web-flask 导致找不到文件）
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES_DIR = os.path.join(_APP_DIR, 'templates')

# ==================== JWT 工具 ====================

def _jwt_b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode()


def create_jwt(payload: dict) -> str:
    """生成 HS256 JWT，有效期由 JWT_EXPIRE_SECONDS 控制。"""
    header = _jwt_b64url(b'{"alg":"HS256","typ":"JWT"}')
    body = _jwt_b64url(
        _json.dumps({**payload, 'exp': int(time.time()) + JWT_EXPIRE_SECONDS},
                    ensure_ascii=False).encode()
    )
    msg = f"{header}.{body}".encode()
    sig = _jwt_b64url(
        _hmac.new(JWT_SECRET.encode(), msg, hashlib.sha256).digest()
    )
    return f"{header}.{body}.{sig}"


def verify_jwt(token: str) -> dict | None:
    """验证 JWT；过期或签名错误返回 None。"""
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        header, body, sig = parts
        msg = f"{header}.{body}".encode()
        expected = _jwt_b64url(
            _hmac.new(JWT_SECRET.encode(), msg, hashlib.sha256).digest()
        )
        if not _hmac.compare_digest(sig, expected):
            return None
        payload = _json.loads(base64.urlsafe_b64decode(body + '=='))
        if payload.get('exp', 0) < time.time():
            return None
        return payload
    except Exception:
        return None


app = Flask(__name__)
CORS(app)

# ==================== 页面路由 ====================

@app.route('/')
@app.route('/login.html')
def index():
    """登录页面"""
    login_html_path = os.path.join(_TEMPLATES_DIR, 'login.html')
    with open(login_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/images/<filename>')
def serve_images(filename):
    """提供images目录下的静态文件"""
    images_path = os.path.join(_TEMPLATES_DIR, 'images')
    return send_from_directory(images_path, filename)


@app.route('/detect.html')
def detect_html():
    """茶叶病害检测页面"""
    detect_html_path = os.path.join(_TEMPLATES_DIR, 'detect.html')
    with open(detect_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}



@app.route('/model-data.html')
def model_data_html():
    """模型数据查看页面"""
    model_data_html_path = os.path.join(_TEMPLATES_DIR, 'model-data.html')
    with open(model_data_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/video-detect.html')
def video_detect_html():
    """视频病害检测页面"""
    video_detect_html_path = os.path.join(_TEMPLATES_DIR, 'video-detect.html')
    with open(video_detect_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/realtime-detect.html')
def realtime_detect_html():
    """实时摄像头检测页面"""
    realtime_detect_html_path = os.path.join(_TEMPLATES_DIR, 'realtime-detect.html')
    with open(realtime_detect_html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/detection-history')
@app.route('/detection-history.html')
def detection_history_html():
    """检测历史记录页面"""
    path = os.path.join(_TEMPLATES_DIR, 'detection-history.html')
    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/tea-knowledge')
@app.route('/tea-knowledge.html')
def tea_knowledge_html():
    """茶树病虫害相关知识与资料页面"""
    path = os.path.join(_TEMPLATES_DIR, 'tea-knowledge.html')
    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/tea-knowledge-admin')
@app.route('/tea-knowledge-admin.html')
def tea_knowledge_admin_html():
    """病虫害知识库管理（管理员）"""
    path = os.path.join(_TEMPLATES_DIR, 'tea-knowledge-admin.html')
    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/user-manage')
@app.route('/user-manage.html')
def user_manage_html():
    """用户管理（管理员）"""
    path = os.path.join(_TEMPLATES_DIR, 'user-manage.html')
    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route('/log-manage')
@app.route('/log-manage.html')
def log_manage_html():
    """系统日志管理（管理员）"""
    path = os.path.join(_TEMPLATES_DIR, 'log-manage.html')
    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}


def _list_pdf_books():
    """列出 data/pdf 目录下的 PDF 文件名与大小（仅一层目录，防路径穿越）。"""
    out = []
    if not os.path.isdir(PDF_KNOWLEDGE_DIR):
        return out
    for name in sorted(os.listdir(PDF_KNOWLEDGE_DIR), key=lambda x: x.lower()):
        if not name.lower().endswith('.pdf'):
            continue
        fp = os.path.join(PDF_KNOWLEDGE_DIR, name)
        if not os.path.isfile(fp):
            continue
        title = os.path.splitext(name)[0]
        out.append({
            'filename': name,
            'title': title,
            'size_bytes': os.path.getsize(fp),
        })
    return out


# ==================== 用户API ====================

@app.route('/api/user/login', methods=['POST'])
def api_login():
    """用户登录 API，成功后返回 JWT token。"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        result = login_user(username, password)
        if result.get('code') == 200:
            token = create_jwt({
                'sub': username,
                'role': result['data']['user']['role'],
                'realName': result['data']['user'].get('realName', username),
            })
            result['data']['token'] = token
        write_log(
            "user_action", "login",
            f"用户 {username} 登录{'成功' if result.get('code') == 200 else '失败'}",
            operator=username,
            result="success" if result.get("code") == 200 else "fail",
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'登录失败: {str(e)}'})


@app.route('/api/user/register', methods=['POST'])
@app.route('/api/register', methods=['POST'])
def api_register():
    """用户注册API"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        real_name = data.get('realName')
        result = register_user(username, password, real_name)
        write_log(
            "user_action", "register",
            f"新用户注册：{username}，结果：{result.get('message')}",
            operator=username,
            result="success" if result.get("code") == 200 else "fail",
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'注册失败: {str(e)}'})


# ==================== 植物病害检测API ====================

@app.route('/api/models', methods=['GET'])
def api_get_models():
    """获取可用模型列表"""
    result = get_models()
    return jsonify(result)


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """茶叶病害检测接口"""
    try:
        data = request.json
        model_name = data.get('model', 'ready-model')
        image_data = data.get('image')

        if not image_data:
            return jsonify({'code': 400, 'message': '请提供图像数据'})

        result = detect_objects(model_name, image_data)
        if result.get('code') == 200 and isinstance(result.get('data'), dict):
            hist = save_detection_record(model_name, image_data, result['data'])
            if hist.get('code') == 200:
                result['data']['history_record_id'] = hist.get('record_id')
        return jsonify(result)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'茶叶病害检测失败: {str(e)}'})


@app.route('/api/detection-history', methods=['GET'])
def api_detection_history_list():
    """检测历史列表（摘要）。type=image|video|realtime 筛选。"""
    try:
        limit = request.args.get('limit', 200, type=int)
        src = request.args.get('type') or request.args.get('source')
        if src not in (None, '', 'image', 'video', 'realtime'):
            src = None
        return jsonify(list_detection_history(limit=limit, source=src))
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/detection-history/<record_id>', methods=['GET'])
def api_detection_history_detail(record_id):
    """单条检测历史详情（不含 frame_details，避免过大）"""
    try:
        return jsonify(get_detection_record(record_id))
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/detection-history/<record_id>/frames', methods=['GET'])
def api_detection_history_frames(record_id):
    """视频记录的分页帧数据。?page=1&size=50&only_detections=0"""
    try:
        page = request.args.get('page', 1, type=int)
        size = request.args.get('size', 50, type=int)
        only_det = request.args.get('only_detections', '0') in ('1', 'true', 'yes')
        return jsonify(get_detection_frames(record_id, page=page, size=size,
                                            only_detections=only_det))
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/detection-history', methods=['DELETE'])
def api_detection_history_delete():
    """批量删除检测记录。Body: {"ids": ["id1", "id2", ...]}"""
    try:
        body = request.get_json(silent=True) or {}
        ids = body.get('ids') or []
        return jsonify(delete_detection_records(ids))
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/detection-history/images/<filename>')
def api_detection_history_image(filename):
    """历史记录中的图片（原图、标注图、视频封面、实时帧）"""
    safe = safe_image_filename(filename)
    if not safe:
        return jsonify({'code': 400, 'message': '无效的文件名'}), 400
    return send_from_directory(DETECTION_IMAGES_DIR, safe)


@app.route('/api/detection-history/videos/<filename>')
def api_detection_history_video(filename):
    """历史记录中保存的处理后视频"""
    safe = safe_video_filename(filename)
    if not safe:
        return jsonify({'code': 400, 'message': '无效的文件名'}), 400
    return send_from_directory(DETECTION_VIDEOS_DIR, safe)


@app.route('/api/tea-knowledge/pdfs', methods=['GET'])
def api_tea_knowledge_pdfs():
    """列出 data/pdf 中的茶树病虫害相关 PDF 资料。"""
    try:
        books = _list_pdf_books()
        return jsonify({'code': 200, 'data': books})
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/tea-knowledge/pdf', methods=['GET'])
def api_tea_knowledge_pdf():
    """在线预览或下载 PDF。参数 f=文件名（单层，禁止路径符号）。"""
    name = request.args.get('f', '') or ''
    if not name or '/' in name or '\\' in name or '..' in name:
        return jsonify({'code': 400, 'message': '无效的文件名'}), 400
    fp = os.path.join(PDF_KNOWLEDGE_DIR, name)
    if not os.path.isfile(fp):
        return jsonify({'code': 404, 'message': '文件不存在'}), 404
    return send_from_directory(PDF_KNOWLEDGE_DIR, name, mimetype='application/pdf')


def _require_admin():
    """
    管理员鉴权：优先 Bearer JWT，兼容 HTTP Basic（API 调试工具用）。
    """
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        payload = verify_jwt(auth_header[7:])
        if payload and payload.get('role') == 'admin':
            return None
        return jsonify({'code': 403, 'message': '无管理员权限或登录已过期，请重新登录'}), 403
    # 兼容 HTTP Basic（仅用于 curl / Postman 等工具调试）
    auth = request.authorization
    if not auth or not auth.username or not auth.password:
        resp = jsonify({'code': 401, 'message': '需要管理员认证'})
        resp.status_code = 401
        resp.headers['WWW-Authenticate'] = 'Basic realm="Admin"'
        return resp
    if not verify_admin_credentials(auth.username, auth.password):
        return jsonify({'code': 403, 'message': '无管理员权限或账号已禁用'}), 403
    return None


def _get_operator() -> str:
    """从 Bearer JWT 或 HTTP Basic Auth 中提取操作者用户名。"""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        payload = verify_jwt(auth_header[7:])
        if payload:
            return payload.get('sub', '')
    if request.authorization:
        return request.authorization.username or ''
    return ''


@app.route('/api/knowledge/items', methods=['GET'])
def api_knowledge_items_list():
    """前台：仅上架条目，分页与关键词、类型筛选。"""
    try:
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 20, type=int)
        q = request.args.get('q', '') or ''
        kind = request.args.get('kind', '') or ''
        return jsonify(
            list_knowledge_items(
                page=page,
                page_size=page_size,
                q=q or None,
                kind=kind or None,
                public_only=True,
            )
        )
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/knowledge/items/<int:item_id>', methods=['GET'])
def api_knowledge_item_detail(item_id):
    """前台：单条详情（仅上架）。"""
    try:
        return jsonify(get_knowledge_item(item_id, public_only=True))
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/knowledge/images/<filename>')
def api_knowledge_image(filename):
    """知识库参考图静态访问。"""
    safe = safe_knowledge_image_name(filename)
    if not safe:
        return jsonify({'code': 400, 'message': '无效的文件名'}), 400
    fp = os.path.join(KNOWLEDGE_UPLOAD_DIR, safe)
    if not os.path.isfile(fp):
        return jsonify({'code': 404, 'message': '文件不存在'}), 404
    return send_from_directory(KNOWLEDGE_UPLOAD_DIR, safe)


@app.route('/api/admin/knowledge/items', methods=['GET'])
def api_admin_knowledge_list():
    """管理员：全部状态，可筛选。"""
    err = _require_admin()
    if err:
        return err
    try:
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 20, type=int)
        q = request.args.get('q', '') or ''
        kind = request.args.get('kind', '') or ''
        status = request.args.get('status', type=int)
        time_from = request.args.get('time_from', '') or ''
        time_to = request.args.get('time_to', '') or ''
        return jsonify(
            list_knowledge_items(
                page=page,
                page_size=page_size,
                q=q or None,
                kind=kind or None,
                status=status,
                public_only=False,
                time_from=time_from or None,
                time_to=time_to or None,
            )
        )
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/admin/knowledge/items', methods=['POST'])
def api_admin_knowledge_create():
    err = _require_admin()
    if err:
        return err
    try:
        data = request.get_json(silent=True) or {}
        r = create_knowledge_item(
            title=data.get('title'),
            kind=data.get('kind') or '病害',
            symptom=data.get('symptom') or '',
            pattern_text=data.get('pattern_text') or '',
            harm=data.get('harm') or '',
            prevention=data.get('prevention') or '',
            images=data.get('images') or [],
            status=data.get('status', 1),
        )
        if r.get('code') == 200:
            operator = _get_operator()
            write_log("admin_action", "create_knowledge",
                      f"新增知识条目「{data.get('title')}」",
                      operator=operator, target_type="knowledge_item",
                      target_id=r.get('data', {}).get('id'))
        return jsonify(r)
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/admin/knowledge/items/<int:item_id>', methods=['PUT'])
def api_admin_knowledge_update(item_id):
    err = _require_admin()
    if err:
        return err
    try:
        data = request.get_json(silent=True) or {}
        allowed_keys = (
            'title', 'kind', 'symptom', 'pattern_text', 'harm',
            'prevention', 'images', 'status',
        )
        payload = {k: data[k] for k in allowed_keys if k in data}
        r = update_knowledge_item(item_id, **payload)
        if r.get('code') == 200:
            operator = _get_operator()
            action = "toggle_knowledge_status" if list(payload.keys()) == ['status'] else "update_knowledge"
            write_log("admin_action", action,
                      f"编辑知识条目 ID={item_id}，字段：{list(payload.keys())}",
                      operator=operator, target_type="knowledge_item", target_id=item_id)
        return jsonify(r)
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/admin/knowledge/items/<int:item_id>', methods=['DELETE'])
def api_admin_knowledge_delete(item_id):
    err = _require_admin()
    if err:
        return err
    try:
        r = delete_knowledge_item(item_id)
        if r.get('code') == 200:
            operator = _get_operator()
            write_log("admin_action", "delete_knowledge",
                      f"删除知识条目 ID={item_id}",
                      operator=operator, target_type="knowledge_item", target_id=item_id)
        return jsonify(r)
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/admin/knowledge/upload', methods=['POST'])
def api_admin_knowledge_upload():
    err = _require_admin()
    if err:
        return err
    try:
        if 'file' not in request.files:
            return jsonify({'code': 400, 'message': '缺少 file 字段'})
        return jsonify(save_knowledge_image(request.files['file']))
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


# ─────────────── 管理员用户管理 API ───────────────

@app.route('/api/admin/users', methods=['GET'])
def api_admin_list_users():
    err = _require_admin()
    if err:
        return err
    try:
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 20, type=int)
        q = request.args.get('q', '') or ''
        return jsonify(list_users(page=page, page_size=page_size, q=q or None))
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/admin/users/<username>/status', methods=['PUT'])
def api_admin_set_user_status(username):
    err = _require_admin()
    if err:
        return err
    try:
        data = request.get_json(force=True) or {}
        status = data.get('status')
        if status is None:
            return jsonify({'code': 400, 'message': '缺少 status 字段'})
        operator = _get_operator()
        r = set_user_status(username, status, operator=operator or None)
        if r.get('code') == 200:
            label = '启用' if int(status) else '禁用'
            write_log("admin_action", "set_user_status",
                      f"管理员 {operator} 将用户 {username} 设为{label}",
                      operator=operator, target_type="user", target_id=username)
        return jsonify(r)
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/admin/users/<username>/password', methods=['PUT'])
def api_admin_reset_password(username):
    err = _require_admin()
    if err:
        return err
    try:
        data = request.get_json(force=True) or {}
        new_password = data.get('new_password', '').strip()
        operator = _get_operator()
        r = reset_user_password(username, new_password)
        if r.get('code') == 200:
            write_log("admin_action", "reset_password",
                      f"管理员 {operator} 重置了用户 {username} 的密码",
                      operator=operator, target_type="user", target_id=username)
        return jsonify(r)
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/admin/users/<username>', methods=['DELETE'])
def api_admin_delete_user(username):
    err = _require_admin()
    if err:
        return err
    try:
        operator = _get_operator()
        r = delete_user(username, operator=operator or None)
        if r.get('code') == 200:
            write_log("admin_action", "delete_user",
                      f"管理员 {operator} 删除了用户 {username}",
                      operator=operator, target_type="user", target_id=username)
        return jsonify(r)
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


# ─────────────── 系统日志 API ───────────────

@app.route('/api/admin/logs', methods=['GET'])
def api_admin_logs():
    err = _require_admin()
    if err:
        return err
    try:
        return jsonify(list_logs(
            page=request.args.get('page', 1, type=int),
            page_size=request.args.get('page_size', 20, type=int),
            log_type=request.args.get('log_type') or None,
            operator=request.args.get('operator') or None,
            q=request.args.get('q') or None,
            time_from=request.args.get('time_from') or None,
            time_to=request.args.get('time_to') or None,
        ))
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/admin/logs/<int:log_id>', methods=['GET'])
def api_admin_log_detail(log_id):
    err = _require_admin()
    if err:
        return err
    try:
        return jsonify(get_log(log_id))
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/admin/logs/export', methods=['GET'])
def api_admin_logs_export():
    err = _require_admin()
    if err:
        return err
    try:
        csv_text = export_logs_csv(
            log_type=request.args.get('log_type') or None,
            operator=request.args.get('operator') or None,
            q=request.args.get('q') or None,
            time_from=request.args.get('time_from') or None,
            time_to=request.args.get('time_to') or None,
        )
        from datetime import date
        fname = f"sys_logs_{date.today().isoformat()}.csv"
        return Response(
            '\ufeff' + csv_text,  # UTF-8 BOM，让 Excel 正确识别中文
            mimetype='text/csv; charset=utf-8-sig',
            headers={'Content-Disposition': f'attachment; filename="{fname}"'},
        )
    except Exception as e:
        return jsonify({'code': 500, 'message': str(e)})


@app.route('/api/model-data', methods=['GET'])
def api_get_model_data():
    """获取模型训练和验证数据"""
    try:
        data_type = request.args.get('type')  # 'training' or 'validation'
        model_key = request.args.get('model')

        if not data_type or not model_key:
            return jsonify({'code': 400, 'message': '缺少必要参数'})

        result = get_model_data(data_type, model_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'获取模型数据失败: {str(e)}'})


# ==================== 视频检测API ====================

@app.route('/api/video/process', methods=['POST'])
def api_video_process():
    """视频处理接口（流式响应）"""

    try:
        # 检查是否有上传的文件
        if 'video' not in request.files:
            return jsonify({'code': 400, 'message': '请上传视频文件'})

        video_file = request.files['video']
        model_name = request.form.get('model', 'ready-model')

        if video_file.filename == '':
            return jsonify({'code': 400, 'message': '请选择视频文件'})

        # 生成唯一的会话ID
        session_id = str(uuid.uuid4())

        # 保存上传的视频文件到临时目录
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, f"input_{session_id}.mp4")
        video_file.save(video_path)

        # 开始异步处理视频
        processor = start_video_processing(session_id, model_name, video_path)

        def generate_progress():
            """生成进度流"""
            yield "data: " + json.dumps({
                'type': 'start',
                'session_id': session_id,
                'message': '开始处理视频...'
            }) + "\n\n"

            last_progress = -1
            while True:
                status = processor.get_status()

                if status['error']:
                    yield "data: " + json.dumps({
                        'type': 'error',
                        'message': status['error']
                    }) + "\n\n"
                    break

                # 发送进度更新
                if status['progress'] != last_progress:
                    yield "data: " + json.dumps({
                        'type': 'progress',
                        'progress': status['progress'],
                        'message': status['message']
                    }) + "\n\n"
                    last_progress = status['progress']

                # 检查是否完成
                if not status['is_processing'] and status['result']:
                    # 生成视频访问URL
                    video_url = f"/api/video/download/{session_id}"

                    result_data = status['result'].copy()
                    try:
                        # 把原始视频路径也传入，方便历史记录同时保存原视频
                        result_data['original_video_path'] = video_path
                        save_video_history_record(model_name, session_id, result_data)
                    except Exception as ex:
                        print('保存视频检测历史失败:', ex)

                    # SSE 推送给浏览器时去掉超大的全量帧数据，避免前端卡顿
                    stream_data = {k: v for k, v in result_data.items()
                                   if k not in ('frame_details', 'original_video_path')}
                    stream_data['frame_details'] = (result_data.get('frame_details') or [])[:100]
                    stream_data['processed_video_url'] = video_url
                    stream_data['session_id'] = session_id

                    yield "data: " + json.dumps({
                        'type': 'result',
                        'data': stream_data
                    }) + "\n\n"
                    break

                if not status['is_processing'] and not status['result']:
                    yield "data: " + json.dumps({
                        'type': 'error',
                        'message': '处理失败，未知错误'
                    }) + "\n\n"
                    break

                time.sleep(1)  # 每秒检查一次状态

        return Response(
            generate_progress(),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )

    except Exception as e:
        return jsonify({'code': 500, 'message': f'视频处理失败: {str(e)}'})


@app.route('/api/video/download/<session_id>')
def api_video_download(session_id):
    """视频下载接口，支持范围请求以实现视频流播放"""
    try:
        status_result = get_processing_status(session_id)

        if status_result['code'] != 200:
            return jsonify(status_result)

        status = status_result['data']

        if not status['result']:
            return jsonify({'code': 404, 'message': '视频文件不存在'})

        processed_video_path = status['result']['processed_video_path']

        if not os.path.exists(processed_video_path):
            return jsonify({'code': 404, 'message': '处理后的视频文件不存在'})

        # 获取文件信息
        file_size = os.path.getsize(processed_video_path)
        file_ext = os.path.splitext(processed_video_path)[1].lower()

        # 确定MIME类型
        if file_ext == '.mp4':
            mimetype = 'video/mp4'
        elif file_ext == '.avi':
            mimetype = 'video/x-msvideo'
        elif file_ext == '.mov':
            mimetype = 'video/quicktime'
        else:
            mimetype = 'video/mp4'  # 默认

        # 处理Range请求
        range_header = request.headers.get('Range', None)
        if range_header:
            # 解析Range头
            byte_start = 0
            byte_end = file_size - 1

            if range_header:
                match = re.match(r'bytes=(\d+)-(\d*)', range_header)
                if match:
                    byte_start = int(match.group(1))
                    if match.group(2):
                        byte_end = int(match.group(2))

            # 确保范围有效
            byte_start = max(0, byte_start)
            byte_end = min(file_size - 1, byte_end)
            content_length = byte_end - byte_start + 1

            # 读取文件的指定范围
            with open(processed_video_path, 'rb') as f:
                f.seek(byte_start)
                data = f.read(content_length)

            # 创建部分内容响应
            response = Response(
                data,
                206,  # Partial Content
                {
                    'Content-Type': mimetype,
                    'Accept-Ranges': 'bytes',
                    'Content-Length': str(content_length),
                    'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                    'Cache-Control': 'no-cache'
                }
            )
            return response

        # 没有Range请求，返回完整文件
        response = send_from_directory(
            os.path.dirname(processed_video_path),
            os.path.basename(processed_video_path),
            as_attachment=False,
            mimetype=mimetype
        )

        # 添加支持范围请求的头
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Type'] = mimetype
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Content-Length'] = str(file_size)

        return response

    except Exception as e:
        return jsonify({'code': 500, 'message': f'视频下载失败: {str(e)}'})


@app.route('/api/video/status/<session_id>')
def api_video_status(session_id):
    """获取视频处理状态"""
    try:
        result = get_processing_status(session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'获取状态失败: {str(e)}'})


# ==================== 实时检测API ====================

@app.route('/api/realtime/detect', methods=['POST'])
def api_realtime_detect():
    """实时摄像头检测接口"""
    try:
        data = request.json
        model_name = data.get('model', 'ready-model')
        image_data = data.get('image')

        if not image_data:
            return jsonify({'code': 400, 'message': '请提供图像数据'})

        result = detect_objects_realtime(model_name, image_data)
        if result.get('code') == 200 and isinstance(result.get('data'), dict):
            try:
                save_realtime_history_record(model_name, image_data, result['data'])
            except Exception as ex:
                print('保存实时检测历史失败:', ex)
        return jsonify(result)
    except Exception as e:
        return jsonify({'code': 500, 'message': f'实时检测失败: {str(e)}'})


# ==================== 应用启动 ====================

if __name__ == '__main__':
    # 初始化默认用户、知识库表、日志表、检测历史表
    init_default_users()
    init_knowledge_table()
    init_log_table()
    init_detection_history_table()

    # 收集所有模板文件，使 Flask reloader 在 HTML 修改后也自动重启
    _tmpl_dir = os.path.join(os.path.dirname(__file__), 'templates')
    _extra = [
        os.path.join(_tmpl_dir, f)
        for f in os.listdir(_tmpl_dir)
        if f.endswith('.html')
    ]
    app.run(host='0.0.0.0', port=5011, debug=True, use_reloader=True, extra_files=_extra)

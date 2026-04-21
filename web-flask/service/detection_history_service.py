"""
检测历史：图像 / 视频 / 实时三种记录格式，使用 SQLite 持久化。

record_json 字段存储完整 JSON（替代原来的 records/{id}.json 文件）。
图片 / 视频二进制文件仍保存在文件系统。

表结构：
  detection_history(id, source, created_at, model, summary,
                    total_detections, preview_filename, preview_kind,
                    top_detections TEXT, record_json TEXT, md5)

迁移：首次调用 init_detection_history_table() 时，若 index.json 存在
且表为空，自动将历史数据导入 SQLite。
"""
import base64
import contextlib
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from io import BytesIO

import cv2
from PIL import Image

from config import (
    DETECTION_DB_PATH,
    DETECTION_HISTORY_DIR,
    DETECTION_IMAGES_DIR,
    DETECTION_INDEX_FILE,
    DETECTION_RECORDS_DIR,
    DETECTION_VIDEOS_DIR,
)


# ──────────────────── 目录 ────────────────────
def _ensure_dirs():
    os.makedirs(DETECTION_RECORDS_DIR, exist_ok=True)
    os.makedirs(DETECTION_IMAGES_DIR, exist_ok=True)
    os.makedirs(DETECTION_VIDEOS_DIR, exist_ok=True)


# ──────────────────── SQLite ────────────────────
@contextlib.contextmanager
def _connect():
    _ensure_dirs()
    conn = sqlite3.connect(DETECTION_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_detection_history_table():
    """建表；若 index.json 存在且表为空则自动迁移旧数据。"""
    _ensure_dirs()
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS detection_history (
                id               TEXT PRIMARY KEY,
                source           TEXT NOT NULL DEFAULT 'image',
                created_at       TEXT NOT NULL,
                model            TEXT,
                summary          TEXT,
                total_detections INTEGER DEFAULT 0,
                preview_filename TEXT,
                preview_kind     TEXT DEFAULT 'image',
                top_detections   TEXT DEFAULT '[]',
                record_json      TEXT DEFAULT '{}',
                md5              TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_dh_created_at
                ON detection_history(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_dh_source
                ON detection_history(source);
            CREATE INDEX IF NOT EXISTS idx_dh_model
                ON detection_history(model);
        """)

        # 若表为空，尝试从 index.json 迁移
        n = conn.execute("SELECT COUNT(*) FROM detection_history").fetchone()[0]
        if n == 0 and os.path.isfile(DETECTION_INDEX_FILE):
            _migrate_from_json(conn)


def _migrate_from_json(conn):
    """将 index.json + records/*.json 的旧数据迁移进 SQLite。"""
    try:
        with open(DETECTION_INDEX_FILE, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    except Exception:
        return
    if not isinstance(entries, list):
        return

    migrated = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rid = entry.get('id', '')
        if not rid:
            continue

        # 读取对应的完整 record JSON
        record_json = {}
        rec_path = os.path.join(DETECTION_RECORDS_DIR, f'{rid}.json')
        if os.path.isfile(rec_path):
            try:
                with open(rec_path, 'r', encoding='utf-8') as f:
                    record_json = json.load(f)
            except Exception:
                record_json = {}

        top_dets = entry.get('top_detections') or []
        try:
            conn.execute(
                """INSERT OR IGNORE INTO detection_history
                   (id, source, created_at, model, summary, total_detections,
                    preview_filename, preview_kind, top_detections, record_json, md5)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rid,
                    entry.get('source') or 'image',
                    entry.get('created_at') or '',
                    entry.get('model') or '',
                    entry.get('summary') or '',
                    int(entry.get('total_detections') or 0),
                    entry.get('preview_filename') or '',
                    entry.get('preview_kind') or 'image',
                    json.dumps(top_dets, ensure_ascii=False),
                    json.dumps(record_json, ensure_ascii=False),
                    record_json.get('md5') or entry.get('md5') or '',
                )
            )
            migrated += 1
        except Exception:
            pass

    print(f"[detection_history] 从 index.json 迁移了 {migrated} 条历史记录到 SQLite")


def _insert_record(entry: dict, record: dict):
    """向 detection_history 表插入一条记录。"""
    top_dets = entry.get('top_detections') or []
    with _connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO detection_history
               (id, source, created_at, model, summary, total_detections,
                preview_filename, preview_kind, top_detections, record_json, md5)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                entry['id'],
                entry.get('source') or 'image',
                entry.get('created_at') or '',
                entry.get('model') or '',
                entry.get('summary') or '',
                int(entry.get('total_detections') or 0),
                entry.get('preview_filename') or '',
                entry.get('preview_kind') or 'image',
                json.dumps(top_dets, ensure_ascii=False),
                json.dumps(record, ensure_ascii=False),
                record.get('md5') or '',
            )
        )


# ──────────────────── 图片工具 ────────────────────
def _decode_base64_raw(image_data):
    if image_data.startswith('data:image'):
        image_data = image_data.split(',', 1)[1]
    return base64.b64decode(image_data)


def _guess_ext_from_bytes(raw_bytes):
    try:
        img = Image.open(BytesIO(raw_bytes))
        fmt = (img.format or 'JPEG').upper()
        if fmt == 'JPEG': return 'jpg'
        if fmt == 'PNG':  return 'png'
        if fmt == 'WEBP': return 'webp'
        if fmt == 'BMP':  return 'bmp'
    except Exception:
        pass
    return 'jpg'


# ──────────────────── 扁平化检测数据 ────────────────────
def _flatten_image_detect(detect_data, record_id):
    if not isinstance(detect_data, dict):
        return 0, [], None
    raw_dets = detect_data.get('detections') or []
    total = int(detect_data.get('total_detections') or len(raw_dets))
    detections = []
    for i, d in enumerate(raw_dets, start=1):
        try:
            conf = float(d.get('confidence', 0))
        except (TypeError, ValueError):
            conf = 0.0
        detections.append({
            'rank': i,
            'class_name_zh': d.get('class_name_zh') or '',
            'class_name': d.get('class_name') or '',
            'confidence': round(conf, 4),
            'percentage': d.get('percentage') or f"{conf * 100:.2f}%",
        })
    annotated_file = None
    di = detect_data.get('detection_image')
    if di and isinstance(di, str) and 'base64' in di:
        try:
            b64 = di.split(',', 1)[1] if 'data:image' in di else di
            raw = base64.b64decode(b64)
            ann_name = f'{record_id}_annotated.jpg'
            ann_path = os.path.join(DETECTION_IMAGES_DIR, ann_name)
            with open(ann_path, 'wb') as f:
                f.write(raw)
            annotated_file = f'images/{ann_name}'
        except Exception:
            pass
    return total, detections, annotated_file


def _flatten_realtime_detect(data):
    raw_dets = data.get('detections') or []
    detections = []
    for i, d in enumerate(raw_dets, start=1):
        try:
            conf = float(d.get('confidence', 0))
        except (TypeError, ValueError):
            conf = 0.0
        detections.append({
            'rank': i,
            'class_name_zh': d.get('class_name_zh') or '',
            'class_name': d.get('class_name_en') or d.get('class_name') or '',
            'confidence': round(conf, 4),
            'percentage': f"{conf * 100:.2f}%",
        })
    return len(detections), detections


def _flatten_video_stats(result_data):
    if not isinstance(result_data, dict):
        return {}
    stats = result_data.get('detection_statistics') or {}
    classes = stats.get('class_statistics') or []
    total = int(result_data.get('total_detections') or 0)
    top_classes = []
    for i, c in enumerate(classes[:8], start=1):
        count = int(c.get('count') or 0)
        pct_val = round(count / total * 100, 1) if total > 0 else 0.0
        top_classes.append({
            'rank': i,
            'class_name_zh': c.get('class_name_zh') or '',
            'class_name': c.get('class_name_en') or c.get('class_name') or '',
            'count': count,
            'percentage': f"{pct_val:.1f}%",
            'pct_val': pct_val,
        })
    vi = result_data.get('video_info') or {}
    return {
        'total_frames': result_data.get('total_frames'),
        'total_detections': total,
        'frames_with_detections': result_data.get('frames_with_detections'),
        'processing_time': result_data.get('processing_time'),
        'duration_sec': round(float(vi.get('duration') or 0), 2),
        'top_classes': top_classes,
    }


def _full_video_stats(result_data):
    if not isinstance(result_data, dict):
        return {}
    det_stats = result_data.get('detection_statistics') or {}
    vi = result_data.get('video_info') or {}
    return {
        'video_info': vi,
        'class_statistics': det_stats.get('class_statistics') or [],
        'detection_rate': det_stats.get('detection_rate'),
        'avg_detections_per_frame': det_stats.get('avg_detections_per_frame'),
        'total_classes_detected': det_stats.get('total_classes_detected'),
        'frame_details': result_data.get('frame_details') or [],
    }


# ──────────────────── top_detections 生成 ────────────────────
def _make_top_for_index(detections, n=3, video_top_classes=None):
    source = detections if video_top_classes is None else video_top_classes
    result = []
    for d in (source or [])[:n]:
        if video_top_classes is not None:
            pct_val = float(d.get('pct_val') or 0)
        else:
            pct_val = round(float(d.get('confidence') or 0) * 100, 1)
        result.append({
            'rank': d.get('rank', len(result) + 1),
            'class_name_zh': d.get('class_name_zh') or '',
            'class_name': d.get('class_name') or d.get('class_name_en') or '',
            'percentage': d.get('percentage') or f"{pct_val:.1f}%",
            'pct_val': pct_val,
        })
    return result


# ──────────────────── 摘要 ────────────────────
def _summary_from_detections(total, detections, prefix="检出"):
    if total == 0 or not detections:
        return '未检出病害'
    names = []
    for d in detections:
        z = d.get('class_name_zh') or d.get('class_name') or ''
        if z and z not in names:
            names.append(z)
        if len(names) >= 3:
            break
    more = '等' if len(detections) > 3 else ''
    return f"{prefix} {total} 处：{'、'.join(names)}{more}"


def _summary_video(stats):
    total = int(stats.get('total_detections') or 0)
    if total == 0:
        return '视频中未检出病害'
    names = [x.get('class_name_zh') or x.get('class_name') or ''
             for x in stats.get('top_classes') or [] if x.get('class_name_zh') or x.get('class_name')]
    names = [n for n in names if n][:3]
    tail = '等' if len(stats.get('top_classes') or []) > 3 else ''
    dur = stats.get('duration_sec')
    dur_s = f"，时长 {dur:.1f}s" if dur else ''
    return f"共检出 {total} 次{dur_s}，涉及：{'、'.join(names)}{tail}"


# ──────────────────── 保存记录 ────────────────────
def save_detection_record(model_name, image_data, detect_data):
    """图像检测：保存原图 + 标注图 + 扁平化结果入库。"""
    try:
        raw_bytes = _decode_base64_raw(image_data)
    except Exception as e:
        return {'code': 400, 'message': f'图片数据无效: {e}'}
    if not raw_bytes:
        return {'code': 400, 'message': '图片数据为空'}

    md5_hex = hashlib.md5(raw_bytes).hexdigest()
    ext = _guess_ext_from_bytes(raw_bytes)
    record_id = f"{int(time.time() * 1000)}_{md5_hex[:8]}"

    _ensure_dirs()
    image_filename = f"{record_id}.{ext}"
    image_rel = f"images/{image_filename}"
    abs_image = os.path.join(DETECTION_IMAGES_DIR, image_filename)
    try:
        with open(abs_image, 'wb') as f:
            f.write(raw_bytes)
    except OSError as e:
        return {'code': 500, 'message': f'保存图片失败: {e}'}

    created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
    total, detections, annotated_file = _flatten_image_detect(detect_data, record_id)
    summary = _summary_from_detections(total, detections)

    record = {
        'id': record_id,
        'source': 'image',
        'created_at': created_at,
        'md5': md5_hex,
        'model': model_name,
        'summary': summary,
        'total_detections': total,
        'image_file': image_rel,
        'annotated_file': annotated_file,
        'detections': detections,
    }

    preview_name = (annotated_file or image_rel).replace('images/', '', 1)
    entry = {
        'id': record_id,
        'created_at': created_at,
        'source': 'image',
        'model': model_name,
        'summary': summary,
        'total_detections': total,
        'preview_filename': preview_name,
        'preview_kind': 'image',
        'top_detections': _make_top_for_index(detections),
    }
    try:
        _insert_record(entry, record)
    except Exception as e:
        try: os.remove(abs_image)
        except OSError: pass
        return {'code': 500, 'message': f'写入数据库失败: {e}'}

    return {'code': 200, 'record_id': record_id}


def save_video_history_record(model_name, session_id, result_data):
    """视频处理完成后：复制视频 + 封面 + 写入 SQLite。"""
    if not isinstance(result_data, dict):
        return {'code': 400, 'message': '无效数据'}
    src_path = result_data.get('processed_video_path')
    if not src_path or not os.path.isfile(src_path):
        return {'code': 400, 'message': '无有效视频文件'}

    record_id = f"{int(time.time() * 1000)}_{hashlib.md5(session_id.encode('utf-8')).hexdigest()[:8]}"
    _ensure_dirs()

    ext = os.path.splitext(src_path)[1].lower() or '.mp4'
    if ext not in ('.mp4', '.avi', '.mov'): ext = '.mp4'

    video_filename = f'{record_id}{ext}'
    dst_video = os.path.join(DETECTION_VIDEOS_DIR, video_filename)
    try:
        shutil.copy2(src_path, dst_video)
    except OSError as e:
        return {'code': 500, 'message': f'保存检测视频失败: {e}'}

    orig_video_file = None
    orig_path = result_data.get('original_video_path')
    if orig_path and os.path.isfile(orig_path):
        orig_ext = os.path.splitext(orig_path)[1].lower() or '.mp4'
        if orig_ext not in ('.mp4', '.avi', '.mov'): orig_ext = '.mp4'
        orig_filename = f'{record_id}_orig{orig_ext}'
        dst_orig = os.path.join(DETECTION_VIDEOS_DIR, orig_filename)
        try:
            shutil.copy2(orig_path, dst_orig)
            orig_video_file = f'videos/{orig_filename}'
        except OSError:
            pass

    thumb_name = f'{record_id}_vthumb.jpg'
    thumb_rel = f'images/{thumb_name}'
    thumb_path = os.path.join(DETECTION_IMAGES_DIR, thumb_name)
    try:
        cap = cv2.VideoCapture(dst_video)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            cv2.imwrite(thumb_path, frame)
        else:
            thumb_rel = None
    except Exception:
        thumb_rel = None

    stats = _flatten_video_stats(result_data)
    full = _full_video_stats(result_data)
    summary = _summary_video(stats)
    created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
    total = int(stats.get('total_detections') or 0)

    record = {
        'id': record_id,
        'source': 'video',
        'created_at': created_at,
        'model': model_name,
        'summary': summary,
        'session_id': session_id,
        'video_file': f'videos/{video_filename}',
        'original_video_file': orig_video_file,
        'thumb_file': thumb_rel,
        **stats,
        **full,
    }

    preview_filename = thumb_name if thumb_rel else video_filename
    entry = {
        'id': record_id,
        'created_at': created_at,
        'source': 'video',
        'model': model_name,
        'summary': summary,
        'total_detections': total,
        'preview_filename': preview_filename,
        'preview_kind': 'video',
        'top_detections': _make_top_for_index(None, video_top_classes=stats.get('top_classes')),
    }
    try:
        _insert_record(entry, record)
    except Exception as e:
        try: os.remove(dst_video)
        except OSError: pass
        return {'code': 500, 'message': f'写入数据库失败: {e}'}

    return {'code': 200, 'record_id': record_id}


def save_realtime_history_record(model_name, image_data, data):
    """实时检测：仅在有检出时保存，避免空帧刷屏。"""
    if not isinstance(data, dict):
        return {'code': 400, 'message': '无效数据'}
    total, detections = _flatten_realtime_detect(data)
    if total <= 0:
        return {'code': 200, 'record_id': None, 'skipped': True}

    try:
        raw_bytes = _decode_base64_raw(image_data)
    except Exception as e:
        return {'code': 400, 'message': str(e)}

    md5_hex = hashlib.md5(raw_bytes).hexdigest()
    record_id = f"{int(time.time() * 1000)}_rt{md5_hex[:7]}"
    ext = _guess_ext_from_bytes(raw_bytes)
    image_filename = f"{record_id}.{ext}"
    abs_image = os.path.join(DETECTION_IMAGES_DIR, image_filename)

    _ensure_dirs()
    try:
        with open(abs_image, 'wb') as f:
            f.write(raw_bytes)
    except OSError as e:
        return {'code': 500, 'message': str(e)}

    created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
    summary = _summary_from_detections(total, detections, prefix="实时检出")

    record = {
        'id': record_id,
        'source': 'realtime',
        'created_at': created_at,
        'model': model_name,
        'summary': summary,
        'total_detections': total,
        'image_file': f'images/{image_filename}',
        'detections': detections,
    }
    entry = {
        'id': record_id,
        'created_at': created_at,
        'source': 'realtime',
        'model': model_name,
        'summary': summary,
        'total_detections': total,
        'preview_filename': image_filename,
        'preview_kind': 'image',
        'top_detections': _make_top_for_index(detections),
    }
    try:
        _insert_record(entry, record)
    except Exception as e:
        try: os.remove(abs_image)
        except OSError: pass
        return {'code': 500, 'message': str(e)}

    return {'code': 200, 'record_id': record_id}


# ──────────────────── 对外查询 API ────────────────────
def list_detection_history(limit=200, source=None):
    """按时间倒序；source 为 image / video / realtime，None 表示全部。"""
    try:
        with _connect() as conn:
            if source:
                rows = conn.execute(
                    "SELECT * FROM detection_history WHERE source=? ORDER BY created_at DESC LIMIT ?",
                    (source, max(1, min(limit, 500)))
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM detection_history ORDER BY created_at DESC LIMIT ?",
                    (max(1, min(limit, 500)),)
                ).fetchall()

        out = []
        for row in rows:
            try:
                top_dets = json.loads(row['top_detections'] or '[]')
            except Exception:
                top_dets = []
            out.append({
                'id':               row['id'],
                'source':           row['source'],
                'created_at':       row['created_at'],
                'model':            row['model'],
                'summary':          row['summary'],
                'total_detections': row['total_detections'],
                'preview_filename': row['preview_filename'],
                'preview_kind':     row['preview_kind'],
                'top_detections':   top_dets,
            })
        return {'code': 200, 'data': out}
    except Exception as e:
        return {'code': 500, 'message': str(e), 'data': []}


def get_detection_record(record_id):
    """获取单条详情（不含 frame_details 大字段）。"""
    if not record_id:
        return {'code': 400, 'message': '无效的记录 ID'}
    if not re.match(r'^[0-9]+_(rt[a-f0-9]{7}|[a-f0-9]{8})$', str(record_id)):
        return {'code': 400, 'message': '无效的记录 ID'}
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT record_json FROM detection_history WHERE id=?",
                (record_id,)
            ).fetchone()
        if not row:
            return {'code': 404, 'message': '记录不存在'}

        record = json.loads(row['record_json'] or '{}')

        # 去掉大字段
        record.pop('frame_details', None)

        # 旧格式兼容：把 result.* 提升到顶层
        if 'result' in record and isinstance(record.get('result'), dict):
            r = record['result']
            if record.get('source') != 'video':
                if not record.get('detections') and r.get('detections'):
                    record['detections'] = r['detections']
                if not record.get('total_detections') and r.get('total_detections') is not None:
                    record['total_detections'] = r['total_detections']
                if not record.get('annotated_file') and r.get('detection_image_file'):
                    record['annotated_file'] = r['detection_image_file']
            else:
                for k in ('total_frames', 'total_detections', 'frames_with_detections',
                          'processing_time', 'duration_sec', 'top_classes'):
                    if k not in record and k in r:
                        record[k] = r[k]

        return {'code': 200, 'data': record}
    except Exception as e:
        return {'code': 500, 'message': str(e)}


def get_detection_frames(record_id, page=1, size=50, only_detections=False):
    """分页获取视频记录的 frame_details。"""
    if not record_id:
        return {'code': 400, 'message': '无效的记录 ID'}
    if not re.match(r'^[0-9]+_(rt[a-f0-9]{7}|[a-f0-9]{8})$', str(record_id)):
        return {'code': 400, 'message': '无效的记录 ID'}
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT record_json FROM detection_history WHERE id=?",
                (record_id,)
            ).fetchone()
        if not row:
            return {'code': 404, 'message': '记录不存在'}

        record = json.loads(row['record_json'] or '{}')
        frames = record.get('frame_details') or []
        if only_detections:
            frames = [f for f in frames if f.get('detections_count', 0) > 0]
        total = len(frames)
        page = max(1, int(page))
        size = max(1, min(int(size), 200))
        start = (page - 1) * size
        return {
            'code': 200,
            'total': total,
            'page': page,
            'size': size,
            'pages': (total + size - 1) // size if size else 1,
            'data': frames[start: start + size],
        }
    except Exception as e:
        return {'code': 500, 'message': str(e)}


# ──────────────────── 批量删除 ────────────────────
def delete_detection_records(ids):
    """
    批量删除：从 SQLite 移除记录，同时删除关联图片 / 视频文件。
    返回 {'deleted': [...], 'failed': [...]}
    """
    if not ids or not isinstance(ids, list):
        return {'code': 400, 'message': 'ids 不能为空'}

    valid_pattern = re.compile(r'^[0-9]+_(rt[a-f0-9]{7}|[a-f0-9]{8})$')
    deleted, failed = [], []

    for rid in ids:
        rid = str(rid).strip()
        if not valid_pattern.match(rid):
            failed.append({'id': rid, 'reason': '无效 ID'})
            continue

        # 先从数据库读取 record_json 以获取关联文件路径
        try:
            with _connect() as conn:
                row = conn.execute(
                    "SELECT record_json FROM detection_history WHERE id=?",
                    (rid,)
                ).fetchone()
        except Exception as e:
            failed.append({'id': rid, 'reason': str(e)})
            continue

        # 删除关联文件
        if row:
            try:
                rec = json.loads(row['record_json'] or '{}')
                for key in ('image_file', 'annotated_file', 'thumb_file'):
                    val = rec.get(key) or ''
                    fn = val.replace('images/', '', 1)
                    if fn:
                        fp = os.path.join(DETECTION_IMAGES_DIR, fn)
                        try:
                            if os.path.isfile(fp): os.remove(fp)
                        except OSError:
                            pass
                for key in ('video_file', 'original_video_file'):
                    val = rec.get(key) or ''
                    fn = val.replace('videos/', '', 1)
                    if fn:
                        fp = os.path.join(DETECTION_VIDEOS_DIR, fn)
                        try:
                            if os.path.isfile(fp): os.remove(fp)
                        except OSError:
                            pass
            except Exception:
                pass

        # 从数据库删除
        try:
            with _connect() as conn:
                conn.execute("DELETE FROM detection_history WHERE id=?", (rid,))
            deleted.append(rid)
        except Exception as e:
            failed.append({'id': rid, 'reason': str(e)})

    return {'code': 200, 'deleted': deleted, 'failed': failed}


# ──────────────────── 文件名校验 ────────────────────
def safe_image_filename(name):
    if not name or '/' in name or '\\' in name or '..' in name:
        return None
    if re.match(r'^[0-9]+_[a-f0-9]{8}\.(jpg|jpeg|png|webp|bmp)$', name, re.I):
        return name
    if re.match(r'^[0-9]+_[a-f0-9]{8}_annotated\.jpg$', name, re.I):
        return name
    if re.match(r'^[0-9]+_[a-f0-9]{8}_vthumb\.jpg$', name, re.I):
        return name
    if re.match(r'^[0-9]+_rt[a-f0-9]{7}\.(jpg|jpeg|png|webp|bmp)$', name, re.I):
        return name
    return None


def safe_video_filename(name):
    if not name or '/' in name or '\\' in name or '..' in name:
        return None
    if re.match(r'^[0-9]+_[a-f0-9]{8}\.(mp4|avi|mov)$', name, re.I):
        return name
    if re.match(r'^[0-9]+_[a-f0-9]{8}_orig\.(mp4|avi|mov)$', name, re.I):
        return name
    return None

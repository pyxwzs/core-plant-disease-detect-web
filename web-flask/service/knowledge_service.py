"""
病虫害结构化知识库（SQLite + 本地图片）
"""
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime

from config import DATABASE_PATH, KNOWLEDGE_UPLOAD_DIR

_ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _connect():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_knowledge_table():
    """创建 knowledge_items 表。"""
    os.makedirs(KNOWLEDGE_UPLOAD_DIR, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT '病害',
                symptom TEXT,
                pattern_text TEXT,
                harm TEXT,
                prevention TEXT,
                images_json TEXT NOT NULL DEFAULT '[]',
                status INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_status ON knowledge_items(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_kind ON knowledge_items(kind)"
        )


def _parse_images(images_json_str):
    try:
        data = json.loads(images_json_str or "[]")
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _image_urls(filenames):
    out = []
    for fn in filenames:
        if isinstance(fn, str) and fn and "/" not in fn and "\\" not in fn and ".." not in fn:
            out.append("/api/knowledge/images/" + fn)
    return out


def _row_to_dict(row, include_body=True):
    images = _parse_images(row["images_json"])
    d = {
        "id": row["id"],
        "title": row["title"],
        "kind": row["kind"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "image_urls": _image_urls(images),
    }
    if include_body:
        d["symptom"] = row["symptom"] or ""
        d["pattern_text"] = row["pattern_text"] or ""
        d["harm"] = row["harm"] or ""
        d["prevention"] = row["prevention"] or ""
    return d


def list_knowledge_items(
    *,
    page=1,
    page_size=20,
    q=None,
    kind=None,
    status=None,
    public_only=False,
    time_from=None,
    time_to=None,
):
    """
    public_only=True 时仅 status=1。
    status: None 表示不过滤；0/1 为指定状态（管理员用）。
    time_from / time_to: ISO 日期字符串，按 updated_at 区间过滤（管理员用）。
    """
    init_knowledge_table()
    page = max(1, int(page or 1))
    page_size = min(100, max(1, int(page_size or 20)))
    offset = (page - 1) * page_size

    where = []
    args = []
    if public_only:
        where.append("status = 1")
    elif status is not None:
        where.append("status = ?")
        args.append(int(status))

    if kind:
        where.append("kind = ?")
        args.append(kind.strip())

    if q and q.strip():
        where.append("title LIKE ?")
        args.append("%" + q.strip().replace("%", "\\%") + "%")

    if time_from and time_from.strip():
        where.append("updated_at >= ?")
        args.append(time_from.strip())
    if time_to and time_to.strip():
        end = time_to.strip()
        if len(end) == 10:
            end += "T23:59:59"
        where.append("updated_at <= ?")
        args.append(end)

    wh = (" WHERE " + " AND ".join(where)) if where else ""

    with _connect() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM knowledge_items{wh}", args
        ).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT id, title, kind, symptom, pattern_text, harm, prevention,
                   images_json, status, created_at, updated_at
            FROM knowledge_items
            {wh}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            args + [page_size, offset],
        ).fetchall()

    items = [_row_to_dict(r) for r in rows]
    return {
        "code": 200,
        "data": {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
        },
    }


def get_knowledge_item(item_id, *, public_only=False):
    init_knowledge_table()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, title, kind, symptom, pattern_text, harm, prevention,
                   images_json, status, created_at, updated_at
            FROM knowledge_items WHERE id = ?
            """,
            (int(item_id),),
        ).fetchone()
    if not row:
        return {"code": 404, "message": "记录不存在"}
    if public_only and row["status"] != 1:
        return {"code": 404, "message": "记录不存在"}
    return {"code": 200, "data": _row_to_dict(row)}


def _delete_image_files(filenames):
    for fn in filenames:
        if not isinstance(fn, str) or "/" in fn or "\\" in fn or ".." in fn:
            continue
        fp = os.path.join(KNOWLEDGE_UPLOAD_DIR, fn)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except OSError:
                pass


def create_knowledge_item(
    title,
    kind="病害",
    symptom="",
    pattern_text="",
    harm="",
    prevention="",
    images=None,
    status=1,
):
    init_knowledge_table()
    title = (title or "").strip()
    if not title:
        return {"code": 400, "message": "名称不能为空"}
    kind = (kind or "病害").strip() or "病害"
    images = images if isinstance(images, list) else []
    now = datetime.now().isoformat()
    imgs_json = json.dumps(images, ensure_ascii=False)
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO knowledge_items (
                title, kind, symptom, pattern_text, harm, prevention,
                images_json, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                kind,
                symptom or "",
                pattern_text or "",
                harm or "",
                prevention or "",
                imgs_json,
                1 if int(status) else 0,
                now,
                now,
            ),
        )
        conn.commit()
        new_id = cur.lastrowid
    return {"code": 200, "message": "创建成功", "data": {"id": new_id}}


def update_knowledge_item(item_id, **fields):
    init_knowledge_table()
    allowed = {
        "title",
        "kind",
        "symptom",
        "pattern_text",
        "harm",
        "prevention",
        "images",
        "status",
    }
    sets = []
    args = []
    old_images = []
    with _connect() as conn:
        row = conn.execute(
            "SELECT images_json FROM knowledge_items WHERE id = ?",
            (int(item_id),),
        ).fetchone()
        if not row:
            return {"code": 404, "message": "记录不存在"}
        old_images = _parse_images(row["images_json"])

        for k, v in fields.items():
            if k not in allowed or v is None:
                continue
            if k == "images":
                sets.append("images_json = ?")
                args.append(json.dumps(v, ensure_ascii=False))
            elif k == "status":
                sets.append("status = ?")
                args.append(1 if int(v) else 0)
            elif k == "title":
                sets.append("title = ?")
                args.append((v or "").strip())
            else:
                sets.append(f"{k} = ?")
                args.append(v if isinstance(v, str) else str(v))

        if not sets:
            return {"code": 400, "message": "无有效字段"}

        sets.append("updated_at = ?")
        args.append(datetime.now().isoformat())
        args.append(int(item_id))

        conn.execute(
            f"UPDATE knowledge_items SET {', '.join(sets)} WHERE id = ?",
            args,
        )
        conn.commit()

        if "images" in fields and isinstance(fields["images"], list):
            new_set = set(fields["images"])
            for fn in old_images:
                if fn not in new_set:
                    _delete_image_files([fn])

    return {"code": 200, "message": "更新成功"}


def delete_knowledge_item(item_id):
    init_knowledge_table()
    with _connect() as conn:
        row = conn.execute(
            "SELECT images_json FROM knowledge_items WHERE id = ?",
            (int(item_id),),
        ).fetchone()
        if not row:
            return {"code": 404, "message": "记录不存在"}
        imgs = _parse_images(row["images_json"])
        conn.execute("DELETE FROM knowledge_items WHERE id = ?", (int(item_id),))
        conn.commit()
    _delete_image_files(imgs)
    return {"code": 200, "message": "已删除"}


def save_knowledge_image(file_storage):
    """
    保存上传图片，返回 { code, data: { filename } } 或错误信息。
    """
    init_knowledge_table()
    if not file_storage or not file_storage.filename:
        return {"code": 400, "message": "未选择文件"}
    orig = file_storage.filename
    ext = os.path.splitext(orig)[1].lower()
    if ext not in _ALLOWED_EXT:
        return {"code": 400, "message": "仅支持 jpg/png/gif/webp"}
    raw = file_storage.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        return {"code": 400, "message": "单文件不超过 5MB"}
    fn = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(KNOWLEDGE_UPLOAD_DIR, fn)
    with open(path, "wb") as out:
        out.write(raw)
    return {"code": 200, "data": {"filename": fn}}


def safe_knowledge_image_name(filename):
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        return None
    if not re.match(r"^[a-zA-Z0-9_\-\.]+\.(jpg|jpeg|png|gif|webp)$", filename):
        return None
    return filename

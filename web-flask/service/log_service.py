"""
系统日志服务（SQLite 持久化）

日志类型：
  admin_action  —— 管理员后台操作（用户管理、知识库管理）
  user_action   —— 用户行为（登录、注册、检测）
  sys_error     —— 系统异常（5xx、断言失败等）
"""
import csv
import io
import json
import os
import sqlite3
from datetime import datetime

from config import DATABASE_PATH


def _connect():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_log_table():
    """创建 sys_logs 表与索引（幂等）。"""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sys_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                log_type    TEXT NOT NULL,
                operator    TEXT,
                action      TEXT NOT NULL,
                target_type TEXT,
                target_id   TEXT,
                message     TEXT NOT NULL,
                extra       TEXT,
                result      TEXT NOT NULL DEFAULT 'success',
                created_at  TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_type    ON sys_logs(log_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_operator ON sys_logs(operator)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_created  ON sys_logs(created_at)"
        )


def write_log(
    log_type: str,
    action: str,
    message: str,
    *,
    operator: str = None,
    target_type: str = None,
    target_id=None,
    extra: dict = None,
    result: str = "success",
):
    """
    写入一条系统日志，失败时静默忽略（不影响主业务）。

    参数：
      log_type    : admin_action | user_action | sys_error
      action      : 简短动作编码，如 login / create_knowledge / delete_user
      message     : 可读描述
      operator    : 操作者用户名
      target_type : 目标资源类型，如 user / knowledge_item
      target_id   : 目标资源 ID
      extra       : 任意附加字典，序列化为 JSON 文本
      result      : success | fail | error
    """
    try:
        init_log_table()
        extra_str = (
            json.dumps(extra, ensure_ascii=False) if extra is not None else None
        )
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO sys_logs
                    (log_type, operator, action, target_type, target_id,
                     message, extra, result, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log_type,
                    operator,
                    action,
                    target_type,
                    str(target_id) if target_id is not None else None,
                    message,
                    extra_str,
                    result,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
    except Exception:
        pass


def list_logs(
    *,
    page=1,
    page_size=20,
    log_type=None,
    operator=None,
    q=None,
    time_from=None,
    time_to=None,
):
    """分页查询日志（管理员用）。"""
    init_log_table()
    page = max(1, int(page or 1))
    page_size = min(200, max(1, int(page_size or 20)))
    offset = (page - 1) * page_size

    where, args = [], []
    if log_type and log_type.strip():
        where.append("log_type = ?")
        args.append(log_type.strip())
    if operator and operator.strip():
        where.append("operator LIKE ?")
        args.append("%" + operator.strip().replace("%", "\\%") + "%")
    if q and q.strip():
        where.append("(message LIKE ? OR action LIKE ?)")
        like = "%" + q.strip().replace("%", "\\%") + "%"
        args += [like, like]
    if time_from and time_from.strip():
        where.append("created_at >= ?")
        args.append(time_from.strip())
    if time_to and time_to.strip():
        end = time_to.strip()
        if len(end) == 10:
            end += "T23:59:59"
        where.append("created_at <= ?")
        args.append(end)

    wh = (" WHERE " + " AND ".join(where)) if where else ""
    with _connect() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM sys_logs{wh}", args
        ).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT id, log_type, operator, action, target_type, target_id,
                   message, result, created_at
            FROM sys_logs{wh}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            args + [page_size, offset],
        ).fetchall()

    return {
        "code": 200,
        "data": {
            "items": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
        },
    }


def get_log(log_id):
    """获取单条日志详情（含 extra 字段，JSON 已反序列化）。"""
    init_log_table()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM sys_logs WHERE id = ?", (int(log_id),)
        ).fetchone()
    if not row:
        return {"code": 404, "message": "记录不存在"}
    d = dict(row)
    if d.get("extra"):
        try:
            d["extra"] = json.loads(d["extra"])
        except Exception:
            pass
    return {"code": 200, "data": d}


def export_logs_csv(
    *,
    log_type=None,
    operator=None,
    q=None,
    time_from=None,
    time_to=None,
):
    """
    按当前筛选条件导出最多 5000 条日志，返回 CSV 文本字符串。
    调用方负责设置响应头 Content-Type 与 Content-Disposition。
    """
    init_log_table()
    where, args = [], []
    if log_type and log_type.strip():
        where.append("log_type = ?")
        args.append(log_type.strip())
    if operator and operator.strip():
        where.append("operator LIKE ?")
        args.append("%" + operator.strip().replace("%", "\\%") + "%")
    if q and q.strip():
        like = "%" + q.strip().replace("%", "\\%") + "%"
        where.append("(message LIKE ? OR action LIKE ?)")
        args += [like, like]
    if time_from and time_from.strip():
        where.append("created_at >= ?")
        args.append(time_from.strip())
    if time_to and time_to.strip():
        end = time_to.strip()
        if len(end) == 10:
            end += "T23:59:59"
        where.append("created_at <= ?")
        args.append(end)

    wh = (" WHERE " + " AND ".join(where)) if where else ""
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, log_type, operator, action, target_type, target_id,
                   message, result, created_at, extra
            FROM sys_logs{wh}
            ORDER BY id DESC LIMIT 5000
            """,
            args,
        ).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["ID", "日志类型", "操作者", "动作", "目标类型", "目标ID",
         "消息", "结果", "时间", "扩展信息"]
    )
    for r in rows:
        writer.writerow(
            [
                r["id"],
                r["log_type"],
                r["operator"] or "",
                r["action"],
                r["target_type"] or "",
                r["target_id"] or "",
                r["message"],
                r["result"],
                r["created_at"],
                r["extra"] or "",
            ]
        )
    return buf.getvalue()

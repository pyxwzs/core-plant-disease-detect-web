# MD5: 4ee205ed1c0ce729b7180d30a09644c5


"""
用户服务模块（SQLite 持久化）
"""
import binascii
import hashlib
import json
import os
import secrets
import sqlite3
from datetime import datetime

from config import DATABASE_PATH, LEGACY_USERS_JSON

# ─────────────── 密码哈希（PBKDF2-SHA256） ───────────────

_PBKDF2_ITER = 260_000
_HASH_PREFIX = "pbkdf2:"


def _hash_password(password: str) -> str:
    """返回形如 pbkdf2:<iters>:<salt_hex>:<dk_hex> 的哈希字符串。"""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITER)
    return f"{_HASH_PREFIX}{_PBKDF2_ITER}:{salt.hex()}:{dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    """校验密码；兼容旧版明文（自动识别，登录时会升级）。"""
    if stored.startswith(_HASH_PREFIX):
        try:
            _, iters_s, salt_hex, dk_hex = stored.split(":", 3)
            salt = bytes.fromhex(salt_hex)
            dk = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), salt, int(iters_s)
            )
            return dk.hex() == dk_hex
        except Exception:
            return False
    # 旧版明文兼容
    return stored == password


def _is_hashed(stored: str) -> bool:
    return stored.startswith(_HASH_PREFIX)


def _connect():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_user_role_column(conn):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "role" not in cols:
        conn.execute(
            "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'"
        )
        conn.execute("UPDATE users SET role = 'admin' WHERE username = 'admin'")


def init_db():
    """创建用户表与索引。"""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                real_name TEXT,
                created_at TEXT NOT NULL,
                status INTEGER NOT NULL DEFAULT 1,
                detection_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        _ensure_user_role_column(conn)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)"
        )


def migrate_from_legacy_json():
    """
    若存在旧版 users.json 且当前库中无用户，则导入一次。
    导入成功后不自动删除 JSON 文件，由部署者自行清理。
    """
    if not os.path.isfile(LEGACY_USERS_JSON):
        return
    with _connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if n > 0:
            return
        try:
            with open(LEGACY_USERS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        now = datetime.now().isoformat()
        for username, u in data.items():
            if not isinstance(u, dict):
                continue
            pwd = u.get("password") or ""
            real = u.get("realName") or username
            created = u.get("createTime") or now
            try:
                conn.execute(
                    """
                    INSERT INTO users (username, password, real_name, created_at, status, detection_count, role)
                    VALUES (?, ?, ?, ?, 1, 0, 'user')
                    """,
                    (username, pwd, real, created),
                )
            except sqlite3.IntegrityError:
                continue
        conn.commit()


def login_user(username, password):
    """用户登录；旧版明文密码验证通过后自动升级为哈希。"""
    if not username or not password:
        return {"code": 400, "message": "用户名和密码不能为空"}

    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT password, real_name, status, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if not row:
            return {"code": 400, "message": "用户名或密码错误"}

        if row["status"] != 1:
            return {"code": 403, "message": "账号已被禁用，请联系管理员"}

        if not _verify_password(password, row["password"]):
            return {"code": 400, "message": "用户名或密码错误"}

        # 旧版明文 → 自动升级为哈希
        if not _is_hashed(row["password"]):
            conn.execute(
                "UPDATE users SET password = ? WHERE username = ?",
                (_hash_password(password), username),
            )
            conn.commit()

    role = row["role"] if row["role"] else "user"
    return {
        "code": 200,
        "message": "登录成功",
        "data": {
            "user": {
                "id": username,
                "username": username,
                "realName": row["real_name"] or username,
                "role": role,
            }
        },
    }


def register_user(username, password, real_name=None):
    """用户注册，密码哈希存储。"""
    if not username or not password:
        return {"code": 400, "message": "用户名和密码不能为空"}

    if len(username) < 3 or len(password) < 6:
        return {"code": 400, "message": "用户名至少3位，密码至少6位"}

    init_db()
    created = datetime.now().isoformat()
    display = real_name or username
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO users (username, password, real_name, created_at, status, detection_count, role)
                VALUES (?, ?, ?, ?, 1, 0, 'user')
                """,
                (username, _hash_password(password), display, created),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return {"code": 400, "message": "用户名已存在"}

    return {"code": 200, "message": "注册成功"}


def verify_admin_credentials(username, password):
    """校验是否为已启用且角色为 admin 的账号；兼容明文并自动升级。"""
    if not username or not password:
        return False
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT password, status, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if not row or row["status"] != 1:
            return False
        if not _verify_password(password, row["password"]):
            return False
        # 自动升级明文
        if not _is_hashed(row["password"]):
            conn.execute(
                "UPDATE users SET password = ? WHERE username = ?",
                (_hash_password(password), username),
            )
            conn.commit()
    role = row["role"] or "user"
    return role == "admin"


def increment_detection_count(username):
    """成功完成一次检测时增加该用户的累计检测次数（可选调用）。"""
    if not username:
        return
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE users SET detection_count = detection_count + 1
            WHERE username = ? AND status = 1
            """,
            (username,),
        )
        conn.commit()


# ─────────────── 管理员用户管理 ───────────────

def list_users(*, page=1, page_size=20, q=None):
    """分页列出所有用户（管理员用）。"""
    init_db()
    page = max(1, int(page or 1))
    page_size = min(100, max(1, int(page_size or 20)))
    offset = (page - 1) * page_size
    where, args = [], []
    if q and q.strip():
        like = "%" + q.strip().replace("%", "\\%") + "%"
        where.append("(username LIKE ? OR real_name LIKE ?)")
        args += [like, like]
    wh = (" WHERE " + " AND ".join(where)) if where else ""
    with _connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM users{wh}", args).fetchone()[0]
        rows = conn.execute(
            f"""SELECT id, username, real_name, created_at, status,
                       detection_count, role
                FROM users{wh} ORDER BY id ASC LIMIT ? OFFSET ?""",
            args + [page_size, offset],
        ).fetchall()
    items = [
        {
            "id": r["id"],
            "username": r["username"],
            "realName": r["real_name"] or r["username"],
            "created_at": r["created_at"],
            "status": r["status"],
            "detection_count": r["detection_count"],
            "role": r["role"] or "user",
        }
        for r in rows
    ]
    return {"code": 200, "data": {"items": items, "total": total,
                                   "page": page, "page_size": page_size}}


def set_user_status(username, status, operator=None):
    """启用（status=1）或禁用（status=0）账号；不允许操作自身。"""
    init_db()
    if operator and operator == username:
        return {"code": 400, "message": "不能对自身账号执行启用/禁用"}
    with _connect() as conn:
        r = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if not r:
            return {"code": 404, "message": "用户不存在"}
        conn.execute("UPDATE users SET status=? WHERE username=?",
                     (1 if int(status) else 0, username))
        conn.commit()
    return {"code": 200, "message": "操作成功"}


def reset_user_password(username, new_password):
    """管理员重置指定用户密码（哈希存储）。"""
    init_db()
    if not new_password or len(new_password) < 6:
        return {"code": 400, "message": "新密码至少6位"}
    with _connect() as conn:
        r = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if not r:
            return {"code": 404, "message": "用户不存在"}
        conn.execute("UPDATE users SET password=? WHERE username=?",
                     (_hash_password(new_password), username))
        conn.commit()
    return {"code": 200, "message": "密码已重置"}


def delete_user(username, operator=None):
    """删除用户；不允许删除自身及最后一个 admin。"""
    init_db()
    if operator and operator == username:
        return {"code": 400, "message": "不能删除自身账号"}
    with _connect() as conn:
        r = conn.execute("SELECT role FROM users WHERE username=?", (username,)).fetchone()
        if not r:
            return {"code": 404, "message": "用户不存在"}
        if r["role"] == "admin":
            cnt = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role='admin' AND status=1"
            ).fetchone()[0]
            if cnt <= 1:
                return {"code": 400, "message": "系统至少保留一个管理员账号"}
        conn.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()
    return {"code": 200, "message": "已删除"}


def init_default_users():
    """初始化数据库、迁移旧数据、并在无用户时创建默认账号。"""
    init_db()
    migrate_from_legacy_json()

    with _connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if n > 0:
            return

        now = datetime.now().isoformat()
        seeds = (
            ("admin", "管理员", "admin"),
            ("user", "普通用户", "user"),
        )
        for uname, real, role in seeds:
            conn.execute(
                """
                INSERT INTO users (username, password, real_name, created_at, status, detection_count, role)
                VALUES (?, ?, ?, ?, 1, 0, ?)
                """,
                (uname, _hash_password("123456"), real, now, role),
            )
        conn.commit()

    print("已创建默认用户：")
    print("用户名: admin, 密码: 123456")
    print("用户名: user, 密码: 123456")

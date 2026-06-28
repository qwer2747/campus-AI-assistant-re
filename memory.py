# ==================== memory.py ====================
"""
数据库层：负责用户认证、对话记忆、用户画像的持久化存储
重启后数据完全保留（SQLite文件持久化）
"""
import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = "campus_assistant.db"

# ==================== 数据库初始化 ====================
def get_conn():
    """获取线程安全的数据库连接"""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    """建表（首次运行创建，之后幂等）"""
    with get_conn() as conn:
        conn.executescript('''
            -- 用户表（登录认证）
            CREATE TABLE IF NOT EXISTS users (
                user_id     TEXT PRIMARY KEY,
                username    TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name  TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            -- 对话历史（持久化跨会话记忆）
            CREATE TABLE IF NOT EXISTS chat_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                role        TEXT NOT NULL,       -- user / assistant
                content     TEXT NOT NULL,
                tool_calls  TEXT,               -- JSON序列化的工具调用记录
                timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            -- 用户画像（长期记忆：专业/年级/偏好等）
            CREATE TABLE IF NOT EXISTS user_facts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                fact        TEXT NOT NULL,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            -- 纠错反馈日志
            CREATE TABLE IF NOT EXISTS feedback_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT,
                question        TEXT,
                ai_answer       TEXT,
                correct_answer  TEXT,
                timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        ''')

# ==================== 认证函数 ====================
def _hash(password: str) -> str:
    """SHA-256 密码哈希"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def register_user(username: str, password: str, display_name: str = "") -> tuple:
    """
    注册新用户
    返回: (success: bool, message: str, user_id: str)
    """
    if len(username) < 2:
        return False, "用户名至少2个字符", ""
    if len(password) < 6:
        return False, "密码至少6位", ""
    
    user_id = f"u_{_hash(username)[:12]}"  # 确定性ID，防重复
    display = display_name.strip() or username
    
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (user_id, username, password_hash, display_name) VALUES (?,?,?,?)",
                (user_id, username.strip(), _hash(password), display)
            )
        return True, "注册成功！", user_id
    except sqlite3.IntegrityError:
        return False, "用户名已被占用，请换一个", ""
    except Exception as e:
        return False, f"注册失败：{e}", ""

def login_user(username: str, password: str) -> tuple:
    """
    用户登录验证
    返回: (success: bool, user_id: str, display_name: str)
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT user_id, display_name FROM users WHERE username=? AND password_hash=?",
            (username.strip(), _hash(password))
        )
        row = cur.fetchone()
    
    if row:
        return True, row[0], row[1]
    return False, "", ""

# ==================== 对话历史函数 ====================
def save_message(user_id: str, role: str, content: str, tool_calls: str = None):
    """保存单条消息到数据库"""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_history (user_id, role, content, tool_calls) VALUES (?,?,?,?)",
            (user_id, role, content, tool_calls)
        )

def load_chat_history(user_id: str, limit: int = 50) -> list:
    """
    加载用户历史对话（最新limit条，按时间正序返回）
    返回: [{"role": str, "content": str, "tool_calls": str|None}]
    """
    with get_conn() as conn:
        cur = conn.cursor()
        # 取最新N条，再反转为正序
        cur.execute('''
            SELECT role, content, tool_calls 
            FROM chat_history 
            WHERE user_id=? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        rows = cur.fetchall()
    
    return [
        {"role": r[0], "content": r[1], "tool_calls": r[2]}
        for r in reversed(rows)
    ]

def clear_chat_history(user_id: str):
    """清空用户对话历史"""
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_history WHERE user_id=?", (user_id,))

def get_chat_stats(user_id: str) -> dict:
    """获取用户对话统计"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM chat_history WHERE user_id=? AND role='user'",
            (user_id,)
        )
        msg_count = cur.fetchone()[0]
        cur.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM chat_history WHERE user_id=?",
            (user_id,)
        )
        dates = cur.fetchone()
    return {
        "message_count": msg_count,
        "first_chat": dates[0],
        "last_chat": dates[1]
    }

# ==================== 用户画像函数 ====================
def save_user_fact(user_id: str, fact: str):
    """存储用户的一条个人信息"""
    # 避免重复存相同内容
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM user_facts WHERE user_id=? AND fact=?",
            (user_id, fact)
        )
        if not cur.fetchone():
            conn.execute(
                "INSERT INTO user_facts (user_id, fact) VALUES (?,?)",
                (user_id, fact)
            )

def get_user_facts(user_id: str, limit: int = 20) -> list:
    """获取用户的所有已知个人信息"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT fact, created_at FROM user_facts WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        )
        return [{"fact": r[0], "time": r[1]} for r in cur.fetchall()]

def delete_user_fact(user_id: str, fact: str):
    """删除某条用户画像记录"""
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM user_facts WHERE user_id=? AND fact=?",
            (user_id, fact)
        )

# ==================== 反馈日志 ====================
def save_feedback(user_id: str, question: str, ai_answer: str, correct_answer: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback_log (user_id, question, ai_answer, correct_answer) VALUES (?,?,?,?)",
            (user_id, question, ai_answer, correct_answer)
        )

# 模块加载时自动建表
init_db()
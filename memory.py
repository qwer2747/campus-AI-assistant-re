# ==================== memory.py ====================
"""
数据库层：SQLite → Supabase 持久化版本
"""
import hashlib
from supabase import create_client
import streamlit as st

# ==================== 初始化 Supabase ====================
@st.cache_resource
def get_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

def init_db():
    pass  # 表已在Supabase建好，无需操作

# ==================== 用户认证 ====================
def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username: str, password: str) -> tuple[bool, str]:
    sb = get_supabase()
    # 检查用户名是否已存在
    res = sb.table("users").select("id").eq("username", username).execute()
    if res.data:
        return False, "用户名已存在"
    sb.table("users").insert({
        "username": username,
        "password_hash": _hash(password)
    }).execute()
    return True, "注册成功"

def login_user(username: str, password: str) -> tuple[bool, str, int | None]:
    sb = get_supabase()
    res = sb.table("users").select("*").eq("username", username).eq("password_hash", _hash(password)).execute()
    if not res.data:
        return False, "用户名或密码错误", None
    user = res.data[0]
    return True, "登录成功", user["id"]

# ==================== 对话历史 ====================
def save_message(user_id: int, role: str, content: str):
    sb = get_supabase()
    sb.table("chat_history").insert({
        "user_id": user_id,
        "role": role,
        "content": content
    }).execute()

def load_history(user_id: int, limit: int = 20) -> list[dict]:
    sb = get_supabase()
    res = sb.table("chat_history")\
        .select("role, content")\
        .eq("user_id", user_id)\
        .order("created_at", desc=False)\
        .limit(limit)\
        .execute()
    return res.data if res.data else []

def clear_history(user_id: int):
    sb = get_supabase()
    sb.table("chat_history").delete().eq("user_id", user_id).execute()

# ==================== 用户画像 ====================
def save_user_fact(user_id: int, fact: str):
    sb = get_supabase()
    # 避免重复
    res = sb.table("user_facts").select("id").eq("user_id", user_id).eq("fact", fact).execute()
    if not res.data:
        sb.table("user_facts").insert({
            "user_id": user_id,
            "fact": fact
        }).execute()

def load_user_facts(user_id: int) -> list[str]:
    sb = get_supabase()
    res = sb.table("user_facts")\
        .select("fact")\
        .eq("user_id", user_id)\
        .execute()
    return [r["fact"] for r in res.data] if res.data else []

# ==================== 反馈学习 ====================
def save_feedback(user_id: int, question: str, ai_answer: str, correct_answer: str):
    sb = get_supabase()
    sb.table("feedback_log").insert({
        "user_id": user_id,
        "question": question,
        "ai_answer": ai_answer,
        "correct_answer": correct_answer
    }).execute()

init_db()
# 兼容别名
get_user_facts = load_user_facts

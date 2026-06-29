# ==================== memory.py ====================
import hashlib
from supabase import create_client
import streamlit as st

@st.cache_resource
def get_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

def init_db():
    pass

# ==================== 用户认证 ====================
def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username: str, password: str, display_name: str = "") -> tuple:
    sb = get_supabase()
    res = sb.table("users").select("user_id").eq("username", username).execute()
    if res.data:
        return False, "用户名已存在", None
    sb.table("users").insert({
        "username": username,
        "password_hash": _hash(password),
        "display_name": display_name or username
    }).execute()
    return True, "注册成功", None

def login_user(username: str, password: str) -> tuple:
    sb = get_supabase()
    res = sb.table("users").select("*")\
        .eq("username", username)\
        .eq("password_hash", _hash(password))\
        .execute()
    if not res.data:
        return False, None, None
    user = res.data[0]
    display_name = user.get("display_name") or username
    return True, user["user_id"], display_name

# ==================== 对话历史 ====================
def save_message(user_id: int, role: str, content: str, tool_calls_log=None):
    sb = get_supabase()
    sb.table("chat_history").insert({
        "user_id": user_id,
        "role": role,
        "content": content,
        "tool_calls_log": tool_calls_log
    }).execute()

def load_history(user_id: int, limit: int = 20) -> list:
    sb = get_supabase()
    res = sb.table("chat_history")\
        .select("role, content, tool_calls_log")\
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
    res = sb.table("user_facts").select("user_id")\
        .eq("user_id", user_id).eq("fact", fact).execute()
    if not res.data:
        sb.table("user_facts").insert({
            "user_id": user_id,
            "fact": fact
        }).execute()

def load_user_facts(user_id: int) -> list:
    sb = get_supabase()
    res = sb.table("user_facts")\
        .select("fact")\
        .eq("user_id", user_id)\
        .execute()
    # 返回字典列表，兼容 web7.py 的 f['fact'] 用法
    return res.data if res.data else []

def delete_user_fact(user_id: int, fact: str):
    sb = get_supabase()
    sb.table("user_facts").delete()\
        .eq("user_id", user_id)\
        .eq("fact", fact)\
        .execute()

# ==================== 统计 ====================
def get_chat_stats(user_id: int) -> dict:
    sb = get_supabase()
    res = sb.table("chat_history")\
        .select("user_id")\
        .eq("user_id", user_id)\
        .eq("role", "user")\
        .execute()
    return {"message_count": len(res.data) if res.data else 0}

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
get_user_facts    = load_user_facts
load_chat_history = load_history
clear_chat_history = clear_history

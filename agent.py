# ==================== agent.py ====================
"""
ReAct Agent：Reasoning + Acting
显式捕获每一步 Thought → Action → Observation → Answer
"""
import os
import json
import math
import time
import requests
import pandas as pd
import chromadb
import streamlit as st
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

from memory import save_user_fact, get_user_facts

# ==================== 配置 ====================
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
CHROMA_DB_PATH   = "./chroma_db"
CSV_PATH         = "campus_qa.csv"
EMBEDDING_MODEL  = "BAAI/bge-small-zh"
MAX_TOOL_ROUNDS  = 6
TEMPERATURE      = 0.3

# ==================== 模型加载（独立，带缓存）====================
@st.cache_resource(show_spinner="⚙️ 正在加载AI模型...")
def _load_model():
    """只负责加载嵌入模型，不涉及知识库"""
    return SentenceTransformer(EMBEDDING_MODEL)

# ==================== 知识库加载（只读，不构建）====================
def _load_collection():
    """只负责连接知识库，不负责构建"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name="campus_qa")
    return collection

# ==================== 向量化函数 ====================
def _embed(texts):
    """使用独立的模型进行向量化"""
    model = _load_model()
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts, normalize_embeddings=True).tolist()

# ==================== 知识库检索 ====================
def tool_search_knowledge(query: str, k: int = 5) -> str:
    collection = _load_collection()
    
    # 检查知识库是否为空
    if collection.count() == 0:
        return "📭 知识库为空，请先运行 数据-chroma.py 构建知识库。"
    
    results = collection.query(query_embeddings=_embed([query]), n_results=k)
    docs = results['documents'][0] if results['documents'] else []
    
    if not docs:
        return "📭 知识库中暂未找到相关资料。"
    return "\n\n".join([f"[资料{i+1}] {doc}" for i, doc in enumerate(docs)])

# ==================== 时间工具（已修正时区）====================
def tool_get_datetime() -> str:
    now = datetime.now() + timedelta(hours=8)  # 北京时间
    weekdays = ['一', '二', '三', '四', '五', '六', '日']
    return (f"📅 当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}，"
            f"星期{weekdays[now.weekday()]}，第{now.isocalendar()[1]}周")

# ==================== 计算工具 ====================
def tool_calculate(expression: str) -> str:
    try:
        safe_ns = {k: v for k, v in math.__dict__.items() if not k.startswith('_')}
        result = eval(expression, {"__builtins__": {}}, safe_ns)
        return f"🔢 {expression} = {result}"
    except ZeroDivisionError:
        return "❌ 除数不能为零"
    except Exception as e:
        return f"❌ 计算错误：{e}"

# ==================== 记忆工具 ====================
def tool_remember_fact(user_id: str, fact: str) -> str:
    save_user_fact(user_id, fact)
    return f"✅ 已记住：「{fact}」"

def tool_recall_facts(user_id: str) -> str:
    facts = get_user_facts(user_id)
    if not facts:
        return "🧠 暂无关于你的记忆，可以主动告诉我你的专业、年级等！"
    return "🧠 我记得关于你的信息：\n" + "\n".join([f"- {f['fact']}" for f in facts])

# ==================== 工具分发 ====================
def _dispatch_tool(name: str, args: dict, user_id: str) -> str:
    dispatch_map = {
        "search_knowledge": lambda: tool_search_knowledge(args.get("query", "")),
        "get_datetime":     lambda: tool_get_datetime(),
        "calculate":        lambda: tool_calculate(args.get("expression", "1+1")),
        "remember_fact":    lambda: tool_remember_fact(user_id, args.get("fact", "")),
        "recall_facts":     lambda: tool_recall_facts(user_id),
    }
    handler = dispatch_map.get(name)
    return handler() if handler else f"⚠️ 未知工具：{name}"

# ==================== 工具注册表 ====================
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索校园知识库。当你需要查询图书馆、食堂、宿舍、选课、校园卡等校园信息时，必须调用这个工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，越具体越好"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "仅在用户明确询问当前时间、日期、星期、今天几号、现在几点时使用。",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "数学计算，支持四则运算、幂运算、sqrt()、sin()、cos()、log()等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "记住用户告诉你的个人信息（专业、年级、兴趣、偏好等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "要记住的事实"}
                },
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_facts",
            "description": "回忆当前用户之前告诉过你的所有个人信息",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# ==================== ReAct 系统提示 ====================
REACT_SYSTEM_PROMPT = """你是校园新生助手「小慧」，帮助{display_name}解决校园生活问题。

## ⚡ 你必须遵循 ReAct 推理框架

每次行动前，必须写出你的 Thought（思考过程）。

### 标准格式：
**调用工具时：**
Thought: [分析问题，说明为什么要用这个工具]
Action: search_knowledge(query="关键词")

**最终回答时：**
Thought: [整理答案]
Answer: [给用户的正式回答]

## 工具使用规则
1. 校园相关问题 → 必须先调用 search_knowledge
2. 时间问题 → 调用 get_datetime
3. 个人信息 → 调用 remember_fact / recall_facts

## 当前用户：{display_name}"""


# ==================== ReAct Agent 主循环 ====================
def run_agent(
    user_message: str,
    user_id: str,
    chat_history: list,
    display_name: str = "同学"
) -> tuple:
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
    except:
        return "❌ API密钥未配置，请在 secrets.toml 中设置 DEEPSEEK_API_KEY", []
    
    system_content = REACT_SYSTEM_PROMPT.format(display_name=display_name)
    messages = [{"role": "system", "content": system_content}]
    
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    
    react_steps = []
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    for round_num in range(MAX_TOOL_ROUNDS):
        payload = {
            "model":       "deepseek-chat",
            "messages":    messages,
            "tools":       TOOL_SCHEMAS,
            "tool_choice": "auto",
            "temperature": TEMPERATURE,
            "max_tokens":  1500
        }
        
        try:
            resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=45)
        except requests.Timeout:
            return "⏱️ 请求超时，请重试。", react_steps
        except Exception as e:
            return f"🌐 网络错误：{e}", react_steps
        
        if resp.status_code != 200:
            return f"❌ API错误 {resp.status_code}：{resp.text[:200]}", react_steps
        
        resp_json = resp.json()
        choice = resp_json['choices'][0]
        finish_reason = choice['finish_reason']
        ai_message = choice['message']
        
        raw_content = ai_message.get("content") or ""
        thought = _extract_thought(raw_content)
        
        messages.append(ai_message)
        
        if finish_reason == "stop" or not ai_message.get("tool_calls"):
            final_answer = _extract_answer(raw_content)
            if thought and react_steps:
                react_steps[-1]["final_thought"] = thought
            elif thought and not react_steps:
                react_steps.append({
                    "thought": thought,
                    "action": None,
                    "action_input": None,
                    "observation": None,
                })
            return final_answer, react_steps
        
        for tc in ai_message.get("tool_calls", []):
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}
            
            observation = _dispatch_tool(tool_name, tool_args, user_id)
            
            react_steps.append({
                "thought": thought,
                "action": tool_name,
                "action_input": tool_args,
                "observation": observation,
            })
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": observation
            })
    
    return "⚠️ 推理轮数超限，请换个方式提问。", react_steps


def _extract_thought(content: str) -> str:
    if not content:
        return ""
    content = content.strip()
    if "Thought:" in content:
        thought_part = content.split("Thought:")[-1]
        if "Answer:" in thought_part:
            thought_part = thought_part.split("Answer:")[0]
        return thought_part.strip()
    if "Answer:" not in content:
        return content.strip()
    return ""


def _extract_answer(content: str) -> str:
    if not content:
        return "（无回复）"
    if "Answer:" in content:
        return content.split("Answer:")[-1].strip()
    if content.startswith("Thought:"):
        after_thought = content.replace("Thought:", "", 1).strip()
        if len(after_thought) > 20:
            return after_thought
    return content.strip()


# ==================== 知识库在线学习 ====================
def learn_new_knowledge(question: str, correct_answer: str):
    collection = _load_collection()
    content = f"问题：{question}\n答案：{correct_answer}（用户补充）"
    doc_id = f"qa_learned_{int(time.time())}"
    embedding = _embed([content])
    collection.add(
        documents=[content],
        embeddings=embedding,
        ids=[doc_id]
    )
    print(f"✅ 纠错学习：{question[:30]}...")

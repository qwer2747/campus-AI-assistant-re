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

# ==================== 知识库加载（自动从 CSV 构建）====================
def _load_collection():
    """加载知识库，如果为空则自动从 CSV 构建"""
    print("📚 正在连接知识库...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name="campus_qa")
    
    print(f"📊 当前记录数: {collection.count()}")
    
    if collection.count() == 0 and os.path.exists(CSV_PATH):
        print("📦 知识库为空，正在从 CSV 自动构建...")
        
        # 直接加载模型（不通过 _embed 函数）
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        try:
            df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(CSV_PATH, encoding='gbk')
        
        documents = [f"问题：{r['question']}\n答案：{r['answer']}" for _, r in df.iterrows()]
        ids = [f"qa_{i}" for i in range(len(documents))]
        embeddings = model.encode(documents, normalize_embeddings=True).tolist()
        
        collection.add(documents=documents, embeddings=embeddings, ids=ids)
        print(f"✅ 构建完成，共 {len(documents)} 条数据")
    
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
    
    if collection.count() == 0:
        return "📭 知识库暂无数据，请确保 campus_qa.csv 文件存在。"
    
    results = collection.query(query_embeddings=_embed([query]), n_results=k)
    docs = results['documents'][0] if results['documents'] else []
    
    if not docs:
        return "📭 知识库中暂未找到相关资料。"
    return "\n\n".join([f"[资料{i+1}] {doc}" for i, doc in enumerate(docs)])

# ==================== 时间工具 ====================
def tool_get_datetime() -> str:
    now = datetime.now() + timedelta(hours=8)
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
            "description": "搜索校园知识库。当用户问图书馆、食堂、宿舍、选课、校园卡、报到、学分等校园问题时，必须调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜索关键词"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "仅在用户明确询问当前时间、日期、星期时使用。",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "数学计算",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "记住用户告诉你的个人信息",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_facts",
            "description": "回忆当前用户的个人信息",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# ==================== ReAct 系统提示 ====================
REACT_SYSTEM_PROMPT = """你是校园新生助手「小慧」，帮助{display_name}解决校园生活问题。

## 工具使用规则
1. 校园相关问题 → 必须先调用 search_knowledge
2. 时间问题 → 调用 get_datetime
3. 用户说个人信息 → 调用 remember_fact

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
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    
    for _ in range(MAX_TOOL_ROUNDS):
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "temperature": TEMPERATURE,
            "max_tokens": 1500
        }
        
        try:
            resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=45)
        except requests.Timeout:
            return "⏱️ 请求超时，请重试。", react_steps
        except Exception as e:
            return f"🌐 网络错误：{e}", react_steps
        
        if resp.status_code != 200:
            return f"❌ API错误 {resp.status_code}", react_steps
        
        ai_message = resp.json()['choices'][0]['message']
        raw_content = ai_message.get("content") or ""
        thought = _extract_thought(raw_content)
        
        messages.append(ai_message)
        
        if not ai_message.get("tool_calls"):
            final_answer = _extract_answer(raw_content)
            return final_answer, react_steps
        
        for tc in ai_message.get("tool_calls", []):
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except:
                tool_args = {}
            
            observation = _dispatch_tool(tool_name, tool_args, user_id)
            react_steps.append({"thought": thought, "action": tool_name, "action_input": tool_args, "observation": observation})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": observation})
    
    return "⚠️ 推理轮数超限，请换个方式提问。", react_steps

def _extract_thought(content: str) -> str:
    if not content:
        return ""
    if "Thought:" in content:
        return content.split("Thought:")[-1].split("Answer:")[0].strip()
    return content.strip()

def _extract_answer(content: str) -> str:
    if not content:
        return "（无回复）"
    if "Answer:" in content:
        return content.split("Answer:")[-1].strip()
    return content.strip()

# ==================== 知识库在线学习 ====================
def learn_new_knowledge(question: str, correct_answer: str):
    collection = _load_collection()
    content = f"问题：{question}\n答案：{correct_answer}（用户补充）"
    doc_id = f"qa_learned_{int(time.time())}"
    embedding = _embed([content])
    collection.add(documents=[content], embeddings=embedding, ids=[doc_id])
    print(f"✅ 纠错学习：{question[:30]}...")

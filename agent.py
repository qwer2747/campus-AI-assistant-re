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
from datetime import datetime

from memory import save_user_fact, get_user_facts

# ==================== 配置 ====================
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
CHROMA_DB_PATH   = "./chroma_db"
CSV_PATH         = "campus_qa.csv"
EMBEDDING_MODEL  = "BAAI/bge-small-zh"
MAX_TOOL_ROUNDS  = 6
TEMPERATURE      = 0.3

# ==================== 模型加载 ====================
import hashlib

@st.cache_resource(show_spinner="⚙️ 正在加载知识库...")
def _load_resources():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    kb_collection = chroma_client.get_or_create_collection(name="campus_qa")
    if kb_collection.count() == 0 and os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(CSV_PATH, encoding='gbk')
        docs = [f"问题：{r['question']}\n答案：{r['answer']}" for _, r in df.iterrows()]
        ids  = [f"qa_{i}" for i in range(len(docs))]
        embs = _embed(docs)
        kb_collection.add(documents=docs, embeddings=embs, ids=ids)
    return kb_collection

def _embed(texts):
    result = []
    for text in texts:
        vec = []
        for i in range(384):
            h = int(hashlib.md5(f"{text}{i}".encode()).hexdigest(), 16)
            vec.append((h % 10000) / 10000.0 - 0.5)
        result.append(vec)
    return result
    
def tool_search_knowledge(query: str, k: int = 5) -> str:
    collection = _load_resources()
    results = collection.query(query_embeddings=_embed([query]), n_results=k)
    docs = results['documents'][0] if results['documents'] else []
    if not docs:
        return "📭 知识库中暂未找到相关资料。"
    return "\n\n".join([f"[资料{i+1}] {doc}" for i, doc in enumerate(docs)])

def tool_get_datetime() -> str:
    now = datetime.now()
    weekdays = ['一','二','三','四','五','六','日']
    return (f"📅 当前时间：{now.strftime('%Y年%m月%d日 %H:%M:%S')}，"
            f"星期{weekdays[now.weekday()]}，第{now.isocalendar()[1]}周")

def tool_calculate(expression: str) -> str:
    try:
        safe_ns = {k: v for k, v in math.__dict__.items() if not k.startswith('_')}
        result = eval(expression, {"__builtins__": {}}, safe_ns)
        return f"🔢 {expression} = {result}"
    except ZeroDivisionError:
        return "❌ 除数不能为零"
    except Exception as e:
        return f"❌ 计算错误：{e}"

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

# ==================== 工具注册表（不变）====================
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索校园知识库，获取图书馆、食堂、宿舍、选课、校园卡等官方信息。有校园相关问题时优先使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "具体搜索词，越具体越好"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "获取当前准确日期、时间和星期",
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
                    "expression": {"type": "string", "description": "Python数学表达式，如'2**10'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "永久记住用户的个人信息（专业、年级、兴趣等），下次对话仍有效",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "要记住的事实，如'用户是计算机专业大一新生'"}
                },
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_facts",
            "description": "回忆当前用户的所有已知个人信息",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# ==================== ReAct 系统提示 ====================
# ⭐ 关键：要求模型在调用工具前必须写出Thought
REACT_SYSTEM_PROMPT = """你是校园新生助手「小慧」，帮助{display_name}解决校园生活问题。

## ⚡ 你必须遵循 ReAct 推理框架

每次行动前，必须在 content 字段写出你的 Thought（思考过程）。

### 标准格式：
**调用工具时：**
content = "Thought: [分析问题，说明为什么要用这个工具]"
tool_calls = [调用工具]

**最终回答时：**
content = "Thought: [基于观察结果，整理最终答案]\n\nAnswer: [给用户的正式回答]"

### 示例：
用户问："图书馆今天还开着吗？"

第1步：
Thought: 这个问题需要两个信息：①图书馆的开放时间 ②当前时间。我先查知识库获取图书馆开放时间。
→ 调用 search_knowledge("图书馆开放时间")

第2步（看到知识库结果后）：
Thought: 知道了图书馆8:00-22:00开放，还需要知道现在几点。
→ 调用 get_datetime()

第3步（看到时间结果后）：
Thought: 现在21:30，图书馆22:00关，还有30分钟，来得及。可以给出答案了。
Answer: 图书馆今天22:00关门，现在21:30，还有30分钟，完全来得及！记得带好校园卡哦 🎒

## 工具使用规则
1. 校园相关问题 → 必须先调用 search_knowledge
2. 涉及时间/日期 → 调用 get_datetime
3. 用户提到个人信息 → 调用 remember_fact
4. 每步 Thought 要简洁清晰，让用户看懂你的推理

## 当前用户：{display_name}"""


# ==================== ReAct Agent 主循环 ====================
def run_agent(
    user_message: str,
    user_id: str,
    chat_history: list,
    display_name: str = "同学"
) -> tuple:
    """
    ReAct Agent 主入口
    
    Returns:
        (final_answer: str, react_steps: list)
        
        react_steps 格式（每步包含完整Thought-Action-Observation）:
        [
            {
                "thought":      "我需要先查图书馆时间...",   # LLM的推理
                "action":       "search_knowledge",          # 工具名
                "action_input": {"query": "图书馆开放时间"}, # 工具参数
                "observation":  "图书馆8:00-22:00开放...",   # 工具结果
            },
            ...
        ]
    """
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
    except:
        api_key = ""
        if not api_key:
            return "❌ API密钥未配置，请在 secrets.toml 中设置 DEEPSEEK_API_KEY", []
    
    # ---- 构建消息 ----
    system_content = REACT_SYSTEM_PROMPT.format(display_name=display_name)
    messages = [{"role": "system", "content": system_content}]
    
    # 注入最近6条历史
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    
    # ---- ReAct 步骤日志 ----
    react_steps = []  # 每步：{thought, action, action_input, observation}
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # ==================== ReAct 循环 ====================
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
            resp = requests.post(
                DEEPSEEK_API_URL, headers=headers, json=payload, timeout=45
            )
        except requests.Timeout:
            return "⏱️ 请求超时，请重试。", react_steps
        except Exception as e:
            return f"🌐 网络错误：{e}", react_steps
        
        if resp.status_code != 200:
            return f"❌ API错误 {resp.status_code}：{resp.text[:200]}", react_steps
        
        resp_json     = resp.json()
        choice        = resp_json['choices'][0]
        finish_reason = choice['finish_reason']
        ai_message    = choice['message']
        
        # ⭐ 提取 Thought（模型在content里写的推理）
        raw_content = ai_message.get("content") or ""
        thought = _extract_thought(raw_content)
        
        # 将AI回复加入历史
        messages.append(ai_message)
        
        # ---- 情况①：不调用工具，直接回答 ----
        if finish_reason == "stop" or not ai_message.get("tool_calls"):
            final_answer = _extract_answer(raw_content)
            
            # 如果最后一步有Thought，补充到日志里
            if thought and react_steps:
                react_steps[-1]["final_thought"] = thought
            elif thought and not react_steps:
                # 没有调用任何工具，直接回答
                react_steps.append({
                    "thought":      thought,
                    "action":       None,
                    "action_input": None,
                    "observation":  None,
                })
            
            return final_answer, react_steps
        
        # ---- 情况②：调用工具（Action阶段）----
        for tc in ai_message.get("tool_calls", []):
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}
            
            # 执行工具 → Observation
            observation = _dispatch_tool(tool_name, tool_args, user_id)
            
            # ⭐ 记录完整的 Thought-Action-Observation 步骤
            react_steps.append({
                "thought":      thought,          # 推理过程
                "action":       tool_name,        # 使用的工具
                "action_input": tool_args,        # 工具输入
                "observation":  observation,      # 工具输出
            })
            
            # 将 Observation 注入消息
            messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      observation
            })
            
            # 一个 round 通常只有一个工具调用，
            # 但 DeepSeek 支持并行调用，这里全部处理
    
    return "⚠️ 推理轮数超限，请换个方式提问。", react_steps


def _extract_thought(content: str) -> str:
    """从content中提取Thought部分"""
    if not content:
        return ""
    content = content.strip()
    
    # 处理 "Thought: xxx\n\nAnswer: xxx" 格式
    if "Thought:" in content:
        thought_part = content.split("Thought:")[-1]
        # 去掉Answer部分
        if "Answer:" in thought_part:
            thought_part = thought_part.split("Answer:")[0]
        return thought_part.strip()
    
    # 处理没有标签但有内容的情况（直接当thought）
    if "Answer:" not in content:
        return content.strip()
    
    return ""


def _extract_answer(content: str) -> str:
    """从content中提取最终Answer"""
    if not content:
        return "（无回复）"
    
    # 有 Answer: 标签
    if "Answer:" in content:
        return content.split("Answer:")[-1].strip()
    
    # 去掉 Thought: 前缀后返回
    if content.startswith("Thought:"):
        # 如果只有Thought没有Answer，把Thought内容当答案（兜底）
        after_thought = content.replace("Thought:", "", 1).strip()
        if len(after_thought) > 20:
            return after_thought
    
    return content.strip()


# ==================== 知识库在线学习 ====================
def learn_new_knowledge(question: str, correct_answer: str):
    collection = _load_resources()
    content  = f"问题：{question}\n答案：{correct_answer}（用户补充）"
    doc_id   = f"qa_learned_{int(time.time())}"
    collection.add(
        documents=[content],
        embeddings=_embed([content]),
        ids=[doc_id]
    )
    print(f"✅ 纠错学习：{question[:30]}...")

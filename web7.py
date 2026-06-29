# ==================== web7.py ====================
"""
只需替换工具调用展示部分，其余代码完全不变
"""
import json
import streamlit as st
from datetime import datetime

from agent import run_agent, learn_new_knowledge
from memory import (
    register_user, login_user,
    save_message, load_chat_history, clear_chat_history,
    get_user_facts, delete_user_fact,
    get_chat_stats, save_feedback
)

# ==================== 页面配置（不变）====================
st.set_page_config(
    page_title="校园智能助手",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 全局样式（新增ReAct展示样式）====================
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); }

.login-card {
    background: white; border-radius: 20px; padding: 40px;
    box-shadow: 0 20px 60px rgba(102,126,234,0.2); margin: 20px 0;
}
.login-title {
    text-align: center; font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.login-subtitle {
    text-align: center; color: #888; margin-bottom: 30px; font-size: 1rem;
}

/* ---- ReAct 步骤卡片 ---- */
.react-step {
    border-left: 3px solid #667eea;
    padding: 10px 15px;
    margin: 8px 0;
    background: #f8f7ff;
    border-radius: 0 8px 8px 0;
}
.react-thought {
    border-left: 3px solid #f59e0b;  /* 黄色 = 思考 */
    background: #fffbeb;
    padding: 8px 14px;
    border-radius: 0 6px 6px 0;
    margin: 4px 0;
    font-size: 0.9rem;
    color: #92400e;
}
.react-action {
    border-left: 3px solid #667eea;  /* 蓝色 = 行动 */
    background: #eff6ff;
    padding: 8px 14px;
    border-radius: 0 6px 6px 0;
    margin: 4px 0;
}
.react-observation {
    border-left: 3px solid #10b981;  /* 绿色 = 观察 */
    background: #f0fdf4;
    padding: 8px 14px;
    border-radius: 0 6px 6px 0;
    margin: 4px 0;
    font-size: 0.85rem;
    color: #065f46;
}
.fact-tag {
    background: #f0e6ff; border: 1px solid #c084fc; border-radius: 15px;
    padding: 4px 12px; margin: 3px; font-size: 0.85rem; color: #7c3aed;
    display: inline-block;
}
[data-testid="stChatMessage"] {
    border-radius: 12px; margin: 8px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# ==================== Session State 初始化（不变）====================
def _init_session():
    defaults = {
        "logged_in": False, "user_id": None, "username": None,
        "display_name": None, "messages": [], "quick_prompt": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ==================== ⭐ ReAct 步骤展示组件（核心新增）====================
def render_react_steps(react_steps: list):
    """
    将 Thought-Action-Observation 链可视化展示
    
    react_steps 格式：
    [{"thought": str, "action": str, "action_input": dict, "observation": str}]
    """
    if not react_steps:
        return
    
    # 工具图标映射
    tool_icons = {
        "search_knowledge": "🔍 搜索知识库",
        "get_datetime":     "📅 查询时间",
        "calculate":        "🔢 数学计算",
        "remember_fact":    "💾 记住信息",
        "recall_facts":     "🧠 回忆记忆",
    }
    
    # 过滤掉没有action的纯thought步骤
    real_steps = [s for s in react_steps if s.get("action")]
    
    if not real_steps:
        return
    
    with st.expander(
        f"🧠 推理过程（{len(real_steps)}步）— 点击查看",
        expanded=False
    ):
        for i, step in enumerate(real_steps):
            st.markdown(f"**第 {i+1} 步**")
            
            # 💭 Thought
            if step.get("thought"):
                st.markdown(
                    f"<div class='react-thought'>"
                    f"💭 <b>Thought（思考）</b><br>{step['thought']}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            # ⚡ Action
            action_label = tool_icons.get(step['action'], f"🔧 {step['action']}")
            action_input_str = ""
            if step.get("action_input"):
                # 只展示参数值，不展示key
                vals = list(step["action_input"].values())
                if vals:
                    action_input_str = f"：`{vals[0]}`" if len(str(vals[0])) < 60 else ""
            
            st.markdown(
                f"<div class='react-action'>"
                f"⚡ <b>Action（行动）</b> → {action_label}{action_input_str}"
                f"</div>",
                unsafe_allow_html=True
            )
            
            # 展示详细参数（可折叠）
            if step.get("action_input"):
                with st.expander("查看完整参数", expanded=False):
                    st.json(step["action_input"])
            
            # 👁️ Observation
            if step.get("observation"):
                obs_text = step["observation"]
                # 超长结果截断显示
                display_obs = obs_text[:400] + ("..." if len(obs_text) > 400 else "")
                st.markdown(
                    f"<div class='react-observation'>"
                    f"👁️ <b>Observation（观察）</b><br>"
                    f"<pre style='margin:0;font-size:0.8rem;white-space:pre-wrap'>"
                    f"{display_obs}</pre>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            # 最后一步不加分隔线
            if i < len(real_steps) - 1:
                st.markdown(
                    "<div style='text-align:center;color:#ccc;margin:4px 0'>↓</div>",
                    unsafe_allow_html=True
                )
        
        # 最终思考（如果有）
        last_step = react_steps[-1]
        if last_step.get("final_thought"):
            st.markdown("---")
            st.markdown(
                f"<div class='react-thought'>"
                f"💡 <b>Final Thought（最终推理）</b><br>{last_step['final_thought']}"
                f"</div>",
                unsafe_allow_html=True
            )


# ==================== 登录/注册页====================
def show_auth_page():
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">🎓 校园智能助手</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">你的专属AI小助手，记住你的每一次问答</div>', unsafe_allow_html=True)
        
        tab_login, tab_register = st.tabs(["🔑 登录", "✨ 注册"])
        
        with tab_login:
            st.markdown("#### 欢迎回来")
            with st.form("login_form"):
                username = st.text_input("学号 / 用户名", placeholder="请输入学号或用户名")
                password = st.text_input("密码", type="password", placeholder="请输入密码")
                col_btn, col_hint = st.columns([1, 1])
                with col_btn:
                    submitted = st.form_submit_button("🚀 登录", use_container_width=True, type="primary")
                with col_hint:
                    st.markdown("<p style='color:#aaa;font-size:0.8rem;padding-top:8px;'>首次使用？前往注册 →</p>", unsafe_allow_html=True)
            
            if submitted:
                if not username or not password:
                    st.error("请填写用户名和密码")
                else:
                    success, user_id, display_name = login_user(username, password)
                    if success:
                        st.session_state.logged_in    = True
                        st.session_state.user_id      = user_id
                        st.session_state.username     = username
                        st.session_state.display_name = display_name
                        st.session_state.messages     = load_chat_history(user_id)
                        st.success(f"✅ 登录成功！欢迎回来，{display_name}")
                        st.rerun()
                    else:
                        st.error("❌ 用户名或密码错误")
        
        with tab_register:
            st.markdown("#### 创建新账号")
            with st.form("register_form", clear_on_submit=True):
                reg_display  = st.text_input("昵称", placeholder="如：张同学、小明")
                reg_username = st.text_input("用户名 / 学号", placeholder="用于登录，至少2位")
                reg_pass1    = st.text_input("设置密码", type="password", placeholder="至少6位")
                reg_pass2    = st.text_input("确认密码", type="password", placeholder="再次输入密码")
                reg_submitted = st.form_submit_button("🎉 立即注册", use_container_width=True, type="primary")
            
            if reg_submitted:
                if not all([reg_username, reg_pass1, reg_pass2]):
                    st.error("请填写所有必填项")
                elif reg_pass1 != reg_pass2:
                    st.error("❌ 两次密码不一致")
                else:
                    ok, msg, user_id = register_user(reg_username, reg_pass1, reg_display)
                    if ok:
                        st.success(f"✅ {msg}，请切换到登录标签页登录")
                        st.balloons()
                    else:
                        st.error(f"❌ {msg}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        feat_cols = st.columns(3)
        features = [
            ("🤖", "AI小助手", "思考→行动→观察"),
            ("🧠", "持久化记忆", "重启不丢，记住你的偏好"),
            ("📚", "知识库", "基于校园官方和非官方数据回答"),
        ]
        for col, (icon, title, desc) in zip(feat_cols, features):
            with col:
                st.markdown(
                    f"<div style='text-align:center;padding:15px;background:white;"
                    f"border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.06)'>"
                    f"<div style='font-size:2rem'>{icon}</div><b>{title}</b><br>"
                    f"<span style='color:#888;font-size:0.85rem'>{desc}</span></div>",
                    unsafe_allow_html=True
                )


# ==================== 主聊天页 ====================
def show_chat_page():
    user_id      = st.session_state.user_id
    display_name = st.session_state.display_name
    
    # ---- 侧边栏（完全不变）----
    with st.sidebar:
        stats = get_chat_stats(user_id)
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#667eea,#764ba2);"
            f"border-radius:12px;padding:15px;color:white;margin-bottom:15px'>"
            f"<div style='font-size:1.3rem;font-weight:bold'>👤 {display_name}</div>"
            f"<div style='opacity:0.85;font-size:0.85rem'>@{st.session_state.username}</div>"
            f"<div style='margin-top:10px;font-size:0.85rem'>💬 累计提问：{stats['message_count']}次</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        if st.button("🚪 退出登录", use_container_width=True):
            for key in ["logged_in","user_id","username","display_name","messages","quick_prompt"]:
                st.session_state[key] = None if key != "logged_in" else False
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ⚡ 快捷提问")
        quick_questions = {
            "📚 图书馆": "图书馆几点开门，现在还来得及去吗？",
            "🍔 食堂":   "学校有几个食堂？",
            "💳 校园卡": "校园卡丢失了怎么办？",
            "📖 选课":   "选课系统怎么登录，什么时候开始选课？",
            "🏠 宿舍":   "宿舍几点关门？",
            "🎓 学籍":   "学籍注册需要准备什么材料？",
        }
        cols = st.columns(2)
        for idx, (label, question) in enumerate(quick_questions.items()):
            with cols[idx % 2]:
                if st.button(label, use_container_width=True, key=f"quick_{idx}"):
                    st.session_state.quick_prompt = question
        
        st.markdown("---")
        st.markdown("### 🧠 我的记忆")
        facts = get_user_facts(user_id)
        if facts:
            for f in facts[:8]:
                col_f, col_del = st.columns([4, 1])
                with col_f:
                    st.markdown(
                        f"<div class='fact-tag'>{f['fact'][:20]}{'...' if len(f['fact'])>20 else ''}</div>",
                        unsafe_allow_html=True
                    )
                with col_del:
                    if st.button("×", key=f"del_fact_{f['fact'][:10]}", help="删除"):
                        delete_user_fact(user_id, f['fact'])
                        st.rerun()
        else:
            st.caption("💡 告诉我你的专业、年级，我会记住！")
        
        st.markdown("---")
        if st.button("🗑️ 清空对话历史", use_container_width=True, type="secondary"):
            clear_chat_history(user_id)
            st.session_state.messages = []
            st.rerun()
    
    # ---- 主区域标题 ----
    st.markdown(
        f"<h2 style='text-align:center;color:#333;margin-bottom:5px'>"
        f"💬 校园智能助手</h2>"
        f"<p style='text-align:center;color:#888;margin-bottom:20px'>"
        f"你好，{display_name}！我是你的AI小助手，"
        f"我会努力为你提供你想要的答案 🧠</p>",
        unsafe_allow_html=True
    )
    
    # ---- 渲染历史消息 ----
    for i, msg in enumerate(st.session_state.messages):
        role   = msg["role"]
        avatar = "🧑‍🎓" if role == "user" else "🤖"
        
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            
            # ⭐ 展示 ReAct 推理链（历史消息）
            if role == "assistant" and msg.get("tool_calls_log"):
                try:
                    react_steps = (
                        json.loads(msg["tool_calls_log"])
                        if isinstance(msg["tool_calls_log"], str)
                        else msg["tool_calls_log"]
                    )
                    render_react_steps(react_steps)
                except Exception:
                    pass
            
            # 最后一条AI回复：纠错入口
            if role == "assistant" and i == len(st.session_state.messages) - 1 and i > 0:
                last_user_msg = st.session_state.messages[i-1]["content"]
                with st.expander("💡 答案有误？帮我纠错"):
                    correct_ans = st.text_area("正确答案应该是：", key=f"corr_{i}", height=80)
                    if st.button("✅ 提交纠错并让我学习", key=f"submit_corr_{i}"):
                        if correct_ans.strip():
                            learn_new_knowledge(last_user_msg, correct_ans)
                            save_feedback(user_id, last_user_msg, msg["content"], correct_ans)
                            st.success("🎉 感谢纠错！已写入知识库。")
    
    # ---- 处理输入 ----
    prompt = st.session_state.quick_prompt or st.chat_input("💬 问我任何校园问题...")
    if st.session_state.quick_prompt:
        st.session_state.quick_prompt = None
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(user_id, "user", prompt)
        
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            # ⭐ 动态状态：显示ReAct阶段
            status = st.empty()
            status.markdown("💭 *Thought：正在分析问题...*")
            
            answer, react_steps = run_agent(
                user_message=prompt,
                user_id=user_id,
                chat_history=st.session_state.messages[:-1],
                display_name=display_name
            )
            
            status.empty()
            st.markdown(answer)
            
            # ⭐ 展示ReAct推理链
            render_react_steps(react_steps)
        
        # 保存
        react_json = json.dumps(react_steps, ensure_ascii=False) if react_steps else None
        st.session_state.messages.append({
            "role":           "assistant",
            "content":        answer,
            "tool_calls_log": react_steps
        })
        save_message(user_id, "assistant", answer, react_json)
        st.rerun()


# ==================== 路由 ====================
if not st.session_state.logged_in:
    show_auth_page()
else:
    show_chat_page()

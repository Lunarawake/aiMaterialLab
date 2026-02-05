"""
NEXUS 通用材料研发平台
双视图架构：智能仪表盘 + 数据工作台
支持定量目标设定
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="NEXUS 材料研发平台",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 自定义 CSS - 极简白 + 科研蓝
# ============================================================
st.markdown("""
<style>
    /* ====== 全局样式 ====== */
    .stApp {
        background-color: #FFFFFF;
    }
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif !important;
        color: #333333;
    }
    
    /* ====== Header ====== */
    .app-header {
        padding: 1.25rem 0;
        border-bottom: 1px solid #e8e8e8;
        margin-bottom: 1.5rem;
    }
    
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a1a;
        letter-spacing: 1px;
    }
    
    .app-title span {
        color: #2563eb;
    }
    
    .app-subtitle {
        font-size: 0.9rem;
        color: #888888;
        margin-top: 0.25rem;
    }
    
    /* ====== 区域标题 ====== */
    .area-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333333;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2563eb;
        display: inline-block;
    }
    
    .area-number {
        color: #2563eb;
        font-weight: 700;
    }
    
    /* ====== 分隔线 ====== */
    .section-divider {
        border: none;
        border-top: 1px solid #e8e8e8;
        margin: 2rem 0;
    }
    
    /* ====== 项目信息卡片 ====== */
    .project-card {
        background: linear-gradient(135deg, #f8faff 0%, #f0f5ff 100%);
        border: 1px solid #d0e0f5;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .project-label {
        font-size: 0.7rem;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .project-value {
        font-size: 1rem;
        font-weight: 600;
        color: #333333;
        margin-top: 0.2rem;
    }
    
    /* ====== 目标卡片 ====== */
    .target-card {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    
    .target-label {
        font-size: 0.75rem;
        color: #166534;
        font-weight: 600;
    }
    
    .target-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #15803d;
    }
    
    .current-value {
        font-size: 0.8rem;
        color: #666666;
    }
    
    /* ====== 数据摘要卡片 ====== */
    .data-summary {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    
    .summary-item {
        display: inline-block;
        margin-right: 2rem;
    }
    
    .summary-label {
        font-size: 0.7rem;
        color: #888888;
        text-transform: uppercase;
    }
    
    .summary-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #333333;
    }
    
    /* ====== AI 分析卡片 ====== */
    .insight-card {
        background: linear-gradient(135deg, #fafbff 0%, #f5f8ff 100%);
        border: 1px solid #d0e0f5;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .insight-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #2563eb;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }
    
    .action-card {
        background: linear-gradient(135deg, #f8fffe 0%, #f0fdf9 100%);
        border: 1px solid #a7e8d8;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .action-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #10b981;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }
    
    /* ====== 映射信息 ====== */
    .mapping-info {
        background: #fff8f0;
        border: 1px solid #ffd6a5;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #666666;
        margin-bottom: 1rem;
    }
    
    .mapping-tag {
        display: inline-block;
        background: #e8e8e8;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    
    .mapping-tag.input {
        background: #dbeafe;
        color: #1d4ed8;
    }
    
    .mapping-tag.output {
        background: #d1fae5;
        color: #047857;
    }
    
    /* ====== 目标设定区域 ====== */
    .target-section {
        background: #fefce8;
        border: 1px solid #fef08a;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
    }
    
    .target-section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #854d0e;
        margin-bottom: 0.75rem;
    }
    
    /* ====== 按钮样式 ====== */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }
    
    /* ====== 输入框 ====== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background-color: #FFFFFF;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        color: #333333;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1);
    }
    
    /* ====== 提示框 ====== */
    .hint-box {
        background: #f0f7ff;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #1e40af;
        margin-bottom: 1rem;
    }
    
    /* ====== 占位符 ====== */
    .placeholder-box {
        background: #fafbfc;
        border: 1px dashed #d0d0d0;
        border-radius: 8px;
        padding: 2.5rem;
        text-align: center;
        color: #999999;
    }
    
    /* ====== 页脚 ====== */
    .footer {
        text-align: center;
        color: #aaaaaa;
        font-size: 0.8rem;
        padding: 2rem 0;
        border-top: 1px solid #e8e8e8;
        margin-top: 2rem;
    }
    
    /* ====== 间距 ====== */
    .block-container {
        padding: 1.5rem 2.5rem 2rem 2.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 初始化 Session State
# ============================================================

def init_session_state():
    """初始化所有 session state 变量"""
    
    # 视图状态
    if 'current_view' not in st.session_state:
        st.session_state['current_view'] = 'dashboard'
    
    # 项目信息（移除 optimization_goal，改用 target_values）
    if 'material_name' not in st.session_state:
        st.session_state['material_name'] = ''
    if 'equipment_name' not in st.session_state:
        st.session_state['equipment_name'] = ''
    
    # 实验数据
    if 'experiment_data' not in st.session_state:
        st.session_state['experiment_data'] = pd.DataFrame({
            '温度(°C)': [1800, 1850, 1900, 1950, 2000],
            '压力(mbar)': [50, 55, 60, 65, 70],
            'Ar流量(sccm)': [100, 100, 120, 120, 150],
            '生长时间(h)': [24, 24, 30, 30, 36],
            '生长速率(um/h)': [80, 95, 110, 105, 98],
            '微管密度(cm-2)': [5.2, 4.1, 2.8, 3.5, 4.0]
        })
    
    # 列映射
    if 'input_columns' not in st.session_state:
        st.session_state['input_columns'] = []
    if 'output_columns' not in st.session_state:
        st.session_state['output_columns'] = []
    
    # 定量目标值字典
    if 'target_values' not in st.session_state:
        st.session_state['target_values'] = {}
    
    # AI 分析结果
    if 'ai_result' not in st.session_state:
        st.session_state['ai_result'] = None
    
    # API Key
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ''


# ============================================================
# 工具函数
# ============================================================

def create_trend_chart(df: pd.DataFrame, output_cols: list, target_values: dict) -> go.Figure:
    """创建实验趋势图，包含目标线"""
    fig = go.Figure()
    
    if not output_cols or df.empty:
        fig.add_annotation(text="请选择输出列以显示趋势图", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=300)
        return fig
    
    x_data = list(range(1, len(df) + 1))
    colors = ['#2563eb', '#10b981', '#f59e0b', '#ef4444']
    
    for idx, col in enumerate(output_cols[:4]):
        if col in df.columns:
            y_data = pd.to_numeric(df[col], errors='coerce')
            color = colors[idx % len(colors)]
            
            # 数据线
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='lines+markers',
                name=col,
                line=dict(color=color, width=2),
                marker=dict(size=7)
            ))
            
            # 目标线
            if col in target_values and target_values[col]:
                try:
                    target_val = float(target_values[col])
                    fig.add_hline(
                        y=target_val,
                        line_dash="dash",
                        line_color=color,
                        annotation_text=f"{col} 目标: {target_val}",
                        annotation_position="right",
                        annotation_font_color=color
                    )
                except:
                    pass
    
    fig.update_layout(
        template='simple_white',
        title=dict(text='实验结果趋势（虚线为目标值）', font=dict(size=14)),
        xaxis_title='实验编号',
        yaxis_title='数值',
        height=320,
        margin=dict(t=50, b=40, l=50, r=100),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    fig.update_xaxes(gridcolor='#f0f0f0')
    fig.update_yaxes(gridcolor='#f0f0f0')
    
    return fig


def analyze_with_ai(
    df: pd.DataFrame,
    material_name: str,
    equipment_name: str,
    input_cols: list,
    output_cols: list,
    target_values: dict,
    api_key: str
) -> dict:
    """使用定量目标感知的 AI 进行分析"""
    try:
        genai.configure(api_key=api_key)
        
        # 数据转 CSV
        data_csv = df.to_csv(index=False)
        
        # 构建量化目标描述
        target_descriptions = []
        for col in output_cols:
            if col in df.columns:
                current_avg = pd.to_numeric(df[col], errors='coerce').mean()
                current_max = pd.to_numeric(df[col], errors='coerce').max()
                current_min = pd.to_numeric(df[col], errors='coerce').min()
                
                target_val = target_values.get(col, '')
                if target_val:
                    gap = float(target_val) - current_avg
                    gap_pct = (gap / current_avg * 100) if current_avg != 0 else 0
                    target_descriptions.append(
                        f"- {col}：目标值 = {target_val}，当前平均值 = {current_avg:.2f}，"
                        f"当前最优 = {current_max:.2f}，差距 = {gap:.2f} ({gap_pct:+.1f}%)"
                    )
                else:
                    target_descriptions.append(
                        f"- {col}：未设定目标，当前平均值 = {current_avg:.2f}，当前最优 = {current_max:.2f}"
                    )
        
        target_str = "\n".join(target_descriptions) if target_descriptions else "（用户未设定具体目标）"
        
        # 构建 System Prompt
        system_prompt = f"""你是一位世界顶级的材料科学家和工艺工程师。

用户正在进行【{material_name if material_name else '材料'}】的研究。
使用的设备/工艺是：【{equipment_name if equipment_name else '实验设备'}】。

你的任务是帮助用户达成他们设定的**量化目标**。
你的分析必须：
1. 精确指出当前数据与目标值的差距
2. 结合物理/化学原理解释瓶颈
3. 给出能够逼近目标值的具体参数建议
4. 如果目标不切实际，诚实指出"""

        # 构建 User Prompt
        input_cols_str = ', '.join(input_cols) if input_cols else '（用户未指定）'
        
        user_prompt = f"""## 实验数据
```csv
{data_csv}
```

## 数据列说明
- **实验参数列 (可调变量)**：{input_cols_str}

## 用户的量化目标
{target_str}

---

请按以下结构分析：

### 一、目标差距诊断
针对每个设定了目标的指标，分析：
1. 当前距离目标还有多大差距？
2. 从数据趋势看，哪些参数组合表现最好？
3. 是否存在参数间的权衡关系？

### 二、瓶颈机理分析
结合【{material_name}】的物理/化学原理：
1. 什么因素阻碍了目标的达成？
2. 当前工艺的主要限制是什么？

### 三、精准参数建议
为了逼近设定的目标值，建议下一次实验采用：
（请给出每个参数的具体数值，并解释为什么这样设置能帮助达成目标）

### 四、预期效果评估
1. 按照建议调整后，各指标预计可以达到多少？
2. 距离目标还有多少差距？
3. 是否需要多轮迭代？"""

        model = genai.GenerativeModel('gemini-2.0-flash', system_instruction=system_prompt)
        response = model.generate_content(user_prompt)
        
        full_response = response.text
        
        # 解析响应
        analysis_part = ""
        suggestion_part = ""
        
        if "### 三" in full_response:
            parts = full_response.split("### 三")
            analysis_part = parts[0].strip()
            suggestion_part = "### 三" + parts[1] if len(parts) > 1 else ""
        else:
            analysis_part = full_response
        
        return {
            'success': True,
            'analysis': analysis_part,
            'suggestions': suggestion_part,
            'full_response': full_response
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# ============================================================
# View B: 数据工作台 (Data Studio)
# ============================================================

def render_data_studio():
    """渲染数据工作台视图"""
    
    # Header
    st.markdown("""
    <div class="app-header">
        <div class="app-title"><span>NEXUS</span> 数据工作台</div>
        <div class="app-subtitle">Data Studio - 实验数据管理与目标设定</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== 区域一：实验背景 ==========
    st.markdown('<div class="area-title"><span class="area-number">1.</span> 实验背景</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        material_name = st.text_input(
            "材料/项目名称",
            value=st.session_state.get('material_name', ''),
            placeholder="例如：碳化硅 SiC、GaN 外延片、钙钛矿太阳能电池"
        )
    
    with col2:
        equipment_name = st.text_input(
            "实验设备/工艺",
            value=st.session_state.get('equipment_name', ''),
            placeholder="例如：PVT 长晶炉、MOCVD、磁控溅射"
        )
    
    # 分隔线
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # ========== 区域二：数据全量编辑 ==========
    st.markdown('<div class="area-title"><span class="area-number">2.</span> 实验数据录入</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="hint-box">
        您可以在下方直接编辑数据、添加/删除行、修改数值。支持从 Excel 复制粘贴整列数据。
    </div>
    """, unsafe_allow_html=True)
    
    # 数据编辑器
    edited_df = st.data_editor(
        st.session_state['experiment_data'],
        num_rows="dynamic",
        use_container_width=True,
        height=350,
        key="data_studio_editor"
    )
    
    # 添加列功能
    with st.expander("添加新列"):
        col_add1, col_add2, col_add3 = st.columns([2, 1, 1])
        with col_add1:
            new_col_name = st.text_input("新列名称", placeholder="例如：催化剂浓度", key="new_col_name")
        with col_add2:
            new_col_default = st.number_input("默认值", value=0.0, key="new_col_default")
        with col_add3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("添加列", key="add_col_btn"):
                if new_col_name and new_col_name not in edited_df.columns:
                    edited_df[new_col_name] = new_col_default
                    st.session_state['experiment_data'] = edited_df
                    st.rerun()
    
    # 分隔线
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # ========== 区域三：列映射与目标设定 ==========
    st.markdown('<div class="area-title"><span class="area-number">3.</span> 列映射与目标设定</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="mapping-info">
        <strong>第一步：</strong>告诉 AI 哪些列是"可调参数"，哪些列是"实验结果"。<br>
        <strong>第二步：</strong>为每个结果列设定您想达成的目标值。
    </div>
    """, unsafe_allow_html=True)
    
    all_columns = edited_df.columns.tolist()
    
    # 第一步：选择参数列和结果列
    col_map1, col_map2 = st.columns(2)
    
    with col_map1:
        input_columns = st.multiselect(
            "选择实验参数列 (Inputs)",
            options=all_columns,
            default=[c for c in st.session_state.get('input_columns', []) if c in all_columns],
            help="这些是您在实验中可以控制的变量"
        )
    
    with col_map2:
        available_outputs = [c for c in all_columns if c not in input_columns]
        output_columns = st.multiselect(
            "选择实验结果列 (Outputs)",
            options=available_outputs,
            default=[c for c in st.session_state.get('output_columns', []) if c in available_outputs],
            help="这些是您想要优化的目标指标"
        )
    
    # 显示映射预览
    if input_columns or output_columns:
        preview_html = ""
        if input_columns:
            preview_html += "参数: " + " ".join([f'<span class="mapping-tag input">{c}</span>' for c in input_columns])
        if output_columns:
            preview_html += " → 结果: " + " ".join([f'<span class="mapping-tag output">{c}</span>' for c in output_columns])
        st.markdown(preview_html, unsafe_allow_html=True)
    
    # 第二步：为每个 Output 列设定目标值
    target_values = {}
    
    if output_columns:
        st.markdown("""
        <div class="target-section">
            <div class="target-section-title">设定各指标的目标值</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 根据 output 列数量动态生成输入框
        num_outputs = len(output_columns)
        cols_per_row = min(num_outputs, 3)  # 每行最多3个
        
        # 分行显示
        for i in range(0, num_outputs, cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(output_columns[i:i+cols_per_row]):
                with cols[j]:
                    # 获取当前值作为参考
                    if col_name in edited_df.columns:
                        current_avg = pd.to_numeric(edited_df[col_name], errors='coerce').mean()
                        current_max = pd.to_numeric(edited_df[col_name], errors='coerce').max()
                    else:
                        current_avg = 0
                        current_max = 0
                    
                    # 获取之前保存的目标值
                    saved_target = st.session_state.get('target_values', {}).get(col_name, '')
                    
                    target_val = st.text_input(
                        f"【{col_name}】目标值",
                        value=str(saved_target) if saved_target else '',
                        placeholder=f"当前均值: {current_avg:.2f}",
                        help=f"当前均值: {current_avg:.2f}，最优: {current_max:.2f}",
                        key=f"target_{col_name}"
                    )
                    
                    target_values[col_name] = target_val
                    
                    # 显示当前值参考
                    st.caption(f"当前: 均值 {current_avg:.2f} / 最优 {current_max:.2f}")
    
    # 分隔线
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # ========== 保存并返回 ==========
    col_save1, col_save2, col_save3 = st.columns([1, 2, 1])
    
    with col_save2:
        if st.button("保存并返回仪表盘", use_container_width=True, type="primary"):
            # 保存所有数据到 session state
            st.session_state['material_name'] = material_name
            st.session_state['equipment_name'] = equipment_name
            st.session_state['experiment_data'] = edited_df
            st.session_state['input_columns'] = input_columns
            st.session_state['output_columns'] = output_columns
            st.session_state['target_values'] = target_values
            st.session_state['current_view'] = 'dashboard'
            st.rerun()


# ============================================================
# View A: 智能仪表盘 (Dashboard)
# ============================================================

def render_dashboard():
    """渲染智能仪表盘视图"""
    
    # Header
    st.markdown("""
    <div class="app-header">
        <div class="app-title"><span>NEXUS</span> 智能仪表盘</div>
        <div class="app-subtitle">AI-Powered Materials Research Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 项目信息卡片
    material = st.session_state.get('material_name', '')
    equipment = st.session_state.get('equipment_name', '')
    target_values = st.session_state.get('target_values', {})
    output_cols = st.session_state.get('output_columns', [])
    df = st.session_state['experiment_data']
    
    if material or equipment:
        st.markdown(f"""
        <div class="project-card">
            <div style="display: flex; gap: 3rem; flex-wrap: wrap;">
                <div>
                    <div class="project-label">研究项目</div>
                    <div class="project-value">{material if material else '—'}</div>
                </div>
                <div>
                    <div class="project-label">设备/工艺</div>
                    <div class="project-value">{equipment if equipment else '—'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 显示目标值卡片
    if target_values and any(target_values.values()):
        st.markdown("**量化目标**")
        target_cols = st.columns(len([v for v in target_values.values() if v]))
        col_idx = 0
        for col_name, target_val in target_values.items():
            if target_val and col_name in df.columns:
                current_avg = pd.to_numeric(df[col_name], errors='coerce').mean()
                with target_cols[col_idx]:
                    st.markdown(f"""
                    <div class="target-card">
                        <div class="target-label">{col_name}</div>
                        <div class="target-value">目标: {target_val}</div>
                        <div class="current-value">当前均值: {current_avg:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                col_idx += 1
    
    # 操作按钮行
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("编辑/导入数据", use_container_width=True):
            st.session_state['current_view'] = 'data_studio'
            st.rerun()
    
    with col_btn2:
        analyze_btn = st.button("开始 AI 深度分析", use_container_width=True, type="primary")
    
    with col_btn3:
        api_key_input = st.text_input(
            "API Key",
            value=st.session_state.get('api_key', ''),
            type="password",
            placeholder="输入 Gemini API Key",
            label_visibility="collapsed"
        )
        st.session_state['api_key'] = api_key_input
    
    # 处理 AI 分析
    if analyze_btn:
        api_key = st.session_state.get('api_key', '')
        if not api_key:
            st.warning("请输入 Gemini API Key")
        elif df.empty:
            st.warning("请先添加实验数据")
        else:
            with st.spinner("AI 正在分析目标差距并生成优化建议..."):
                result = analyze_with_ai(
                    df,
                    st.session_state.get('material_name', ''),
                    st.session_state.get('equipment_name', ''),
                    st.session_state.get('input_columns', []),
                    st.session_state.get('output_columns', []),
                    st.session_state.get('target_values', {}),
                    api_key
                )
            st.session_state['ai_result'] = result
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # 数据摘要
    input_cols = st.session_state.get('input_columns', [])
    
    st.markdown(f"""
    <div class="data-summary">
        <span class="summary-item">
            <span class="summary-label">实验次数</span><br>
            <span class="summary-value">{len(df)}</span>
        </span>
        <span class="summary-item">
            <span class="summary-label">参数列</span><br>
            <span class="summary-value">{len(input_cols)}</span>
        </span>
        <span class="summary-item">
            <span class="summary-label">结果列</span><br>
            <span class="summary-value">{len(output_cols)}</span>
        </span>
        <span class="summary-item">
            <span class="summary-label">已设目标</span><br>
            <span class="summary-value">{len([v for v in target_values.values() if v])}</span>
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # 数据表格和图表
    col_table, col_chart = st.columns([1, 1])
    
    with col_table:
        st.markdown("**实验数据预览**")
        st.dataframe(df, use_container_width=True, height=280)
    
    with col_chart:
        st.markdown("**结果趋势与目标**")
        fig = create_trend_chart(df, output_cols, target_values)
        st.plotly_chart(fig, use_container_width=True)
    
    # AI 分析结果
    ai_result = st.session_state.get('ai_result')
    
    if ai_result is not None:
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        if ai_result.get('success'):
            col_insight, col_action = st.columns([1, 1])
            
            with col_insight:
                st.markdown("""
                <div class="insight-card">
                    <div class="insight-title">目标差距诊断与机理分析</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(ai_result.get('analysis', ''))
            
            with col_action:
                st.markdown("""
                <div class="action-card">
                    <div class="action-title">精准参数建议与预期效果</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(ai_result.get('suggestions', ai_result.get('full_response', '')))
            
            with st.expander("查看完整 AI 报告"):
                st.markdown(ai_result.get('full_response', ''))
        else:
            st.error(f"AI 分析失败: {ai_result.get('error', '未知错误')}")
    else:
        st.markdown("""
        <div class="placeholder-box">
            设定目标值后，点击「开始 AI 深度分析」，获取针对性的优化建议
        </div>
        """, unsafe_allow_html=True)
    
    # 页脚
    st.markdown("""
    <div class="footer">
        NEXUS 材料研发平台 · Quantitative Target Optimization · Powered by Gemini AI
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 主程序
# ============================================================

def main():
    """主程序入口"""
    init_session_state()
    
    if st.session_state['current_view'] == 'data_studio':
        render_data_studio()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()

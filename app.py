"""
NEXUS Lab — 双模式材料研发平台
Dual-Mode (Guest / Admin) Materials R&D Platform

Guest : 本地数据操作（内存），无法同步云端
Admin : 拥有所有 Guest 功能 + Google Sheets 读取/保存

依赖：streamlit, pandas, numpy, plotly, google-generativeai, Pillow, streamlit-gsheets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
from PIL import Image
import io

# Google Sheets 连接（可选依赖，缺失时优雅降级）
try:
    from streamlit_gsheets import GSheetsConnection
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="NEXUS Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS — 极简白 SaaS 风格
# ============================================================
st.markdown("""
<style>
    /* === 隐藏 Streamlit 默认 Hamburger 菜单 & Footer === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* === 全局 === */
    .stApp {background-color: #FFFFFF;}
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
                     'Hiragino Sans GB', 'Microsoft YaHei', sans-serif !important;
        color: #333;
    }

    /* === Navbar === */
    .navbar-logo {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1a1a1a;
        letter-spacing: 0.5px;
        line-height: 2.4rem;
    }
    .navbar-logo .accent {color: #2563eb;}
    .navbar-badge {
        display: inline-block;
        font-size: 0.6rem;
        padding: 0.12rem 0.45rem;
        border-radius: 99px;
        margin-left: 0.5rem;
        font-weight: 600;
        vertical-align: middle;
    }
    .badge-guest  {background: #f1f5f9; color: #64748b;}
    .badge-admin  {background: #dbeafe; color: #1d4ed8;}

    /* === 区域标题 === */
    .area-title {
        font-size: 1.05rem; font-weight: 600; color: #333;
        margin-bottom: 0.8rem; padding-bottom: 0.4rem;
        border-bottom: 2px solid #2563eb; display: inline-block;
    }
    .area-number {color: #2563eb; font-weight: 700;}

    /* === 分隔线 === */
    .section-divider {border: none; border-top: 1px solid #e8e8e8; margin: 1.5rem 0;}

    /* === 项目信息卡片 === */
    .project-card {
        background: linear-gradient(135deg, #f8faff 0%, #f0f5ff 100%);
        border: 1px solid #d0e0f5; border-radius: 10px;
        padding: 1.1rem 1.4rem; margin-bottom: 1.2rem;
    }
    .project-label {font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 1px;}
    .project-value {font-size: 1rem; font-weight: 600; color: #333; margin-top: 0.15rem;}

    /* === 目标卡片 === */
    .target-card {
        background: #f0fdf4; border: 1px solid #86efac;
        border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.5rem;
    }
    .target-label  {font-size: 0.75rem; color: #166534; font-weight: 600;}
    .target-value  {font-size: 1.1rem;  font-weight: 700; color: #15803d;}
    .current-value {font-size: 0.8rem;  color: #666;}

    /* === 数据摘要 === */
    .data-summary {
        background: #f8f9fa; border: 1px solid #e0e0e0;
        border-radius: 8px; padding: 0.9rem 1.2rem; margin-bottom: 1rem;
    }
    .summary-item  {display: inline-block; margin-right: 2rem;}
    .summary-label {font-size: 0.7rem; color: #888; text-transform: uppercase;}
    .summary-value {font-size: 1.2rem; font-weight: 700; color: #333;}

    /* === AI 分析卡片 === */
    .insight-card {
        background: linear-gradient(135deg, #fafbff, #f5f8ff);
        border: 1px solid #d0e0f5; border-left: 4px solid #2563eb;
        border-radius: 8px; padding: 1.4rem; margin-bottom: 1rem;
    }
    .insight-title {
        font-size: 0.85rem; font-weight: 700; color: #2563eb;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem;
    }
    .action-card {
        background: linear-gradient(135deg, #f8fffe, #f0fdf9);
        border: 1px solid #a7e8d8; border-left: 4px solid #10b981;
        border-radius: 8px; padding: 1.4rem; margin-bottom: 1rem;
    }
    .action-title {
        font-size: 0.85rem; font-weight: 700; color: #10b981;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem;
    }

    /* === 映射标签 === */
    .mapping-info {
        background: #fff8f0; border: 1px solid #ffd6a5;
        border-radius: 6px; padding: 0.7rem 1rem;
        font-size: 0.85rem; color: #666; margin-bottom: 1rem;
    }
    .mapping-tag {
        display: inline-block; border-radius: 4px;
        padding: 0.15rem 0.45rem; font-size: 0.8rem; margin: 0.15rem;
    }
    .mapping-tag.input  {background: #dbeafe; color: #1d4ed8;}
    .mapping-tag.output {background: #fff0e6; color: #c2410c;}

    /* === 目标设定区域 === */
    .target-section {
        background: #fefce8; border: 1px solid #fef08a;
        border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 0.8rem;
    }
    .target-section-title {font-size: 0.85rem; font-weight: 600; color: #854d0e; margin-bottom: 0.6rem;}

    /* === 按钮 === */
    .stButton > button {border-radius: 8px; font-weight: 600; transition: all 0.2s ease;}

    /* === 提示框 === */
    .hint-box {
        background: #f0f7ff; border: 1px solid #bfdbfe;
        border-radius: 6px; padding: 0.7rem 1rem;
        font-size: 0.85rem; color: #1e40af; margin-bottom: 0.8rem;
    }

    /* === 占位符 === */
    .placeholder-box {
        background: #fafbfc; border: 1px dashed #d0d0d0;
        border-radius: 8px; padding: 2.5rem; text-align: center; color: #999;
    }

    /* === 页脚 === */
    .app-footer {
        text-align: center; color: #aaa; font-size: 0.8rem;
        padding: 1.5rem 0; border-top: 1px solid #e8e8e8; margin-top: 1.5rem;
    }

    /* === 间距 === */
    .block-container {padding: 1rem 2.5rem 2rem 2.5rem;}

    /* === 输入框 === */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; color: #333;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37,99,235,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State 初始化
# ============================================================
def init_session_state():
    """首次运行时初始化所有状态；后续刷新保持不变。"""
    defaults = {
        # 身份
        "user_role": "guest",           # "guest" | "admin"
        # 项目信息
        "material_name": "",
        "equipment_name": "",
        # 核心数据
        "df": pd.DataFrame({
            "温度(°C)":      [1800, 1850, 1900, 1950, 2000],
            "压力(mbar)":    [50,   55,   60,   65,   70],
            "Ar流量(sccm)":  [100,  100,  120,  120,  150],
            "生长时间(h)":   [24,   24,   30,   30,   36],
            "生长速率(um/h)": [80,   95,   110,  105,  98],
            "微管密度(cm-2)": [5.2,  4.1,  2.8,  3.5,  4.0],
        }),
        # 语义映射
        "input_columns":  [],
        "output_columns": [],
        "target_values":  {},           # {col_name: target_string}
        # 样品图片
        "sample_image":      None,      # bytes
        "sample_image_name": None,
        # AI
        "ai_result": None,
        "api_key":   "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ============================================================
# 工具函数
# ============================================================
def _clear_editor_widget():
    """删除 data_editor 的 widget state，防止外部修改 df 后 key 冲突。"""
    if "editor" in st.session_state:
        del st.session_state["editor"]


def style_dataframe(df: pd.DataFrame, input_cols: list, output_cols: list):
    """Pandas Styler: Input 列 → 浅蓝 #e6f3ff, Output 列 → 浅橙 #fff0e6."""
    def _color(col: pd.Series) -> list[str]:
        if col.name in input_cols:
            return ["background-color: #e6f3ff"] * len(col)
        if col.name in output_cols:
            return ["background-color: #fff0e6"] * len(col)
        return [""] * len(col)
    return df.style.apply(_color, axis=0)


def create_trend_chart(
    df: pd.DataFrame, output_cols: list, target_values: dict
) -> go.Figure:
    """创建实验结果趋势图（含目标虚线）。"""
    fig = go.Figure()

    if not output_cols or df.empty:
        fig.add_annotation(
            text="请在数据工作台选择 Output 列以显示趋势图",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#999"),
        )
        fig.update_layout(height=300)
        return fig

    x = list(range(1, len(df) + 1))
    palette = ["#2563eb", "#10b981", "#f59e0b", "#ef4444"]

    for i, col in enumerate(output_cols[:4]):
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        c = palette[i % len(palette)]

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=col,
            line=dict(color=c, width=2), marker=dict(size=7),
        ))

        tv = target_values.get(col, "")
        if tv:
            try:
                fig.add_hline(
                    y=float(tv), line_dash="dash", line_color=c,
                    annotation_text=f"目标: {tv}",
                    annotation_position="right",
                    annotation_font_color=c,
                )
            except (ValueError, TypeError):
                pass

    fig.update_layout(
        template="simple_white",
        title=dict(text="结果趋势（虚线 = 目标值）", font=dict(size=14)),
        xaxis_title="实验编号", yaxis_title="数值",
        height=320, margin=dict(t=50, b=40, l=50, r=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#f0f0f0")
    fig.update_yaxes(gridcolor="#f0f0f0")
    return fig


# ============================================================
# AI 分析（Gemini）
# ============================================================
def analyze_with_ai(
    df: pd.DataFrame,
    material: str,
    equipment: str,
    input_cols: list,
    output_cols: list,
    target_values: dict,
    api_key: str,
    image_bytes: bytes = None,
) -> dict:
    """调用 Gemini AI 进行目标感知分析（支持图像）。"""
    try:
        genai.configure(api_key=api_key)
        csv_str = df.to_csv(index=False)

        # ---- 构建量化目标描述 ----
        t_lines: list[str] = []
        for col in output_cols:
            if col not in df.columns:
                continue
            avg = pd.to_numeric(df[col], errors="coerce").mean()
            best = pd.to_numeric(df[col], errors="coerce").max()
            tv = target_values.get(col, "")
            if tv:
                gap = float(tv) - avg
                pct = (gap / avg * 100) if avg != 0 else 0
                t_lines.append(
                    f"- {col}：目标值 = {tv}，当前均值 = {avg:.2f}，"
                    f"最优 = {best:.2f}，差距 = {gap:.2f} ({pct:+.1f}%)"
                )
            else:
                t_lines.append(
                    f"- {col}：未设定目标，当前均值 = {avg:.2f}，最优 = {best:.2f}"
                )
        target_str = "\n".join(t_lines) if t_lines else "（用户未设定具体目标）"

        # ---- System Prompt ----
        has_image = image_bytes is not None
        img_instr = ""
        if has_image:
            img_instr = (
                "\n5. 仔细观察用户上传的样品微观结构图（如 SEM / 光学显微镜图像）"
                "\n6. 分析图像中的形貌特征（晶粒大小、裂纹、孔隙、颜色异常等）"
                "\n7. 将图像观察与实验参数关联，推断工艺-形貌-性能的因果关系"
            )

        system_prompt = (
            f"你是一位世界顶级的材料科学家和工艺工程师。\n"
            f"用户正在进行【{material or '材料'}】的研究。\n"
            f"使用的设备/工艺是：【{equipment or '实验设备'}】。\n\n"
            f"你的任务是帮助用户达成他们设定的**量化目标**。\n"
            f"你的分析必须：\n"
            f"1. 精确指出当前数据与目标值的差距\n"
            f"2. 结合物理/化学原理解释瓶颈\n"
            f"3. 给出能够逼近目标值的具体参数建议\n"
            f"4. 如果目标不切实际，诚实指出{img_instr}"
        )

        # ---- User Prompt ----
        in_str = ", ".join(input_cols) if input_cols else "（用户未指定）"

        if has_image:
            user_prompt = (
                f"## 实验数据\n```csv\n{csv_str}\n```\n\n"
                f"## 数据列说明\n- **实验参数列 (可调变量)**：{in_str}\n\n"
                f"## 用户的量化目标\n{target_str}\n\n"
                f"## 样品图像\n用户上传了一张样品的微观结构图。请仔细观察图像中的形貌特征。\n\n---\n\n"
                f"请按以下结构分析：\n\n"
                f"### 一、图像形貌分析\n观察上传的样品图像：\n"
                f"1. 描述主要形貌特征（晶粒、表面、缺陷等）\n"
                f"2. 是否存在裂纹、孔隙、颜色不均匀等异常？\n"
                f"3. 这些形貌特征对应的可能原因？\n\n"
                f"### 二、数据-图像关联分析\n"
                f"1. 图像异常是否对应特定参数区间？\n"
                f"2. 哪些参数最可能影响微观结构？\n"
                f"3. 当前距离目标还有多大差距？\n\n"
                f"### 三、瓶颈机理分析\n"
                f"结合【{material}】的物理/化学原理分析瓶颈。\n\n"
                f"### 四、精准参数建议\n"
                f"给出每个参数的具体数值，解释如何改善形貌。\n\n"
                f"### 五、预期效果评估\n"
                f"1. 微观结构预计如何改善？\n2. 各指标预计可达到多少？"
            )
        else:
            user_prompt = (
                f"## 实验数据\n```csv\n{csv_str}\n```\n\n"
                f"## 数据列说明\n- **实验参数列 (可调变量)**：{in_str}\n\n"
                f"## 用户的量化目标\n{target_str}\n\n---\n\n"
                f"请按以下结构分析：\n\n"
                f"### 一、目标差距诊断\n"
                f"1. 当前距离目标还有多大差距？\n"
                f"2. 哪些参数组合表现最好？\n"
                f"3. 是否存在参数间的权衡关系？\n\n"
                f"### 二、瓶颈机理分析\n"
                f"结合【{material}】的物理/化学原理分析瓶颈。\n\n"
                f"### 三、精准参数建议\n"
                f"给出每个参数的具体数值，解释为什么这样设置能帮助达成目标。\n\n"
                f"### 四、预期效果评估\n"
                f"1. 各指标预计可达到多少？\n2. 距离目标还有多少差距？\n"
                f"3. 是否需要多轮迭代？"
            )

        # ---- 调用模型 ----
        model = genai.GenerativeModel(
            "gemini-2.0-flash", system_instruction=system_prompt
        )
        if has_image:
            img = Image.open(io.BytesIO(image_bytes))
            response = model.generate_content([user_prompt, img])
        else:
            response = model.generate_content(user_prompt)

        full = response.text

        # ---- 拆分 analysis / suggestions ----
        split_mk = "### 四" if has_image else "### 三"
        if split_mk in full:
            parts = full.split(split_mk, 1)
            analysis = parts[0].strip()
            suggestion = split_mk + parts[1]
        else:
            analysis, suggestion = full, ""

        return {
            "success": True,
            "analysis": analysis,
            "suggestions": suggestion,
            "full_response": full,
            "has_image": has_image,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# 顶部导航栏 (Navbar)
# ============================================================
def render_navbar():
    """Logo + Popover 用户中心（登录/登出）。"""
    role = st.session_state.get("user_role", "guest")
    is_admin = role == "admin"

    badge_html = (
        '<span class="navbar-badge badge-admin">Admin</span>'
        if is_admin
        else '<span class="navbar-badge badge-guest">Guest</span>'
    )

    nav_left, _, nav_right = st.columns([5, 3, 1.5])

    with nav_left:
        st.markdown(
            f'<div class="navbar-logo">'
            f'🧪 <span class="accent">NEXUS</span> Lab {badge_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with nav_right:
        popover_label = "👨‍🔬 Admin" if is_admin else "👤 Guest"
        with st.popover(popover_label, use_container_width=True):
            if is_admin:
                st.markdown("✅ 已登录为 **Admin**")
                st.caption("拥有 Google Sheets 云端同步权限")
                if st.button("退出登录", key="logout_btn", use_container_width=True):
                    st.session_state["user_role"] = "guest"
                    st.rerun()
            else:
                st.markdown("**登录为 Admin**")
                st.caption("解锁 Google Sheets 云端读取 / 保存功能")
                pwd = st.text_input(
                    "密码", type="password", key="login_pwd",
                    placeholder="输入管理密码",
                )
                if st.button("登录", key="login_btn", use_container_width=True):
                    try:
                        correct = st.secrets["general"]["password"]
                        if pwd == correct:
                            st.session_state["user_role"] = "admin"
                            st.rerun()
                        else:
                            st.error("密码错误")
                    except (KeyError, FileNotFoundError):
                        st.error(
                            "未配置管理密码。请在 `.streamlit/secrets.toml` 中添加：\n\n"
                            '```\n[general]\npassword = "your_password"\n```'
                        )

    # Navbar 底部分割线
    st.markdown(
        '<hr style="margin:0 0 0.6rem 0; border:none; border-top:1px solid #e8e8e8;">',
        unsafe_allow_html=True,
    )


# ============================================================
# 侧边栏 — 数据存取 (所有人可用 + Admin 专属)
# ============================================================
def render_sidebar():
    """Sidebar: CSV 上传/下载（全部可用）+ Google Sheets（仅 Admin）。"""
    is_admin = st.session_state.get("user_role") == "admin"
    df = st.session_state["df"]

    with st.sidebar:
        st.markdown("### 📁 数据管理")
        st.caption(f"当前数据：{len(df)} 行 × {len(df.columns)} 列")

        # ---- 📥 下载 CSV 备份 (所有人) ----
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 下载 CSV 备份", csv_bytes, "nexus_backup.csv",
            "text/csv", use_container_width=True,
        )

        # ---- 📂 上传 CSV 恢复 (所有人) ----
        st.markdown("---")
        uploaded = st.file_uploader(
            "📂 上传 CSV 恢复", type=["csv"], key="csv_uploader",
        )
        if uploaded is not None:
            try:
                preview_df = pd.read_csv(uploaded)
                st.info(f"检测到 {len(preview_df)} 行 × {len(preview_df.columns)} 列")
                if st.button("✅ 确认导入此文件", key="confirm_csv_import"):
                    st.session_state["df"] = preview_df
                    # 重置映射（列可能完全不同了）
                    st.session_state["input_columns"] = []
                    st.session_state["output_columns"] = []
                    st.session_state["target_values"] = {}
                    _clear_editor_widget()
                    st.rerun()
            except Exception as e:
                st.error(f"CSV 解析失败: {e}")

        # ---- ☁️ Google Sheets 云端同步 (仅 Admin) ----
        if is_admin:
            st.markdown("---")
            st.markdown("### ☁️ 云端同步")

            if not GSHEETS_AVAILABLE:
                st.warning(
                    "未安装 `streamlit-gsheets`。\n\n"
                    "运行 `pip install streamlit-gsheets` 后重启。"
                )
            else:
                # 加载
                if st.button(
                    "☁️ 从 Google Sheets 加载",
                    use_container_width=True, key="gs_load",
                ):
                    try:
                        conn = st.connection("gsheets", type=GSheetsConnection)
                        cloud_df = conn.read(worksheet="Sheet1", ttl=0)
                        cloud_df = cloud_df.dropna(how="all")
                        if cloud_df.empty:
                            st.warning("Sheet1 为空或无法读取。")
                        else:
                            st.session_state["df"] = cloud_df
                            st.session_state["input_columns"] = []
                            st.session_state["output_columns"] = []
                            st.session_state["target_values"] = {}
                            _clear_editor_widget()
                            st.success(f"已加载 {len(cloud_df)} 行 × {len(cloud_df.columns)} 列")
                            st.rerun()
                    except Exception as e:
                        st.error(f"加载失败: {e}")

                # 保存
                if st.button(
                    "💾 保存到 Google Sheets",
                    use_container_width=True, key="gs_save",
                ):
                    try:
                        conn = st.connection("gsheets", type=GSheetsConnection)
                        conn.update(worksheet="Sheet1", data=df)
                        st.success("✓ 已保存到 Google Sheets")
                    except Exception as e:
                        st.error(f"保存失败: {e}")


# ============================================================
# Tab 1: 数据工作台 (Data Studio)
# ============================================================
def render_data_studio():
    """列管理 → 数据编辑 → 语义映射 → 目标设定。"""
    df = st.session_state["df"]

    # ========== 1. 实验背景 ==========
    st.markdown(
        '<div class="area-title"><span class="area-number">1.</span> 实验背景</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        mat = st.text_input(
            "材料 / 项目名称",
            value=st.session_state["material_name"],
            placeholder="例如：碳化硅 SiC、GaN 外延片、钙钛矿太阳能电池",
            key="ds_material",
        )
    with c2:
        eqp = st.text_input(
            "实验设备 / 工艺",
            value=st.session_state["equipment_name"],
            placeholder="例如：PVT 长晶炉、MOCVD、磁控溅射",
            key="ds_equipment",
        )
    st.session_state["material_name"] = mat
    st.session_state["equipment_name"] = eqp

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ========== 2. 列管理 + 数据编辑 ==========
    st.markdown(
        '<div class="area-title"><span class="area-number">2.</span> 列管理与数据编辑</div>',
        unsafe_allow_html=True,
    )

    # ---- 列管理 Expander ----
    with st.expander("🛠️ 列管理 (修改列名 / 删除列)", expanded=False):
        cols_list = df.columns.tolist()

        # -- 功能 A: 重命名列 --
        st.markdown("**重命名列**")
        rc1, rc2, rc3 = st.columns([2, 2, 1])
        with rc1:
            old_name = st.selectbox("选择列", cols_list, key="rename_select")
        with rc2:
            new_name_input = st.text_input(
                "新列名", key="rename_input", placeholder="输入新名称",
            )
        with rc3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("确认重命名", key="rename_btn"):
                nn = (new_name_input or "").strip()
                if nn and nn != old_name:
                    st.session_state["df"] = df.rename(columns={old_name: nn})
                    # 同步映射
                    st.session_state["input_columns"] = [
                        nn if c == old_name else c
                        for c in st.session_state["input_columns"]
                    ]
                    st.session_state["output_columns"] = [
                        nn if c == old_name else c
                        for c in st.session_state["output_columns"]
                    ]
                    tv = st.session_state["target_values"]
                    if old_name in tv:
                        tv[nn] = tv.pop(old_name)
                    _clear_editor_widget()
                    st.rerun()
                elif nn == old_name:
                    st.warning("新旧列名相同，无需修改。")
                else:
                    st.warning("请输入有效的新列名。")

        st.markdown("---")

        # -- 功能 B: 删除列 --
        st.markdown("**删除列**")
        del_cols = st.multiselect(
            "选择要删除的列（可多选）", cols_list, key="del_cols_select",
        )
        if del_cols:
            if st.button("🗑️ 确认删除所选列", key="del_cols_btn", type="primary"):
                st.session_state["df"] = df.drop(columns=del_cols, errors="ignore")
                st.session_state["input_columns"] = [
                    c for c in st.session_state["input_columns"] if c not in del_cols
                ]
                st.session_state["output_columns"] = [
                    c for c in st.session_state["output_columns"] if c not in del_cols
                ]
                for c in del_cols:
                    st.session_state["target_values"].pop(c, None)
                _clear_editor_widget()
                st.rerun()

    # ---- 全功能数据编辑器 ----
    st.markdown(
        '<div class="hint-box">'
        '直接编辑下方表格：增删行、修改数值、从 Excel 复制粘贴均可。'
        '</div>',
        unsafe_allow_html=True,
    )

    edited_df = st.data_editor(
        st.session_state["df"],
        num_rows="dynamic",
        use_container_width=True,
        height=360,
        key="editor",
    )
    # 实时同步回 session_state
    st.session_state["df"] = edited_df

    # ---- 添加列 + 图片上传 ----
    exp1, exp2 = st.columns(2)
    with exp1:
        with st.expander("➕ 添加新列"):
            ac1, ac2, ac3 = st.columns([2, 1, 1])
            with ac1:
                ncn = st.text_input(
                    "列名", key="new_col_name", placeholder="例如：催化剂浓度",
                )
            with ac2:
                ncv = st.number_input("默认值", value=0.0, key="new_col_val")
            with ac3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("添加", key="add_col_btn"):
                    name = (ncn or "").strip()
                    if name and name not in edited_df.columns:
                        new = edited_df.copy()
                        new[name] = ncv
                        st.session_state["df"] = new
                        _clear_editor_widget()
                        st.rerun()
                    elif not name:
                        st.warning("请输入列名。")
                    else:
                        st.warning("该列名已存在。")

    with exp2:
        with st.expander("📷 样品图片（可选）"):
            st.caption("上传 SEM / 光学显微镜图片，AI 将结合图像形貌分析")
            up_img = st.file_uploader(
                "上传图片", type=["png", "jpg", "jpeg"],
                key="img_uploader", label_visibility="collapsed",
            )
            if up_img is not None:
                img = Image.open(up_img)
                st.image(img, caption=f"已上传: {up_img.name}", use_container_width=True)
                st.session_state["sample_image"] = up_img.getvalue()
                st.session_state["sample_image_name"] = up_img.name
            elif st.session_state.get("sample_image"):
                img = Image.open(io.BytesIO(st.session_state["sample_image"]))
                st.image(
                    img,
                    caption=f"已保存: {st.session_state.get('sample_image_name', '')}",
                    use_container_width=True,
                )
                if st.button("移除图片", key="rm_img_btn"):
                    st.session_state["sample_image"] = None
                    st.session_state["sample_image_name"] = None
                    st.rerun()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ========== 3. 语义映射与目标设定 ==========
    st.markdown(
        '<div class="area-title">'
        '<span class="area-number">3.</span> 语义映射与目标设定'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="mapping-info">
        <strong>第一步：</strong>选择参数列 (Inputs) 和结果列 (Outputs)。
        <strong>第二步：</strong>为结果列设定量化目标值。
    </div>
    """, unsafe_allow_html=True)

    all_cols = edited_df.columns.tolist()

    mc1, mc2 = st.columns(2)
    with mc1:
        inp = st.multiselect(
            "Inputs (参数列) — 蓝色标记", all_cols,
            default=[c for c in st.session_state["input_columns"] if c in all_cols],
            help="实验中可以控制的变量",
            key="sel_inputs",
        )
    with mc2:
        avail_out = [c for c in all_cols if c not in inp]
        out = st.multiselect(
            "Outputs (结果列) — 橙色标记", avail_out,
            default=[c for c in st.session_state["output_columns"] if c in avail_out],
            help="想要优化的目标指标",
            key="sel_outputs",
        )

    st.session_state["input_columns"] = inp
    st.session_state["output_columns"] = out

    # 映射标签预览
    if inp or out:
        tag_html = ""
        if inp:
            tag_html += "Inputs: " + " ".join(
                f'<span class="mapping-tag input">{c}</span>' for c in inp
            )
        if out:
            tag_html += " → Outputs: " + " ".join(
                f'<span class="mapping-tag output">{c}</span>' for c in out
            )
        st.markdown(tag_html, unsafe_allow_html=True)
        # 可视化提示
        st.info(
            f"已将 [{', '.join(inp) or '无'}] 标记为 🔵 蓝色，"
            f"[{', '.join(out) or '无'}] 标记为 🟠 橙色。"
            f"切换到「智能仪表盘」标签页查看彩色表格效果。"
        )

    # ---- 动态目标设定 ----
    tvs = dict(st.session_state.get("target_values", {}))

    if out:
        st.markdown(
            '<div class="target-section">'
            '<div class="target-section-title">设定各指标的目标值</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        per_row = min(len(out), 3)
        for i in range(0, len(out), per_row):
            cols = st.columns(per_row)
            for j, cn in enumerate(out[i : i + per_row]):
                with cols[j]:
                    if cn in edited_df.columns:
                        avg = pd.to_numeric(edited_df[cn], errors="coerce").mean()
                        mx  = pd.to_numeric(edited_df[cn], errors="coerce").max()
                    else:
                        avg, mx = 0.0, 0.0

                    saved = tvs.get(cn, "")
                    val = st.text_input(
                        f"【{cn}】目标值",
                        value=str(saved) if saved else "",
                        placeholder=f"均值 {avg:.2f}",
                        help=f"当前均值: {avg:.2f}，最优: {mx:.2f}",
                        key=f"tgt_{cn}",
                    )
                    tvs[cn] = val
                    st.caption(f"均值 {avg:.2f} / 最优 {mx:.2f}")

    # 只保留当前 output 列的目标
    st.session_state["target_values"] = {k: v for k, v in tvs.items() if k in out}


# ============================================================
# Tab 2: 智能仪表盘 (Dashboard)
# ============================================================
def render_dashboard():
    """数据概览 + 彩色表格 + 趋势图 + AI 深度分析。"""
    df   = st.session_state["df"]
    inp  = st.session_state["input_columns"]
    out  = st.session_state["output_columns"]
    tvs  = st.session_state["target_values"]
    mat  = st.session_state["material_name"]
    eqp  = st.session_state["equipment_name"]

    # ---- 项目信息卡片 ----
    if mat or eqp:
        st.markdown(f"""
        <div class="project-card">
            <div style="display:flex; gap:3rem; flex-wrap:wrap;">
                <div>
                    <div class="project-label">研究项目</div>
                    <div class="project-value">{mat or '—'}</div>
                </div>
                <div>
                    <div class="project-label">设备 / 工艺</div>
                    <div class="project-value">{eqp or '—'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- 量化目标卡片 ----
    active_t = {k: v for k, v in tvs.items() if v}
    if active_t:
        st.markdown("**量化目标**")
        t_cols = st.columns(len(active_t))
        for idx, (cn, tv) in enumerate(active_t.items()):
            if cn in df.columns:
                avg = pd.to_numeric(df[cn], errors="coerce").mean()
                with t_cols[idx]:
                    st.markdown(f"""
                    <div class="target-card">
                        <div class="target-label">{cn}</div>
                        <div class="target-value">目标: {tv}</div>
                        <div class="current-value">当前均值: {avg:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ---- AI 控制行 ----
    bc1, bc2, bc3 = st.columns([1, 1, 2])
    with bc1:
        analyze_btn = st.button(
            "🔬 AI 深度分析", use_container_width=True, type="primary",
        )
    with bc2:
        pass  # 占位
    with bc3:
        api = st.text_input(
            "Gemini API Key",
            value=st.session_state.get("api_key", ""),
            type="password",
            placeholder="输入 Gemini API Key",
            label_visibility="collapsed",
            key="api_key_input",
        )
        st.session_state["api_key"] = api

    # 处理 AI 分析
    if analyze_btn:
        key = st.session_state.get("api_key", "")
        if not key:
            st.warning("请先输入 Gemini API Key。")
        elif df.empty:
            st.warning("请先在数据工作台录入实验数据。")
        else:
            img_bytes = st.session_state.get("sample_image")
            spinner = (
                "AI 正在分析数据、图像与目标差距…"
                if img_bytes
                else "AI 正在分析目标差距并生成优化建议…"
            )
            with st.spinner(spinner):
                result = analyze_with_ai(
                    df, mat, eqp, inp, out, tvs, key, img_bytes,
                )
            st.session_state["ai_result"] = result

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ---- 数据摘要 ----
    img_bytes = st.session_state.get("sample_image")
    img_status = "已上传" if img_bytes else "无"
    st.markdown(f"""
    <div class="data-summary">
        <span class="summary-item">
            <span class="summary-label">实验次数</span><br>
            <span class="summary-value">{len(df)}</span>
        </span>
        <span class="summary-item">
            <span class="summary-label">参数列</span><br>
            <span class="summary-value">{len(inp)}</span>
        </span>
        <span class="summary-item">
            <span class="summary-label">结果列</span><br>
            <span class="summary-value">{len(out)}</span>
        </span>
        <span class="summary-item">
            <span class="summary-label">已设目标</span><br>
            <span class="summary-value">{len(active_t)}</span>
        </span>
        <span class="summary-item">
            <span class="summary-label">样品图片</span><br>
            <span class="summary-value" style="color:{'#10b981' if img_bytes else '#999'}">
                {img_status}
            </span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ---- 彩色表格 + 趋势图 (+ 图片预览) ----
    if img_bytes:
        col_tbl, col_chart, col_img = st.columns([1, 1, 0.8])
    else:
        col_tbl, col_chart = st.columns([1, 1])
        col_img = None

    with col_tbl:
        st.markdown("**实验数据预览**")
        if inp or out:
            st.dataframe(
                style_dataframe(df, inp, out),
                use_container_width=True, height=280,
            )
        else:
            st.dataframe(df, use_container_width=True, height=280)

    with col_chart:
        st.markdown("**结果趋势与目标**")
        st.plotly_chart(
            create_trend_chart(df, out, tvs), use_container_width=True,
        )

    if col_img is not None and img_bytes:
        with col_img:
            st.markdown("**样品图片**")
            st.image(
                Image.open(io.BytesIO(img_bytes)),
                caption=st.session_state.get("sample_image_name", ""),
                use_container_width=True,
            )

    # ---- AI 分析结果 ----
    ai_result = st.session_state.get("ai_result")

    if ai_result is not None:
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        if ai_result.get("success"):
            has_img = ai_result.get("has_image", False)
            title_l = "图像形貌与数据关联分析" if has_img else "目标差距诊断与机理分析"
            title_r = "形貌改善与参数建议"     if has_img else "精准参数建议与预期效果"

            a_left, a_right = st.columns(2)
            with a_left:
                st.markdown(
                    f'<div class="insight-card">'
                    f'<div class="insight-title">{title_l}</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(ai_result.get("analysis", ""))
            with a_right:
                st.markdown(
                    f'<div class="action-card">'
                    f'<div class="action-title">{title_r}</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    ai_result.get("suggestions", ai_result.get("full_response", ""))
                )

            with st.expander("查看完整 AI 报告"):
                st.markdown(ai_result.get("full_response", ""))
        else:
            st.error(f"AI 分析失败: {ai_result.get('error', '未知错误')}")
    else:
        st.markdown(
            '<div class="placeholder-box">'
            '设定目标后，点击「🔬 AI 深度分析」获取科学原理溯源与参数优化建议'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# 主程序入口
# ============================================================
def main():
    init_session_state()

    # 顶部导航栏
    render_navbar()

    # 侧边栏 — 数据存取
    render_sidebar()

    # 主内容 — 双标签页
    tab_dashboard, tab_studio = st.tabs([
        "📊 智能仪表盘 (Dashboard)",
        "🛠️ 数据工作台 (Data Studio)",
    ])

    with tab_dashboard:
        render_dashboard()

    with tab_studio:
        render_data_studio()

    # 页脚
    st.markdown(
        '<div class="app-footer">'
        'NEXUS Lab · Dual-Mode Materials R&D Platform · Powered by Gemini AI'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

"""
NEXUS Lab — Dual-Mode Materials R&D Platform
Clinical White Theme | Guest / Admin | Google Sheets Sync
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
from PIL import Image
import io

try:
    from streamlit_gsheets import GSheetsConnection
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="NEXUS Lab",
    page_icon=None,
    layout="wide",
)


# ============================================================
# CSS — Clinical White Theme
# ============================================================
ACCENT = "#007AFF"

st.markdown(f"""
<style>
    /* === Hide Streamlit native UI === */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{
        visibility: hidden !important; height: 0 !important;
        min-height: 0 !important; padding: 0 !important;
        margin: 0 !important; overflow: hidden !important;
    }}
    .stDeployButton {{display: none !important;}}
    section[data-testid="stSidebar"] {{display: none !important;}}
    button[data-testid="stSidebarCollapseButton"] {{display: none !important;}}

    /* === Global White === */
    .stApp {{background-color: #FFFFFF;}}
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                     'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei',
                     sans-serif !important;
        color: #333333;
    }}

    /* === Navbar visual bar (pure HTML) === */
    .nexus-navbar {{
        background: #FFFFFF;
        border-bottom: 2px solid {ACCENT};
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        padding: 0.6rem 0; margin-bottom: 0.8rem;
    }}
    .navbar-logo {{
        font-size: 1.4rem; font-weight: 800; color: #1a1a1a;
        letter-spacing: 0.5px; line-height: 2.2rem;
    }}
    .navbar-logo .accent {{color: {ACCENT};}}
    .navbar-badge {{
        display: inline-block; font-size: 0.6rem; padding: 0.1rem 0.45rem;
        border-radius: 99px; margin-left: 0.5rem; font-weight: 600;
        vertical-align: middle; letter-spacing: 0.5px;
    }}
    .badge-guest {{background: #F1F5F9; color: #64748B; border: 1px solid #E2E8F0;}}
    .badge-admin {{background: #EFF6FF; color: {ACCENT}; border: 1px solid #BFDBFE;}}

    /* === Popover === */
    [data-testid="stPopover"] > div {{min-width: 260px;}}

    /* === Section Titles === */
    .area-title {{
        font-size: 1.05rem; font-weight: 600; color: #333;
        margin-bottom: 0.8rem; padding-bottom: 0.4rem;
        border-bottom: 2px solid {ACCENT}; display: inline-block;
    }}
    .area-number {{color: {ACCENT}; font-weight: 700;}}

    /* === Divider === */
    .section-divider {{border: none; border-top: 1px solid #E8E8E8; margin: 1.5rem 0;}}

    /* === Project Card === */
    .project-card {{
        background: linear-gradient(135deg, #F8FAFF 0%, #F0F5FF 100%);
        border: 1px solid #D0E0F5; border-radius: 10px;
        padding: 1.1rem 1.4rem; margin-bottom: 1.2rem;
    }}
    .project-label {{font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 1px;}}
    .project-value {{font-size: 1rem; font-weight: 600; color: #333; margin-top: 0.15rem;}}

    /* === Target Card === */
    .target-card {{
        background: #F0FDF4; border: 1px solid #86EFAC;
        border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.5rem;
    }}
    .target-label  {{font-size: 0.75rem; color: #166534; font-weight: 600;}}
    .target-value  {{font-size: 1.1rem; font-weight: 700; color: #15803D;}}
    .current-value {{font-size: 0.8rem; color: #666;}}

    /* === Data Summary === */
    .data-summary {{
        background: #F8F9FA; border: 1px solid #E0E0E0;
        border-radius: 8px; padding: 0.9rem 1.2rem; margin-bottom: 1rem;
    }}
    .summary-item  {{display: inline-block; margin-right: 2rem;}}
    .summary-label {{font-size: 0.7rem; color: #888; text-transform: uppercase;}}
    .summary-value {{font-size: 1.2rem; font-weight: 700; color: #333;}}

    /* === AI Cards === */
    .insight-card {{
        background: linear-gradient(135deg, #FAFBFF, #F5F8FF);
        border: 1px solid #D0E0F5; border-left: 4px solid {ACCENT};
        border-radius: 8px; padding: 1.4rem; margin-bottom: 1rem;
    }}
    .insight-title {{
        font-size: 0.85rem; font-weight: 700; color: {ACCENT};
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem;
    }}
    .action-card {{
        background: linear-gradient(135deg, #F8FFFE, #F0FDF9);
        border: 1px solid #A7E8D8; border-left: 4px solid #10B981;
        border-radius: 8px; padding: 1.4rem; margin-bottom: 1rem;
    }}
    .action-title {{
        font-size: 0.85rem; font-weight: 700; color: #10B981;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem;
    }}

    /* === Mapping Tags === */
    .mapping-info {{
        background: #FFF8F0; border: 1px solid #FFD6A5;
        border-radius: 6px; padding: 0.7rem 1rem;
        font-size: 0.85rem; color: #666; margin-bottom: 1rem;
    }}
    .mapping-tag {{
        display: inline-block; border-radius: 4px;
        padding: 0.15rem 0.45rem; font-size: 0.8rem; margin: 0.15rem;
    }}
    .mapping-tag.input  {{background: #DBEAFE; color: #1D4ED8;}}
    .mapping-tag.output {{background: #FFF0E6; color: #C2410C;}}

    /* === Target Setting === */
    .target-section {{
        background: #FEFCE8; border: 1px solid #FEF08A;
        border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 0.8rem;
    }}
    .target-section-title {{
        font-size: 0.85rem; font-weight: 600; color: #854D0E; margin-bottom: 0.6rem;
    }}

    /* === Buttons === */
    .stButton > button {{
        border-radius: 4px; font-weight: 600; transition: all 0.15s ease;
        background: #FFFFFF; color: #333333; border: 1px solid #D0D0D0;
    }}
    .stButton > button:hover {{
        border-color: {ACCENT}; color: {ACCENT};
    }}
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background: {ACCENT} !important; color: #FFFFFF !important;
        border: none !important; border-radius: 4px !important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background: #0066DD !important;
    }}

    /* === Tabs === */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        border-bottom-color: {ACCENT} !important; color: {ACCENT} !important;
        font-weight: 600;
    }}
    .stTabs [data-baseweb="tab-list"] button {{color: #888;}}

    /* === Hint Box === */
    .hint-box {{
        background: #F0F7FF; border: 1px solid #BFDBFE;
        border-radius: 6px; padding: 0.7rem 1rem;
        font-size: 0.85rem; color: #1E40AF; margin-bottom: 0.8rem;
    }}

    /* === Placeholder === */
    .placeholder-box {{
        background: #FAFBFC; border: 1px dashed #D0D0D0;
        border-radius: 8px; padding: 2.5rem; text-align: center; color: #999;
    }}

    /* === Footer === */
    .app-footer {{
        text-align: center; color: #AAA; font-size: 0.8rem;
        padding: 1.5rem 0; border-top: 1px solid #E8E8E8; margin-top: 1.5rem;
    }}

    /* === Layout Spacing === */
    .main .block-container {{
        padding-top: 1.5rem !important;
        padding-left: 2.5rem; padding-right: 2.5rem; padding-bottom: 2rem;
    }}

    /* === Inputs === */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {{
        background: #FFFFFF; border: 1px solid #E0E0E0;
        border-radius: 4px; color: #333;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: {ACCENT}; box-shadow: 0 0 0 2px rgba(0,122,255,0.1);
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
def init_session_state():
    defaults = {
        "user_role": "guest",
        "material_name": "",
        "equipment_name": "",
        "df": pd.DataFrame({
            "温度(C)":       [1800, 1850, 1900, 1950, 2000],
            "压力(mbar)":    [50,   55,   60,   65,   70],
            "Ar流量(sccm)":  [100,  100,  120,  120,  150],
            "生长时间(h)":   [24,   24,   30,   30,   36],
            "生长速率(um/h)": [80,   95,   110,  105,  98],
            "微管密度(cm-2)": [5.2,  4.1,  2.8,  3.5,  4.0],
        }),
        "input_columns":  [],
        "output_columns": [],
        "target_values":  {},
        "sample_image":      None,
        "sample_image_name": None,
        "ai_result": None,
        "api_key":   "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ============================================================
# Utilities
# ============================================================
def _clear_editor_widget():
    if "editor" in st.session_state:
        del st.session_state["editor"]


def style_dataframe(df: pd.DataFrame, input_cols: list, output_cols: list):
    """Light-mode Styler: Input cols -> light blue, Output cols -> light orange."""
    def _color(col: pd.Series) -> list[str]:
        if col.name in input_cols:
            return ["background-color: #E6F3FF"] * len(col)
        if col.name in output_cols:
            return ["background-color: #FFF0E6"] * len(col)
        return [""] * len(col)
    return df.style.apply(_color, axis=0)


def create_trend_chart(
    df: pd.DataFrame, output_cols: list, target_values: dict
) -> go.Figure:
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
    palette = [ACCENT, "#10B981", "#F59E0B", "#EF4444"]

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
                    annotation_position="right", annotation_font_color=c,
                )
            except (ValueError, TypeError):
                pass

    fig.update_layout(
        template="simple_white",
        title=dict(text="结果趋势 (虚线 = 目标值)", font=dict(size=14)),
        xaxis_title="实验编号", yaxis_title="数值",
        height=320, margin=dict(t=50, b=40, l=50, r=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#F0F0F0")
    fig.update_yaxes(gridcolor="#F0F0F0")
    return fig


# ============================================================
# AI Analysis (Gemini)
# ============================================================
def analyze_with_ai(
    df: pd.DataFrame, material: str, equipment: str,
    input_cols: list, output_cols: list, target_values: dict,
    api_key: str, image_bytes: bytes = None,
) -> dict:
    try:
        genai.configure(api_key=api_key)
        csv_str = df.to_csv(index=False)

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
                    f"- {col}: 目标值={tv}, 当前均值={avg:.2f}, "
                    f"最优={best:.2f}, 差距={gap:.2f} ({pct:+.1f}%)"
                )
            else:
                t_lines.append(
                    f"- {col}: 未设定目标, 当前均值={avg:.2f}, 最优={best:.2f}"
                )
        target_str = "\n".join(t_lines) if t_lines else "(用户未设定具体目标)"

        has_image = image_bytes is not None
        img_instr = ""
        if has_image:
            img_instr = (
                "\n5. 仔细观察用户上传的样品微观结构图"
                "\n6. 分析图像中的形貌特征（晶粒大小、裂纹、孔隙、颜色异常等）"
                "\n7. 将图像观察与实验参数关联，推断工艺-形貌-性能的因果关系"
            )

        system_prompt = (
            f"你是一位世界顶级的材料科学家和工艺工程师。\n"
            f"用户正在进行【{material or '材料'}】的研究。\n"
            f"使用的设备/工艺是：【{equipment or '实验设备'}】。\n\n"
            f"你的任务是帮助用户达成量化目标。\n"
            f"1. 精确指出当前数据与目标值的差距\n"
            f"2. 结合物理/化学原理解释瓶颈\n"
            f"3. 给出能够逼近目标值的具体参数建议\n"
            f"4. 如果目标不切实际，诚实指出{img_instr}"
        )

        in_str = ", ".join(input_cols) if input_cols else "(用户未指定)"

        if has_image:
            user_prompt = (
                f"## 实验数据\n```csv\n{csv_str}\n```\n\n"
                f"## 数据列说明\n- 实验参数列 (可调变量): {in_str}\n\n"
                f"## 用户的量化目标\n{target_str}\n\n"
                f"## 样品图像\n用户上传了一张样品的微观结构图。请仔细观察。\n\n---\n\n"
                f"请按以下结构分析:\n\n"
                f"### 一、图像形貌分析\n### 二、数据-图像关联分析\n"
                f"### 三、瓶颈机理分析\n### 四、精准参数建议\n### 五、预期效果评估"
            )
        else:
            user_prompt = (
                f"## 实验数据\n```csv\n{csv_str}\n```\n\n"
                f"## 数据列说明\n- 实验参数列 (可调变量): {in_str}\n\n"
                f"## 用户的量化目标\n{target_str}\n\n---\n\n"
                f"请按以下结构分析:\n\n"
                f"### 一、目标差距诊断\n### 二、瓶颈机理分析\n"
                f"### 三、精准参数建议\n### 四、预期效果评估"
            )

        model = genai.GenerativeModel(
            "gemini-2.0-flash", system_instruction=system_prompt
        )
        if has_image:
            img = Image.open(io.BytesIO(image_bytes))
            response = model.generate_content([user_prompt, img])
        else:
            response = model.generate_content(user_prompt)

        full = response.text
        split_mk = "### 四" if has_image else "### 三"
        if split_mk in full:
            parts = full.split(split_mk, 1)
            analysis = parts[0].strip()
            suggestion = split_mk + parts[1]
        else:
            analysis, suggestion = full, ""

        return {
            "success": True, "analysis": analysis, "suggestions": suggestion,
            "full_response": full, "has_image": has_image,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# Navbar (4-column layout in main flow)
# ============================================================
def render_navbar():
    """4-column navbar: Logo | spacer | user popover | right buffer."""
    role = st.session_state.get("user_role", "guest")
    is_admin = role == "admin"
    badge = (
        '<span class="navbar-badge badge-admin">ADMIN</span>'
        if is_admin
        else '<span class="navbar-badge badge-guest">GUEST</span>'
    )

    # Visual top bar
    st.markdown(
        f'<div class="nexus-navbar">'
        f'  <div class="navbar-logo">'
        f'    <span class="accent">NEXUS</span> Lab {badge}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 4-column layout: logo label | spacer | popover | right buffer
    col1, col2, col3, col4 = st.columns([3, 4, 2, 1])

    with col1:
        pass  # Logo already rendered as HTML above

    with col2:
        pass  # Spacer

    with col3:
        popover_label = "Admin" if is_admin else "未登录"
        with st.popover(popover_label, width="stretch"):
            if is_admin:
                st.markdown("已登录为 **Admin**")
                st.caption("已启用 Google Sheets 云端同步权限")
                if st.button("退出登录", key="logout_btn", width="stretch"):
                    st.session_state["user_role"] = "guest"
                    st.rerun()
            else:
                st.markdown("**管理员登录**")
                st.caption("解锁 Google Sheets 云端读取 / 保存功能")
                pwd = st.text_input(
                    "密码", type="password", key="login_pwd",
                    placeholder="输入管理密码",
                )
                if st.button("登录", key="login_btn", width="stretch"):
                    try:
                        correct = st.secrets["general"]["password"]
                        if pwd == correct:
                            st.session_state["user_role"] = "admin"
                            st.rerun()
                        else:
                            st.error("密码错误")
                    except Exception:
                        st.warning(
                            "未配置管理密码。请在 `.streamlit/secrets.toml` 中添加:\n\n"
                            '```toml\n[general]\npassword = "your_password"\n```'
                        )

    with col4:
        pass  # Right buffer


# ============================================================
# Tab: 数据工作台 (Data Studio)
# ============================================================
def render_data_studio():
    df = st.session_state["df"]

    # === 1. 实验背景 ===
    st.markdown(
        '<div class="area-title"><span class="area-number">01</span> 实验背景</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        mat = st.text_input(
            "材料 / 项目名称",
            value=st.session_state["material_name"],
            placeholder="例如: 碳化硅 SiC、GaN 外延片、钙钛矿太阳能电池",
            key="ds_material",
        )
    with c2:
        eqp = st.text_input(
            "实验设备 / 工艺",
            value=st.session_state["equipment_name"],
            placeholder="例如: PVT 长晶炉、MOCVD、磁控溅射",
            key="ds_equipment",
        )
    st.session_state["material_name"] = mat
    st.session_state["equipment_name"] = eqp

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # === 2. 列管理工具栏 + 数据编辑 ===
    st.markdown(
        '<div class="area-title">'
        '<span class="area-number">02</span> 列管理与数据编辑'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- 紧凑列操作工具栏 ----
    cols_list = df.columns.tolist()
    tb1, tb2, tb3 = st.columns([2, 2, 3])

    with tb1:
        st.caption("新增列")
        ncn = st.text_input(
            "新列名", key="new_col_name", placeholder="输入列名",
            label_visibility="collapsed",
        )
        if st.button("添加列", key="add_col_btn"):
            name = (ncn or "").strip()
            if name and name not in df.columns:
                new = df.copy()
                new[name] = 0.0
                st.session_state["df"] = new
                _clear_editor_widget()
                st.rerun()
            elif not name:
                st.warning("请输入列名。")
            else:
                st.warning("该列名已存在。")

    with tb2:
        st.caption("重命名列")
        old_name = st.selectbox(
            "选择列", cols_list, key="rename_select",
            label_visibility="collapsed",
        )
        new_name_input = st.text_input(
            "新名称", key="rename_input", placeholder="输入新名称",
            label_visibility="collapsed",
        )
        if st.button("重命名", key="rename_btn"):
            nn = (new_name_input or "").strip()
            if nn and nn != old_name:
                st.session_state["df"] = df.rename(columns={old_name: nn})
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

    with tb3:
        st.caption("删除列")
        del_cols = st.multiselect(
            "选择要删除的列", cols_list, key="del_cols_select",
            label_visibility="collapsed",
        )
        if del_cols:
            if st.button("删除选中列", key="del_cols_btn", type="primary"):
                st.session_state["df"] = df.drop(
                    columns=del_cols, errors="ignore"
                )
                st.session_state["input_columns"] = [
                    c for c in st.session_state["input_columns"]
                    if c not in del_cols
                ]
                st.session_state["output_columns"] = [
                    c for c in st.session_state["output_columns"]
                    if c not in del_cols
                ]
                for c in del_cols:
                    st.session_state["target_values"].pop(c, None)
                _clear_editor_widget()
                st.rerun()

    st.caption("注: 由于组件限制，请在此处管理列结构，在下方表格编辑数据。")

    # ---- 数据存取区 (Data IO) ----
    with st.container(border=True):
        # 云端同步 (仅管理员可见)
        if st.session_state.get("user_role") == "admin":
            st.caption("云端数据库 (管理员模式)")
            gc1, gc2 = st.columns(2)
            with gc1:
                if st.button(
                    "从 Google Sheets 加载",
                    width="stretch", key="gs_load",
                ):
                    if not GSHEETS_AVAILABLE:
                        st.error("未安装 st-gsheets-connection，无法连接云端。")
                    else:
                        try:
                            conn = st.connection(
                                "gsheets", type=GSheetsConnection
                            )
                            df_cloud = conn.read()
                            df_cloud = df_cloud.dropna(how="all")
                            if df_cloud.empty:
                                st.warning("云端工作表为空或无法读取。")
                            else:
                                st.session_state["df"] = df_cloud
                                st.session_state["input_columns"] = []
                                st.session_state["output_columns"] = []
                                st.session_state["target_values"] = {}
                                _clear_editor_widget()
                                st.success(
                                    f"已同步云端数据 "
                                    f"({len(df_cloud)} 行 x "
                                    f"{len(df_cloud.columns)} 列)"
                                )
                                st.rerun()
                        except Exception as e:
                            st.error(f"加载失败: {str(e)}")

            with gc2:
                if st.button(
                    "保存到 Google Sheets",
                    width="stretch", key="gs_save", type="primary",
                ):
                    if not GSHEETS_AVAILABLE:
                        st.error("未安装 st-gsheets-connection，无法连接云端。")
                    else:
                        try:
                            conn = st.connection(
                                "gsheets", type=GSheetsConnection
                            )
                            conn.update(data=st.session_state["df"])
                            st.success("保存成功，Google Sheets 已更新。")
                        except Exception as e:
                            st.error(f"保存失败: {str(e)}")
                            st.markdown(
                                "**排查建议**: 请检查 Google Sheet 是否已"
                                "分享给 Service Account 邮箱，并赋予 "
                                "**Editor** 权限。"
                            )
            st.divider()

        # 本地备份 (所有人可见)
        st.caption("本地文件存取")
        lc1, lc2 = st.columns(2)
        with lc1:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "下载 CSV 备份", csv_bytes, "nexus_backup.csv",
                "text/csv", width="stretch",
            )
        with lc2:
            uploaded = st.file_uploader(
                "上传 CSV 恢复", type=["csv"], key="csv_uploader",
            )
            if uploaded is not None:
                try:
                    preview_df = pd.read_csv(uploaded)
                    st.info(
                        f"检测到 {len(preview_df)} 行 x "
                        f"{len(preview_df.columns)} 列"
                    )
                    if st.button("确认导入", key="confirm_csv_import"):
                        st.session_state["df"] = preview_df
                        st.session_state["input_columns"] = []
                        st.session_state["output_columns"] = []
                        st.session_state["target_values"] = {}
                        _clear_editor_widget()
                        st.rerun()
                except Exception as e:
                    st.error(f"CSV 解析失败: {e}")

    # ---- 全功能数据编辑器 ----
    st.markdown(
        '<div class="hint-box">'
        '直接编辑下方表格: 增删行、修改数值、从 Excel 复制粘贴均可。'
        '</div>',
        unsafe_allow_html=True,
    )

    edited_df = st.data_editor(
        st.session_state["df"],
        num_rows="dynamic", width="stretch", height=360, key="editor",
    )
    st.session_state["df"] = edited_df

    # ---- 样品图片 (可选) ----
    with st.expander("样品图片 (可选)"):
        st.caption("上传 SEM / 光学显微镜图片，AI 将结合图像形貌分析")
        up_img = st.file_uploader(
            "上传图片", type=["png", "jpg", "jpeg"],
            key="img_uploader", label_visibility="collapsed",
        )
        if up_img is not None:
            img = Image.open(up_img)
            st.image(img, caption=f"已上传: {up_img.name}", width="stretch")
            st.session_state["sample_image"] = up_img.getvalue()
            st.session_state["sample_image_name"] = up_img.name
        elif st.session_state.get("sample_image"):
            img = Image.open(io.BytesIO(st.session_state["sample_image"]))
            st.image(
                img,
                caption=f"已保存: {st.session_state.get('sample_image_name', '')}",
                width="stretch",
            )
            if st.button("移除图片", key="rm_img_btn"):
                st.session_state["sample_image"] = None
                st.session_state["sample_image_name"] = None
                st.rerun()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # === 3. 语义映射与目标设定 ===
    st.markdown(
        '<div class="area-title">'
        '<span class="area-number">03</span> 语义映射与目标设定'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="mapping-info">
        <strong>第一步:</strong> 选择参数列 (Inputs) 和结果列 (Outputs)。
        <strong>第二步:</strong> 为结果列设定量化目标值。
    </div>
    """, unsafe_allow_html=True)

    all_cols = edited_df.columns.tolist()
    mc1, mc2 = st.columns(2)
    with mc1:
        inp = st.multiselect(
            "Inputs (参数列) — 蓝色标记", all_cols,
            default=[
                c for c in st.session_state["input_columns"] if c in all_cols
            ],
            help="实验中可以控制的变量",
            key="sel_inputs",
        )
    with mc2:
        avail_out = [c for c in all_cols if c not in inp]
        out = st.multiselect(
            "Outputs (结果列) — 橙色标记", avail_out,
            default=[
                c for c in st.session_state["output_columns"]
                if c in avail_out
            ],
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
            tag_html += " &rarr; Outputs: " + " ".join(
                f'<span class="mapping-tag output">{c}</span>' for c in out
            )
        st.markdown(tag_html, unsafe_allow_html=True)
        st.info(
            f"已将 [{', '.join(inp) or '无'}] 标记为蓝色, "
            f"[{', '.join(out) or '无'}] 标记为橙色。"
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
            for j, cn in enumerate(out[i: i + per_row]):
                with cols[j]:
                    if cn in edited_df.columns:
                        avg = pd.to_numeric(
                            edited_df[cn], errors="coerce"
                        ).mean()
                        mx = pd.to_numeric(
                            edited_df[cn], errors="coerce"
                        ).max()
                    else:
                        avg, mx = 0.0, 0.0
                    saved = tvs.get(cn, "")
                    val = st.text_input(
                        f"[{cn}] 目标值",
                        value=str(saved) if saved else "",
                        placeholder=f"均值 {avg:.2f}",
                        help=f"当前均值: {avg:.2f}, 最优: {mx:.2f}",
                        key=f"tgt_{cn}",
                    )
                    tvs[cn] = val
                    st.caption(f"均值 {avg:.2f} / 最优 {mx:.2f}")

    st.session_state["target_values"] = {
        k: v for k, v in tvs.items() if k in out
    }


# ============================================================
# Visual Analytics Module
# ============================================================
def _render_visual_analytics(df: pd.DataFrame):
    """散点图 + 相关性热力图，放在 Dashboard 的表格下方、AI 报告上方。"""
    if df.empty or len(df.columns) < 2:
        return

    # 筛选数值列
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return

    all_cols = df.columns.tolist()

    with st.expander("可视化分析 (Visual Analytics)", expanded=True):

        # ---- 交互控件: X / Y / Color ----
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            x_col = st.selectbox(
                "X 轴", num_cols,
                index=0,
                key="va_x",
            )
        with ctrl2:
            y_default = min(len(num_cols) - 1, max(0, len(num_cols) - 1))
            y_col = st.selectbox(
                "Y 轴", num_cols,
                index=y_default,
                key="va_y",
            )
        with ctrl3:
            color_options = ["(无)"] + all_cols
            color_sel = st.selectbox(
                "颜色映射 (可选)", color_options,
                index=0,
                key="va_color",
            )

        color_arg = color_sel if color_sel != "(无)" else None

        # ---- 散点图 + 趋势线 ----
        try:
            scatter_kwargs = dict(
                data_frame=df, x=x_col, y=y_col,
                color=color_arg,
                color_continuous_scale="Blues",
                template="simple_white",
                title=f"{y_col}  vs  {x_col}",
            )
            # OLS 趋势线需要 statsmodels；缺失时降级为无趋势线
            try:
                import statsmodels  # noqa: F401
                scatter_kwargs["trendline"] = "ols"
            except ImportError:
                pass

            fig_scatter = px.scatter(**scatter_kwargs)
            fig_scatter.update_traces(
                marker=dict(size=9, line=dict(width=1, color="#FFFFFF")),
            )
            fig_scatter.update_layout(
                height=380,
                margin=dict(t=50, b=40, l=50, r=30),
                xaxis=dict(gridcolor="#F0F0F0"),
                yaxis=dict(gridcolor="#F0F0F0"),
                font=dict(color="#333"),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
            )
            # 趋势线颜色统一为强调蓝
            for trace in fig_scatter.data:
                if hasattr(trace, "mode") and trace.mode == "lines":
                    trace.line.color = ACCENT

            st.plotly_chart(fig_scatter, width="stretch")
        except Exception as exc:
            st.warning(f"散点图绘制失败: {exc}")

        # ---- 相关性热力图 ----
        if len(num_cols) >= 2:
            st.markdown("**参数相关性热力图 (Correlation Heatmap)**")
            corr = df[num_cols].corr()

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale=[
                    [0.0, "#F0F7FF"],
                    [0.5, "#7ABAFF"],
                    [1.0, "#005CBF"],
                ],
                zmin=-1, zmax=1,
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont=dict(size=12, color="#333"),
                hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
                colorbar=dict(title="r"),
            ))
            fig_heatmap.update_layout(
                height=max(320, 50 * len(num_cols)),
                margin=dict(t=30, b=30, l=80, r=30),
                xaxis=dict(tickangle=-40),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#333"),
            )
            st.plotly_chart(fig_heatmap, width="stretch")

        st.caption(
            "提示: 散点图展示双变量关系 (含 OLS 趋势线, 需安装 statsmodels); "
            "热力图展示所有数值列之间的 Pearson 相关系数, "
            "绝对值越接近 1 表示线性相关性越强。"
        )


# ============================================================
# Tab: 智能仪表盘 (Dashboard)
# ============================================================
def render_dashboard():
    df = st.session_state["df"]
    inp = st.session_state["input_columns"]
    out = st.session_state["output_columns"]
    tvs = st.session_state["target_values"]
    mat = st.session_state["material_name"]
    eqp = st.session_state["equipment_name"]

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
            "AI 深度分析", width="stretch", type="primary",
        )
    with bc2:
        pass
    with bc3:
        api = st.text_input(
            "Gemini API Key",
            value=st.session_state.get("api_key", ""),
            type="password", placeholder="输入 Gemini API Key",
            label_visibility="collapsed", key="api_key_input",
        )
        st.session_state["api_key"] = api

    if analyze_btn:
        key = st.session_state.get("api_key", "")
        if not key:
            st.warning("请先输入 Gemini API Key。")
        elif df.empty:
            st.warning("请先在数据工作台录入实验数据。")
        else:
            img_bytes = st.session_state.get("sample_image")
            spinner = (
                "AI 正在分析数据、图像与目标差距..."
                if img_bytes
                else "AI 正在分析目标差距并生成优化建议..."
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
            <span class="summary-value" style="color:{'#10B981' if img_bytes else '#999'}">
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
                width="stretch", height=280,
            )
        else:
            st.dataframe(df, width="stretch", height=280)

    with col_chart:
        st.markdown("**结果趋势与目标**")
        st.plotly_chart(
            create_trend_chart(df, out, tvs), width="stretch",
        )

    if col_img is not None and img_bytes:
        with col_img:
            st.markdown("**样品图片**")
            st.image(
                Image.open(io.BytesIO(img_bytes)),
                caption=st.session_state.get("sample_image_name", ""),
                width="stretch",
            )

    # ---- 可视化分析 (Visual Analytics) ----
    _render_visual_analytics(df)

    # ---- AI 分析结果 ----
    ai_result = st.session_state.get("ai_result")
    if ai_result is not None:
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        if ai_result.get("success"):
            has_img = ai_result.get("has_image", False)
            title_l = (
                "图像形貌与数据关联分析"
                if has_img
                else "目标差距诊断与机理分析"
            )
            title_r = (
                "形貌改善与参数建议"
                if has_img
                else "精准参数建议与预期效果"
            )
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
                    ai_result.get(
                        "suggestions", ai_result.get("full_response", "")
                    )
                )
            with st.expander("查看完整 AI 报告"):
                st.markdown(ai_result.get("full_response", ""))
        else:
            st.error(
                f"AI 分析失败: {ai_result.get('error', '未知错误')}"
            )
    else:
        st.markdown(
            '<div class="placeholder-box">'
            '设定目标后，点击「AI 深度分析」获取科学原理溯源与参数优化建议'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# Main
# ============================================================
def main():
    init_session_state()

    # 顶部导航栏
    render_navbar()

    # 主内容 — 双标签页
    tab_dashboard, tab_studio = st.tabs([
        "智能仪表盘 (Dashboard)",
        "数据工作台 (Data Studio)",
    ])

    with tab_dashboard:
        render_dashboard()
    with tab_studio:
        render_data_studio()

    # 页脚
    st.markdown(
        '<div class="app-footer">'
        'NEXUS Lab | Dual-Mode Materials R&D Platform | Powered by Gemini AI'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

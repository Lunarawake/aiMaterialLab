"""
JPZ 科研平台 — Dual-Mode Materials R&D Platform
Clinical White Theme | Guest / Admin | Google Sheets Sync
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
from PIL import Image
from datetime import datetime
import io
import sqlite3
import os

from translations import TRANSLATIONS

try:
    from streamlit_gsheets import GSheetsConnection
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

try:
    from streamlit_sortables import sort_items
    SORTABLES_AVAILABLE = True
except ImportError:
    SORTABLES_AVAILABLE = False


# --- i18n Helper ---
def T(key: str, **kwargs) -> str:
    """Return the translated string for the current language."""
    lang = st.session_state.get("language", "en")
    text = TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


# --- SQLite Local Database ---
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_APP_DIR, "lab_storage.db")
_OLD_DB_PATH = os.path.join(_APP_DIR, "research_data.db")


def _db_conn() -> sqlite3.Connection:
    """获取 SQLite 连接 (每次调用新建，避免跨线程问题)。"""
    return sqlite3.connect(DB_PATH)


def _db_init():
    """创建数据表 (如不存在)。"""
    conn = _db_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiment_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_json TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """清洗 DataFrame: 删除全空行, 数值列 NaN->0, 文本列 NaN->空字符串。"""
    df = df.dropna(how="all").reset_index(drop=True)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("")
    return df


def db_save(df: pd.DataFrame):
    """清洗后将 DataFrame 以 JSON 写入 SQLite (覆盖式)。"""
    df = _clean_df(df)
    conn = _db_conn()
    try:
        data_json = df.to_json(orient="split", force_ascii=False)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("DELETE FROM experiment_logs")
        conn.execute(
            "INSERT INTO experiment_logs (data_json, timestamp) VALUES (?, ?)",
            (data_json, ts),
        )
        conn.commit()
    finally:
        conn.close()


def db_load() -> pd.DataFrame | None:
    """从 SQLite 读取最新 DataFrame，无数据时返回 None。"""
    conn = _db_conn()
    try:
        row = conn.execute(
            "SELECT data_json FROM experiment_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            df = pd.read_json(io.StringIO(row[0]), orient="split")
            if df.empty or len(df.columns) == 0:
                return None
            return df
        return None
    except Exception:
        return None
    finally:
        conn.close()


# 程序启动时确保数据库和表存在
_db_init()


def _migrate_old_db():
    """如果旧 research_data.db 存在而 lab_storage.db 中无数据，则迁移旧数据。"""
    if not os.path.isfile(_OLD_DB_PATH):
        return
    # lab_storage.db 中已有数据则跳过迁移
    if db_load() is not None:
        return
    try:
        old_conn = sqlite3.connect(_OLD_DB_PATH)
        row = old_conn.execute(
            "SELECT data_json FROM experiment_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        old_conn.close()
        if row:
            df = pd.read_json(io.StringIO(row[0]), orient="split")
            if not df.empty and len(df.columns) > 0:
                db_save(df)
    except Exception:
        pass  # 旧数据库格式不兼容时静默跳过


_migrate_old_db()


# --- Page Config ---
st.set_page_config(page_title="JPZ Platform", page_icon=None, layout="wide")

ACCENT = "#0047AB"

st.markdown(f"""
<style>
    #MainMenu {{visibility:hidden;}}
    footer {{visibility:hidden;}}
    header[data-testid="stHeader"] {{visibility:hidden!important;height:0!important;min-height:0!important;padding:0!important;margin:0!important;overflow:hidden!important;}}
    .stDeployButton {{display:none!important;}}
    section[data-testid="stSidebar"] {{display:none!important;}}
    button[data-testid="stSidebarCollapseButton"] {{display:none!important;}}

    .stApp {{background:#FFFFFF;}}
    html,body,[class*="css"] {{
        font-family:'PingFang SC','Microsoft YaHei',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;
        color:#1D1D1F;
    }}

    .portal-header {{padding:0.7rem 0;margin-bottom:0.5rem;border-bottom:1px solid #E5E5E5;}}
    .portal-header .logo {{font-size:1.35rem;font-weight:800;color:#1D1D1F;letter-spacing:-0.3px;}}
    .portal-header .logo .accent {{color:{ACCENT};}}
    .portal-header .badge {{
        display:inline-block;font-size:0.55rem;padding:0.1rem 0.5rem;border-radius:99px;
        margin-left:0.5rem;font-weight:600;vertical-align:middle;letter-spacing:0.5px;
    }}
    .badge-guest {{background:#F5F5F7;color:#86868B;border:1px solid #E5E5E5;}}
    .badge-admin {{background:#EEF2FF;color:{ACCENT};border:1px solid #C7D2FE;}}

    .stats-bar {{
        background:#FFFFFF;border:1px solid #E5E5E5;border-radius:14px;
        padding:1rem 1.5rem;margin-bottom:1.5rem;
        display:flex;gap:3rem;flex-wrap:wrap;align-items:center;
    }}
    .stat-label {{font-size:0.6rem;color:#86868B;text-transform:uppercase;letter-spacing:1px;}}
    .stat-value {{font-size:1.3rem;font-weight:800;color:#1D1D1F;}}
    .stat-value.accent {{color:{ACCENT};}}

    [data-testid="stPopover"] > div {{min-width:260px;}}

    .area-title {{
        font-size:0.95rem;font-weight:700;color:#1D1D1F;
        margin-bottom:1rem;padding-bottom:0.35rem;
        border-bottom:2px solid {ACCENT};display:inline-block;
    }}
    .area-number {{color:{ACCENT};font-weight:800;}}
    .section-divider {{border:none;border-top:1px solid #E5E5E5;margin:2rem 0;}}

    .project-card {{
        background:#FFFFFF;border:1px solid #E5E5E5;border-radius:14px;
        padding:1.2rem 1.5rem;margin-bottom:1rem;
    }}
    .project-label {{font-size:0.65rem;color:#86868B;text-transform:uppercase;letter-spacing:1px;}}
    .project-value {{font-size:1rem;font-weight:600;color:#1D1D1F;margin-top:0.15rem;}}

    .target-card {{
        background:#FFFFFF;border:1px solid #E5E5E5;border-left:3px solid {ACCENT};
        border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.5rem;
    }}
    .target-label {{font-size:0.7rem;color:#86868B;font-weight:600;}}
    .target-value {{font-size:1.05rem;font-weight:700;color:{ACCENT};}}
    .current-value {{font-size:0.75rem;color:#86868B;}}

    .data-summary {{
        background:#FFFFFF;border:1px solid #E5E5E5;border-radius:14px;
        padding:1rem 1.3rem;margin-bottom:1rem;
    }}
    .summary-item {{display:inline-block;margin-right:2.5rem;}}
    .summary-label {{font-size:0.6rem;color:#86868B;text-transform:uppercase;letter-spacing:0.5px;}}
    .summary-value {{font-size:1.2rem;font-weight:800;color:#1D1D1F;}}

    .insight-card {{
        background:#FFFFFF;border:1px solid #E5E5E5;border-left:3px solid {ACCENT};
        border-radius:14px;padding:1.5rem;margin-bottom:1rem;
    }}
    .insight-title {{font-size:0.8rem;font-weight:700;color:{ACCENT};text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;}}
    .action-card {{
        background:#FFFFFF;border:1px solid #E5E5E5;border-left:3px solid #10B981;
        border-radius:14px;padding:1.5rem;margin-bottom:1rem;
    }}
    .action-title {{font-size:0.8rem;font-weight:700;color:#10B981;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;}}

    .mapping-info {{
        background:#FFFFFF;border:1px solid #E5E5E5;border-left:3px solid {ACCENT};
        border-radius:12px;padding:0.8rem 1rem;font-size:0.85rem;color:#86868B;margin-bottom:1rem;
    }}
    .mapping-tag {{display:inline-block;border-radius:6px;padding:0.15rem 0.5rem;font-size:0.75rem;font-weight:600;margin:0.15rem;}}
    .mapping-tag.input {{background:#EEF2FF;color:{ACCENT};}}
    .mapping-tag.output {{background:#FFF7ED;color:#C2410C;}}

    .target-section {{
        background:#FFFFFF;border:1px solid #E5E5E5;border-radius:12px;
        padding:1rem 1.1rem;margin-top:0.8rem;
    }}
    .target-section-title {{font-size:0.8rem;font-weight:600;color:#1D1D1F;margin-bottom:0.6rem;}}

    .hint-box {{
        background:#FFFFFF;border:1px solid #E5E5E5;border-left:3px solid {ACCENT};
        border-radius:12px;padding:0.75rem 1rem;font-size:0.82rem;color:#1D1D1F;margin-bottom:1rem;
    }}
    .placeholder-box {{
        background:#FFFFFF;border:1px dashed #D0D0D0;border-radius:14px;
        padding:3rem;text-align:center;color:#86868B;font-size:0.9rem;
    }}

    .recent-title {{font-size:0.9rem;font-weight:700;color:#1D1D1F;margin-bottom:0.5rem;}}

    .stButton > button {{
        border-radius:8px;font-weight:600;transition:all 0.15s ease;
        background:#FFFFFF;color:#1D1D1F;border:1px solid #E5E5E5;
    }}
    .stButton > button:hover {{border-color:{ACCENT};color:{ACCENT};}}
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background:{ACCENT}!important;color:#FFF!important;
        border:none!important;border-radius:8px!important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background:#003A8C!important;
    }}

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        border-bottom-color:{ACCENT}!important;color:{ACCENT}!important;font-weight:600;
    }}
    .stTabs [data-baseweb="tab-list"] button {{color:#86868B;}}

    .app-footer {{text-align:center;color:#86868B;font-size:0.75rem;padding:2rem 0;margin-top:2rem;}}

    .main .block-container {{
        padding-top:1.2rem!important;
        padding-left:3rem;padding-right:3rem;padding-bottom:2rem;
    }}

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {{
        background:#FFFFFF;border:1px solid #E5E5E5;border-radius:8px;color:#1D1D1F;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {{
        border-color:{ACCENT};box-shadow:0 0 0 2px rgba(0,71,171,0.08);
    }}
</style>
""", unsafe_allow_html=True)


# --- Session State ---
def _get_sample_df() -> pd.DataFrame:
    """最小示例数据，确保 DataFrame 至少有列名和一行数据。"""
    return pd.DataFrame({
        "温度(C)":       [1800, 1850, 1900, 1950, 2000],
        "压力(mbar)":    [50,   55,   60,   65,   70],
        "Ar流量(sccm)":  [100,  100,  120,  120,  150],
        "生长时间(h)":   [24,   24,   30,   30,   36],
        "生长速率(um/h)": [80,   95,   110,  105,  98],
        "微管密度(cm-2)": [5.2,  4.1,  2.8,  3.5,  4.0],
    })


def init_session_state():
    # 优先从本地 SQLite 恢复; 若为空则用示例数据兜底
    restored_df = db_load()
    if restored_df is not None and not restored_df.empty and len(restored_df.columns) > 0:
        starting_df = restored_df
    else:
        starting_df = _get_sample_df()
        # 将示例数据写入 SQLite, 防止下次启动仍为空
        db_save(starting_df)

    defaults = {
        "language": "en",
        "user_role": "guest",
        "material_name": "",
        "equipment_name": "",
        "df": starting_df,
        "input_columns":  [],
        "output_columns": [],
        "target_values":  {},
        "target_memory":  {},            # 持久化记忆: 即使取消勾选也保留旧目标值
        "sample_image":      None,
        "sample_image_name": None,
        "ai_result": None,
        "api_key":   "",
        "editor_version": 0,
        "db_ready": True,
        "active_view": "home",       # home | dashboard | data_studio | visual | settings
        "last_sync_time": None,      # 最近云端同步时间
        "experiment_chat_history": [],   # AI 诊断页 - 实验数据追问
        "guide_chat_history": [],        # 首页 - 平台使用向导
        "custom_column_order": starting_df.columns.tolist(),  # 用户自定义列顺序
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# --- Utilities ---
def _clear_editor_widget():
    """清除 data_editor widget state 并递增版本号，强制组件刷新。"""
    ver = st.session_state.get("editor_version", 0)
    old_key = f"editor_{ver}"
    if old_key in st.session_state:
        del st.session_state[old_key]
    st.session_state["editor_version"] = ver + 1


def _sync_column_order():
    """防御性同步: 确保 custom_column_order 与 df.columns 一致。"""
    df_cols = st.session_state["df"].columns.tolist()
    order = st.session_state.get("custom_column_order", [])
    # 保留顺序中仍存在的列, 追加新出现的列
    synced = [c for c in order if c in df_cols]
    for c in df_cols:
        if c not in synced:
            synced.append(c)
    st.session_state["custom_column_order"] = synced


def _on_editor_change():
    """data_editor 的 on_change 回调: 一旦用户修改单元格, 立即保存。"""
    ver = st.session_state.get("editor_version", 0)
    editor_key = f"editor_{ver}"
    editor_state = st.session_state.get(editor_key)

    if editor_state is None:
        return

    df = st.session_state["df"].copy()

    # 应用 edited_rows
    edited_rows = editor_state.get("edited_rows", {})
    for row_idx_str, changes in edited_rows.items():
        row_idx = int(row_idx_str)
        if row_idx < len(df):
            for col, val in changes.items():
                if col in df.columns:
                    df.at[row_idx, col] = val

    # 应用 added_rows
    added_rows = editor_state.get("added_rows", [])
    if added_rows:
        new_rows = pd.DataFrame(added_rows)
        # 确保新行包含所有列
        for col in df.columns:
            if col not in new_rows.columns:
                new_rows[col] = 0 if pd.api.types.is_numeric_dtype(df[col]) else ""
        df = pd.concat([df, new_rows], ignore_index=True)

    # 应用 deleted_rows
    deleted_rows = editor_state.get("deleted_rows", [])
    if deleted_rows:
        df = df.drop(index=deleted_rows, errors="ignore").reset_index(drop=True)

    # 清洗并保存
    df = _clean_df(df)
    st.session_state["df"] = df
    db_save(df)
    st.session_state["_save_status"] = "saved"

    # Admin: 同步到云端
    if st.session_state.get("user_role") == "admin" and GSHEETS_AVAILABLE:
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            clean = df.fillna("")
            conn.update(data=clean)
        except Exception:
            pass  # 静默失败, 不阻断本地保存


def style_dataframe(df: pd.DataFrame, input_cols: list, output_cols: list):
    def _color(col: pd.Series) -> list[str]:
        if col.name in input_cols:
            return ["background-color: #EEF2FF"] * len(col)
        if col.name in output_cols:
            return ["background-color: #FFF7ED"] * len(col)
        return [""] * len(col)
    return df.style.apply(_color, axis=0)


def create_trend_chart(
    df: pd.DataFrame, output_cols: list, target_values: dict
) -> go.Figure:
    fig = go.Figure()

    if not output_cols or df.empty:
        fig.add_annotation(
            text=T("no_output_for_trend"),
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#86868B"),
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
                        annotation_text=T("target_annotation", val=tv),
                        annotation_position="right", annotation_font_color=c,
                    )
            except (ValueError, TypeError):
                pass

    fig.update_layout(
        template="simple_white",
        title=dict(text=T("trend_chart_title"), font=dict(size=14)),
        xaxis_title=T("experiment_number"), yaxis_title=T("value_label"),
        height=320, margin=dict(t=50, b=40, l=50, r=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#F0F0F0")
    fig.update_yaxes(gridcolor="#F0F0F0")
    return fig


# --- AI Analysis (Gemini) ---
def analyze_with_ai(
    df: pd.DataFrame, material: str, equipment: str,
    input_cols: list, output_cols: list, target_values: dict,
    api_key: str, image_bytes: bytes = None,
    custom_prompt: str = "",
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
                    T("ai_target_line", col=col, tv=tv, avg=avg, best=best, gap=gap, pct=pct)
                )
            else:
                t_lines.append(
                    T("ai_target_line_no_target", col=col, avg=avg, best=best)
                )
        target_str = "\n".join(t_lines) if t_lines else T("ai_no_target")

        has_image = image_bytes is not None
        img_instr = ""
        if has_image:
            img_instr = T("ai_img_instr")

        custom_instr = ""
        if custom_prompt and custom_prompt.strip():
            custom_instr = T("ai_custom_instr", prompt=custom_prompt.strip())

        _mat_default = "Material" if st.session_state.get("language", "en") == "en" else "材料"
        _eq_default = "Equipment" if st.session_state.get("language", "en") == "en" else "实验设备"
        system_prompt = T(
            "ai_system_prompt",
            material=material or _mat_default,
            equipment=equipment or _eq_default,
            img_instr=img_instr,
            custom_instr=custom_instr,
        )

        in_str = ", ".join(input_cols) if input_cols else T("ai_input_not_specified")

        if has_image:
            user_prompt = T(
                "ai_user_prompt_img", csv=csv_str, inputs=in_str, targets=target_str,
            )
        else:
            user_prompt = T(
                "ai_user_prompt_no_img", csv=csv_str, inputs=in_str, targets=target_str,
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
        split_mk = T("ai_split_marker_img") if has_image else T("ai_split_marker_no_img")
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


# --- Portal Header (Logo + Badge + Popover) ---
def render_header():
    role = st.session_state.get("user_role", "guest")
    is_admin = role == "admin"
    badge_cls = "badge-admin" if is_admin else "badge-guest"
    badge_txt = "ADMIN" if is_admin else "GUEST"

    st.markdown(
        f'<div class="portal-header">'
        f'  <span class="logo"><span class="accent">JPZ</span>{T("logo_suffix")}</span>'
        f'  <span class="badge {badge_cls}">{badge_txt}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _, lang_col, pop_col = st.columns([7, 1, 2])
    with lang_col:
        lang_options = ["English", "中文"]
        current_lang = st.session_state.get("language", "en")
        current_idx = 0 if current_lang == "en" else 1
        selected = st.selectbox(
            T("lang_label"),
            options=lang_options,
            index=current_idx,
            key="lang_selector",
            label_visibility="collapsed",
        )
        new_lang = "en" if selected == "English" else "cn"
        if new_lang != current_lang:
            st.session_state["language"] = new_lang
            st.rerun()
    with pop_col:
        popover_label = T("popover_admin") if is_admin else T("popover_guest")
        with st.popover(popover_label, width="stretch"):
            if is_admin:
                st.markdown(T("logged_in_as_admin"))
                st.caption(T("cloud_sync_enabled"))
                if st.button(T("logout"), key="logout_btn", width="stretch"):
                    st.session_state["user_role"] = "guest"
                    st.rerun()
            else:
                st.markdown(T("admin_login"))
                st.caption(T("unlock_gsheets"))
                pwd = st.text_input(
                    T("password"), type="password", key="login_pwd",
                    placeholder=T("password_placeholder"),
                )
                if st.button(T("login"), key="login_btn", width="stretch"):
                    try:
                        correct = st.secrets["general"]["password"]
                        if pwd == correct:
                            st.session_state["user_role"] = "admin"
                            st.rerun()
                        else:
                            st.error(T("wrong_password"))
                    except Exception:
                        st.warning(T("no_password_config"))


def render_stats_bar():
    df = st.session_state["df"]
    rows, cols = df.shape
    sync_time = st.session_state.get("last_sync_time")
    sync_display = sync_time if sync_time else "--"
    role_display = T("role_admin") if st.session_state.get("user_role") == "admin" else T("role_guest")
    db_status = T("storage_active") if st.session_state.get("db_ready") else T("storage_disconnected")

    st.markdown(
        f'<div class="stats-bar">'
        f'  <div class="stat-item"><div class="stat-label">{T("stat_records")}</div><div class="stat-value accent">{rows}</div></div>'
        f'  <div class="stat-item"><div class="stat-label">{T("stat_columns")}</div><div class="stat-value">{cols}</div></div>'
        f'  <div class="stat-item"><div class="stat-label">{T("stat_identity")}</div><div class="stat-value">{role_display}</div></div>'
        f'  <div class="stat-item"><div class="stat-label">{T("stat_storage")}</div><div class="stat-value">{db_status}</div></div>'
        f'  <div class="stat-item"><div class="stat-label">{T("stat_cloud_sync")}</div><div class="stat-value">{sync_display}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# --- Platform Guide Dialog (平台向导弹窗) ---
def _guide_dialog_body():
    """Platform guide dialog body (language-aware)."""

    if st.button(T("close"), key="close_guide"):
        st.session_state["show_guide_dialog"] = False
        st.rerun()

    st.markdown(T("guide_intro"))

    # 显示历史对话
    chat_area = st.container(height=350)
    with chat_area:
        if not st.session_state["guide_chat_history"]:
            st.caption(T("guide_no_history"))
        for msg in st.session_state["guide_chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input(
        T("guide_input_placeholder"),
        key="guide_chat_input",
    )

    if user_input:
        api_key = st.session_state.get("api_key", "")
        st.session_state["guide_chat_history"].append(
            {"role": "user", "content": user_input}
        )
        if not api_key:
            answer = T("guide_no_api_key")
        else:
            system_prompt = T("guide_system_prompt")
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    "gemini-2.0-flash", system_instruction=system_prompt,
                )
                response = model.generate_content(user_input)
                answer = response.text
            except Exception as e:
                answer = T("guide_ai_error", error=str(e))

        st.session_state["guide_chat_history"].append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()


@st.dialog("JPZ Platform Assistant", width="large")
def guide_dialog_en():
    _guide_dialog_body()


@st.dialog("JPZ 平台使用助手", width="large")
def guide_dialog_cn():
    _guide_dialog_body()


def guide_dialog():
    """Dispatch to the correct language dialog."""
    if st.session_state.get("language", "en") == "cn":
        guide_dialog_cn()
    else:
        guide_dialog_en()


def render_portal_home():
    is_admin = st.session_state.get("user_role") == "admin"
    df = st.session_state["df"]
    rows, cols = df.shape if df is not None and not df.empty else (0, 0)
    num_cols_count = len(df.select_dtypes(include=["number"]).columns) if df is not None else 0
    sync_time = st.session_state.get("last_sync_time", "--")

    st.markdown(
        f'<p style="font-size:1.5rem;font-weight:800;color:#1D1D1F;margin-bottom:0.1rem;">'
        f'{T("welcome_title")}</p>'
        f'<p style="font-size:0.85rem;color:#86868B;margin-bottom:1.8rem;">'
        f'{T("welcome_subtitle")}</p>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div style="background:#FFF;border:1px solid #E5E5E5;border-radius:14px;padding:1.2rem 1.4rem;">'
            f'<div style="font-size:2rem;font-weight:800;color:#0047AB;">{rows}</div>'
            f'<div style="font-size:0.7rem;color:#86868B;margin-top:0.2rem;">{T("metric_records")}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div style="background:#FFF;border:1px solid #E5E5E5;border-radius:14px;padding:1.2rem 1.4rem;">'
            f'<div style="font-size:2rem;font-weight:800;color:#0047AB;">{cols}</div>'
            f'<div style="font-size:0.7rem;color:#86868B;margin-top:0.2rem;">{T("metric_columns")}</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div style="background:#FFF;border:1px solid #E5E5E5;border-radius:14px;padding:1.2rem 1.4rem;">'
            f'<div style="font-size:2rem;font-weight:800;color:#0047AB;">{num_cols_count}</div>'
            f'<div style="font-size:0.7rem;color:#86868B;margin-top:0.2rem;">{T("metric_numeric")}</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div style="background:#FFF;border:1px solid #E5E5E5;border-radius:14px;padding:1.2rem 1.4rem;">'
            f'<div style="font-size:1.1rem;font-weight:700;color:#1D1D1F;margin-top:0.4rem;">{sync_time}</div>'
            f'<div style="font-size:0.7rem;color:#86868B;margin-top:0.2rem;">{T("metric_last_sync")}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

    qa, qb = st.columns(2)
    with qa:
        if st.button(T("btn_guide"), key="tile_guide", width="stretch"):
            st.session_state["show_guide_dialog"] = True
            st.rerun()
    with qb:
        report_html = generate_html_report()
        html_filename = f"Lab_Report_{datetime.now().strftime('%Y%m%d')}.html"
        st.download_button(
            T("btn_export_report"),
            data=report_html.encode("utf-8"),
            file_name=html_filename, mime="text/html",
            key="tile_report", width="stretch",
        )

    if st.session_state.get("show_guide_dialog", False):
        guide_dialog()

    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button(T("btn_data_studio"), key="tile_studio", width="stretch"):
            st.session_state["active_view"] = "data_studio"
            st.rerun()
    with c2:
        if st.button(T("btn_ai_diagnosis"), key="tile_ai", width="stretch"):
            st.session_state["active_view"] = "dashboard"
            st.rerun()
    with c3:
        if st.button(T("btn_visual"), key="tile_visual", width="stretch"):
            st.session_state["active_view"] = "visual"
            st.rerun()
    with c4:
        if st.button(T("btn_settings"), key="tile_settings", width="stretch"):
            st.session_state["active_view"] = "settings"
            st.rerun()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'<div class="recent-title">{T("recent_data")}</div>', unsafe_allow_html=True)
    if df is not None and not df.empty:
        st.dataframe(df.tail(5), width="stretch", hide_index=True)
    else:
        st.info(T("no_data_hint"))


def render_settings():
    if st.button(T("back_home"), key="back_settings"):
        st.session_state["active_view"] = "home"
        st.rerun()

    st.markdown(f'<span class="area-title">{T("settings_title")}</span>', unsafe_allow_html=True)

    st.subheader(T("project_info"))
    col_a, col_b = st.columns(2)
    with col_a:
        mat = st.text_input(
            T("material_name_label"),
            value=st.session_state.get("material_name", ""),
            key="settings_mat",
        )
        if mat != st.session_state.get("material_name"):
            st.session_state["material_name"] = mat
    with col_b:
        eq = st.text_input(
            T("equipment_name_label"),
            value=st.session_state.get("equipment_name", ""),
            key="settings_eq",
        )
        if eq != st.session_state.get("equipment_name"):
            st.session_state["equipment_name"] = eq

    st.divider()
    st.subheader(T("gemini_api_key"))
    api = st.text_input(
        T("api_key_label"),
        value=st.session_state.get("api_key", ""),
        type="password",
        key="settings_api",
    )
    if api != st.session_state.get("api_key"):
        st.session_state["api_key"] = api
    st.caption(T("api_key_hint"))


# --- Visual Analytics (独立页面) ---
def render_visual_page():
    """图表可视化独立页面。"""
    if st.button(T("back_home"), key="back_visual"):
        st.session_state["active_view"] = "home"
        st.rerun()

    st.markdown(f'<span class="area-title">{T("visual_title")}</span>', unsafe_allow_html=True)

    df = st.session_state["df"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    if len(numeric_cols) < 1:
        st.info(T("need_numeric_col"))
        return

    _none = T("none_option")
    c1, c2, c3 = st.columns(3)
    with c1:
        x_col = st.selectbox(T("x_axis"), options=all_cols, index=0, key="vis_x")
    with c2:
        y_col = st.selectbox(
            T("y_axis"), options=numeric_cols,
            index=min(len(numeric_cols) - 1, 0), key="vis_y",
        )
    with c3:
        color_col = st.selectbox(
            T("color_map"), options=[_none] + all_cols, index=0, key="vis_color",
        )

    color = color_col if color_col != _none else None
    try:
        fig_sc = px.scatter(
            df, x=x_col, y=y_col, color=color, trendline="ols",
            template="simple_white",
            color_continuous_scale="Blues",
        )
        fig_sc.update_layout(
            margin=dict(l=40, r=20, t=30, b=40),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_sc, width="stretch")
    except Exception:
        fig_sc = px.scatter(df, x=x_col, y=y_col, color=color, template="simple_white")
        fig_sc.update_layout(
            margin=dict(l=40, r=20, t=30, b=40),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig_sc, width="stretch")

    st.markdown(f"**{T('correlation_heatmap')}**")
    corr = df[numeric_cols].corr()
    fig_hm = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0.0, "#F0F4FF"], [0.5, "#6B9FD6"], [1.0, "#0047AB"]],
        zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
    ))
    fig_hm.update_layout(
        margin=dict(l=60, r=20, t=30, b=60), height=420,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    )
    st.plotly_chart(fig_hm, width="stretch")


# --- Tab: 数据工作台 (Data Studio) ---
def render_data_studio():
    if st.button(T("back_home"), key="back_studio"):
        st.session_state["active_view"] = "home"
        st.rerun()

    st.markdown(f'<span class="area-title">{T("data_studio_title")}</span>', unsafe_allow_html=True)

    df = st.session_state["df"]

    # --- 第一层: 数据存取区 (IO Zone) — 云端 + 本地统一置顶 ---
    with st.container(border=True):
        st.markdown(
            f'<div class="area-title"><span class="area-number">{T("io_zone_title")}</span>{T("io_zone_label")}</div>',
            unsafe_allow_html=True,
        )

        # ---- 云端同步 (仅管理员可见) ----
        if st.session_state.get("user_role") == "admin" and GSHEETS_AVAILABLE:
            sc1, sc2 = st.columns(2)
            with sc1:
                if st.button(
                    T("pull_from_cloud"),
                    width="stretch", key="gs_pull",
                ):
                    try:
                        st.cache_data.clear()
                        conn = st.connection("gsheets", type=GSheetsConnection)
                        df_cloud = conn.read()
                        df_cloud = df_cloud.dropna(how="all").reset_index(drop=True)
                        for col in df_cloud.columns:
                            if pd.api.types.is_numeric_dtype(df_cloud[col]):
                                df_cloud[col] = df_cloud[col].fillna(0)
                            else:
                                df_cloud[col] = df_cloud[col].fillna("")
                        if df_cloud.empty or len(df_cloud.columns) == 0:
                            st.warning(T("cloud_empty_warning"))
                        else:
                            st.session_state["df"] = df_cloud
                            st.session_state["input_columns"] = []
                            st.session_state["output_columns"] = []
                            st.session_state["target_values"] = {}
                            st.session_state["custom_column_order"] = df_cloud.columns.tolist()
                            db_save(df_cloud)
                            _clear_editor_widget()
                            st.session_state["last_sync_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                            st.success(
                                T("pull_success", rows=len(df_cloud), cols=len(df_cloud.columns))
                            )
                            st.rerun()
                    except Exception as e:
                        st.error(T("pull_failed", error=str(e)))

            with sc2:
                if st.button(
                    T("push_to_cloud"),
                    width="stretch", key="gs_push", type="primary",
                ):
                    try:
                        current_df = st.session_state["df"]
                        db_save(current_df)
                        conn = st.connection("gsheets", type=GSheetsConnection)
                        conn.update(data=current_df.fillna(""))
                        st.session_state["last_sync_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                        st.toast(T("push_toast"))
                    except Exception as e:
                        st.error(T("push_failed", error=str(e)))
                        st.markdown(T("push_troubleshoot"))
            st.caption(T("cloud_caption"))
            st.markdown(
                '<hr style="border:none; border-top:1px solid #E5E5E5; margin:0.5rem 0;">',
                unsafe_allow_html=True,
            )

        # ---- 本地文件存取 (所有人可见) ----
        lc1, lc2 = st.columns(2)
        with lc1:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                T("download_csv"), csv_bytes, "jpz_backup.csv",
                "text/csv", width="stretch",
            )
        with lc2:
            uploaded = st.file_uploader(
                T("upload_csv"), type=["csv"], key="csv_uploader",
            )
            if uploaded is not None:
                try:
                    preview_df = pd.read_csv(uploaded)
                    st.info(
                        T("detected_rows_cols", rows=len(preview_df), cols=len(preview_df.columns))
                    )
                    if st.button(T("confirm_import"), key="confirm_csv_import"):
                        st.session_state["df"] = preview_df
                        st.session_state["input_columns"] = []
                        st.session_state["output_columns"] = []
                        st.session_state["target_values"] = {}
                        st.session_state["custom_column_order"] = preview_df.columns.tolist()
                        db_save(preview_df)
                        _clear_editor_widget()
                        st.rerun()
                except Exception as e:
                    st.error(T("csv_parse_failed", error=str(e)))

    # --- 第二层: 表格结构与定义 (Schema & Definition) ---
    st.markdown(
        f'<div class="area-title">'
        f'<span class="area-number">{T("schema_zone_title")}</span>{T("schema_zone_label")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    schema_left, schema_right = st.columns([1, 2])

    # ---- 左侧: Popover 列管理 ----
    with schema_left:
        cols_list = df.columns.tolist()

        with st.popover(T("table_structure_btn"), help=T("table_structure_help")):
            pop_tabs = st.tabs([T("tab_add"), T("tab_rename"), T("tab_delete"), T("tab_formula"), T("tab_reorder")])

            with pop_tabs[0]:
                ncn = st.text_input(T("new_col_name"), key="new_col_name", placeholder=T("new_col_name_placeholder"))
                _dtype_opts = [T("dtype_number"), T("dtype_text")]
                col_dtype = st.radio(
                    T("data_type"), _dtype_opts,
                    index=0, key="new_col_dtype", horizontal=True,
                )
                if st.button(T("create_now"), key="add_col_btn", type="primary", width="stretch"):
                    name = (ncn or "").strip()
                    if name and name not in df.columns:
                        new = df.copy()
                        if col_dtype == _dtype_opts[1]:  # Text
                            new[name] = ""
                        else:
                            new[name] = 0.0
                        st.session_state["df"] = new
                        st.session_state.get("custom_column_order", []).append(name)
                        _sync_column_order()
                        db_save(new)
                        _clear_editor_widget()
                        st.rerun()
                    elif not name:
                        st.warning(T("enter_col_name"))
                    else:
                        st.warning(T("col_exists"))

            with pop_tabs[1]:
                old_name = st.selectbox(T("select_column"), cols_list, key="rename_select")
                new_name_input = st.text_input(T("new_name"), key="rename_input", placeholder=T("new_name_placeholder"))
                if st.button(T("confirm_rename"), key="rename_btn", width="stretch"):
                    nn = (new_name_input or "").strip()
                    if nn and nn != old_name:
                        st.session_state["df"] = df.rename(columns={old_name: nn})
                        st.session_state["input_columns"] = [
                            nn if c == old_name else c for c in st.session_state["input_columns"]
                        ]
                        st.session_state["output_columns"] = [
                            nn if c == old_name else c for c in st.session_state["output_columns"]
                        ]
                        tv = st.session_state["target_values"]
                        if old_name in tv:
                            tv[nn] = tv.pop(old_name)
                        st.session_state["custom_column_order"] = [
                            nn if c == old_name else c
                            for c in st.session_state.get("custom_column_order", [])
                        ]
                        db_save(st.session_state["df"])
                        _clear_editor_widget()
                        st.rerun()
                    elif nn == old_name:
                        st.warning(T("same_name_warning"))
                    else:
                        st.warning(T("enter_valid_name"))

            with pop_tabs[2]:
                del_cols = st.multiselect(T("select_cols_delete"), cols_list, key="del_cols_select")
                if st.button(T("confirm_delete"), key="del_cols_btn", type="primary", width="stretch"):
                    if del_cols:
                        st.session_state["df"] = df.drop(columns=del_cols, errors="ignore")
                        st.session_state["input_columns"] = [
                            c for c in st.session_state["input_columns"] if c not in del_cols
                        ]
                        st.session_state["output_columns"] = [
                            c for c in st.session_state["output_columns"] if c not in del_cols
                        ]
                        for c in del_cols:
                            st.session_state["target_values"].pop(c, None)
                        st.session_state["custom_column_order"] = [
                            c for c in st.session_state.get("custom_column_order", [])
                            if c not in del_cols
                        ]
                        db_save(st.session_state["df"])
                        _clear_editor_widget()
                        st.rerun()
                    else:
                        st.warning(T("select_cols_first"))

            with pop_tabs[3]:
                formula_col_name = st.text_input(T("formula_col_name"), key="formula_col_name", placeholder=T("formula_expr_placeholder"))
                formula_expr = st.text_input(
                    T("formula_expr"), key="formula_expr",
                    placeholder=T("formula_expr_placeholder"),
                )
                st.caption(T("formula_help"))
                if st.button(T("calc_and_add"), key="calc_col_btn", type="primary", width="stretch"):
                    f_name = (formula_col_name or "").strip()
                    f_expr = (formula_expr or "").strip()
                    if not f_name:
                        st.error(T("enter_formula_col_name"))
                    elif not f_expr:
                        st.error(T("enter_formula_expr"))
                    elif f_name in df.columns:
                        st.error(T("col_already_exists", name=f_name))
                    else:
                        try:
                            result = df.eval(f_expr)
                            new_df = df.copy()
                            new_df[f_name] = result
                            st.session_state["df"] = new_df
                            st.session_state.get("custom_column_order", []).append(f_name)
                            _sync_column_order()
                            db_save(new_df)
                            _clear_editor_widget()
                            st.rerun()
                        except Exception as e1:
                            try:
                                alias_map = {col: f"_c{i}_" for i, col in enumerate(df.columns)}
                                safe_df = df.rename(columns=alias_map)
                                safe_expr = f_expr
                                for col in sorted(df.columns, key=len, reverse=True):
                                    safe_expr = safe_expr.replace(f"`{col}`", alias_map[col])
                                result = safe_df.eval(safe_expr)
                                new_df = df.copy()
                                new_df[f_name] = result
                                st.session_state["df"] = new_df
                                st.session_state.get("custom_column_order", []).append(f_name)
                                _sync_column_order()
                                db_save(new_df)
                                _clear_editor_widget()
                                st.rerun()
                            except Exception:
                                st.error(
                                    T("formula_failed", error=str(e1), cols=", ".join(df.columns.tolist()))
                                )

            with pop_tabs[4]:
                _sync_column_order()
                current_order = st.session_state["custom_column_order"]

                if SORTABLES_AVAILABLE:
                    st.caption(T("drag_sort_caption"))
                    sorted_cols = sort_items(current_order, direction="vertical")
                    if sorted_cols != current_order:
                        st.session_state["custom_column_order"] = sorted_cols
                        st.session_state["df"] = st.session_state["df"][sorted_cols]
                        db_save(st.session_state["df"])
                        _clear_editor_widget()
                        st.rerun()
                else:
                    st.caption(T("need_sortables"))
                    st.code("pip install streamlit-sortables", language="bash")
                    st.caption(T("install_sortables_hint"))
                    new_order = st.multiselect(
                        T("col_order_label"),
                        options=current_order,
                        default=current_order,
                        key="reorder_cols",
                        label_visibility="collapsed",
                    )
                    if st.button(T("apply_order"), key="apply_order_btn", type="primary", width="stretch"):
                        if set(new_order) == set(current_order) and len(new_order) == len(current_order):
                            st.session_state["custom_column_order"] = new_order
                            st.session_state["df"] = st.session_state["df"][new_order]
                            db_save(st.session_state["df"])
                            _clear_editor_widget()
                            st.rerun()
                        else:
                            st.warning(
                                T("keep_all_cols_warning", n=len(current_order))
                            )

        # 样品图片 (可选)
        with st.expander(T("sample_image_expander")):
            st.caption(T("sample_image_caption"))
            up_img = st.file_uploader(
                T("upload_image"), type=["png", "jpg", "jpeg"],
                key="img_uploader", label_visibility="collapsed",
            )
            if up_img is not None:
                img = Image.open(up_img)
                st.image(img, caption=T("uploaded_prefix", name=up_img.name), width="stretch")
                st.session_state["sample_image"] = up_img.getvalue()
                st.session_state["sample_image_name"] = up_img.name
            elif st.session_state.get("sample_image"):
                img = Image.open(io.BytesIO(st.session_state["sample_image"]))
                st.image(
                    img,
                    caption=T("saved_prefix", name=st.session_state.get('sample_image_name', '')),
                    width="stretch",
                )
                if st.button(T("remove_image"), key="rm_img_btn"):
                    st.session_state["sample_image"] = None
                    st.session_state["sample_image_name"] = None
                    st.rerun()

    # ---- 右侧: 语义映射与目标设定 ----
    with schema_right:
        st.markdown(f"""
        <div class="mapping-info">
            {T("mapping_info_html")}
        </div>
        """, unsafe_allow_html=True)

        all_cols = st.session_state["df"].columns.tolist()
        mc1, mc2 = st.columns(2)
        with mc1:
            inp = st.multiselect(
                T("inputs_label"), all_cols,
                default=[c for c in st.session_state["input_columns"] if c in all_cols],
                help=T("inputs_help"),
                key="sel_inputs",
            )
        with mc2:
            avail_out = [c for c in all_cols if c not in inp]
            out = st.multiselect(
                T("outputs_label"), avail_out,
                default=[c for c in st.session_state["output_columns"] if c in avail_out],
                help=T("outputs_help"),
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

        # 动态目标设定 (带记忆)
        memory = dict(st.session_state.get("target_memory", {}))
        tvs = dict(st.session_state.get("target_values", {}))
        if out:
            st.markdown(
                f'<div class="target-section">'
                f'<div class="target-section-title">{T("target_section_title")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            per_row = min(len(out), 3)
            for i in range(0, len(out), per_row):
                cols = st.columns(per_row)
                for j, cn in enumerate(out[i: i + per_row]):
                    with cols[j]:
                        if cn in st.session_state["df"].columns:
                            avg = pd.to_numeric(
                                st.session_state["df"][cn], errors="coerce"
                            ).mean()
                            mx = pd.to_numeric(
                                st.session_state["df"][cn], errors="coerce"
                            ).max()
                        else:
                            avg, mx = 0.0, 0.0
                        # 优先从 memory 恢复, 其次从 tvs
                        remembered = memory.get(cn, tvs.get(cn, ""))
                        val = st.text_input(
                            T("target_input_label", col=cn),
                            value=str(remembered) if remembered else "",
                            placeholder=T("target_placeholder", avg=avg),
                            help=T("target_help", avg=avg, mx=mx),
                            key=f"tgt_{cn}",
                        )
                        tvs[cn] = val
                        # 写入持久记忆 (无论是否在当前 Outputs 中)
                        memory[cn] = val
                        st.caption(T("target_caption", avg=avg, mx=mx))

        st.session_state["target_memory"] = memory
        st.session_state["target_values"] = {k: v for k, v in tvs.items() if k in out}

    # --- 配置状态栏 (Active Configuration) ---
    _inp = st.session_state.get("input_columns", [])
    _out = st.session_state.get("output_columns", [])
    _tvs = st.session_state.get("target_values", {})
    _active_goals = {k: v for k, v in _tvs.items() if v}

    if _inp or _out or _active_goals:
        cfg_parts = []
        if _inp:
            inp_badges = " ".join(
                f'<span class="mapping-tag input">{c}</span>' for c in _inp
            )
            cfg_parts.append(f"<b>Inputs:</b> {inp_badges}")
        if _out:
            out_badges = " ".join(
                f'<span class="mapping-tag output">{c}</span>' for c in _out
            )
            cfg_parts.append(f"<b>Outputs:</b> {out_badges}")
        if _active_goals:
            goal_str = ", ".join(f"{k} → {v}" for k, v in _active_goals.items())
            cfg_parts.append(f"<b>Goals:</b> {goal_str}")
        st.markdown(
            '<div class="hint-box">' + " &nbsp;|&nbsp; ".join(cfg_parts) + '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="hint-box" style="color:#86868B;">{T("config_waiting")}</div>',
            unsafe_allow_html=True,
        )

    # --- 第三层: 数据表格 (The Grid) ---
    st.markdown(
        f'<div class="area-title">'
        f'<span class="area-number">{T("grid_zone_title")}</span>{T("grid_zone_label")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    status_area = st.empty()
    save_status = st.session_state.pop("_save_status", None)
    if save_status == "saved":
        status_area.markdown(
            f'<div class="hint-box"><strong>{T("status_label")}</strong> {T("save_status_saved")}</div>',
            unsafe_allow_html=True,
        )
    else:
        status_area.markdown(
            f'<div class="hint-box">'
            f'<strong>{T("status_label")}</strong> {T("save_status_ready")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # 应用自定义列顺序
    _sync_column_order()
    col_order = st.session_state["custom_column_order"]
    df_ordered = st.session_state["df"][col_order]

    # 强制所有数值列为 float, 防止 int 导致小数点被截断
    for _nc in df_ordered.select_dtypes(include=["number"]).columns:
        df_ordered[_nc] = df_ordered[_nc].astype(float)
    st.session_state["df"] = df_ordered

    # 动态生成 column_config: 数值列允许高精度输入
    _col_cfg = {}
    for _nc in df_ordered.select_dtypes(include=["number"]).columns:
        _col_cfg[_nc] = st.column_config.NumberColumn(
            label=_nc,
            step=0.0001,
        )

    editor_ver = st.session_state.get("editor_version", 0)
    editor_key = f"editor_{editor_ver}"

    st.data_editor(
        df_ordered,
        num_rows="dynamic", width="stretch", height=420,
        key=editor_key, on_change=_on_editor_change,
        column_config=_col_cfg,
    )


# --- Visual Analytics Module ---
def _render_visual_analytics(df: pd.DataFrame):
    """散点图 + 相关性热力图，放在 Dashboard 的表格下方、AI 报告上方。"""
    if df.empty or len(df.columns) < 2:
        return

    # 筛选数值列
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 1:
        return

    all_cols = df.columns.tolist()

    _none = T("none_option")
    with st.expander(T("visual_analytics_expander"), expanded=True):

        # ---- 交互控件: X / Y / Color ----
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            x_col = st.selectbox(
                T("x_axis"), all_cols,
                index=0,
                key="va_x",
            )
        with ctrl2:
            y_default = min(len(num_cols) - 1, 0)
            y_col = st.selectbox(
                T("y_axis"), num_cols,
                index=y_default,
                key="va_y",
            )
        with ctrl3:
            color_options = [_none] + all_cols
            color_sel = st.selectbox(
                T("color_map"), color_options,
                index=0,
                key="va_color",
            )

        color_arg = color_sel if color_sel != _none else None

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
                font=dict(color="#1D1D1F"),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
            )
            # 趋势线颜色统一为强调蓝
            for trace in fig_scatter.data:
                if hasattr(trace, "mode") and trace.mode == "lines":
                    trace.line.color = ACCENT

            st.plotly_chart(fig_scatter, width="stretch")
        except Exception as exc:
            st.warning(T("scatter_failed", error=str(exc)))

        # ---- 相关性热力图 ----
        if len(num_cols) >= 2:
            st.markdown(T("correlation_heatmap_title"))
            corr = df[num_cols].corr()

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale=[
                    [0.0, "#F0F4FF"],
                    [0.5, "#6B9FD6"],
                    [1.0, "#0047AB"],
                ],
                zmin=-1, zmax=1,
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont=dict(size=12, color="#1D1D1F"),
                hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
                colorbar=dict(title="r"),
            ))
            fig_heatmap.update_layout(
                height=max(320, 50 * len(num_cols)),
                margin=dict(t=30, b=30, l=80, r=30),
                xaxis=dict(tickangle=-40),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#1D1D1F"),
            )
            st.plotly_chart(fig_heatmap, width="stretch")

        st.caption(T("visual_tip"))


# --- Report Generation (HTML) ---
def generate_html_report() -> str:
    """生成完整 HTML 报告，模拟 A4 纸排版，浏览器直接打开即可阅读。"""
    now = datetime.now()
    role = st.session_state.get("user_role", "guest").upper()
    _ns = T("report_not_specified")
    mat = st.session_state.get("material_name", "") or _ns
    eqp = st.session_state.get("equipment_name", "") or _ns
    inp = st.session_state.get("input_columns", [])
    out = st.session_state.get("output_columns", [])
    tvs = st.session_state.get("target_values", {})
    df = st.session_state.get("df", pd.DataFrame())
    ai_result = st.session_state.get("ai_result")

    # ---- 数据表 HTML ----
    if not df.empty:
        data_table = df.to_html(
            index=False, classes="styled-table", border=0,
        )
        desc_table = df.describe().to_html(
            classes="styled-table", border=0,
        )
    else:
        data_table = f"<p>{T('report_no_data')}</p>"
        desc_table = f"<p>{T('report_no_data')}</p>"

    # ---- 目标表 HTML ----
    active_t = {k: v for k, v in tvs.items() if v}
    if active_t:
        target_rows = ""
        for cn, tv in active_t.items():
            avg_val = ""
            if cn in df.columns and not df.empty:
                avg_val = f"{pd.to_numeric(df[cn], errors='coerce').mean():.2f}"
            target_rows += (
                f"<tr><td>{cn}</td><td>{tv}</td>"
                f"<td>{avg_val}</td></tr>\n"
            )
        target_html = (
            '<table class="styled-table">'
            f"<thead><tr><th>{T('report_metric')}</th><th>{T('report_target_value')}</th>"
            f"<th>{T('report_current_avg')}</th></tr></thead>"
            f"<tbody>{target_rows}</tbody></table>"
        )
    else:
        target_html = f"<p>{T('report_no_targets')}</p>"

    # ---- AI 分析内容 ----
    if ai_result and ai_result.get("success"):
        ai_text = ai_result.get("full_response", "")
        # 简单将换行转为 <br>，段落用 <p>
        ai_html = ""
        for para in ai_text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if para.startswith("###"):
                heading = para.lstrip("# ").strip()
                ai_html += f"<h3>{heading}</h3>\n"
            elif para.startswith("##"):
                heading = para.lstrip("# ").strip()
                ai_html += f"<h3>{heading}</h3>\n"
            elif para.startswith("#"):
                heading = para.lstrip("# ").strip()
                ai_html += f"<h3>{heading}</h3>\n"
            else:
                ai_html += f"<p>{para.replace(chr(10), '<br>')}</p>\n"
    elif ai_result and not ai_result.get("success"):
        ai_html = f"<p style='color:#c00;'>{T('report_ai_failed', error=ai_result.get('error', 'unknown'))}</p>"
    else:
        ai_html = f"<p style='color:#86868B;'>{T('report_ai_pending')}</p>"

    # ---- 完整 HTML ----
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{T("report_title")} — {now.strftime('%Y-%m-%d')}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
        background: #F0F0F0; color: #333; line-height: 1.7;
        padding: 20px;
    }}
    .paper {{
        width: 210mm; max-width: 100%;
        margin: 0 auto; background: #FFFFFF;
        padding: 20mm; border-radius: 4px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
    }}
    h1 {{
        font-size: 1.8rem; font-weight: 800; color: #1a1a1a;
        border-bottom: 3px solid #0047AB; padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }}
    h1 .accent {{ color: #0047AB; }}
    h2 {{
        font-size: 1.2rem; font-weight: 700; color: #0047AB;
        margin-top: 2rem; margin-bottom: 0.8rem;
        padding-bottom: 0.3rem; border-bottom: 1px solid #E0E0E0;
    }}
    h3 {{
        font-size: 1.05rem; font-weight: 600; color: #333;
        margin-top: 1.2rem; margin-bottom: 0.5rem;
    }}
    p {{ margin-bottom: 0.8rem; }}
    .meta {{
        display: flex; gap: 2rem; flex-wrap: wrap;
        margin-bottom: 1.5rem; font-size: 0.9rem; color: #555;
    }}
    .meta-item {{ }}
    .meta-label {{
        font-size: 0.75rem; color: #888; text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .meta-value {{ font-weight: 600; color: #333; }}
    .styled-table {{
        width: 100%; border-collapse: collapse;
        margin: 0.8rem 0 1.2rem 0; font-size: 0.9rem;
    }}
    .styled-table th {{
        background: #0047AB; color: #FFFFFF;
        padding: 0.6rem 0.8rem; text-align: left;
        font-weight: 600;
    }}
    .styled-table td {{
        padding: 0.5rem 0.8rem; border-bottom: 1px solid #E8E8E8;
    }}
    .styled-table tbody tr:nth-child(even) {{
        background: #F8FAFF;
    }}
    .styled-table tbody tr:hover {{
        background: #EEF4FF;
    }}
    .ai-section {{
        background: #FAFBFF; border: 1px solid #D0E0F5;
        border-left: 4px solid #0047AB; border-radius: 6px;
        padding: 1.2rem 1.5rem; margin-top: 0.8rem;
    }}
    footer {{
        text-align: center; color: #AAA; font-size: 0.8rem;
        margin-top: 3rem; padding-top: 1rem;
        border-top: 1px solid #E0E0E0;
    }}
    @media print {{
        body {{ background: #FFF; padding: 0; }}
        .paper {{ box-shadow: none; padding: 15mm; width: 100%; }}
    }}
</style>
</head>
<body>
<div class="paper">

    <h1><span class="accent">JPZ</span> {T("report_title")}</h1>

    <div class="meta">
        <div class="meta-item">
            <div class="meta-label">{T("report_generated")}</div>
            <div class="meta-value">{now.strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">{T("report_identity")}</div>
            <div class="meta-value">{role}</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">{T("report_material")}</div>
            <div class="meta-value">{mat}</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">{T("report_equipment")}</div>
            <div class="meta-value">{eqp}</div>
        </div>
    </div>

    <h2>{T("report_section_mapping")}</h2>
    <p>
        <strong>{T("report_inputs_label")}</strong> {', '.join(inp) if inp else _ns}<br>
        <strong>{T("report_outputs_label")}</strong> {', '.join(out) if out else _ns}
    </p>

    <h2>{T("report_section_targets")}</h2>
    {target_html}

    <h2>{T("report_section_data", rows=len(df), cols=len(df.columns))}</h2>
    {data_table}

    <h2>{T("report_section_stats")}</h2>
    {desc_table}

    <h2>{T("report_section_ai")}</h2>
    <div class="ai-section">
        {ai_html}
    </div>

    <footer>
        {T("report_footer")} &mdash; {now.strftime('%Y-%m-%d')}
    </footer>

</div>
</body>
</html>"""

    return html


# --- Tab: 智能仪表盘 (Dashboard) ---
def render_dashboard():
    if st.button(T("back_home"), key="back_dashboard"):
        st.session_state["active_view"] = "home"
        st.rerun()

    st.markdown(f'<span class="area-title">{T("dashboard_title")}</span>', unsafe_allow_html=True)

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
                    <div class="project-label">{T("research_project")}</div>
                    <div class="project-value">{mat or '—'}</div>
                </div>
                <div>
                    <div class="project-label">{T("equipment_process")}</div>
                    <div class="project-value">{eqp or '—'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- 量化目标卡片 ----
    active_t = {k: v for k, v in tvs.items() if v}
    if active_t:
        st.markdown(T("quantitative_targets"))
        t_cols = st.columns(len(active_t))
        for idx, (cn, tv) in enumerate(active_t.items()):
            if cn in df.columns:
                avg = pd.to_numeric(df[cn], errors="coerce").mean()
                with t_cols[idx]:
                    st.markdown(f"""
                    <div class="target-card">
                        <div class="target-label">{cn}</div>
                        <div class="target-value">{T("target_prefix", val=tv)}</div>
                        <div class="current-value">{T("current_avg_prefix", avg=avg)}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ---- 补充分析要求 (Custom Prompt) ----
    custom_req = st.text_area(
        T("custom_prompt_label"),
        value="",
        placeholder=T("custom_prompt_placeholder"),
        height=80,
        key="custom_ai_prompt",
    )

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    analyze_btn = st.button(T("btn_ai_analysis"), type="primary", width="stretch")

    if analyze_btn:
        key = st.session_state.get("api_key", "")
        if not key:
            st.warning(T("warning_no_api_key"))
        elif df.empty:
            st.warning(T("warning_no_data"))
        else:
            img_bytes = st.session_state.get("sample_image")
            spinner = (
                T("spinner_with_image")
                if img_bytes
                else T("spinner_no_image")
            )
            with st.spinner(spinner):
                result = analyze_with_ai(
                    df, mat, eqp, inp, out, tvs, key, img_bytes,
                    custom_prompt=custom_req,
                )
            st.session_state["ai_result"] = result

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ---- 数据摘要 ----
    img_bytes = st.session_state.get("sample_image")
    img_status = T("img_uploaded") if img_bytes else T("img_none")
    st.markdown(f"""
    <div class="data-summary">
        <span class="summary-item">
            <span class="summary-label">{T("summary_experiments")}</span><br>
            <span class="summary-value">{len(df)}</span>
        </span>
        <span class="summary-item">
            <span class="summary-label">{T("summary_param_cols")}</span><br>
            <span class="summary-value">{len(inp)}</span>
        </span>
        <span class="summary-item">
            <span class="summary-label">{T("summary_result_cols")}</span><br>
            <span class="summary-value">{len(out)}</span>
        </span>
        <span class="summary-item">
            <span class="summary-label">{T("summary_targets_set")}</span><br>
            <span class="summary-value">{len(active_t)}</span>
        </span>
        <span class="summary-item">
            <span class="summary-label">{T("summary_sample_img")}</span><br>
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
        st.markdown(T("data_preview"))
        if inp or out:
            st.dataframe(
                style_dataframe(df, inp, out),
                width="stretch", height=280,
            )
        else:
            st.dataframe(df, width="stretch", height=280)

    with col_chart:
        st.markdown(T("trend_and_target"))
        st.plotly_chart(
            create_trend_chart(df, out, tvs), width="stretch",
        )

    if col_img is not None and img_bytes:
        with col_img:
            st.markdown(T("sample_image_label"))
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
                T("ai_title_img_analysis")
                if has_img
                else T("ai_title_no_img_analysis")
            )
            title_r = (
                T("ai_title_img_suggestion")
                if has_img
                else T("ai_title_no_img_suggestion")
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
            with st.expander(T("view_full_report")):
                st.markdown(ai_result.get("full_response", ""))
        else:
            st.error(
                T("ai_analysis_failed", error=ai_result.get('error', 'unknown'))
            )
    else:
        st.markdown(
            f'<div class="placeholder-box">'
            f'{T("dashboard_placeholder")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- 实验数据追问 (Experiment Q&A) — 默认折叠 ---
    st.divider()

    with st.expander(T("experiment_qa"), expanded=False):
        chat_area = st.container(height=400)
        with chat_area:
            history = st.session_state.get("experiment_chat_history", [])
            if not history:
                st.markdown(
                    f'<div style="text-align:center; color:#86868B; padding:2rem 0;">'
                    f'{T("qa_no_history")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            for msg in history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        exp_input = st.chat_input(
            T("qa_input_placeholder"),
            key="exp_chat_input",
        )

        if exp_input:
            api_key = st.session_state.get("api_key", "")
            st.session_state["experiment_chat_history"].append(
                {"role": "user", "content": exp_input}
            )
            if not api_key:
                answer = T("qa_no_api_key")
            else:
                csv_data = st.session_state["df"].to_csv(index=False)
                _mat = st.session_state.get("material_name", "")
                _inp = st.session_state.get("input_columns", [])
                _out = st.session_state.get("output_columns", [])

                _not_specified = T("report_not_specified")
                sys_prompt = T(
                    "qa_system_prompt",
                    material=_mat or _not_specified,
                    inputs=", ".join(_inp) or _not_specified,
                    outputs=", ".join(_out) or _not_specified,
                    csv=csv_data,
                )

                recent = st.session_state["experiment_chat_history"][-20:]
                conversation = ""
                for m in recent[:-1]:
                    label = T("qa_label_user") if m["role"] == "user" else T("qa_label_assistant")
                    conversation += f"{label}: {m['content']}\n\n"
                conversation += f"{T('qa_label_user')}: {exp_input}"

                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(
                        "gemini-2.0-flash", system_instruction=sys_prompt,
                    )
                    response = model.generate_content(conversation)
                    answer = response.text
                except Exception as e:
                    answer = T("qa_ai_error", error=str(e))

            st.session_state["experiment_chat_history"].append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()



# --- Main — Portal Routing ---
def main():
    init_session_state()

    # 顶部 Header + 统计栏
    render_header()
    render_stats_bar()

    # 根据 active_view 路由
    view = st.session_state.get("active_view", "home")

    if view == "home":
        render_portal_home()

    elif view == "data_studio":
        render_data_studio()

    elif view == "dashboard":
        render_dashboard()

    elif view == "visual":
        render_visual_page()

    elif view == "settings":
        render_settings()

    else:
        render_portal_home()

    # 页脚
    st.markdown(
        f'<div class="app-footer">'
        f'{T("footer_text")}'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

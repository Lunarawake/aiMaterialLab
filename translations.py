"""
Internationalization (i18n) translations for JPZ Platform.
Supports: English ('en') and Chinese ('cn').
"""

TRANSLATIONS = {
    # =========================================================================
    #  ENGLISH
    # =========================================================================
    "en": {
        # --- Page / Global ---
        "page_title": "JPZ Platform",
        "logo_suffix": " Platform",
        "footer_text": "JPZ Platform | Materials R&D Data Management & Intelligent Analysis",

        # --- Header / Auth ---
        "badge_admin": "ADMIN",
        "badge_guest": "GUEST",
        "popover_admin": ":material/account_circle: Admin",
        "popover_guest": ":material/account_circle: Not Logged In",
        "logged_in_as_admin": "Logged in as **Administrator**",
        "cloud_sync_enabled": "Google Sheets cloud sync enabled",
        "logout": "Log Out",
        "admin_login": "**Admin Login**",
        "unlock_gsheets": "Unlock Google Sheets cloud read / write",
        "password": "Password",
        "password_placeholder": "Enter admin password",
        "login": "Log In",
        "wrong_password": "Incorrect password",
        "no_password_config": (
            "Admin password not configured. Please add to "
            "`.streamlit/secrets.toml`:\n\n"
            '```toml\n[general]\npassword = "your_password"\n```'
        ),

        # --- Stats Bar ---
        "stat_records": "RECORDS",
        "stat_columns": "COLUMNS",
        "stat_identity": "IDENTITY",
        "stat_storage": "STORAGE",
        "stat_cloud_sync": "CLOUD SYNC",
        "role_admin": "Admin",
        "role_guest": "Guest",
        "storage_active": "Auto-saving",
        "storage_disconnected": "Disconnected",

        # --- Language Switcher ---
        "lang_label": "Language / 语言",

        # --- Portal Home ---
        "welcome_title": "Welcome back to JPZ Platform",
        "welcome_subtitle": "Materials Science Data Management & Intelligent Analysis",
        "metric_records": "Records",
        "metric_columns": "Columns",
        "metric_numeric": "Numeric Params",
        "metric_last_sync": "Last Sync",
        "btn_guide": ":material/smart_toy:  Platform Guide",
        "btn_export_report": ":material/download:  Export Report",
        "btn_data_studio": ":material/table_chart:  Data Studio",
        "btn_ai_diagnosis": ":material/psychology:  AI Diagnosis",
        "btn_visual": ":material/insights:  Visualization",
        "btn_settings": ":material/tune:  Settings",
        "recent_data": "Recent Data",
        "no_data_hint": "No data yet. Go to Data Studio to add experiment records.",

        # --- Guide Dialog ---
        "guide_title": "JPZ Platform Assistant",
        "close": "Close",
        "guide_intro": (
            "I'm your platform assistant. You can ask any questions about "
            "how to use this platform, e.g.: How to add a new column? "
            "How to sync data? How to set target values?"
        ),
        "guide_no_history": "No conversation yet. Enter your question below.",
        "guide_input_placeholder": "Enter a question, e.g.: How to use formula columns?",
        "guide_no_api_key": (
            "Please configure Gemini API Key in Settings first "
            "to enable the AI assistant."
        ),
        "guide_ai_error": "AI response failed: {error}",
        "guide_system_prompt": (
            "You are an assistant for a lab data management platform (JPZ Platform).\n"
            "Platform features:\n"
            "- Home: Quick navigation cards to each module\n"
            "- Data Studio: Manage column structure (add/rename/delete/formula), "
            "semantic mapping (set Inputs/Outputs), target values, data editor, "
            "cloud sync (Pull/Push, admin only), CSV import/export\n"
            "- AI Diagnosis: Gemini-powered deep analysis, custom prompts, "
            "image analysis, experiment Q&A\n"
            "- Visualization: Scatter plots (with trendlines), correlation heatmaps\n"
            "- Settings: Project info, Gemini API Key configuration\n"
            "- Report Export: Download HTML lab reports from Home\n\n"
            "Please answer user questions about platform usage in clear, concise English."
        ),

        # --- Settings ---
        "back_home": ":material/arrow_back: Back to Home",
        "settings_title": "System Settings",
        "project_info": "Project Info",
        "material_name_label": "Material Name",
        "equipment_name_label": "Equipment / Process",
        "gemini_api_key": "Gemini API Key",
        "api_key_label": "API Key",
        "api_key_hint": (
            "API Key is stored in session memory only and will be "
            "cleared when you close the page."
        ),

        # --- Visual Page ---
        "visual_title": "Visualization",
        "need_numeric_col": "At least 1 numeric column is required to draw charts.",
        "x_axis": "X-Axis (text columns supported)",
        "y_axis": "Y-Axis (numeric only)",
        "color_map": "Color Map (optional)",
        "none_option": "(None)",
        "correlation_heatmap": "Correlation Heatmap",

        # --- Data Studio ---
        "data_studio_title": "Data Studio",
        "io_zone_title": "01",
        "io_zone_label": " Data I/O",
        "pull_from_cloud": "Pull from Cloud",
        "push_to_cloud": "Push to Cloud",
        "cloud_empty_warning": "Cloud sheet is empty, no overwrite performed.",
        "pull_success": "Pulled from cloud and overwrote local ({rows} rows x {cols} cols)",
        "pull_failed": "Cloud pull failed: {error}",
        "push_toast": "Cloud sync completed",
        "push_failed": "Cloud sync failed: {error}",
        "push_troubleshoot": (
            "**Troubleshooting**: Please check that the Google Sheet "
            "is shared with the Service Account email and has **Editor** permission."
        ),
        "cloud_caption": (
            "Pull = Cloud overwrites local | Push = Local overwrites cloud | "
            "Daily edits auto-save to local DB."
        ),
        "download_csv": "Download CSV Backup",
        "upload_csv": "Upload CSV",
        "detected_rows_cols": "Detected {rows} rows x {cols} columns",
        "confirm_import": "Confirm Import",
        "csv_parse_failed": "CSV parse failed: {error}",
        "schema_zone_title": "02",
        "schema_zone_label": " Schema & Semantic Mapping",
        "table_structure_btn": ":material/settings: Table Structure",
        "table_structure_help": "Add, rename, delete, reorder columns, or create formula columns",
        "tab_add": "Add",
        "tab_rename": "Rename",
        "tab_delete": "Delete",
        "tab_formula": "Formula",
        "tab_reorder": "Reorder",
        "new_col_name": "New Column Name",
        "new_col_name_placeholder": "Enter column name",
        "data_type": "Data Type",
        "dtype_number": "Number",
        "dtype_text": "Text",
        "create_now": "Create Now",
        "enter_col_name": "Please enter a column name.",
        "col_exists": "Column name already exists.",
        "select_column": "Select Column",
        "new_name": "New Name",
        "new_name_placeholder": "Enter new name",
        "confirm_rename": "Confirm Rename",
        "same_name_warning": "New and old names are the same.",
        "enter_valid_name": "Please enter a valid new name.",
        "select_cols_delete": "Select columns to delete",
        "confirm_delete": "Confirm Delete",
        "select_cols_first": "Please select columns to delete first.",
        "formula_col_name": "New Column Name",
        "formula_expr": "Formula",
        "formula_expr_placeholder": "e.g.: `col_A` / `col_B`",
        "formula_help": "Wrap column names with backticks `, supports +, -, *, / and parentheses.",
        "calc_and_add": "Calculate & Add",
        "enter_formula_col_name": "Please enter a new column name.",
        "enter_formula_expr": "Please enter a formula.",
        "col_already_exists": 'Column "{name}" already exists.',
        "formula_failed": (
            "Formula calculation failed: {error}\n\n"
            "**Correct format examples:**\n"
            "- `` `col_A` + `col_B` ``\n"
            "- `` `col_A` / `col_B` * 100 ``\n\n"
            "Available column names: {cols}"
        ),
        "drag_sort_caption": "Drag to reorder columns",
        "need_sortables": "Drag-sort requires streamlit-sortables",
        "install_sortables_hint": "Install and restart to use drag-sort. Fallback below:",
        "col_order_label": "Column order (drag tags to reorder)",
        "apply_order": "Apply Order",
        "keep_all_cols_warning": "Please keep all {n} columns, just reorder them.",
        "sample_image_expander": "Sample Image (optional)",
        "sample_image_caption": "Upload SEM / optical microscope images for AI morphology analysis",
        "upload_image": "Upload Image",
        "uploaded_prefix": "Uploaded: {name}",
        "saved_prefix": "Saved: {name}",
        "remove_image": "Remove Image",
        "mapping_info_html": (
            "<strong>Step 1:</strong> Select parameter columns (Inputs) and result columns (Outputs). "
            "<strong>Step 2:</strong> Set quantitative target values for result columns."
        ),
        "inputs_label": "Inputs (Parameters) — Blue",
        "inputs_help": "Variables you can control in experiments",
        "outputs_label": "Outputs (Results) — Orange",
        "outputs_help": "Target metrics to optimize",
        "target_section_title": "Set target values for each metric",
        "target_input_label": "[{col}] Target",
        "target_placeholder": "Avg {avg:.2f}",
        "target_help": "Current avg: {avg:.2f}, Best: {mx:.2f}",
        "target_caption": "Avg {avg:.2f} / Best {mx:.2f}",
        "config_waiting": (
            "Configuration: Waiting... "
            "(Set parameter columns, result columns, and target values above)"
        ),
        "grid_zone_title": "03",
        "grid_zone_label": " Data Grid",
        "status_label": "Status:",
        "save_status_saved": "All changes saved.",
        "save_status_ready": (
            "Local DB ready (auto-saving) — "
            "Edit to save, data persists on page refresh."
        ),

        # --- Trend Chart ---
        "no_output_for_trend": "Select Output columns in Data Studio to show trend chart",
        "target_annotation": "Target: {val}",
        "trend_chart_title": "Result Trends (dashed = target)",
        "experiment_number": "Experiment #",
        "value_label": "Value",

        # --- Dashboard ---
        "dashboard_title": "Smart Dashboard",
        "research_project": "Research Project",
        "equipment_process": "Equipment / Process",
        "quantitative_targets": "**Quantitative Targets**",
        "target_prefix": "Target: {val}",
        "current_avg_prefix": "Current avg: {avg:.2f}",
        "custom_prompt_label": "Additional Analysis Requirements (optional)",
        "custom_prompt_placeholder": (
            "e.g.: Focus on the nonlinear effect of temperature on hardness, "
            "or compare experiment group #3 vs #5..."
        ),
        "btn_ai_analysis": ":material/neurology: Start AI Deep Analysis",
        "warning_no_api_key": "Please configure Gemini API Key in Settings first.",
        "warning_no_data": "Please enter experiment data in Data Studio first.",
        "spinner_with_image": "AI is analyzing data, images, and target gaps...",
        "spinner_no_image": "AI is analyzing target gaps and generating suggestions...",
        "summary_experiments": "EXPERIMENTS",
        "summary_param_cols": "PARAM COLS",
        "summary_result_cols": "RESULT COLS",
        "summary_targets_set": "TARGETS SET",
        "summary_sample_img": "SAMPLE IMAGE",
        "img_uploaded": "Uploaded",
        "img_none": "None",
        "data_preview": "**Data Preview**",
        "trend_and_target": "**Result Trends & Targets**",
        "sample_image_label": "**Sample Image**",
        "ai_title_img_analysis": "Image Morphology & Data Correlation",
        "ai_title_no_img_analysis": "Target Gap Diagnosis & Mechanism Analysis",
        "ai_title_img_suggestion": "Morphology Improvement & Parameter Suggestions",
        "ai_title_no_img_suggestion": "Precise Parameter Suggestions & Expected Outcomes",
        "view_full_report": "View Full AI Report",
        "ai_analysis_failed": "AI analysis failed: {error}",
        "dashboard_placeholder": (
            "Set targets, then click 'AI Deep Analysis' for "
            "scientific insights and optimization suggestions"
        ),
        "experiment_qa": ":material/forum: Experiment Q&A",
        "qa_no_history": (
            "No conversation yet. Enter your questions about the experiment data below, "
            "and AI will answer based on the current dataset."
        ),
        "qa_input_placeholder": (
            "Enter a question, e.g.: Which experiment group has the lowest micropipe density? "
            "What is the relationship between temperature and growth rate?"
        ),
        "qa_no_api_key": "Please configure Gemini API Key in Settings first.",
        "qa_ai_error": "AI response failed: {error}",

        # --- Visual Analytics (embedded) ---
        "visual_analytics_expander": ":material/insights: Visual Analytics",
        "scatter_failed": "Scatter plot failed: {error}",
        "correlation_heatmap_title": "**Parameter Correlation Heatmap**",
        "visual_tip": (
            "Tip: Scatter plots show bivariate relationships (with OLS trendline if statsmodels installed); "
            "heatmaps show Pearson correlation coefficients between all numeric columns — "
            "values closer to ±1 indicate stronger linear correlation."
        ),

        # --- HTML Report ---
        "report_title": "JPZ Platform Lab Report",
        "report_generated": "Generated",
        "report_identity": "Identity",
        "report_material": "Material",
        "report_equipment": "Equipment / Process",
        "report_not_specified": "Not specified",
        "report_section_mapping": "1. Semantic Mapping",
        "report_inputs_label": "Parameter Columns (Inputs):",
        "report_outputs_label": "Result Columns (Outputs):",
        "report_section_targets": "2. Quantitative Targets",
        "report_metric": "Metric",
        "report_target_value": "Target Value",
        "report_current_avg": "Current Average",
        "report_no_targets": "(No quantitative targets set)",
        "report_section_data": "3. Experiment Data ({rows} rows x {cols} cols)",
        "report_no_data": "(No data)",
        "report_section_stats": "4. Statistical Summary",
        "report_section_ai": "5. AI Analysis Report",
        "report_ai_failed": "Analysis failed: {error}",
        "report_ai_pending": "AI analysis not yet performed",
        "report_footer": "Generated by JPZ Intelligent Assistant",

        # --- AI Analysis System Prompts ---
        "ai_system_prompt": (
            "You are a world-class materials scientist and process engineer.\n"
            "The user is researching [{material}].\n"
            "The equipment/process used: [{equipment}].\n\n"
            "Your task is to help the user achieve quantitative targets.\n"
            "1. Precisely identify the gap between current data and targets\n"
            "2. Explain bottlenecks based on physical/chemical principles\n"
            "3. Provide specific parameter suggestions to approach target values\n"
            "4. Honestly point out if a target is unrealistic{img_instr}{custom_instr}"
        ),
        "ai_img_instr": (
            "\n5. Carefully observe the user-uploaded sample microstructure image"
            "\n6. Analyze morphological features (grain size, cracks, pores, color anomalies, etc.)"
            "\n7. Correlate image observations with experimental parameters to infer "
            "process-morphology-performance causal relationships"
        ),
        "ai_custom_instr": (
            "\n\nThe user has specified the following analysis requirements, "
            "please provide targeted analysis based on data:\n"
            '"{prompt}"'
        ),
        "ai_target_line": (
            "- {col}: target={tv}, current avg={avg:.2f}, "
            "best={best:.2f}, gap={gap:.2f} ({pct:+.1f}%)"
        ),
        "ai_target_line_no_target": (
            "- {col}: no target set, current avg={avg:.2f}, best={best:.2f}"
        ),
        "ai_no_target": "(User has not set specific targets)",
        "ai_input_not_specified": "(Not specified)",
        "ai_user_prompt_img": (
            "## Experiment Data\n```csv\n{csv}\n```\n\n"
            "## Column Description\n- Experimental parameter columns (adjustable variables): {inputs}\n\n"
            "## User's Quantitative Targets\n{targets}\n\n"
            "## Sample Image\nThe user uploaded a sample microstructure image. Please observe carefully.\n\n---\n\n"
            "Please analyze with the following structure:\n\n"
            "### 1. Image Morphology Analysis\n### 2. Data-Image Correlation\n"
            "### 3. Bottleneck Mechanism Analysis\n### 4. Precise Parameter Suggestions\n"
            "### 5. Expected Outcome Assessment"
        ),
        "ai_user_prompt_no_img": (
            "## Experiment Data\n```csv\n{csv}\n```\n\n"
            "## Column Description\n- Experimental parameter columns (adjustable variables): {inputs}\n\n"
            "## User's Quantitative Targets\n{targets}\n\n---\n\n"
            "Please analyze with the following structure:\n\n"
            "### 1. Target Gap Diagnosis\n### 2. Bottleneck Mechanism Analysis\n"
            "### 3. Precise Parameter Suggestions\n### 4. Expected Outcome Assessment"
        ),
        "ai_split_marker_img": "### 4",
        "ai_split_marker_no_img": "### 3",

        # --- Experiment Q&A System Prompt ---
        "qa_system_prompt": (
            "You are a materials science expert and experiment data analyst.\n"
            "Research material: {material}\n"
            "Parameter columns (Inputs): {inputs}\n"
            "Result columns (Outputs): {outputs}\n\n"
            "Please answer user questions based on the following experiment data. "
            "Use concise, professional language. "
            "If the data doesn't contain the answer, be honest about it.\n\n"
            "Experiment data (CSV):\n```\n{csv}\n```"
        ),
        "qa_label_user": "User",
        "qa_label_assistant": "Assistant",
    },

    # =========================================================================
    #  CHINESE (中文)
    # =========================================================================
    "cn": {
        # --- Page / Global ---
        "page_title": "JPZ 科研平台",
        "logo_suffix": " 科研平台",
        "footer_text": "JPZ 科研平台 | 材料研发数据管理与智能分析",

        # --- Header / Auth ---
        "badge_admin": "ADMIN",
        "badge_guest": "GUEST",
        "popover_admin": ":material/account_circle: Admin",
        "popover_guest": ":material/account_circle: 未登录",
        "logged_in_as_admin": "已登录为 **管理员**",
        "cloud_sync_enabled": "已启用 Google Sheets 云端同步权限",
        "logout": "退出登录",
        "admin_login": "**管理员登录**",
        "unlock_gsheets": "解锁 Google Sheets 云端读取 / 保存功能",
        "password": "密码",
        "password_placeholder": "输入管理密码",
        "login": "登录",
        "wrong_password": "密码错误",
        "no_password_config": (
            "未配置管理密码。请在 `.streamlit/secrets.toml` 中添加:\n\n"
            '```toml\n[general]\npassword = "your_password"\n```'
        ),

        # --- Stats Bar ---
        "stat_records": "实验记录数",
        "stat_columns": "数据列",
        "stat_identity": "身份",
        "stat_storage": "存储状态",
        "stat_cloud_sync": "云端同步",
        "role_admin": "管理员",
        "role_guest": "访客",
        "storage_active": "实时保存",
        "storage_disconnected": "未连接",

        # --- Language Switcher ---
        "lang_label": "Language / 语言",

        # --- Portal Home ---
        "welcome_title": "欢迎回到 JPZ 平台",
        "welcome_subtitle": "材料科学实验数据管理与智能分析",
        "metric_records": "实验记录数",
        "metric_columns": "数据列",
        "metric_numeric": "数值参数",
        "metric_last_sync": "最近同步",
        "btn_guide": ":material/smart_toy:  平台向导",
        "btn_export_report": ":material/download:  导出报告",
        "btn_data_studio": ":material/table_chart:  数据工作台",
        "btn_ai_diagnosis": ":material/psychology:  AI 诊断",
        "btn_visual": ":material/insights:  图表可视化",
        "btn_settings": ":material/tune:  系统设置",
        "recent_data": "最近数据",
        "no_data_hint": "暂无数据。请进入数据工作台添加实验记录。",

        # --- Guide Dialog ---
        "guide_title": "JPZ 平台使用助手",
        "close": "关闭",
        "guide_intro": (
            "我是您的平台助手。您可以询问任何关于本平台使用方法的问题，"
            "例如: 如何添加新列？如何同步数据？如何设置目标值？"
        ),
        "guide_no_history": "暂无对话。在下方输入您的问题。",
        "guide_input_placeholder": "输入问题，例如: 如何使用公式计算列？",
        "guide_no_api_key": "请先在「系统设置」页面配置 Gemini API Key，之后即可使用 AI 助手功能。",
        "guide_ai_error": "AI 回答失败: {error}",
        "guide_system_prompt": (
            "你是一个实验数据管理平台 (JPZ 科研平台) 的使用助手。\n"
            "平台功能:\n"
            "- 首页: 功能卡片入口，快速导航到各模块\n"
            "- 数据工作台: 管理列结构(添加/重命名/删除/公式列)、"
            "语义映射(设定 Inputs/Outputs)、目标值设定、数据编辑器、"
            "云端同步(Pull/Push, 仅管理员)、CSV 导入导出\n"
            "- AI 诊断: Gemini 驱动的深度分析，支持自定义分析要求、"
            "图像分析、实验数据追问\n"
            "- 图表可视化: 散点图(支持趋势线)、相关性热力图\n"
            "- 系统设置: 项目信息、Gemini API Key 配置\n"
            "- 报告导出: 首页直接下载 HTML 实验报告\n\n"
            "请用简短、清晰的中文回答用户关于平台使用方法的问题。"
        ),

        # --- Settings ---
        "back_home": ":material/arrow_back: 返回首页",
        "settings_title": "系统设置",
        "project_info": "项目信息",
        "material_name_label": "研究材料名称",
        "equipment_name_label": "设备 / 工艺名称",
        "gemini_api_key": "Gemini API Key",
        "api_key_label": "API Key",
        "api_key_hint": "API Key 仅保留在当前会话内存中，关闭页面后自动清除。",

        # --- Visual Page ---
        "visual_title": "图表可视化",
        "need_numeric_col": "至少需要 1 个数值列才能绘制图表。",
        "x_axis": "X 轴 (支持文本列)",
        "y_axis": "Y 轴 (仅数值列)",
        "color_map": "颜色映射 (可选)",
        "none_option": "(无)",
        "correlation_heatmap": "相关性热力图",

        # --- Data Studio ---
        "data_studio_title": "数据工作台",
        "io_zone_title": "01",
        "io_zone_label": " 数据存取",
        "pull_from_cloud": "从云端拉取 (Pull)",
        "push_to_cloud": "同步到云端 (Push)",
        "cloud_empty_warning": "云端工作表为空, 未执行覆盖。",
        "pull_success": "已从云端拉取并覆盖本地 ({rows} 行 x {cols} 列)",
        "pull_failed": "云端拉取失败: {error}",
        "push_toast": "云端同步已完成",
        "push_failed": "云端同步失败: {error}",
        "push_troubleshoot": (
            "**排查建议**: 请检查 Google Sheet 是否已"
            "分享给 Service Account 邮箱, 并赋予 **Editor** 权限。"
        ),
        "cloud_caption": (
            "Pull = 云端覆盖本地 | Push = 本地覆盖云端 | "
            "日常编辑自动保存至本地数据库。"
        ),
        "download_csv": "下载 CSV 备份",
        "upload_csv": "上传 CSV 恢复",
        "detected_rows_cols": "检测到 {rows} 行 x {cols} 列",
        "confirm_import": "确认导入",
        "csv_parse_failed": "CSV 解析失败: {error}",
        "schema_zone_title": "02",
        "schema_zone_label": " 结构定义与语义映射",
        "table_structure_btn": ":material/settings: 表格结构管理",
        "table_structure_help": "添加、重命名、删除、排序列，或用公式生成新列",
        "tab_add": "新增",
        "tab_rename": "重命名",
        "tab_delete": "删除",
        "tab_formula": "公式列",
        "tab_reorder": "排序",
        "new_col_name": "新列名",
        "new_col_name_placeholder": "输入列名",
        "data_type": "数据类型",
        "dtype_number": "数值 (Number)",
        "dtype_text": "文本 (Text)",
        "create_now": "立即创建",
        "enter_col_name": "请输入列名。",
        "col_exists": "该列名已存在。",
        "select_column": "选择列",
        "new_name": "新名称",
        "new_name_placeholder": "输入新名称",
        "confirm_rename": "确认修改",
        "same_name_warning": "新旧列名相同。",
        "enter_valid_name": "请输入有效的新列名。",
        "select_cols_delete": "选择要删除的列",
        "confirm_delete": "确认删除",
        "select_cols_first": "请先选择要删除的列。",
        "formula_col_name": "新列名",
        "formula_expr": "计算公式",
        "formula_expr_placeholder": "例如: `生长速率(um/h)` / `微管密度(cm-2)`",
        "formula_help": "用反引号 ` 包裹列名, 支持 +, -, *, / 及括号运算。",
        "calc_and_add": "计算并添加",
        "enter_formula_col_name": "请输入新列名。",
        "enter_formula_expr": "请输入计算公式。",
        "col_already_exists": '列 "{name}" 已存在。',
        "formula_failed": (
            "公式计算失败: {error}\n\n"
            "**正确格式示例:**\n"
            "- `` `列A` + `列B` ``\n"
            "- `` `列A` / `列B` * 100 ``\n\n"
            "当前可用列名: {cols}"
        ),
        "drag_sort_caption": "上下拖拽调整列的显示顺序",
        "need_sortables": "拖拽排序需要安装 streamlit-sortables",
        "install_sortables_hint": "安装后重启应用即可使用拖拽排序。以下为备用方案:",
        "col_order_label": "列顺序 (拖拽标签调整)",
        "apply_order": "应用排序",
        "keep_all_cols_warning": "请保留全部 {n} 列，只调整顺序。",
        "sample_image_expander": "样品图片 (可选)",
        "sample_image_caption": "上传 SEM / 光学显微镜图片，AI 将结合图像形貌分析",
        "upload_image": "上传图片",
        "uploaded_prefix": "已上传: {name}",
        "saved_prefix": "已保存: {name}",
        "remove_image": "移除图片",
        "mapping_info_html": (
            "<strong>第一步:</strong> 选择参数列 (Inputs) 和结果列 (Outputs)。"
            "<strong>第二步:</strong> 为结果列设定量化目标值。"
        ),
        "inputs_label": "Inputs (参数列) — 蓝色标记",
        "inputs_help": "实验中可以控制的变量",
        "outputs_label": "Outputs (结果列) — 橙色标记",
        "outputs_help": "想要优化的目标指标",
        "target_section_title": "设定各指标的目标值",
        "target_input_label": "[{col}] 目标值",
        "target_placeholder": "均值 {avg:.2f}",
        "target_help": "当前均值: {avg:.2f}, 最优: {mx:.2f}",
        "target_caption": "均值 {avg:.2f} / 最优 {mx:.2f}",
        "config_waiting": "配置状态: 等待配置... (请在上方设定参数列、结果列及目标值)",
        "grid_zone_title": "03",
        "grid_zone_label": " 数据表格",
        "status_label": "状态:",
        "save_status_saved": "所有更改已保存。",
        "save_status_ready": (
            "本地数据库已就绪 (实时保存中) — "
            "编辑即保存, 刷新页面数据自动恢复。"
        ),

        # --- Trend Chart ---
        "no_output_for_trend": "请在数据工作台选择 Output 列以显示趋势图",
        "target_annotation": "目标: {val}",
        "trend_chart_title": "结果趋势 (虚线 = 目标值)",
        "experiment_number": "实验编号",
        "value_label": "数值",

        # --- Dashboard ---
        "dashboard_title": "智能仪表盘",
        "research_project": "研究项目",
        "equipment_process": "设备 / 工艺",
        "quantitative_targets": "**量化目标**",
        "target_prefix": "目标: {val}",
        "current_avg_prefix": "当前均值: {avg:.2f}",
        "custom_prompt_label": "补充分析要求 (可选)",
        "custom_prompt_placeholder": (
            "例如: 请重点分析温度对硬度的非线性影响, "
            "或对比第3组和第5组实验的差异..."
        ),
        "btn_ai_analysis": ":material/neurology: 开始 AI 深度分析",
        "warning_no_api_key": "请先在「系统设置」页面配置 Gemini API Key。",
        "warning_no_data": "请先在数据工作台录入实验数据。",
        "spinner_with_image": "AI 正在分析数据、图像与目标差距...",
        "spinner_no_image": "AI 正在分析目标差距并生成优化建议...",
        "summary_experiments": "实验次数",
        "summary_param_cols": "参数列",
        "summary_result_cols": "结果列",
        "summary_targets_set": "已设目标",
        "summary_sample_img": "样品图片",
        "img_uploaded": "已上传",
        "img_none": "无",
        "data_preview": "**实验数据预览**",
        "trend_and_target": "**结果趋势与目标**",
        "sample_image_label": "**样品图片**",
        "ai_title_img_analysis": "图像形貌与数据关联分析",
        "ai_title_no_img_analysis": "目标差距诊断与机理分析",
        "ai_title_img_suggestion": "形貌改善与参数建议",
        "ai_title_no_img_suggestion": "精准参数建议与预期效果",
        "view_full_report": "查看完整 AI 报告",
        "ai_analysis_failed": "AI 分析失败: {error}",
        "dashboard_placeholder": "设定目标后，点击「AI 深度分析」获取科学原理溯源与参数优化建议",
        "experiment_qa": ":material/forum: 实验数据追问",
        "qa_no_history": "暂无对话。在下方输入您对实验数据的疑问，AI 将基于当前数据表回答。",
        "qa_input_placeholder": (
            "输入问题，例如: 哪一组实验的微管密度最低？温度和生长速率有什么关系？"
        ),
        "qa_no_api_key": "请先在「系统设置」页面配置 Gemini API Key。",
        "qa_ai_error": "AI 回答失败: {error}",

        # --- Visual Analytics (embedded) ---
        "visual_analytics_expander": ":material/insights: 可视化分析",
        "scatter_failed": "散点图绘制失败: {error}",
        "correlation_heatmap_title": "**参数相关性热力图 (Correlation Heatmap)**",
        "visual_tip": (
            "提示: 散点图展示双变量关系 (含 OLS 趋势线, 需安装 statsmodels); "
            "热力图展示所有数值列之间的 Pearson 相关系数, "
            "绝对值越接近 1 表示线性相关性越强。"
        ),

        # --- HTML Report ---
        "report_title": "JPZ 科研平台 实验报告",
        "report_generated": "生成时间",
        "report_identity": "操作身份",
        "report_material": "研究材料",
        "report_equipment": "设备 / 工艺",
        "report_not_specified": "未指定",
        "report_section_mapping": "1. 语义映射",
        "report_inputs_label": "参数列 (Inputs):",
        "report_outputs_label": "结果列 (Outputs):",
        "report_section_targets": "2. 量化目标",
        "report_metric": "指标",
        "report_target_value": "目标值",
        "report_current_avg": "当前均值",
        "report_no_targets": "（未设定量化目标）",
        "report_section_data": "3. 实验数据 ({rows} 行 x {cols} 列)",
        "report_no_data": "（无数据）",
        "report_section_stats": "4. 数据统计摘要",
        "report_section_ai": "5. AI 分析报告",
        "report_ai_failed": "分析失败: {error}",
        "report_ai_pending": "尚未执行 AI 分析",
        "report_footer": "Generated by JPZ Intelligent Assistant",

        # --- AI Analysis System Prompts ---
        "ai_system_prompt": (
            "你是一位世界顶级的材料科学家和工艺工程师。\n"
            "用户正在进行【{material}】的研究。\n"
            "使用的设备/工艺是：【{equipment}】。\n\n"
            "你的任务是帮助用户达成量化目标。\n"
            "1. 精确指出当前数据与目标值的差距\n"
            "2. 结合物理/化学原理解释瓶颈\n"
            "3. 给出能够逼近目标值的具体参数建议\n"
            "4. 如果目标不切实际，诚实指出{img_instr}{custom_instr}"
        ),
        "ai_img_instr": (
            "\n5. 仔细观察用户上传的样品微观结构图"
            "\n6. 分析图像中的形貌特征（晶粒大小、裂纹、孔隙、颜色异常等）"
            "\n7. 将图像观察与实验参数关联，推断工艺-形貌-性能的因果关系"
        ),
        "ai_custom_instr": (
            "\n\n用户已设定以下具体分析要求，请务必结合数据进行针对性分析:\n"
            "「{prompt}」"
        ),
        "ai_target_line": (
            "- {col}: 目标值={tv}, 当前均值={avg:.2f}, "
            "最优={best:.2f}, 差距={gap:.2f} ({pct:+.1f}%)"
        ),
        "ai_target_line_no_target": (
            "- {col}: 未设定目标, 当前均值={avg:.2f}, 最优={best:.2f}"
        ),
        "ai_no_target": "(用户未设定具体目标)",
        "ai_input_not_specified": "(用户未指定)",
        "ai_user_prompt_img": (
            "## 实验数据\n```csv\n{csv}\n```\n\n"
            "## 数据列说明\n- 实验参数列 (可调变量): {inputs}\n\n"
            "## 用户的量化目标\n{targets}\n\n"
            "## 样品图像\n用户上传了一张样品的微观结构图。请仔细观察。\n\n---\n\n"
            "请按以下结构分析:\n\n"
            "### 一、图像形貌分析\n### 二、数据-图像关联分析\n"
            "### 三、瓶颈机理分析\n### 四、精准参数建议\n### 五、预期效果评估"
        ),
        "ai_user_prompt_no_img": (
            "## 实验数据\n```csv\n{csv}\n```\n\n"
            "## 数据列说明\n- 实验参数列 (可调变量): {inputs}\n\n"
            "## 用户的量化目标\n{targets}\n\n---\n\n"
            "请按以下结构分析:\n\n"
            "### 一、目标差距诊断\n### 二、瓶颈机理分析\n"
            "### 三、精准参数建议\n### 四、预期效果评估"
        ),
        "ai_split_marker_img": "### 四",
        "ai_split_marker_no_img": "### 三",

        # --- Experiment Q&A System Prompt ---
        "qa_system_prompt": (
            "你是一位材料科学专家和实验数据分析师。\n"
            "研究材料: {material}\n"
            "参数列 (Inputs): {inputs}\n"
            "结果列 (Outputs): {outputs}\n\n"
            "请基于以下实验数据回答用户问题。用简洁、专业的语言回答。"
            "如果数据中找不到答案，请诚实告知。\n\n"
            "实验数据 (CSV):\n```\n{csv}\n```"
        ),
        "qa_label_user": "用户",
        "qa_label_assistant": "助手",
    },
}

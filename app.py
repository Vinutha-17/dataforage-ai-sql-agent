import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from langchain_groq import ChatGroq

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="DataForge AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SESSION STATE ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# ── THEME CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:        #0a0c0f;
    --surface:   #111318;
    --panel:     #161b22;
    --border:    #21262d;
    --accent:    #00d4aa;
    --accent2:   #7c3aed;
    --warn:      #f59e0b;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --green:     #3fb950;
    --red:       #f85149;
}

/* ── BASE ── */
html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

/* ── HIDE STREAMLIT CHROME — keep header visible for sidebar toggle ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

.block-container { padding-top: 1.5rem !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
    min-height: 100vh;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── SIDEBAR TOGGLE BUTTON ── */
button[data-testid="baseButton-header"],
button[data-testid="stSidebarNavCollapseButton"],
button[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] button {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--accent) !important;
    cursor: pointer !important;
    z-index: 9999 !important;
    transition: background 0.2s, border-color 0.2s !important;
}

button[data-testid="baseButton-header"]:hover,
button[data-testid="stSidebarNavCollapseButton"]:hover,
button[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="collapsedControl"]:hover {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #000 !important;
}

button[data-testid="baseButton-header"] svg,
button[data-testid="stSidebarNavCollapseButton"] svg,
button[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {
    fill: currentColor !important;
    stroke: currentColor !important;
    visibility: visible !important;
    display: block !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    height: auto !important;
}

/* ── TOPBAR ── */
.topbar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 28px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 24px;
}
.topbar-icon { font-size: 32px; line-height: 1; }
.topbar-name {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 24px;
    color: var(--text);
    letter-spacing: -0.5px;
}
.topbar-badge {
    background: var(--accent);
    color: #000;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 99px;
    margin-left: 4px;
}
.topbar-sub {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-left: auto;
}

/* ── METRIC CARDS ── */
[data-testid="metric-container"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
}
[data-testid="metric-container"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 28px !important;
    color: var(--accent) !important;
}

/* ── SECTION HEADINGS ── */
.section-head {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 14px;
}

/* ── TABS ── */
[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    border-radius: 6px !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: #000 !important;
}

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
textarea {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── CHAT ── */
[data-testid="stChatMessage"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--panel) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
}

/* ── CODE BLOCK ── */
pre, code {
    background: #0d1117 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    color: var(--accent) !important;
}

/* ── STATUS ALERTS ── */
.stSuccess {
    background: rgba(63,185,80,0.1) !important;
    border: 1px solid var(--green) !important;
    border-radius: 8px !important;
    color: var(--green) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}
.stWarning {
    background: rgba(245,158,11,0.1) !important;
    border: 1px solid var(--warn) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}
.stInfo {
    background: rgba(0,212,170,0.08) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}
.stError {
    background: rgba(248,81,73,0.1) !important;
    border: 1px solid var(--red) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}

/* ── SUGGESTION PILL BUTTONS ── */
.pill-btn button {
    background: rgba(0,212,170,0.08) !important;
    border: 1px solid rgba(0,212,170,0.3) !important;
    color: var(--accent) !important;
    border-radius: 99px !important;
    font-size: 11px !important;
    padding: 6px 14px !important;
}
.pill-btn button:hover {
    background: var(--accent) !important;
    color: #000 !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--panel) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}

/* ── SIDEBAR LOGO ── */
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 20px;
    color: var(--text);
    padding: 8px 0 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}
.sidebar-logo span { color: var(--accent); }

/* ── INSIGHT ROW ── */
.insight-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
}
.insight-dot {
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── HISTORY ITEM ── */
.hist-item {
    padding: 12px 16px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 8px;
    margin-bottom: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: var(--muted);
}
.hist-q { color: var(--text); font-size: 13px; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ── TOPBAR ───────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-icon">⬡</div>
  <div>
    <div class="topbar-name">DataForge AI <span class="topbar-badge">SQL AGENT</span></div>
  </div>
  <div class="topbar-sub">Natural Language -> SQL - Auto Visualize - Insight Engine</div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Data<span>Forge</span> ⬡</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-head">API Configuration</div>', unsafe_allow_html=True)
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    st.markdown('<div class="section-head">Model</div>', unsafe_allow_html=True)
    model_option = st.selectbox("", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant"
    ], label_visibility="collapsed")

    st.markdown('<div class="section-head">Upload Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    st.markdown("""
    <div style="margin-top:32px; padding:14px; background:rgba(0,212,170,0.06);
                border:1px solid rgba(0,212,170,0.2); border-radius:8px;
                font-family:'Space Mono',monospace; font-size:11px; color:#8b949e; line-height:1.8;">
        <div style="color:#00d4aa; margin-bottom:6px;">&#9658; CAPABILITIES</div>
        NL -> SQL Query Gen<br>
        Auto Chart Builder<br>
        Correlation Heatmap<br>
        PDF Export<br>
        Data Cleaning<br>
        KPI Dashboard
    </div>
    """, unsafe_allow_html=True)

# ── PDF HELPER ───────────────────────────────────────────────
def generate_pdf(summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # FIX: replaced em-dash with plain hyphen to avoid FPDFUnicodeEncodingException
    pdf.cell(200, 10, txt="DataForge AI - SQL Analysis Report", ln=True)
    pdf.ln(4)
    pdf.multi_cell(0, 10, summary_text)
    pdf.output("report.pdf")

# ── PLOTLY DARK TEMPLATE ─────────────────────────────────────
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#111318",
    plot_bgcolor="#0a0c0f",
    font=dict(family="Space Mono, monospace", color="#e6edf3"),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#00d4aa","#7c3aed","#f59e0b","#3fb950","#f85149","#58a6ff"]
)

# ── MAIN APP ─────────────────────────────────────────────────
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Could not read CSV file")
        st.stop()

    conn = sqlite3.connect(":memory:")
    df.to_sql("data_table", conn, if_exists="replace", index=False)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    tab1, tab2, tab3, tab4 = st.tabs([
        "DASHBOARD", "SQL AGENT", "CHARTS", "EXPORT"
    ])

    # ── DASHBOARD ────────────────────────────────────────────
    with tab1:
        total_rows      = df.shape[0]
        total_columns   = df.shape[1]
        missing_values  = int(df.isnull().sum().sum())
        duplicate_rows  = int(df.duplicated().sum())

        st.markdown('<div class="section-head">Dataset Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Rows",       total_rows)
        with c2: st.metric("Columns",    total_columns)
        with c3: st.metric("Missing",    missing_values)
        with c4: st.metric("Duplicates", duplicate_rows)

        if numeric_cols:
            st.markdown('<div class="section-head">KPI Snapshot - ' + numeric_cols[0] + '</div>', unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3)
            with k1: st.metric("Max", round(df[numeric_cols[0]].max(), 2))
            with k2: st.metric("Min", round(df[numeric_cols[0]].min(), 2))
            with k3: st.metric("Avg", round(df[numeric_cols[0]].mean(), 2))

        left, right = st.columns([3, 2])
        with left:
            st.markdown('<div class="section-head">Data Preview</div>', unsafe_allow_html=True)
            st.dataframe(df.head(20), use_container_width=True)
        with right:
            st.markdown('<div class="section-head">Schema</div>', unsafe_allow_html=True)
            schema_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str)})
            st.dataframe(schema_df, use_container_width=True)

        st.markdown('<div class="section-head">Data Operations</div>', unsafe_allow_html=True)
        col_clean, col_search = st.columns([1, 2])
        with col_clean:
            if st.button("Clean Dataset"):
                before = df.shape[0]
                df = df.drop_duplicates().fillna(0)
                df.to_sql("data_table", conn, if_exists="replace", index=False)
                st.success(f"Removed {before - df.shape[0]} duplicate rows. Nulls filled.")

        with col_search:
            search_col = st.selectbox("Search column", df.columns, key="search_col_select")
            search_val = st.text_input("Search value", placeholder="Type to filter...", key="search_val_input")
            if search_val:
                filtered = df[df[search_col].astype(str).str.contains(search_val, case=False, na=False)]
                st.dataframe(filtered, use_container_width=True)

        if numeric_cols:
            st.markdown('<div class="section-head">Auto Insights</div>', unsafe_allow_html=True)
            for col in numeric_cols[:6]:
                avg_v = round(df[col].mean(), 2)
                max_v = round(df[col].max(), 2)
                min_v = round(df[col].min(), 2)
                st.markdown(f"""
                <div class="insight-row">
                  <div class="insight-dot"></div>
                  <span style="color:#e6edf3;font-weight:700;">{col}</span>
                  <span style="margin-left:auto;color:#8b949e;">avg <span style="color:#00d4aa">{avg_v}</span>
                  &nbsp;·&nbsp; max <span style="color:#3fb950">{max_v}</span>
                  &nbsp;·&nbsp; min <span style="color:#f85149">{min_v}</span></span>
                </div>
                """, unsafe_allow_html=True)

    # ── SQL AGENT ────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-head">Suggested Queries</div>', unsafe_allow_html=True)
        suggestions = []
        for col in numeric_cols[:3]:
            suggestions += [f"What is the average {col}?", f"Show top 10 rows by {col}"]
        suggestions = suggestions[:6]

        s_cols = st.columns(3)
        for i, sug in enumerate(suggestions):
            with s_cols[i % 3]:
                st.markdown('<div class="pill-btn">', unsafe_allow_html=True)
                if st.button(sug, key=f"sug_{i}"):
                    st.session_state["auto_q"] = sug
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-head">Conversation</div>', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask anything about your data...")
        if st.session_state.get("auto_q"):
            question = st.session_state.pop("auto_q")

        if question and groq_api_key:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            try:
                llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_option)

                schema_info = "\n".join(
                    [f"  - {c} ({str(df[c].dtype)})" for c in df.columns]
                )
                sample_rows = df.head(3).to_string(index=False)

                prompt = f"""You are an expert SQLite analyst. Given a table called `data_table`, write a precise SQL query and answer the user's question.

TABLE SCHEMA:
{schema_info}

SAMPLE DATA (first 3 rows):
{sample_rows}

USER QUESTION: {question}

RULES:
1. Write valid SQLite syntax only.
2. Use column names exactly as listed above.
3. Be concise and accurate in the Answer.
4. If the question is ambiguous, make a reasonable assumption and state it.

RESPOND IN EXACTLY THIS FORMAT:
SQL:
<your SQL query here>

Answer:
<your natural language answer here>"""

                response = llm.invoke(prompt).content
                sql_query, final_answer = "", response

                if "SQL:" in response and "Answer:" in response:
                    sql_query    = response.split("SQL:")[1].split("Answer:")[0].strip()
                    final_answer = response.split("Answer:")[1].strip()

                ai_response = (
                    f"**Generated Query**\n\n"
                    f"```sql\n{sql_query}\n```\n\n"
                    f"**Result**\n\n{final_answer}"
                )

                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                st.session_state.query_history.append({"question": question, "sql": sql_query})

                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                    if sql_query.lower().startswith("select"):
                        try:
                            result_df = pd.read_sql_query(sql_query, conn)
                            if not result_df.empty:
                                st.dataframe(result_df, use_container_width=True)
                        except Exception:
                            st.warning("Could not execute the generated SQL on your dataset.")

            except Exception as e:
                st.error(f"Agent error: {e}")

        elif question and not groq_api_key:
            st.warning("Enter your Groq API Key in the sidebar to activate the agent.")

        if st.session_state.query_history:
            st.markdown('<div class="section-head">Query History</div>', unsafe_allow_html=True)
            for item in reversed(st.session_state.query_history[-8:]):
                st.markdown(f"""
                <div class="hist-item">
                  <div class="hist-q">&#9658; {item['question']}</div>
                  <code style="font-size:11px;color:#7c3aed;">{item['sql'][:120]}{'...' if len(item['sql'])>120 else ''}</code>
                </div>
                """, unsafe_allow_html=True)

    # ── CHARTS ───────────────────────────────────────────────
    with tab3:
        if numeric_cols:
            st.markdown('<div class="section-head">Chart Builder</div>', unsafe_allow_html=True)
            cx1, cx2, cx3, cx4 = st.columns([2, 2, 2, 1])
            with cx1: x_axis = st.selectbox("X Axis", df.columns,     key="chart_x")
            with cx2: y_axis = st.selectbox("Y Axis", numeric_cols,    key="chart_y")
            with cx3: chart_type = st.selectbox("Type", ["Bar","Line","Scatter","Histogram","Pie"], key="chart_type")
            with cx4:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                gen = st.button("Generate")

            if gen:
                if chart_type == "Bar":
                    fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                elif chart_type == "Line":
                    fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
                elif chart_type == "Scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                elif chart_type == "Pie":
                    fig = px.pie(df, names=x_axis, values=y_axis, title=f"{y_axis} distribution")
                else:
                    fig = px.histogram(df, x=y_axis, title=f"{y_axis} distribution")
                fig.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        if len(numeric_cols) >= 2:
            st.markdown('<div class="section-head">Correlation Heatmap</div>', unsafe_allow_html=True)
            try:
                corr = df[numeric_cols].corr().fillna(0)
                fig2 = px.imshow(
                    corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale=["#7c3aed","#111318","#00d4aa"],
                    title="Feature Correlation Matrix"
                )
                fig2.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.warning("Could not generate heatmap")

        if len(numeric_cols) >= 1:
            st.markdown('<div class="section-head">Distribution Overview</div>', unsafe_allow_html=True)
            sel_col = st.selectbox("Column", numeric_cols, key="dist_col")
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=df[sel_col], nbinsx=40,
                marker_color="#00d4aa", opacity=0.75,
                name=sel_col
            ))
            # FIX: replaced em-dash with plain hyphen in chart title
            fig3.update_layout(title=f"{sel_col} - Frequency Distribution", **PLOT_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

    # ── EXPORT ───────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-head">Export Options</div>', unsafe_allow_html=True)
        e1, e2 = st.columns(2)

        with e1:
            st.markdown("**PDF Report**")
            st.caption("Summary stats exported as a formatted PDF document.")
            if st.button("Build PDF"):
                # FIX: replaced em-dash with plain hyphen throughout report_text
                report_text = (
                    f"DataForge AI - Analysis Report\n"
                    f"{'='*40}\n"
                    f"Rows:       {df.shape[0]}\n"
                    f"Columns:    {df.shape[1]}\n"
                    f"Missing:    {int(df.isnull().sum().sum())}\n"
                    f"Duplicates: {int(df.duplicated().sum())}\n\n"
                    f"Numeric Columns\n{'-'*40}\n"
                )
                for col in numeric_cols:
                    report_text += (
                        f"{col}:\n"
                        f"  avg={round(df[col].mean(),2)}"
                        f"  max={df[col].max()}"
                        f"  min={df[col].min()}\n"
                    )
                generate_pdf(report_text)
                with open("report.pdf", "rb") as f:
                    st.download_button("Download PDF", f, "DataForge_Report.pdf", "application/pdf")

        with e2:
            st.markdown("**CSV Export**")
            st.caption("Download the current (cleaned) dataset as CSV.")
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_data, "dataforge_export.csv", "text/csv")

        st.markdown('<div class="section-head">Dataset Stats Table</div>', unsafe_allow_html=True)
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)

else:
    # ── EMPTY STATE ──────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                padding:80px 20px; text-align:center;">
      <div style="font-size:64px;margin-bottom:20px;opacity:0.6;">&#11041;</div>
      <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:28px;
                  color:#e6edf3;margin-bottom:12px;">Upload a dataset to begin</div>
      <div style="font-family:'Space Mono',monospace;font-size:13px;color:#8b949e;
                  max-width:480px;line-height:1.8;">
        Drop a <span style="color:#00d4aa">.csv</span> file in the sidebar.<br>
        DataForge will auto-analyse it - then ask anything in plain English.<br>
        The AI SQL Agent converts your question to a query and runs it live.
      </div>
    </div>
    """, unsafe_allow_html=True)

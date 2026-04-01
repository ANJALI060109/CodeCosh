"""
CodeCosh - "Know Your Code. Instantly."
Full Streamlit App — No Flask needed
Deploy directly to Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import predictor

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CodeCosh",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Constants ────────────────────────────────────────────────────────────────
LANG_ICONS = {
    "Python":     "🐍",
    "Java":       "☕",
    "C++":        "⚙️",
    "JavaScript": "🟨",
    "SQL":        "🗄️",
    "Bash":       "🖥️",
}
LANG_COLORS = {
    "Python":     "#3776AB",
    "Java":       "#E76F00",
    "C++":        "#00599C",
    "JavaScript": "#F0DB4F",
    "SQL":        "#CC2927",
    "Bash":       "#4EAA25",
}
MODEL_META = {
    "naive_bayes":         {"label": "Naive Bayes",          "icon": "📊", "color": "#4C8BF5"},
    "logistic_regression": {"label": "Logistic Regression",  "icon": "📈", "color": "#34A853"},
    "ensemble":            {"label": "Ensemble (Best Pick)",  "icon": "🏆", "color": "#F4A827"},
}

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a3a6b 50%, #2e5fa3 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 40px rgba(26,58,107,0.25);
    border: 1px solid rgba(244,168,39,0.2);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🧠";
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.08;
}
.hero h1 { color: #fff; font-size: 2.4rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hero .tagline { color: #F4A827; font-size: 1rem; font-weight: 600; margin: 0.2rem 0 0.5rem; }
.hero p { color: #b8d0f7; font-size: 0.9rem; margin: 0; }

/* ── Cards ── */
.card {
    background: #fff;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    border: 1px solid #e8edf5;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #f0f6ff, #e8f0fe);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #d0e2ff;
}
.metric-card .val { font-size: 2rem; font-weight: 800; color: #1a3a6b; }
.metric-card .lbl { font-size: 0.75rem; color: #5a7fa3; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }

/* ── Result Box ── */
.result-main {
    border-radius: 16px;
    padding: 1.6rem;
    text-align: center;
    margin-bottom: 1rem;
}

/* ── Info / Warn boxes ── */
.info-box {
    background: #f0f6ff;
    border-left: 4px solid #4c8bf5;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.9rem;
    color: #2c3e60;
}
.warn-box {
    background: #fffbf0;
    border-left: 4px solid #F4A827;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.9rem;
    color: #5a4000;
}

/* ── Sidebar ── */
.sidebar-brand {
    text-align: center;
    padding: 1rem 0 0.5rem;
}
.sidebar-brand h2 { color: #1a3a6b; font-size: 1.5rem; font-weight: 800; margin: 0; }
.sidebar-brand .sub { color: #F4A827; font-size: 0.8rem; font-weight: 600; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a3a6b, #2e5fa3) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2e5fa3, #4c8bf5) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(76,139,245,0.35) !important;
}

/* ── Status dot ── */
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.dot-green  { background:#34a853; }
.dot-orange { background:#F4A827; }
.dot-red    { background:#ea4335; }

/* ── Code font ── */
code { font-family: 'JetBrains Mono', monospace !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #aaa;
    font-size: 0.8rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid #eee;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
if "models" not in st.session_state:
    st.session_state.models = predictor.load_models()


def is_trained():
    return st.session_state.models is not None and st.session_state.models.get("trained")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:2.5rem">🧠</div>
        <h2>CodeCosh</h2>
        <div class="sub">"Know Your Code. Instantly."</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if is_trained():
        langs = st.session_state.models["metrics"]["languages"]
        st.markdown(f'<span class="dot dot-green"></span> **Models Ready** — {len(langs)} languages', unsafe_allow_html=True)
        st.markdown("")
        for l in langs:
            st.markdown(f"{LANG_ICONS.get(l,'💻')} &nbsp; {l}", unsafe_allow_html=True)
    else:
        st.markdown('<span class="dot dot-orange"></span> **Not Trained Yet**', unsafe_allow_html=True)
        st.markdown('<div class="warn-box">Upload a CSV in the <b>Train</b> tab to get started!</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📋 CSV Format:**")
    st.code("code,language\n\"print('hi')\",Python\n\"int x=0;\",C++", language="text")
    st.markdown("---")
    st.caption("Built with ❤️ using Python · Scikit-learn · Streamlit")
    st.caption("© 2024 CodeCosh")


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 CodeCosh</h1>
    <div class="tagline">"Know Your Code. Instantly."</div>
    <p>Machine Learning · TF-IDF Vectorization · Naive Bayes · Logistic Regression</p>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Detect Language", "🏋️ Train Model", "📊 Model Metrics", "ℹ️ About"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 ── DETECT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    if not is_trained():
        st.markdown('<div class="warn-box">⚠️ Models not trained yet. Please go to the <b>🏋️ Train Model</b> tab first and upload a dataset.</div>', unsafe_allow_html=True)
    else:
        col_left, col_right = st.columns([1.1, 0.9], gap="large")

        with col_left:
            st.markdown("### 📝 Paste Your Code")

            model_choice = st.selectbox(
                "Prediction model:",
                ["both", "logistic_regression", "naive_bayes"],
                format_func=lambda x: {
                    "both":                "🏆 Both Models + Ensemble",
                    "logistic_regression": "📈 Logistic Regression",
                    "naive_bayes":         "📊 Naive Bayes",
                }[x]
            )

            # Quick example buttons
            st.markdown("**⚡ Quick Examples:**")
            ex1, ex2, ex3 = st.columns(3)
            examples = {
                "🐍 Python":     "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(10))",
                "☕ Java":       "public class Hello {\n    public static void main(String[] args) {\n        System.out.println(\"Hello World\");\n    }\n}",
                "🟨 JavaScript": "const fetchData = async () => {\n    const res = await fetch('/api/data');\n    const json = await res.json();\n    return json;\n};",
            }
            if "code_val" not in st.session_state:
                st.session_state.code_val = ""

            if ex1.button("🐍 Python", use_container_width=True):
                st.session_state.code_val = examples["🐍 Python"]
            if ex2.button("☕ Java", use_container_width=True):
                st.session_state.code_val = examples["☕ Java"]
            if ex3.button("🟨 JS", use_container_width=True):
                st.session_state.code_val = examples["🟨 JavaScript"]

            code_input = st.text_area(
                "Code snippet:",
                value=st.session_state.code_val,
                height=300,
                placeholder="Paste any code here...\n\nExample:\ndef hello():\n    print('Hello World')",
                label_visibility="collapsed"
            )

            detect_btn = st.button("🔍 Detect Language", use_container_width=True, type="primary")

        with col_right:
            st.markdown("### 🎯 Detection Result")

            if detect_btn:
                if not code_input.strip():
                    st.warning("Please paste a code snippet first!")
                elif len(code_input.strip()) < 3:
                    st.warning("Snippet too short. Please add more code.")
                else:
                    with st.spinner("🔍 Analyzing code..."):
                        time.sleep(0.4)
                        result = predictor.predict(st.session_state.models, code_input, model_choice)

                    # Top result
                    top  = result.get("ensemble") or result.get("logistic_regression") or result.get("naive_bayes")
                    lang = top["language"]
                    conf = top["confidence"]
                    icon = LANG_ICONS.get(lang, "💻")
                    col  = LANG_COLORS.get(lang, "#4c8bf5")

                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,{col}22,{col}11);
                                border:2px solid {col}88; border-radius:16px;
                                padding:1.8rem; text-align:center; margin-bottom:1rem;">
                        <div style="font-size:3.5rem; margin-bottom:0.3rem">{icon}</div>
                        <div style="font-size:2rem; font-weight:800; color:{col}">{lang}</div>
                        <div style="font-size:0.95rem; color:#555; margin-top:0.4rem">
                            Confidence: <b style="color:{col}">{conf}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Per model breakdown
                    for mkey, mdata in result.items():
                        if mkey == "ensemble":
                            continue
                        meta   = MODEL_META.get(mkey, {})
                        m_lang = mdata["language"]
                        m_conf = mdata["confidence"]
                        m_col  = meta.get("color", "#888")
                        st.markdown(f"""
                        <div style="background:#f8faff; border:1px solid #dbe8ff;
                                    border-radius:10px; padding:0.9rem 1.1rem; margin-bottom:0.5rem;">
                            <span style="font-weight:700; color:{m_col}">{meta.get('icon','🤖')} {meta.get('label', mkey)}</span>
                            <span style="float:right; font-size:0.85rem; color:#888">{m_conf}% confidence</span><br>
                            <span style="font-size:1.1rem; font-weight:700; color:#1a3a6b">
                                {LANG_ICONS.get(m_lang,'💻')} {m_lang}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Probability chart
                    probs = top["probabilities"]
                    if probs:
                        st.markdown("**📊 Probability Distribution**")
                        df_p = pd.DataFrame(list(probs.items()), columns=["Language","Probability"])
                        df_p = df_p.sort_values("Probability", ascending=True)
                        fig  = go.Figure(go.Bar(
                            x=df_p["Probability"],
                            y=df_p["Language"],
                            orientation='h',
                            marker_color=[LANG_COLORS.get(l,"#4c8bf5") for l in df_p["Language"]],
                            text=[f"{v:.1f}%" for v in df_p["Probability"]],
                            textposition='outside'
                        ))
                        fig.update_layout(
                            margin=dict(l=0,r=50,t=10,b=0),
                            height=max(180, len(probs)*42),
                            xaxis=dict(range=[0,120], showticklabels=False),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.markdown("""
                <div style="text-align:center; color:#bbb; padding:3rem 1rem;">
                    <div style="font-size:3.5rem">🔍</div>
                    <div style="font-size:1rem; margin-top:0.5rem">
                        Paste code on the left<br>and hit <b>Detect Language</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 ── TRAIN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🏋️ Train the Models")
    st.markdown('<div class="info-box">📂 Upload a CSV with <b>code</b> and <b>language</b> columns. Use <code>data/sample_dataset.csv</code> from the repo if you don\'t have one yet.</div>', unsafe_allow_html=True)

    col_up, col_set = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("**1. Upload Dataset CSV**")
        uploaded = st.file_uploader("Choose CSV", type=["csv"])

        if uploaded:
            try:
                df_prev = pd.read_csv(uploaded)
                uploaded.seek(0)

                if "code" not in df_prev.columns or "language" not in df_prev.columns:
                    st.error("❌ CSV must have 'code' and 'language' columns!")
                else:
                    st.success(f"✅ {len(df_prev)} rows · {df_prev['language'].nunique()} languages")
                    counts = df_prev["language"].value_counts()
                    fig_d  = px.bar(
                        x=counts.index, y=counts.values,
                        labels={"x":"Language","y":"Samples"},
                        color=counts.index,
                        color_discrete_map=LANG_COLORS,
                        title="Dataset Distribution"
                    )
                    fig_d.update_layout(
                        showlegend=False,
                        margin=dict(l=0,r=0,t=30,b=0),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_d, use_container_width=True)
                    st.dataframe(df_prev[["code","language"]].head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")

    with col_set:
        st.markdown("**2. Configure Training**")
        test_size    = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        max_features = st.select_slider(
            "TF-IDF max features",
            options=[1000, 2000, 3000, 5000, 8000, 10000],
            value=5000
        )
        st.markdown("**Models being trained:**")
        st.markdown("📊 **Naive Bayes** — fast probabilistic model")
        st.markdown("📈 **Logistic Regression** — higher accuracy linear model")
        st.markdown("🏆 **Ensemble** — picks most confident prediction")

        st.markdown("")
        train_btn = st.button(
            "🚀 Start Training",
            use_container_width=True,
            type="primary",
            disabled=(uploaded is None)
        )

    if train_btn and uploaded:
        uploaded.seek(0)
        df_train = pd.read_csv(uploaded)

        if len(df_train) < 10:
            st.error("Need at least 10 rows to train!")
        else:
            prog = st.progress(0, text="Starting CodeCosh training pipeline...")
            steps = [
                (15, "📥 Loading and validating dataset..."),
                (30, "🔧 Preprocessing code snippets..."),
                (50, "🧮 Fitting TF-IDF vectorizer..."),
                (65, "📊 Training Naive Bayes..."),
                (80, "📈 Training Logistic Regression..."),
                (93, "📐 Evaluating on test set..."),
            ]
            for pct, msg in steps:
                time.sleep(0.35)
                prog.progress(pct, text=msg)

            models = predictor.train_models(df_train, test_size, max_features)
            st.session_state.models = models

            prog.progress(100, text="✅ Training complete!")
            time.sleep(0.5)
            prog.empty()

            nb_acc = models["metrics"]["naive_bayes"]["accuracy"]
            lr_acc = models["metrics"]["logistic_regression"]["accuracy"]
            winner = "Logistic Regression" if lr_acc >= nb_acc else "Naive Bayes"

            st.balloons()
            st.success(f"🎉 Training complete! Best model: **{winner}** with **{max(nb_acc,lr_acc)}%** accuracy")

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-card"><div class="val">{nb_acc}%</div><div class="lbl">Naive Bayes</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="val">{lr_acc}%</div><div class="lbl">Log. Regression</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="val">{models["metrics"]["total_samples"]}</div><div class="lbl">Total Samples</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-card"><div class="val">{len(models["metrics"]["languages"])}</div><div class="lbl">Languages</div></div>', unsafe_allow_html=True)
            st.info("🎯 Switch to **📊 Model Metrics** tab to see the full analysis!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 ── METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Model Performance Analysis")

    if not is_trained():
        st.markdown('<div class="warn-box">⚠️ Train the models first to view metrics.</div>', unsafe_allow_html=True)
    else:
        metrics = st.session_state.models["metrics"]
        nb      = metrics["naive_bayes"]
        lr      = metrics["logistic_regression"]
        langs   = metrics["languages"]

        # Accuracy cards
        st.markdown("#### 🎯 Accuracy Comparison")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="val">{nb["accuracy"]}%</div><div class="lbl">📊 Naive Bayes</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="val">{lr["accuracy"]}%</div><div class="lbl">📈 Log. Regression</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="val">{metrics["train_size"]}</div><div class="lbl">Train Samples</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="val">{metrics["test_size"]}</div><div class="lbl">Test Samples</div></div>', unsafe_allow_html=True)

        st.markdown("")
        fig_acc = go.Figure(go.Bar(
            x=["Naive Bayes", "Logistic Regression"],
            y=[nb["accuracy"], lr["accuracy"]],
            marker_color=["#4C8BF5", "#34A853"],
            text=[f"{nb['accuracy']}%", f"{lr['accuracy']}%"],
            textposition="outside", width=0.35
        ))
        fig_acc.update_layout(
            yaxis=dict(range=[0,115], title="Accuracy (%)"),
            height=300,
            margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        # Per-language metrics
        st.markdown("#### 📋 Per-Language Metrics")
        sel = st.selectbox("View metrics for:", ["Logistic Regression", "Naive Bayes"])
        rep = metrics["logistic_regression" if sel == "Logistic Regression" else "naive_bayes"]["report"]
        rows = []
        for l in langs:
            r = rep.get(l, {})
            rows.append({
                "Language":  f"{LANG_ICONS.get(l,'💻')} {l}",
                "Precision": f"{round(r.get('precision',0)*100,1)}%",
                "Recall":    f"{round(r.get('recall',0)*100,1)}%",
                "F1-Score":  f"{round(r.get('f1-score',0)*100,1)}%",
                "Support":   int(r.get('support', 0)),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Confusion matrix
        st.markdown("#### 🔢 Confusion Matrix")
        cm_sel = st.selectbox("Model:", ["Logistic Regression", "Naive Bayes"], key="cm")
        cm     = metrics["logistic_regression" if cm_sel == "Logistic Regression" else "naive_bayes"]["confusion_matrix"]
        fig_cm = px.imshow(
            cm, x=langs, y=langs,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            text_auto=True, aspect="auto"
        )
        fig_cm.update_layout(
            height=400, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 ── ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ℹ️ About CodeCosh")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card">
            <h4>🧠 What is CodeCosh?</h4>
            <p>
            <b>CodeCosh</b> (Code + कोश) is a machine learning-powered web app
            that detects the programming language of any code snippet instantly.
            </p>
            <p>
            It uses <b>TF-IDF vectorization</b> to extract features from code,
            then classifies using <b>Naive Bayes</b> and <b>Logistic Regression</b>,
            with an ensemble that picks the most confident result.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <h4>⚙️ Tech Stack</h4>
            <p>🐍 <b>Python 3</b> — Core language</p>
            <p>🎨 <b>Streamlit</b> — Frontend UI</p>
            <p>🤖 <b>Scikit-learn</b> — ML models</p>
            <p>📊 <b>Pandas / NumPy</b> — Data handling</p>
            <p>📈 <b>Plotly</b> — Interactive charts</p>
            <p>🔤 <b>TF-IDF</b> — Text vectorization</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>🔄 ML Pipeline</h4>
        <pre style="background:#f5f7fa; padding:1rem; border-radius:8px; font-size:0.85rem; overflow-x:auto;">
Code Input
    ↓
Preprocessor  (remove comments, normalize strings & numbers)
    ↓
TF-IDF Vectorizer  (bigrams, sublinear TF, up to 5000 features)
    ↓
┌─────────────────────┬────────────────────────┐
│   Naive Bayes       │   Logistic Regression  │
│   (MultinomialNB)   │   (max_iter=1000)      │
└─────────────────────┴────────────────────────┘
    ↓
Ensemble → highest-confidence prediction wins
    ↓
Language + Confidence Score + Probability Distribution
        </pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>📋 CSV Format</h4>
        <p>Your dataset must have exactly two columns: <code>code</code> and <code>language</code></p>
        <pre style="background:#f5f7fa; padding:0.8rem; border-radius:8px; font-size:0.85rem;">
code,language
"def hello(): print('hi')",Python
"System.out.println(\"Hi\");",Java
"cout << \"Hello\" << endl;",C++
"console.log('Hello');",JavaScript
"SELECT * FROM users;",SQL
"echo Hello World",Bash
        </pre>
        <p>Minimum <b>10 rows</b> required. At least <b>2 samples per language</b> recommended.</p>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🧠 <b>CodeCosh</b> &nbsp;·&nbsp; "Know Your Code. Instantly."
    &nbsp;·&nbsp; Built with Python & Streamlit
    &nbsp;·&nbsp; © 2024
</div>
""", unsafe_allow_html=True)

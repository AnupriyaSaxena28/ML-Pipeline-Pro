import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, VarianceThreshold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                              r2_score, mean_absolute_error, mean_squared_error,
                              confusion_matrix, classification_report)
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────
st.set_page_config(page_title="ML Pipeline Pro", layout="wide", page_icon="🚀")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0a0c14 0%, #0d1117 50%, #0a0f1e 100%); }

/* Tab pill style */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 8px;
    flex-wrap: nowrap;
    overflow-x: auto;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 10px 18px;
    background: transparent;
    color: #8892a4;
    font-weight: 500;
    font-size: 13px;
    white-space: nowrap;
    border: 1px solid transparent;
    transition: all 0.25s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6c63ff, #00d2ff) !important;
    color: #fff !important;
    border-color: transparent !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.4);
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: rgba(108,99,255,0.15) !important;
    color: #c0cadc !important;
    border-color: rgba(108,99,255,0.3) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* Glass cards */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}
.metric-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(0,210,255,0.08));
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(108,99,255,0.25);
}
.metric-value { font-size: 2rem; font-weight: 700; color: #6c63ff; }
.metric-label { font-size: 0.8rem; color: #8892a4; margin-top: 4px; }

/* Hero problem selector */
.problem-hero {
    background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(0,210,255,0.1));
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
}
.hero-title { font-size: 2.2rem; font-weight: 700; color: #fff; margin-bottom: 8px; }
.hero-sub { font-size: 1rem; color: #8892a4; }

/* Step badge */
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6c63ff, #00d2ff);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Warning / info boxes */
.warn-box {
    background: rgba(255,170,0,0.1);
    border-left: 3px solid #ffaa00;
    border-radius: 8px;
    padding: 12px 16px;
    color: #ffcc55;
    margin: 8px 0;
}
.info-box {
    background: rgba(0,210,255,0.08);
    border-left: 3px solid #00d2ff;
    border-radius: 8px;
    padding: 12px 16px;
    color: #7ee8fa;
    margin: 8px 0;
}
.success-box {
    background: rgba(0,200,100,0.1);
    border-left: 3px solid #00c864;
    border-radius: 8px;
    padding: 12px 16px;
    color: #55dd99;
    margin: 8px 0;
}

/* Sidebar dark */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0c14 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.stMetric { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 12px; }
h1, h2, h3, h4 { color: #e8eaf6 !important; }
.stSelectbox > div, .stMultiSelect > div { background: rgba(255,255,255,0.04) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  HELPER: PLOTLY DARK TEMPLATE DEFAULTS
# ─────────────────────────────────────────
DARK_TEMPLATE = "plotly_dark"
GRAD_COLORS = px.colors.sequential.Plasma

def dark_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(family="Inter", color="#c0cadc"),
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig

def section(title, step=None):
    badge = f'<span class="step-badge">Step {step}</span><br>' if step else ""
    st.markdown(f'{badge}<h3 style="color:#e8eaf6;margin-top:4px">{title}</h3>', unsafe_allow_html=True)

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:20px 0 10px">'
                '<span style="font-size:2rem">🚀</span>'
                '<h2 style="color:#6c63ff;margin:4px 0">ML Pipeline Pro</h2>'
                '<p style="color:#8892a4;font-size:0.8rem">End-to-end machine learning</p>'
                '</div>', unsafe_allow_html=True)
    st.divider()
    problem_type = st.radio("🎯 Problem Type", ["Classification", "Regression"],
                             help="Choose the type of ML problem to solve")
    st.session_state["problem_type"] = problem_type
    st.divider()
    uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
    if uploaded_file:
        raw = pd.read_csv(uploaded_file)
        if "raw_df" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            st.session_state["raw_df"] = raw
            st.session_state["df"] = raw.copy()
            st.session_state["file_name"] = uploaded_file.name
            # Clear downstream state
            for k in ["split_done","model_trained","results","selected_features","final_features"]:
                st.session_state.pop(k, None)
        st.markdown(f'<div class="success-box">✅ Loaded <b>{uploaded_file.name}</b><br>'
                    f'{raw.shape[0]} rows × {raw.shape[1]} cols</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<p style="color:#8892a4;font-size:0.75rem;text-align:center">Navigate through the pipeline tabs →</p>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────
#  TOP BANNER
# ─────────────────────────────────────────
st.markdown("""
<div class="problem-hero">
  <div class="hero-title">🚀 Professional ML Pipeline Dashboard</div>
  <div class="hero-sub">Upload your dataset and walk through each pipeline stage sequentially</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  HORIZONTAL TABS
# ─────────────────────────────────────────
tabs = st.tabs([
    "📥 Data & PCA",
    "🔍 EDA",
    "🛠️ Engineering",
    "🎯 Feature Selection",
    "✂️ Data Split",
    "🤖 Model Select",
    "🏋️ Training & KFold",
    "📊 Performance",
    "⚙️ Hyper-Tuning"
])

# Guard: no data uploaded
if "df" not in st.session_state:
    for tab in tabs:
        with tab:
            st.markdown('<div class="info-box">📂 Please upload a CSV dataset in the sidebar to begin.</div>',
                        unsafe_allow_html=True)
    st.stop()

df = st.session_state["df"]
problem_type = st.session_state.get("problem_type", "Classification")

# ═══════════════════════════════════════════════
#  TAB 1 — DATA INPUT & PCA
# ═══════════════════════════════════════════════
with tabs[0]:
    section("Data Input & Geometric Shape (PCA)", step=1)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        target_col = st.selectbox("🎯 Select Target Variable", df.columns.tolist(),
                                   index=len(df.columns)-1)
        st.session_state["target_col"] = target_col

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feats_available = [c for c in num_cols if c != target_col]
        selected_features = st.multiselect("📌 Features for PCA", feats_available,
                                            default=feats_available[:min(8, len(feats_available))])
        st.session_state["selected_features"] = selected_features

        pca_dims = st.radio("PCA Dimensions", ["2D", "3D"], horizontal=True)
        st.divider()
        st.markdown(f"**Dataset Shape:** `{df.shape[0]} × {df.shape[1]}`")
        st.markdown(f"**Numeric Features:** `{len(num_cols)}`")
        st.markdown(f"**Missing Values:** `{df.isnull().sum().sum()}`")
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        st.markdown(f"**Categorical Cols:** `{len(cat_cols)}`")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        if len(selected_features) < 2:
            st.markdown('<div class="warn-box">⚠️ Select at least 2 numeric features for PCA.</div>',
                        unsafe_allow_html=True)
        else:
            pca_data = df[selected_features].dropna()
            scaled = StandardScaler().fit_transform(pca_data)

            if pca_dims == "2D":
                pca_2 = PCA(n_components=2)
                comps = pca_2.fit_transform(scaled)
                pca_df = pd.DataFrame(comps, columns=["PC1", "PC2"])
                pca_df["Target"] = df.loc[pca_data.index, target_col].astype(str).values
                fig = px.scatter(pca_df, x="PC1", y="PC2", color="Target",
                                  title=f"2D PCA — Variance Explained: {pca_2.explained_variance_ratio_.sum()*100:.1f}%",
                                  template=DARK_TEMPLATE, color_discrete_sequence=px.colors.qualitative.Vivid)
                st.plotly_chart(dark_fig(fig), use_container_width=True)

                ev = pca_2.explained_variance_ratio_ * 100
                fig2 = px.bar(x=["PC1","PC2"], y=ev, labels={"x":"Component","y":"Variance (%)"},
                               title="Explained Variance per Component", template=DARK_TEMPLATE,
                               color=ev, color_continuous_scale="Viridis")
                st.plotly_chart(dark_fig(fig2), use_container_width=True)
            else:
                if scaled.shape[1] < 3:
                    st.warning("Need ≥3 features for 3D PCA.")
                else:
                    pca_3 = PCA(n_components=3)
                    comps3 = pca_3.fit_transform(scaled)
                    pca_df3 = pd.DataFrame(comps3, columns=["PC1","PC2","PC3"])
                    pca_df3["Target"] = df.loc[pca_data.index, target_col].astype(str).values
                    fig3 = px.scatter_3d(pca_df3, x="PC1", y="PC2", z="PC3", color="Target",
                                          title="3D PCA Projection", template=DARK_TEMPLATE,
                                          color_discrete_sequence=px.colors.qualitative.Vivid)
                    st.plotly_chart(dark_fig(fig3), use_container_width=True)

    st.markdown("### 📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ═══════════════════════════════════════════════
#  TAB 2 — EDA
# ═══════════════════════════════════════════════
with tabs[1]:
    section("Exploratory Data Analysis", step=2)
    target_col = st.session_state.get("target_col", df.columns[-1])

    t1, t2, t3, t4 = st.tabs(["📊 Distributions", "🔥 Correlation", "❓ Missing Values", "📦 Box Plots"])

    num_df = df.select_dtypes(include=[np.number])

    with t1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("**Descriptive Statistics**")
        st.dataframe(num_df.describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        feat_dist = st.selectbox("Feature to plot distribution:", num_df.columns.tolist(), key="eda_dist")
        fig_hist = px.histogram(df, x=feat_dist, color=target_col if target_col in df.columns else None,
                                 nbins=40, template=DARK_TEMPLATE,
                                 color_discrete_sequence=px.colors.qualitative.Vivid,
                                 marginal="violin", title=f"Distribution of {feat_dist}")
        st.plotly_chart(dark_fig(fig_hist), use_container_width=True)

    with t2:
        corr = num_df.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              template=DARK_TEMPLATE, title="Feature Correlation Matrix",
                              aspect="auto")
        st.plotly_chart(dark_fig(fig_corr), use_container_width=True)

        # Target correlations
        if target_col in num_df.columns:
            tgt_corr = corr[target_col].drop(target_col).sort_values()
            fig_tc = px.bar(tgt_corr, orientation="h", template=DARK_TEMPLATE,
                             title=f"Feature Correlation with '{target_col}'",
                             color=tgt_corr.values, color_continuous_scale="RdBu_r")
            st.plotly_chart(dark_fig(fig_tc), use_container_width=True)

    with t3:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            st.markdown('<div class="success-box">✅ No missing values found in the dataset!</div>',
                        unsafe_allow_html=True)
        else:
            fig_miss = px.bar(x=missing.index, y=missing.values, template=DARK_TEMPLATE,
                               title="Missing Values per Feature",
                               labels={"x":"Feature","y":"Missing Count"},
                               color=missing.values, color_continuous_scale="Reds")
            st.plotly_chart(dark_fig(fig_miss), use_container_width=True)
            st.dataframe(pd.DataFrame({"Column": missing.index, "Missing": missing.values,
                                        "% Missing": (missing.values/len(df)*100).round(2)}),
                          use_container_width=True)

    with t4:
        box_feat = st.multiselect("Select features for box plot:",
                                   num_df.columns.tolist(),
                                   default=num_df.columns[:min(5,len(num_df.columns))].tolist(),
                                   key="eda_box")
        if box_feat:
            fig_box = px.box(df, y=box_feat, template=DARK_TEMPLATE,
                              title="Feature Box Plots",
                              color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(dark_fig(fig_box), use_container_width=True)

# ═══════════════════════════════════════════════
#  TAB 3 — DATA ENGINEERING & CLEANING
# ═══════════════════════════════════════════════
with tabs[2]:
    section("Data Engineering & Cleaning", step=3)
    df_eng = st.session_state["df"].copy()
    num_cols_eng = df_eng.select_dtypes(include=[np.number]).columns.tolist()

    # ── Imputation ──
    st.markdown("### 1️⃣ Handle Missing Values")
    col_imp1, col_imp2 = st.columns([2, 1])
    with col_imp1:
        cols_to_impute = st.multiselect("Select columns to impute:", num_cols_eng, key="imp_cols")
        imp_method = st.selectbox("Imputation method:", ["Mean", "Median", "Mode"], key="imp_method")
    with col_imp2:
        st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
        if st.button("✅ Apply Imputation", use_container_width=True):
            for col in cols_to_impute:
                if imp_method == "Mean":
                    val = df_eng[col].mean()
                elif imp_method == "Median":
                    val = df_eng[col].median()
                else:
                    val = df_eng[col].mode()[0]
                df_eng[col] = df_eng[col].fillna(val)
            st.session_state["df"] = df_eng
            st.markdown(f'<div class="success-box">✅ Applied <b>{imp_method}</b> imputation to {cols_to_impute}</div>',
                        unsafe_allow_html=True)

    st.divider()

    # ── Outlier Detection ──
    st.markdown("### 2️⃣ Outlier Detection")
    col_o1, col_o2 = st.columns([2, 1])
    with col_o1:
        outlier_method = st.selectbox("Detection Method:",
                                       ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"],
                                       key="outlier_method")
        outlier_features = st.multiselect("Features for outlier detection:", num_cols_eng,
                                           default=num_cols_eng[:min(4, len(num_cols_eng))],
                                           key="outlier_feats")
    with col_o2:
        detect_btn = st.button("🔍 Detect Outliers", use_container_width=True)

    if detect_btn and outlier_features:
        X_out = df_eng[outlier_features].dropna()
        X_scaled = StandardScaler().fit_transform(X_out)

        if outlier_method == "IQR":
            Q1 = X_out.quantile(0.25)
            Q3 = X_out.quantile(0.75)
            IQR = Q3 - Q1
            mask = ((X_out < (Q1 - 1.5 * IQR)) | (X_out > (Q3 + 1.5 * IQR))).any(axis=1)
            labels = np.where(mask, -1, 1)

        elif outlier_method == "Isolation Forest":
            iso = IsolationForest(contamination=0.1, random_state=42)
            labels = iso.fit_predict(X_scaled)

        elif outlier_method == "DBSCAN":
            db = DBSCAN(eps=0.5, min_samples=5)
            labels = db.fit_predict(X_scaled)
            labels = np.where(labels == -1, -1, 1)

        else:  # OPTICS
            op = OPTICS(min_samples=5)
            op.fit(X_scaled)
            labels = np.where(op.labels_ == -1, -1, 1)

        outlier_mask = labels == -1
        n_outliers = outlier_mask.sum()
        st.session_state["outlier_mask"] = outlier_mask
        st.session_state["outlier_indices"] = X_out.index[outlier_mask].tolist()

        st.markdown(f'<div class="warn-box">⚠️ Detected <b>{n_outliers}</b> outliers '
                    f'({n_outliers/len(X_out)*100:.1f}%) using {outlier_method}</div>',
                    unsafe_allow_html=True)

        # Visualize outliers on first two selected features
        if len(outlier_features) >= 2:
            vis_df = X_out.copy()
            vis_df["Outlier"] = np.where(outlier_mask, "Outlier", "Normal")
            fig_out = px.scatter(vis_df, x=outlier_features[0], y=outlier_features[1],
                                  color="Outlier", template=DARK_TEMPLATE,
                                  color_discrete_map={"Outlier":"#ff4d6d","Normal":"#6c63ff"},
                                  title=f"Outlier Scatter — {outlier_method}")
            st.plotly_chart(dark_fig(fig_out), use_container_width=True)

    if "outlier_indices" in st.session_state and len(st.session_state["outlier_indices"]) > 0:
        st.markdown("#### 🗑️ Remove Detected Outliers")
        col_r1, col_r2 = st.columns([2,1])
        with col_r1:
            st.dataframe(st.session_state["df"].iloc[st.session_state["outlier_indices"]].head(20),
                          use_container_width=True)
        with col_r2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">'
                        f'{len(st.session_state["outlier_indices"])}</div>'
                        f'<div class="metric-label">Outlier Rows</div></div>',
                        unsafe_allow_html=True)
            if st.button("🗑️ Remove Outliers", use_container_width=True, type="primary"):
                clean_df = st.session_state["df"].drop(
                    index=st.session_state["outlier_indices"], errors="ignore"
                ).reset_index(drop=True)
                st.session_state["df"] = clean_df
                st.session_state.pop("outlier_indices", None)
                st.markdown(f'<div class="success-box">✅ Outliers removed. New shape: '
                            f'<b>{clean_df.shape}</b></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  TAB 4 — FEATURE SELECTION
# ═══════════════════════════════════════════════
with tabs[3]:
    section("Feature Selection", step=4)
    target_col = st.session_state.get("target_col", st.session_state["df"].columns[-1])
    df_fs = st.session_state["df"].copy()
    num_fs = df_fs.select_dtypes(include=[np.number])
    X_fs = num_fs.drop(columns=[target_col], errors="ignore").dropna()
    y_fs = df_fs.loc[X_fs.index, target_col]

    fs_method = st.radio("🔬 Selection Method:",
                          ["All Features", "Variance Threshold", "Correlation Filter", "Information Gain"],
                          horizontal=True)

    final_features = X_fs.columns.tolist()

    if fs_method == "Variance Threshold":
        threshold = st.slider("Variance Threshold:", 0.0, 1.0, 0.1, 0.01)
        sel = VarianceThreshold(threshold=threshold)
        sel.fit(X_fs)
        final_features = X_fs.columns[sel.get_support()].tolist()
        variances = pd.Series(sel.variances_, index=X_fs.columns).sort_values(ascending=False)
        fig_var = px.bar(variances, labels={"value":"Variance","index":"Feature"},
                          title="Feature Variances", template=DARK_TEMPLATE,
                          color=variances.values, color_continuous_scale="Viridis")
        fig_var.add_hline(y=threshold, line_dash="dash", line_color="#ff4d6d",
                           annotation_text="Threshold")
        st.plotly_chart(dark_fig(fig_var), use_container_width=True)

    elif fs_method == "Correlation Filter":
        corr_thresh = st.slider("Drop features with |correlation| >", 0.5, 1.0, 0.9, 0.01)
        corr_mat = X_fs.corr().abs()
        upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
        drop_cols = [c for c in upper.columns if any(upper[c] > corr_thresh)]
        final_features = [c for c in X_fs.columns if c not in drop_cols]
        st.markdown(f'<div class="warn-box">Dropped <b>{len(drop_cols)}</b> highly-correlated features: '
                    f'{drop_cols}</div>', unsafe_allow_html=True)
        fig_corr2 = px.imshow(X_fs[final_features].corr(), text_auto=".2f",
                               color_continuous_scale="RdBu_r", template=DARK_TEMPLATE,
                               title="Filtered Correlation Matrix")
        st.plotly_chart(dark_fig(fig_corr2), use_container_width=True)

    elif fs_method == "Information Gain":
        try:
            if problem_type == "Classification":
                le = LabelEncoder()
                y_enc = le.fit_transform(y_fs.astype(str))
                scores = mutual_info_classif(X_fs, y_enc, random_state=42)
            else:
                scores = mutual_info_regression(X_fs, y_fs, random_state=42)
            imp = pd.Series(scores, index=X_fs.columns).sort_values(ascending=False)
            top_k = st.slider("Select top K features:", 1, len(imp), min(5, len(imp)))
            final_features = imp.head(top_k).index.tolist()
            fig_ig = px.bar(imp, labels={"value":"Information Gain","index":"Feature"},
                             title="Information Gain (w.r.t. Target)", template=DARK_TEMPLATE,
                             color=imp.values, color_continuous_scale="Plasma")
            st.plotly_chart(dark_fig(fig_ig), use_container_width=True)
        except Exception as e:
            st.error(f"Information Gain error: {e}")

    st.session_state["final_features"] = final_features
    st.markdown(f"**✅ Selected {len(final_features)} features:** `{final_features}`")

# ═══════════════════════════════════════════════
#  TAB 5 — DATA SPLIT
# ═══════════════════════════════════════════════
with tabs[4]:
    section("Train / Test Split", step=5)
    target_col = st.session_state.get("target_col", st.session_state["df"].columns[-1])
    final_features = st.session_state.get("final_features",
                      st.session_state["df"].select_dtypes(include=[np.number])
                        .drop(columns=[target_col], errors="ignore").columns.tolist())

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        test_pct = st.slider("Test set size (%):", 10, 40, 20)
        random_seed = st.number_input("Random Seed:", 0, 9999, 42)
        stratify_flag = st.checkbox("Stratify split (Classification only)",
                                     value=(problem_type=="Classification"))

    df_split = st.session_state["df"].dropna(subset=final_features + [target_col])
    X_all = df_split[final_features]
    y_all = df_split[target_col]

    strat = None
    if stratify_flag and problem_type == "Classification":
        strat = y_all

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=test_pct/100, random_state=int(random_seed), stratify=strat)
    except ValueError as e:
        # Some classes have too few samples for stratified split — fall back gracefully
        st.markdown(
            f'<div class="warn-box">⚠️ Stratified split failed (<code>{e}</code>). '
            f'Falling back to a regular (non-stratified) split.</div>',
            unsafe_allow_html=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=test_pct/100, random_state=int(random_seed), stratify=None)

    st.session_state.update({"X_train": X_train, "X_test": X_test,
                               "y_train": y_train, "y_test": y_test})

    with col_s2:
        sizes = {"Train": len(X_train), "Test": len(X_test)}
        fig_split = px.pie(values=list(sizes.values()), names=list(sizes.keys()),
                            hole=0.5, template=DARK_TEMPLATE,
                            color_discrete_sequence=["#6c63ff","#00d2ff"],
                            title="Dataset Split")
        st.plotly_chart(dark_fig(fig_split), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for card, label, val in [(c1,"Train Samples", len(X_train)),
                               (c2,"Test Samples", len(X_test)),
                               (c3,"Features", len(final_features))]:
        with card:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                        f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    if problem_type == "Classification":
        st.markdown("#### 📊 Class Distribution")
        dist_df = pd.DataFrame({"Split":["Train"]*len(y_train)+["Test"]*len(y_test),
                                 "Class": list(y_train.astype(str))+list(y_test.astype(str))})
        fig_cls = px.histogram(dist_df, x="Class", color="Split", barmode="group",
                                template=DARK_TEMPLATE, title="Class Balance",
                                color_discrete_sequence=["#6c63ff","#00d2ff"])
        st.plotly_chart(dark_fig(fig_cls), use_container_width=True)
    st.session_state["split_done"] = True

# ═══════════════════════════════════════════════
#  TAB 6 — MODEL SELECTION
# ═══════════════════════════════════════════════
with tabs[5]:
    section("Model Selection", step=6)

    if problem_type == "Classification":
        model_options = ["Random Forest", "Logistic Regression", "SVM - Linear",
                          "SVM - RBF", "SVM - Poly", "SVM - Sigmoid"]
    else:
        model_options = ["Random Forest", "Linear Regression", "SVR - Linear",
                          "SVR - RBF", "SVR - Poly"]

    col_m1, col_m2 = st.columns([1, 1])
    with col_m1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        model_name = st.selectbox("🤖 Select Model:", model_options)
        st.session_state["model_name"] = model_name

        model_info = {
            "Random Forest": "Ensemble of decision trees. Great generalization, handles non-linearity.",
            "Logistic Regression": "Linear probabilistic classifier. Fast & interpretable.",
            "Linear Regression": "Fits a linear hyperplane. Best for linear relationships.",
            "SVM - Linear": "Max-margin linear separator. Works well for high-dimensional data.",
            "SVM - RBF": "Radial Basis Function kernel. Handles non-linear boundaries.",
            "SVM - Poly": "Polynomial kernel SVM. Good for polynomial relationships.",
            "SVM - Sigmoid": "Sigmoid kernel — behaves like a neural network.",
            "SVR - Linear": "Linear kernel Support Vector Regression.",
            "SVR - RBF": "RBF kernel SVR. Robust non-linear regression.",
            "SVR - Poly": "Polynomial kernel SVR.",
        }
        st.markdown(f'<div class="info-box">ℹ️ {model_info.get(model_name,"")}</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m2:
        # Model comparison cards
        st.markdown("#### 📊 Model Characteristics")
        chars = {
            "Random Forest":       {"Accuracy": 9, "Speed": 6, "Interpretability": 5},
            "Logistic Regression": {"Accuracy": 6, "Speed": 9, "Interpretability": 9},
            "Linear Regression":   {"Accuracy": 5, "Speed": 9, "Interpretability": 10},
            "SVM - Linear":        {"Accuracy": 7, "Speed": 7, "Interpretability": 6},
            "SVM - RBF":           {"Accuracy": 9, "Speed": 5, "Interpretability": 4},
            "SVM - Poly":          {"Accuracy": 8, "Speed": 5, "Interpretability": 4},
            "SVM - Sigmoid":       {"Accuracy": 7, "Speed": 5, "Interpretability": 4},
            "SVR - Linear":        {"Accuracy": 6, "Speed": 7, "Interpretability": 7},
            "SVR - RBF":           {"Accuracy": 8, "Speed": 5, "Interpretability": 4},
            "SVR - Poly":          {"Accuracy": 7, "Speed": 5, "Interpretability": 4},
        }
        c = chars.get(model_name, {"Accuracy":5,"Speed":5,"Interpretability":5})
        fig_radar = go.Figure(go.Scatterpolar(
            r=list(c.values()), theta=list(c.keys()), fill='toself',
            line_color='#6c63ff', fillcolor='rgba(108,99,255,0.2)'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0,10])),
                                 template=DARK_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                                 font=dict(family="Inter", color="#c0cadc"))
        st.plotly_chart(fig_radar, use_container_width=True)

# ═══════════════════════════════════════════════
#  TAB 7 — TRAINING & KFOLD
# ═══════════════════════════════════════════════
with tabs[6]:
    section("Model Training & K-Fold Cross Validation", step=7)

    if not st.session_state.get("split_done"):
        st.markdown('<div class="warn-box">⚠️ Complete Data Split (Tab 5) first.</div>',
                    unsafe_allow_html=True)
    else:
        col_t1, col_t2 = st.columns([1, 2])

        with col_t1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            k_folds = st.slider("🔁 Number of Folds (K):", 2, 15, 5)
            model_name = st.session_state.get("model_name", model_options[0])
            st.markdown(f"**Model:** `{model_name}`")
            st.markdown(f"**Problem:** `{problem_type}`")
            st.markdown('</div>', unsafe_allow_html=True)
            train_btn = st.button("🚀 Train Model", use_container_width=True, type="primary")

        if train_btn:
            X_tr = st.session_state["X_train"]
            y_tr = st.session_state["y_train"]

            # Build model
            kernel_map = {"SVM - Linear":"linear","SVM - RBF":"rbf",
                          "SVM - Poly":"poly","SVM - Sigmoid":"sigmoid",
                          "SVR - Linear":"linear","SVR - RBF":"rbf","SVR - Poly":"poly"}

            if model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42) if problem_type=="Classification" \
                        else RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name.startswith("SVM"):
                k = kernel_map[model_name]
                model = SVC(kernel=k, probability=True, random_state=42)
            else:
                k = kernel_map[model_name]
                model = SVR(kernel=k)

            # Encode target if classification
            if problem_type == "Classification":
                le = LabelEncoder()
                y_te = st.session_state["y_test"]
                # Fit on ALL labels (train + test) so no label is "unseen"
                # when a non-stratified split leaves some classes only in test
                le.fit(pd.concat([y_tr, y_te]).astype(str))
                y_tr_enc = le.transform(y_tr.astype(str))
                y_te_enc = le.transform(y_te.astype(str))
                st.session_state["label_encoder"] = le
            else:
                y_tr_enc = y_tr
                y_te_enc = st.session_state["y_test"]

            # KFold CV
            with col_t2:
                with st.spinner("Running K-Fold Cross Validation…"):
                    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42) \
                         if problem_type == "Classification" \
                         else KFold(n_splits=k_folds, shuffle=True, random_state=42)

                    scoring = ["accuracy","f1_weighted"] if problem_type=="Classification" \
                               else ["r2","neg_mean_absolute_error"]
                    cv_res = cross_validate(model, X_tr, y_tr_enc, cv=cv, scoring=scoring, return_train_score=True)

                    model.fit(X_tr, y_tr_enc)

                st.session_state["model"] = model
                st.session_state["y_tr_enc"] = y_tr_enc
                st.session_state["y_te_enc"] = y_te_enc
                st.session_state["cv_res"] = cv_res
                st.session_state["model_trained"] = True

                # Plot fold scores
                score_key = "test_accuracy" if problem_type=="Classification" else "test_r2"
                train_key = "train_accuracy" if problem_type=="Classification" else "train_r2"

                fold_df = pd.DataFrame({
                    "Fold": list(range(1, k_folds+1)),
                    "Validation Score": cv_res[score_key],
                    "Train Score": cv_res[train_key]
                })
                fig_folds = px.line(fold_df, x="Fold", y=["Validation Score","Train Score"],
                                     markers=True, template=DARK_TEMPLATE,
                                     title=f"K-Fold Scores per Fold (K={k_folds})",
                                     color_discrete_sequence=["#00d2ff","#6c63ff"])
                fig_folds.update_traces(line=dict(width=2.5))
                st.plotly_chart(dark_fig(fig_folds), use_container_width=True)

                mean_val = cv_res[score_key].mean()
                std_val = cv_res[score_key].std()
                st.markdown(f'<div class="success-box">✅ Training complete! '
                            f'Mean CV Score: <b>{mean_val:.4f} ± {std_val:.4f}</b></div>',
                            unsafe_allow_html=True)
                st.balloons()

# ═══════════════════════════════════════════════
#  TAB 8 — PERFORMANCE METRICS
# ═══════════════════════════════════════════════
with tabs[7]:
    section("Performance Metrics & Overfitting Analysis", step=8)

    if not st.session_state.get("model_trained"):
        st.markdown('<div class="warn-box">⚠️ Train the model in the Training tab first.</div>',
                    unsafe_allow_html=True)
    else:
        model = st.session_state["model"]
        X_tr = st.session_state["X_train"]
        X_te = st.session_state["X_test"]
        y_tr_enc = st.session_state["y_tr_enc"]
        y_te_enc = st.session_state["y_te_enc"]
        cv_res = st.session_state["cv_res"]

        y_pred_train = model.predict(X_tr)
        y_pred_test  = model.predict(X_te)

        if problem_type == "Classification":
            train_acc = accuracy_score(y_tr_enc, y_pred_train)
            test_acc  = accuracy_score(y_te_enc, y_pred_test)
            train_f1  = f1_score(y_tr_enc, y_pred_train, average="weighted", zero_division=0)
            test_f1   = f1_score(y_te_enc, y_pred_test, average="weighted", zero_division=0)
            cv_score  = cv_res["test_accuracy"].mean()

            c1,c2,c3,c4 = st.columns(4)
            for card, label, val in [
                (c1,"Train Accuracy", f"{train_acc*100:.2f}%"),
                (c2,"Test Accuracy",  f"{test_acc*100:.2f}%"),
                (c3,"Test F1 (weighted)", f"{test_f1:.4f}"),
                (c4,"CV Score (mean)", f"{cv_score:.4f}")
            ]:
                with card:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                                f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

            # Overfitting check
            gap = train_acc - test_acc
            st.markdown("#### 🔎 Overfitting / Underfitting Diagnosis")
            if train_acc < 0.6:
                st.markdown('<div class="warn-box">🔴 <b>Underfitting</b>: Model is too simple or data insufficient.</div>',
                            unsafe_allow_html=True)
            elif gap > 0.15:
                st.markdown(f'<div class="warn-box">🟠 <b>Overfitting detected</b>: Train={train_acc:.2%}, '
                            f'Test={test_acc:.2%}, Gap={gap:.2%}. Consider regularization or pruning.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-box">✅ <b>Good fit</b>: Train={train_acc:.2%}, '
                            f'Test={test_acc:.2%}. Generalization looks solid.</div>',
                            unsafe_allow_html=True)

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                # Confusion Matrix
                cm = confusion_matrix(y_te_enc, y_pred_test)
                le = st.session_state.get("label_encoder")
                # Use the classes actually present in encoded data to avoid size mismatch
                present_encoded = np.unique(np.concatenate([y_te_enc, y_pred_test]))
                if le is not None:
                    labels_cm = [le.classes_[i] for i in present_encoded if i < len(le.classes_)]
                else:
                    labels_cm = [str(i) for i in present_encoded]
                # Re-compute CM only on present classes for consistent dimensions
                cm = confusion_matrix(y_te_enc, y_pred_test, labels=present_encoded)
                fig_cm = px.imshow(cm, text_auto=True, template=DARK_TEMPLATE,
                                    x=labels_cm, y=labels_cm,
                                    color_continuous_scale="Purples",
                                    title="Confusion Matrix")
                st.plotly_chart(dark_fig(fig_cm), use_container_width=True)

            with col_p2:
                # Train vs Test radar
                fig_rad = go.Figure()
                fig_rad.add_trace(go.Scatterpolar(
                    r=[train_acc, train_f1, cv_score, train_acc],
                    theta=["Accuracy","F1","CV","Accuracy"],
                    fill='toself', name="Train",
                    line_color='#6c63ff', fillcolor='rgba(108,99,255,0.25)'
                ))
                fig_rad.add_trace(go.Scatterpolar(
                    r=[test_acc, test_f1, cv_score, test_acc],
                    theta=["Accuracy","F1","CV","Accuracy"],
                    fill='toself', name="Test",
                    line_color='#00d2ff', fillcolor='rgba(0,210,255,0.2)'
                ))
                fig_rad.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
                                       template=DARK_TEMPLATE,
                                       paper_bgcolor="rgba(0,0,0,0)",
                                       font=dict(family="Inter", color="#c0cadc"),
                                       title="Train vs Test Radar")
                st.plotly_chart(fig_rad, use_container_width=True)

        else:  # Regression
            train_r2  = r2_score(y_tr_enc, y_pred_train)
            test_r2   = r2_score(y_te_enc, y_pred_test)
            test_mae  = mean_absolute_error(y_te_enc, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_te_enc, y_pred_test))

            c1,c2,c3,c4 = st.columns(4)
            for card, label, val in [
                (c1,"Train R²",   f"{train_r2:.4f}"),
                (c2,"Test R²",    f"{test_r2:.4f}"),
                (c3,"Test MAE",   f"{test_mae:.4f}"),
                (c4,"Test RMSE",  f"{test_rmse:.4f}")
            ]:
                with card:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                                f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

            gap = train_r2 - test_r2
            st.markdown("#### 🔎 Overfitting / Underfitting Diagnosis")
            if test_r2 < 0.4:
                st.markdown('<div class="warn-box">🔴 <b>Underfitting</b>: Low R². Model may be too simple.</div>',
                            unsafe_allow_html=True)
            elif gap > 0.15:
                st.markdown(f'<div class="warn-box">🟠 <b>Overfitting detected</b>: Train R²={train_r2:.4f}, '
                            f'Test R²={test_r2:.4f}, Gap={gap:.4f}.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-box">✅ <b>Good fit</b>: Test R²={test_r2:.4f}.</div>',
                            unsafe_allow_html=True)

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                fig_pred = px.scatter(x=y_te_enc, y=y_pred_test, template=DARK_TEMPLATE,
                                       labels={"x":"Actual","y":"Predicted"},
                                       title="Actual vs Predicted",
                                       color_discrete_sequence=["#6c63ff"])
                mn = min(float(y_te_enc.min()), float(y_pred_test.min()))
                mx = max(float(y_te_enc.max()), float(y_pred_test.max()))
                fig_pred.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                               line=dict(dash="dash",color="#ff4d6d"), name="Ideal"))
                st.plotly_chart(dark_fig(fig_pred), use_container_width=True)

            with col_p2:
                residuals = np.array(y_te_enc) - y_pred_test
                fig_res = px.histogram(residuals, nbins=30, template=DARK_TEMPLATE,
                                        title="Residuals Distribution",
                                        labels={"value":"Residual"},
                                        color_discrete_sequence=["#00d2ff"])
                st.plotly_chart(dark_fig(fig_res), use_container_width=True)

# ═══════════════════════════════════════════════
#  TAB 9 — HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════
with tabs[8]:
    section("Hyperparameter Tuning", step=9)

    if not st.session_state.get("model_trained"):
        st.markdown('<div class="warn-box">⚠️ Train a model first (Tab 7).</div>',
                    unsafe_allow_html=True)
    else:
        model = st.session_state["model"]
        model_name = st.session_state.get("model_name", "")
        X_tr = st.session_state["X_train"]
        y_tr_enc = st.session_state["y_tr_enc"]

        col_ht1, col_ht2 = st.columns([1, 2])
        with col_ht1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            search_method = st.radio("🔬 Search Method:", ["Grid Search", "Random Search"])
            n_iter_rs = st.slider("Iterations (Random Search):", 5, 50, 10) \
                         if search_method == "Random Search" else None
            cv_ht = st.slider("CV Folds for Tuning:", 2, 10, 3)
            tune_btn = st.button("⚙️ Run Tuning", use_container_width=True, type="primary")
            st.markdown('</div>', unsafe_allow_html=True)

        # Build param grids per model
        param_grids = {
            "Random Forest":   {"n_estimators":[50,100,200], "max_depth":[None,5,10,20],
                                 "min_samples_split":[2,5,10]},
            "Logistic Regression": {"C":[0.01,0.1,1,10,100], "solver":["liblinear","lbfgs"]},
            "Linear Regression":   {"fit_intercept":[True,False]},
            "SVM - Linear":   {"C":[0.01,0.1,1,10]},
            "SVM - RBF":      {"C":[0.1,1,10], "gamma":["scale","auto",0.01,0.1]},
            "SVM - Poly":     {"C":[0.1,1,10], "degree":[2,3,4]},
            "SVM - Sigmoid":  {"C":[0.1,1,10], "gamma":["scale","auto"]},
            "SVR - Linear":   {"C":[0.01,0.1,1,10]},
            "SVR - RBF":      {"C":[0.1,1,10], "gamma":["scale","auto",0.01]},
            "SVR - Poly":     {"C":[0.1,1,10], "degree":[2,3]},
        }

        param_grid = param_grids.get(model_name, {})
        with col_ht2:
            st.markdown("**Parameter Grid:**")
            st.json(param_grid)

        if tune_btn and param_grid:
            try:
                with st.spinner("Running hyperparameter search…"):
                    # Correctly pick base score key for classification vs regression
                    cv_res = st.session_state["cv_res"]
                    if problem_type == "Classification":
                        base_score = float(np.mean(cv_res["test_accuracy"]))
                    else:
                        base_score = float(np.mean(cv_res["test_r2"]))

                    if search_method == "Grid Search":
                        searcher = GridSearchCV(model, param_grid, cv=cv_ht,
                                                n_jobs=-1, verbose=0)
                    else:
                        searcher = RandomizedSearchCV(model, param_grid, n_iter=n_iter_rs,
                                                       cv=cv_ht, n_jobs=-1, random_state=42)
                    searcher.fit(X_tr, y_tr_enc)

                best_score = searcher.best_score_
                st.session_state["best_model"] = searcher.best_estimator_

                st.markdown(f'<div class="success-box">✅ Best Params: <b>{searcher.best_params_}</b><br>'
                            f'Best CV Score: <b>{best_score:.4f}</b></div>', unsafe_allow_html=True)

                # Before/After bar chart
                fig_cmp = go.Figure([
                    go.Bar(name="Before Tuning", x=["CV Score"], y=[base_score],
                            marker_color="#8892a4"),
                    go.Bar(name="After Tuning",  x=["CV Score"], y=[best_score],
                            marker_color="#6c63ff")
                ])
                fig_cmp.update_layout(barmode="group", template=DARK_TEMPLATE,
                                       paper_bgcolor="rgba(0,0,0,0)",
                                       font=dict(family="Inter",color="#c0cadc"),
                                       title="Tuning Impact: Before vs After")
                st.plotly_chart(dark_fig(fig_cmp), use_container_width=True)

                # Results table
                results_df = pd.DataFrame(searcher.cv_results_).sort_values("rank_test_score")
                show_cols = [c for c in results_df.columns
                              if c.startswith("param_") or c in
                              ["mean_test_score", "std_test_score", "rank_test_score"]]
                st.markdown("**Top Results:**")
                st.dataframe(results_df[show_cols].head(10), use_container_width=True)

            except Exception as e:
                st.markdown(f'<div class="warn-box">⚠️ Hyperparameter tuning failed: <code>{e}</code></div>',
                            unsafe_allow_html=True)
        elif tune_btn and not param_grid:
            st.markdown('<div class="warn-box">⚠️ No parameter grid defined for this model.</div>',
                        unsafe_allow_html=True)
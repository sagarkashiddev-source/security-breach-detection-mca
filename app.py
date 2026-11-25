import io, time, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Security Breach Detection | ML Research", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
:root { --ink:#e8eefc; --muted:#a9b7d0; --panel:#111d34; --line:rgba(148,163,184,.18); }
.stApp { background: radial-gradient(ellipse at 12% 0%, #172b4c 0%, #0b1324 42%, #080e1a 100%); color:#e8eefc; }
[data-testid="stHeader"] { background:rgba(8,14,26,.72); }
.block-container { padding:1.4rem 2.4rem 3rem; max-width:1800px; }
section[data-testid="stSidebar"] { background:#0b1425; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] * { color:#dce6f8; }
.hero { padding:34px 38px; border-radius:22px; background:linear-gradient(115deg,rgba(22,43,77,.98),rgba(15,83,105,.92)); border:1px solid rgba(125,211,252,.22); box-shadow:0 18px 55px rgba(0,0,0,.25); margin:0 0 22px; }
.hero h1 { color:white; margin:0; font-size:2.45rem; letter-spacing:-.04em; font-weight:800; }
.hero p { color:#d2e7f7; margin:10px 0 0; font-size:1.02rem; }
.eyebrow { color:#7dd3fc; font-size:.76rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; margin-bottom:9px; }
.section-kicker { color:#7dd3fc; font-size:.76rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; margin:1.2rem 0 .25rem; }
h1,h2,h3,h4,p,label,span { color:inherit; }
h2 { font-size:1.55rem !important; font-weight:750 !important; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#14233d,#101b2f); border:1px solid var(--line); border-radius:16px; padding:18px 20px; min-height:112px; }
[data-testid="stMetricLabel"] { color:#a9bdd7 !important; font-size:.9rem !important; }
[data-testid="stMetricValue"] { color:#f8fbff !important; font-size:2rem !important; }
[data-testid="stMetricDelta"] { font-size:.8rem !important; }
[data-testid="stFileUploader"] { background:#111f36; border:1px dashed #42617f; border-radius:14px; padding:12px; }
[data-testid="stExpander"], [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; }
.stButton>button, .stDownloadButton>button { border-radius:12px; min-height:48px; font-weight:750; }
.stButton>button[kind="primary"] { background:linear-gradient(90deg,#0e7490,#2563eb); border:0; }
[data-testid="stTabs"] button { font-size:1rem; font-weight:700; }
.small-note { font-size:.85rem; color:#a9b7d0; }
hr { border-color:var(--line); }
@media(max-width:900px){ .block-container{padding:1rem;} .hero h1{font-size:1.7rem;} }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="eyebrow">MCA Research Laboratory · SPPU · IIMS Chinchwad</div>
<h1>🛡️ Security Breach Detection</h1>
<p>Machine-learning research console · UNSW-NB15 network intrusion benchmark</p>
<p>Compare six supervised classifiers · Inspect detection errors · Export reproducible experiment evidence</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("Research workspace")
    app_mode = st.radio(
        "Choose a section",
        ["Research Portal", "Live ML Experiment"],
        index=0,
        help="Research Portal contains the paper-style explanation, methodology, diagrams and interactive illustrative charts. Live ML Experiment trains models on uploaded CSV files."
    )
    st.divider()
    st.header("Research configuration")
    st.write("**Task:** Binary classification")
    st.write("Normal = 0 · Attack = 1")
    max_train = st.select_slider("Maximum training records per run", options=[2000, 5000, 10000, 15000, 20000], value=10000)
    max_test = st.select_slider("Maximum testing records per run", options=[1000, 2000, 5000, 10000], value=5000)
    random_state = st.number_input("Random seed", min_value=1, max_value=99999, value=42)
    st.caption("For a fast demonstration, begin with 5,000 training and 2,000 testing records. The app reports the actual sample sizes used.")

if app_mode == "Research Portal":
    st.markdown("## Research paper, methodology and viva portal")
    st.caption("This section embeds the supplied interactive research portal. Charts marked illustrative are not experimental findings.")
    portal_file = Path(__file__).parent / "research_portal.html"
    if portal_file.exists():
        portal_html = portal_file.read_text(encoding="utf-8")
        st.components.v1.html(portal_html, height=19000, scrolling=True)
    else:
        st.error("research_portal.html is missing. Keep it in the same folder as app.py.")
    st.stop()

st.markdown("<div class='section-kicker'>01 / Data ingestion</div>", unsafe_allow_html=True)
st.subheader("Load the UNSW-NB15 dataset")
c1, c2 = st.columns(2)
with c1:
    train_file = st.file_uploader("Upload UNSW_NB15_training-set.csv", type=["csv"], key="train")
with c2:
    test_file = st.file_uploader("Upload UNSW_NB15_testing-set.csv", type=["csv"], key="test")

st.info("Download the two official CSV partition files from the UNSW-NB15 dataset page, then upload them here. This application does not fabricate research results.")
if not train_file or not test_file:
    st.markdown("### Dataset download")
    st.markdown("[Open the official UNSW-NB15 dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)")
    st.markdown("""
    1. Download `UNSW_NB15_training-set.csv`.
    2. Download `UNSW_NB15_testing-set.csv`.
    3. Return to this page and upload both files above.
    """)
    st.stop()

@st.cache_data(show_spinner=False)
def read_csv(uploaded_bytes):
    return pd.read_csv(io.BytesIO(uploaded_bytes), low_memory=False)

try:
    train_df = read_csv(train_file.getvalue())
    test_df = read_csv(test_file.getvalue())
except Exception as exc:
    st.error(f"Could not read the uploaded CSV files: {exc}")
    st.stop()

def find_target(df):
    for candidate in ["label", "Label", "class", "Class", "target"]:
        if candidate in df.columns:
            return candidate
    return None

target_col = find_target(train_df)
test_target = find_target(test_df)
if target_col is None or test_target is None:
    st.error("Could not find a binary target column. Expected a column named 'label'.")
    st.stop()

if target_col not in test_df.columns:
    st.error("Training and testing files do not have matching target columns.")
    st.stop()

def normalize_target(series):
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().all() and set(vals.unique()).issubset({0, 1}):
        return vals.astype(int)
    text = series.astype(str).str.strip().str.lower()
    mapping = {"normal": 0, "benign": 0, "0": 0, "attack": 1, "malicious": 1, "1": 1}
    mapped = text.map(mapping)
    if mapped.isna().any():
        raise ValueError("Target must contain binary values 0/1 or Normal/Attack.")
    return mapped.astype(int)

try:
    y_train_all = normalize_target(train_df[target_col])
    y_test_all = normalize_target(test_df[test_target])
except Exception as exc:
    st.error(str(exc))
    st.stop()

# Prevent target leakage from alternative attack labels and record identifiers.
drop_candidates = {target_col, "attack_cat", "attack_category", "id", "label"}
# IP addresses and port numbers can act as capture-specific identifiers; exclude them from this baseline.
drop_candidates.update({"srcip", "dstip", "sport", "dsport"})
feature_cols = [c for c in train_df.columns if c not in drop_candidates and c in test_df.columns]
if not feature_cols:
    st.error("No usable feature columns remain after excluding target and identifier columns.")
    st.stop()

X_train_all = train_df[feature_cols].copy()
X_test_all = test_df[feature_cols].copy()

# Ensure the same column order and numeric coercion for obvious numeric columns.
for col in feature_cols:
    if col not in X_test_all:
        X_test_all[col] = np.nan
X_test_all = X_test_all[feature_cols]

def sample_frame(X, y, cap, seed):
    if len(X) <= cap:
        idx = np.arange(len(X))
    else:
        # Reproducible random sample; stratification is attempted to retain both classes.
        from sklearn.model_selection import train_test_split
        try:
            idx, _ = train_test_split(
                np.arange(len(X)), train_size=cap, random_state=seed,
                stratify=y
            )
        except Exception:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X), size=cap, replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

X_train, y_train = sample_frame(X_train_all, y_train_all, int(max_train), int(random_state))
X_test, y_test = sample_frame(X_test_all, y_test_all, int(max_test), int(random_state)+1)

# Infer column types from the training sample.
numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_cols = [c for c in feature_cols if c not in numeric_cols]
# Coerce mixed columns consistently; high-cardinality identifiers have already been removed.
for col in numeric_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
for col in categorical_cols:
    X_train[col] = X_train[col].astype("string").fillna("__MISSING__")
    X_test[col] = X_test[col].astype("string").fillna("__MISSING__")

st.success("Both dataset files loaded successfully.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Training file rows", f"{len(train_df):,}")
m2.metric("Testing file rows", f"{len(test_df):,}")
m3.metric("Features used", len(feature_cols))
m4.metric("Attack categories", train_df["attack_cat"].nunique() if "attack_cat" in train_df.columns else "Not supplied")

st.markdown("### Dataset visual analytics")
ds_left, ds_right = st.columns(2)
with ds_left:
    class_counts = pd.DataFrame({"Class": ["Normal", "Attack"], "Records": [(y_train_all == 0).sum(), (y_train_all == 1).sum()]})
    class_fig = px.bar(class_counts, x="Class", y="Records", color="Class", title="Training dataset · Normal vs Attack records", text_auto=",")
    class_fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce6f8")
    st.plotly_chart(class_fig, use_container_width=True)
with ds_right:
    if "attack_cat" in train_df.columns:
        attack_counts = train_df["attack_cat"].fillna("Unknown").astype(str).value_counts().rename_axis("Attack category").reset_index(name="Records")
        attack_fig = px.bar(attack_counts, x="Attack category", y="Records", color="Attack category", title="Training dataset · attack category distribution", text_auto=",")
        attack_fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce6f8", xaxis_tickangle=-35)
        st.plotly_chart(attack_fig, use_container_width=True)
    else:
        st.info("The uploaded file has no attack_cat column, so attack-family distribution cannot be plotted.")

with st.expander("Dataset inspection · preview, class balance and excluded fields", expanded=False):
    dleft, dright = st.columns([1, 1.25])
    with dleft:
        st.write("**Training data preview**")
        st.dataframe(train_df.head(8), use_container_width=True)
    with dright:
        counts = pd.DataFrame({"Class": ["Normal", "Attack"], "Records": [(y_train_all==0).sum(), (y_train_all==1).sum()]})
        dist = px.pie(counts, names="Class", values="Records", hole=.58, title="Training partition · class composition")
        dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce6f8")
        st.plotly_chart(dist, use_container_width=True)
    st.caption("Excluded target / identifier fields: " + ", ".join(sorted(drop_candidates.intersection(train_df.columns))))

st.markdown("<div class='section-kicker'>02 / Experiment control</div>", unsafe_allow_html=True)
st.subheader("Train and compare six algorithms")
st.caption(f"Actual run size: {len(X_train):,} training records and {len(X_test):,} testing records. These may be sampled from the uploaded official partitions to keep the demonstration practical.")
run = st.button("🚀 Train all six models and evaluate", type="primary", use_container_width=True)

if run:
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        st.error("Both training and testing samples must contain Normal and Attack records. Increase the sample sizes and try again.")
        st.stop()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", MinMaxScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))
    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=18, min_samples_leaf=2, class_weight="balanced", random_state=int(random_state)),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=int(random_state)),
        "SVM (Linear)": LinearSVC(class_weight="balanced", random_state=int(random_state), max_iter=4000),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="brute", n_jobs=-1),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear", random_state=int(random_state))
    }

    results, fitted, matrices, reports, roc_data = [], {}, {}, {}, {}
    progress = st.progress(0)
    status = st.empty()
    for i, (name, estimator) in enumerate(models.items(), start=1):
        status.write(f"Training {name} ({i}/6)…")
        pipe = Pipeline([("preprocess", clone(preprocess)), ("model", estimator)])
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        pred = pipe.predict(X_test)
        pred_seconds = time.perf_counter() - start

        try:
            if hasattr(pipe, "predict_proba"):
                score = pipe.predict_proba(X_test)[:, 1]
            else:
                score = pipe.decision_function(X_test)
            auc = roc_auc_score(y_test, score)
            fpr_curve, tpr_curve, _ = roc_curve(y_test, score)
            roc_data[name] = (fpr_curve, tpr_curve, auc)
        except Exception:
            auc = np.nan

        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        row = {
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": auc,
            "FPR": fp / (fp + tn) if (fp + tn) else 0,
            "FNR": fn / (fn + tp) if (fn + tp) else 0,
            "Training time (s)": train_seconds,
            "Prediction time (s)": pred_seconds
        }
        results.append(row)
        fitted[name] = pipe
        matrices[name] = cm
        reports[name] = classification_report(y_test, pred, target_names=["Normal", "Attack"], zero_division=0)
        progress.progress(i / 6)

    result_df = pd.DataFrame(results).sort_values("F1-score", ascending=False).reset_index(drop=True)
    st.session_state["result_df"] = result_df
    st.session_state["fitted"] = fitted
    st.session_state["matrices"] = matrices
    st.session_state["reports"] = reports
    st.session_state["roc_data"] = roc_data
    st.session_state["test_X"] = X_test
    st.session_state["test_y"] = y_test
    st.session_state["feature_cols"] = feature_cols
    st.session_state["numeric_cols"] = numeric_cols
    st.session_state["categorical_cols"] = categorical_cols
    st.session_state["run_info"] = {
        "train_rows_used": len(X_train), "test_rows_used": len(X_test),
        "train_file_rows": len(train_df), "test_file_rows": len(test_df),
        "features_used": feature_cols, "excluded_columns": sorted(drop_candidates.intersection(train_df.columns)),
        "random_seed": int(random_state)
    }
    status.success("All six models trained and evaluated. Results below are calculated from this run.")

if "result_df" in st.session_state:
    result_df = st.session_state["result_df"]
    st.markdown("<div class='section-kicker'>03 / Research results</div>", unsafe_allow_html=True)
    st.subheader("Experimental results")
    st.caption("These metrics are calculated from the current trained models and selected test records. They are not pre-entered example values.")
    top = result_df.iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Top F1-score", f"{top['F1-score']:.4f}", top["Algorithm"])
    b.metric("Top accuracy", f"{top['Accuracy']:.4f}")
    c.metric("Top recall", f"{top['Recall']:.4f}")
    d.metric("Models evaluated", "6")
    display_df = result_df.copy()
    for col in ["Accuracy","Precision","Recall","F1-score","ROC-AUC","FPR","FNR"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    for col in ["Training time (s)", "Prediction time (s)"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.3f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    metric_choice = st.selectbox("Compare metric", ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "FPR", "FNR"])
    chart = px.bar(result_df, x="Algorithm", y=metric_choice, color="Algorithm", title=f"Algorithm comparison – {metric_choice}")
    chart.update_layout(showlegend=False, yaxis_range=[0,1] if metric_choice not in ["Training time (s)", "Prediction time (s)"] else None)
    chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce6f8")
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("All classification metrics · side-by-side")
    metric_columns = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    metrics_long = result_df.melt(id_vars="Algorithm", value_vars=metric_columns, var_name="Metric", value_name="Score")
    metrics_fig = px.bar(metrics_long, x="Algorithm", y="Score", color="Metric", barmode="group", title="Five core evaluation metrics by algorithm", hover_data={"Score": ":.4f"})
    metrics_fig.update_yaxes(range=[0, 1])
    metrics_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce6f8")
    st.plotly_chart(metrics_fig, use_container_width=True)

    error_long = result_df.melt(id_vars="Algorithm", value_vars=["FPR", "FNR"], var_name="Error type", value_name="Rate")
    error_fig = px.bar(error_long, x="Algorithm", y="Rate", color="Error type", barmode="group", title="False-positive and false-negative rates")
    error_fig.update_yaxes(tickformat=".1%", rangemode="tozero")
    error_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce6f8")
    st.plotly_chart(error_fig, use_container_width=True)

    runtime_left, runtime_right = st.columns(2)
    with runtime_left:
        runtime_fig = px.bar(result_df, x="Algorithm", y="Training time (s)", title="Training time by algorithm")
        runtime_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce6f8")
        st.plotly_chart(runtime_fig, use_container_width=True)
    with runtime_right:
        pred_fig = px.bar(result_df, x="Algorithm", y="Prediction time (s)", title="Prediction time by algorithm")
        pred_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce6f8")
        st.plotly_chart(pred_fig, use_container_width=True)

    st.subheader("ROC curve comparison")
    roc_fig = go.Figure()
    for model_name, (fpr_vals, tpr_vals, auc_val) in st.session_state.get("roc_data", {}).items():
        roc_fig.add_trace(go.Scatter(x=fpr_vals, y=tpr_vals, mode="lines", name=f"{model_name} · AUC {auc_val:.3f}"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random baseline", line=dict(dash="dash", color="#94a3b8")))
    roc_fig.update_layout(title="Receiver Operating Characteristic", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce6f8", height=460)
    st.plotly_chart(roc_fig, use_container_width=True)

    st.markdown("<div class='section-kicker'>04 / Error analysis</div>", unsafe_allow_html=True)
    st.subheader("Confusion matrices")
    selected_model = st.selectbox("Select model", list(st.session_state["matrices"].keys()))
    cm = st.session_state["matrices"][selected_model]
    cm_fig = px.imshow(cm, text_auto=True, x=["Predicted Normal", "Predicted Attack"],
                       y=["Actual Normal", "Actual Attack"], color_continuous_scale="Blues",
                       title=f"Confusion matrix – {selected_model}")
    st.plotly_chart(cm_fig, use_container_width=True)
    with st.expander("View confusion matrices for all six algorithms"):
        matrix_cols = st.columns(2)
        for matrix_index, (matrix_name, matrix_values) in enumerate(st.session_state["matrices"].items()):
            matrix_fig = px.imshow(matrix_values, text_auto=True, x=["Predicted Normal", "Predicted Attack"], y=["Actual Normal", "Actual Attack"], color_continuous_scale="Blues", title=matrix_name)
            matrix_fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce6f8")
            with matrix_cols[matrix_index % 2]:
                st.plotly_chart(matrix_fig, use_container_width=True)
    with st.expander("Detailed classification report"):
        st.code(st.session_state["reports"][selected_model])

    st.markdown("<div class='section-kicker'>05 / Interactive demonstration</div>", unsafe_allow_html=True)
    st.subheader("Test a network record")
    test_X = st.session_state["test_X"]
    if len(test_X):
        chosen = st.number_input("Testing record row (0-based)", min_value=0, max_value=len(test_X)-1, value=0, step=1)
        selected_row = test_X.iloc[[int(chosen)]]
        selected_pipe = st.session_state["fitted"][selected_model]
        prediction = int(selected_pipe.predict(selected_row)[0])
        label = "ATTACK" if prediction == 1 else "NORMAL"
        if prediction == 1:
            st.error(f"Model prediction: {label}")
        else:
            st.success(f"Model prediction: {label}")
        st.dataframe(selected_row, use_container_width=True)

    st.markdown("<div class='section-kicker'>06 / Research evidence</div>", unsafe_allow_html=True)
    st.subheader("Export evidence")
    run_info = st.session_state["run_info"]
    report_text = [
        "A Comparative Study of Machine Learning Algorithms for Security Breach Detection",
        "Dataset: UNSW-NB15 uploaded training and testing CSV partitions",
        f"Training file rows: {run_info['train_file_rows']}",
        f"Testing file rows: {run_info['test_file_rows']}",
        f"Rows used in this run: train={run_info['train_rows_used']}, test={run_info['test_rows_used']}",
        f"Features used: {len(run_info['features_used'])}",
        f"Random seed: {run_info['random_seed']}",
        "Identifier/target leakage exclusions: " + ", ".join(run_info["excluded_columns"]),
        "",
        result_df.to_string(index=False),
        "",
        "Note: results are specific to this run, preprocessing, sample sizes and model settings."
    ]
    st.download_button("Download results CSV", result_df.to_csv(index=False).encode("utf-8"), "actual_model_comparison_results.csv", "text/csv")
    st.download_button("Download experiment summary TXT", "\\n".join(report_text).encode("utf-8"), "experiment_summary.txt", "text/plain")

    export_figures = [chart, metrics_fig, error_fig, runtime_fig, pred_fig, roc_fig, cm_fig, class_fig]
    if "attack_cat" in train_df.columns:
        export_figures.append(attack_fig)
    graph_html_parts = ["<html><head><meta charset='utf-8'><title>Actual Research Experiment Charts</title></head><body><h1>Actual Research Experiment Charts</h1><p>Generated from this app run; values depend on the uploaded data and sample sizes.</p>"]
    for fig_index, fig in enumerate(export_figures):
        graph_html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if fig_index == 0 else False))
    graph_html_parts.append("</body></html>")
    st.download_button("Download all interactive graphs (HTML)", "\\n".join(graph_html_parts).encode("utf-8"), "research_experiment_graphs.html", "text/html")

st.divider()
st.caption("RESEARCH PROTOTYPE · Offline classification of labelled network-flow records. This application does not capture live packets, block attacks, or provide production-grade network protection.")

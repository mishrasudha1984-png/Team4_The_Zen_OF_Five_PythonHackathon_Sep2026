"""
Cardiac Failure Analytics Dashboard

Run from Anaconda Prompt, inside the folder that holds this file and the CSV:
    python -m streamlit run dashboard.py

Tabs: Descriptive | Prescriptive | Predictive | Summary
Sidebar filters apply to Descriptive, Prescriptive and Summary.
"""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Cardiac Failure Analytics", layout="wide")

# ------------------------------------------------------------------
# SETTINGS  (edit these if your file or columns are named differently)
# ------------------------------------------------------------------
CANDIDATE_FILES = [
    "Team4_The_Zen_Of_Five_cleaned_data.csv",
    "cf_master_dataset.csv",
]

TARGET = "re_admission_within_6_months"

NUM_FEATURES = [
    "cci_score", "systolic_blood_pressure", "diastolic_blood_pressure", "pulse",
    "brain_natriuretic_peptide", "creatinine_enzymatic_method", "sodium",
    "potassium", "hemoglobin", "white_blood_cell", "platelet", "lvef",
    "total_prescriptions_count",
]
CAT_FEATURES = [
    "gender", "bmi_category", "admission_way", "type_of_heart_failure",
    "nyha_cardiac_function_classification",
]

OUTCOMES = {
    "death_within_28_days": "28-day death",
    "re_admission_within_28_days": "28-day readmission",
    "death_within_3_months": "3-month death",
    "re_admission_within_3_months": "3-month readmission",
    "death_within_6_months": "6-month death",
    "re_admission_within_6_months": "6-month readmission",
    "return_to_emergency_department_within_6_months": "6-month ED return",
}

COMORBIDITIES = [
    "diabetes", "myocardial_infarction", "cerebrovascular_disease", "dementia",
    "chronic_obstructive_pulmonary_disease", "connective_tissue_disease",
    "peptic_ulcer_disease", "moderate_to_severe_chronic_kidney_disease",
    "hemiplegia", "leukemia", "malignant_lymphoma", "solid_tumor",
    "liver_disease", "aids",
]

LAB_OPTIONS = [
    "brain_natriuretic_peptide", "sodium", "potassium",
    "creatinine_enzymatic_method", "hemoglobin", "white_blood_cell", "platelet",
    "lactate", "high_sensitivity_troponin", "glucose_blood_gas",
    "systolic_blood_pressure", "diastolic_blood_pressure", "pulse", "lvef",
    "cci_score", "total_prescriptions_count", "dischargeday",
]

# ------------------------------------------------------------------
# DATA LOADING  (looks for the CSV in the same folder as this file)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path_str, mtime):
    return pd.read_csv(path_str)


def find_dataset():
    here = Path(__file__).resolve().parent
    for name in CANDIDATE_FILES:
        p = here / name
        if p.exists():
            return p
    return None


data_path = find_dataset()
if data_path is not None:
    raw = load_csv(str(data_path), data_path.stat().st_mtime)
    data_key = f"{data_path.name}-{data_path.stat().st_mtime}"
    source_name = data_path.name
else:
    st.warning("No dataset found next to dashboard.py. Upload your CSV below, "
               "or copy it into the same folder as dashboard.py.")
    uploaded = st.file_uploader("Upload dataset (CSV)", type="csv")
    if uploaded is None:
        st.stop()
    raw = pd.read_csv(uploaded)
    data_key = f"upload-{uploaded.name}-{len(raw)}"
    source_name = uploaded.name


def g(d, name):
    """Safe column getter: returns NaNs if the column is missing."""
    return d[name] if name in d.columns else pd.Series(np.nan, index=d.index)


# ------------------------------------------------------------------
# DERIVED COLUMNS (flags used across the prescriptive analysis)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def add_features(_df, key):
    d = _df.copy()

    # Guideline-directed medical therapy (GDMT) for HFrEF
    d["on_bb"] = ((g(d, "rx_metoprolol_succinate_sustained-release_tablet") == 1)
                  | (g(d, "rx_metoprolol_tartrate_injection") == 1))
    d["on_acei_arb"] = ((g(d, "rx_valsartan_dispersible_tablet") == 1)
                        | (g(d, "rx_benazepril_hydrochloride_tablet") == 1))
    d["on_mra"] = g(d, "rx_spironolactone_tablet") == 1
    d["gdmt_count"] = d[["on_bb", "on_acei_arb", "on_mra"]].sum(axis=1)
    d["hfref"] = g(d, "lvef") <= 40
    d["gdmt_gap"] = d["hfref"] & (d["gdmt_count"] < 3)
    d["lvef_band"] = pd.cut(g(d, "lvef"), bins=[0, 40, 49, 100],
                            labels=["HFrEF (<=40)", "HFmrEF (41-49)", "HFpEF (>=50)"])

    # Diuretics / renal function (cardiorenal check)
    d["on_diuretic"] = ((g(d, "rx_furosemide_tablet") == 1)
                        | (g(d, "rx_furosemide_injection") == 1)
                        | (g(d, "rx_torasemide_tablet") == 1))
    d["renal_impaired"] = ((g(d, "glomerular_filtration_rate") < 60)
                           | (g(d, "creatinine_enzymatic_method") > 110))

    # Escalation protocol (objective severity criteria)
    d["esc_count"] = ((g(d, "killip_grade") >= 3).astype(int)
                      + (g(d, "lactate") > 2.15).astype(int)
                      + (g(d, "gcs") < 15).astype(int))
    d["esc_flag"] = d["esc_count"] >= 2
    d["in_icu"] = g(d, "admission_ward") == "ICU"

    # Follow-up risk score (0-6)
    cci_q = g(d, "cci_score").quantile(0.75)
    bnp_q = g(d, "brain_natriuretic_peptide").quantile(0.75)
    d["followup_score"] = ((g(d, "killip_grade") >= 3).astype(int)
                           + (g(d, "sodium") < 135).astype(int)
                           + (g(d, "lactate") > 2).astype(int)
                           + (g(d, "cci_score") >= cci_q).astype(int)
                           + (g(d, "brain_natriuretic_peptide") >= bnp_q).astype(int)
                           + d["gdmt_gap"].astype(int))

    d["bad_outcome"] = ((g(d, "death_within_6_months") == 1)
                        | (g(d, "re_admission_within_6_months") == 1))
    return d


df_all = add_features(raw, data_key)
df = df_all  # replaced by the filtered view further below

# ------------------------------------------------------------------
# SMALL HELPERS
# ------------------------------------------------------------------
_chart_n = [0]


def show(fig, height=None):
    """Display a Plotly figure (works on old and new Streamlit versions)."""
    _chart_n[0] += 1
    if height:
        fig.update_layout(height=height)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    key = f"chart_{_chart_n[0]}"
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except TypeError:
        st.plotly_chart(fig, width="stretch", key=key)


def have(*cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.info("Skipped - column(s) not found in this dataset: " + ", ".join(missing))
        return False
    return True


def vc(series, order=None, sort_index=False):
    c = series.dropna().astype(str).value_counts()
    if order is not None:
        c = c.reindex([o for o in order if o in c.index])
    elif sort_index:
        c = c.sort_index()
    out = c.rename_axis("category").reset_index(name="patients")
    out["percent"] = out["patients"] / out["patients"].sum() * 100
    return out


def bar_counts(series, title, order=None, sort_index=False, horizontal=False, color="#4c78a8"):
    data = vc(series, order=order, sort_index=sort_index)
    if horizontal:
        data = data.sort_values("patients")
        fig = px.bar(data, x="patients", y="category", orientation="h",
                     text="patients", title=title)
    else:
        fig = px.bar(data, x="category", y="patients", text="patients", title=title)
    fig.update_traces(marker_color=color, textposition="outside")
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    show(fig)


def rate(mask, col):
    if col not in df.columns:
        return float("nan")
    sub = df.loc[mask, col]
    return float(sub.mean()) if len(sub) else float("nan")


def col_mean(frame, col):
    if col not in frame.columns or len(frame) == 0:
        return float("nan")
    return float(frame[col].mean())


def pct(x):
    return "n/a" if pd.isna(x) else f"{x * 100:.1f}%"


def fold(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return "n/a"
    return f"{a / b:.1f}x"


def age_key(s):
    try:
        return float(str(s).split("-")[0])
    except ValueError:
        return 1e9


def outcome_labels():
    return {k: v for k, v in OUTCOMES.items() if k in df.columns}


# ------------------------------------------------------------------
# MODEL TRAINING (defined early so the Predictive and Summary tabs share it)
# ------------------------------------------------------------------
def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUM_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATURES),
    ])


def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


@st.cache_resource(show_spinner="Training models...")
def train_models(_df, key):
    use_cols = NUM_FEATURES + CAT_FEATURES
    mask = _df[TARGET].notna()
    X = _df.loc[mask, use_cols].copy()
    y = _df.loc[mask, TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    pipes, metrics, probas, preds = {}, {}, {}, {}
    for name, model in make_models().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", model)])
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:, 1]
        yhat = pipe.predict(X_test)
        pipes[name], probas[name], preds[name] = pipe, p, yhat
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, yhat),
            "Precision": precision_score(y_test, yhat, zero_division=0),
            "Recall": recall_score(y_test, yhat, zero_division=0),
            "F1": f1_score(y_test, yhat, zero_division=0),
            "AUC": roc_auc_score(y_test, p),
        }
    return {"pipes": pipes, "metrics": metrics, "probas": probas, "preds": preds,
            "y_test": y_test, "n_train": len(X_train), "n_test": len(X_test)}


@st.cache_resource(show_spinner="Running 5-fold cross-validation...")
def cv_auc(_df, key):
    use_cols = NUM_FEATURES + CAT_FEATURES
    mask = _df[TARGET].notna()
    X = _df.loc[mask, use_cols].copy()
    y = _df.loc[mask, TARGET].astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {}
    for name, model in make_models().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", model)])
        out[name] = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
    return out


def importance_series(pipe):
    names = pd.Index(pipe.named_steps["prep"].get_feature_names_out()).str.replace(
        r"^(num|cat)__", "", regex=True)
    clf = pipe.named_steps["clf"]
    vals = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_importances_
    return pd.Series(vals, index=names)


# ------------------------------------------------------------------
# SIDEBAR: ABOUT + FILTERS
# ------------------------------------------------------------------
st.sidebar.header("About")
st.sidebar.write("Descriptive, prescriptive and predictive analysis of a "
                 "heart-failure patient cohort.")
st.sidebar.caption("Findings are associations in observational data, not proof of causation.")

st.sidebar.header("Filters")
st.sidebar.caption("Filters apply to the Descriptive, Prescriptive and Summary tabs. "
                   "The Predictive models are always trained on all patients.")

FILTERS = [
    ("gender", "Gender"),
    ("agecat", "Age band"),
    ("admission_way", "Admission pathway"),
    ("type_of_heart_failure", "Heart failure type"),
    ("nyha_cardiac_function_classification", "NYHA class"),
    ("killip_grade", "Killip grade"),
    ("admission_ward", "Admission ward"),
    ("lvef_band", "LVEF band"),
]


def filter_options(col):
    s = df_all[col].dropna()
    if s.empty:
        return []
    if col == "agecat":
        opts = sorted(s.unique(), key=age_key)
    elif str(s.dtype) == "category":
        present = set(s.unique())
        opts = [c for c in s.cat.categories if c in present]
    else:
        opts = sorted(s.unique())
        if pd.api.types.is_float_dtype(s) and (s % 1 == 0).all():
            opts = [int(o) for o in opts]
    return [o.item() if hasattr(o, "item") else o for o in opts]


keep = pd.Series(True, index=df_all.index)
for fcol, flabel in FILTERS:
    if fcol not in df_all.columns:
        continue
    fopts = filter_options(fcol)
    if not fopts:
        continue
    picked = st.sidebar.multiselect(flabel, fopts, default=fopts)
    if set(picked) != set(fopts):
        keep &= df_all[fcol].isin(picked)

df = df_all[keep].copy()
if len(df) == 0:
    st.warning("No patients match the current filters. Widen the filters in the sidebar.")
    st.stop()
st.sidebar.success(f"Showing {len(df):,} of {len(df_all):,} patients")

# ------------------------------------------------------------------
# HEADER + TABS
# ------------------------------------------------------------------
st.title("Cardiac Failure Analytics Dashboard")
st.caption(f"{len(df):,} of {len(df_all):,} patients shown | {raw.shape[1]} clinical variables | Data file: {source_name}")

tab_desc, tab_presc, tab_pred, tab_sum = st.tabs(
    ["Descriptive", "Prescriptive", "Predictive", "Summary"])

# ==================================================================
# TAB 1: DESCRIPTIVE
# ==================================================================
with tab_desc:
    st.header("Who are these patients?")

    k = st.columns(5)
    k[0].metric("Patients", f"{len(df):,}")
    k[1].metric("Female", pct((g(df, "gender") == "Female").mean()))
    k[2].metric("Median stay (days)", "n/a" if "dischargeday" not in df.columns else f"{df['dischargeday'].median():.0f}")
    k[3].metric("6-month death", pct(g(df, "death_within_6_months").mean()))
    k[4].metric("6-month readmission", pct(g(df, "re_admission_within_6_months").mean()))

    d_demo, d_clin, d_path, d_lab = st.tabs(
        ["Demographics", "Clinical severity", "Care pathway & outcomes", "Labs, glucose & medications"])

    # ---------------- Demographics ----------------
    with d_demo:
        c1, c2 = st.columns(2)
        with c1:
            if have("agecat"):
                age_order = sorted(df["agecat"].dropna().unique(), key=age_key)
                bar_counts(df["agecat"], "Age band", order=[str(a) for a in age_order])
        with c2:
            if have("gender"):
                fig = px.pie(vc(df["gender"]), names="category", values="patients",
                             hole=0.45, title="Gender")
                show(fig)

        c3, c4 = st.columns(2)
        with c3:
            if have("bmi_category"):
                bar_counts(df["bmi_category"], "BMI category",
                           order=["Underweight", "Normal", "Overweight", "Obese"], color="#54a24b")
        with c4:
            if have("occupation"):
                bar_counts(df["occupation"], "Occupation", horizontal=True, color="#f58518")

        st.subheader("Outcomes by age band")
        if have("agecat", "death_within_6_months", "re_admission_within_6_months"):
            age_order = [str(a) for a in sorted(df["agecat"].dropna().unique(), key=age_key)]
            tmp = (df.assign(agecat=df["agecat"].astype(str))
                   .groupby("agecat")[["death_within_6_months", "re_admission_within_6_months"]]
                   .mean().reindex(age_order) * 100)
            tmp = tmp.reset_index().melt(id_vars="agecat", var_name="Outcome", value_name="Rate (%)")
            tmp["Outcome"] = tmp["Outcome"].map({
                "death_within_6_months": "6-month death",
                "re_admission_within_6_months": "6-month readmission"})
            fig = px.bar(tmp, x="agecat", y="Rate (%)", color="Outcome", barmode="group",
                         title="6-month outcomes by age band")
            fig.update_layout(xaxis_title="Age band")
            show(fig)
            st.caption("The youngest bands contain very few patients, so their rates are unstable.")

    # ---------------- Clinical severity ----------------
    with d_clin:
        c1, c2 = st.columns(2)
        with c1:
            if have("type_of_heart_failure"):
                fig = px.pie(vc(df["type_of_heart_failure"]), names="category", values="patients",
                             hole=0.45, title="Heart failure phenotype (Left / Right / Both)")
                show(fig)
        with c2:
            if have("nyha_cardiac_function_classification"):
                bar_counts(df["nyha_cardiac_function_classification"].dropna().astype(int),
                           "NYHA functional class", sort_index=True, color="#e45756")

        c3, c4 = st.columns(2)
        with c3:
            if have("killip_grade"):
                bar_counts(df["killip_grade"].dropna().astype(int), "Killip grade at admission",
                           sort_index=True, color="#b279a2")
        with c4:
            if have("lvef"):
                fig = px.histogram(df, x="lvef", nbins=30, title="Left ventricular ejection fraction (LVEF)")
                fig.update_traces(marker_color="#4c78a8")
                show(fig)

        c5, c6 = st.columns(2)
        with c5:
            if have("lvef", "death_within_6_months"):
                band = (df.groupby("lvef_band", observed=True)["death_within_6_months"]
                        .agg(["mean", "count"]).reset_index())
                band["Death rate (%)"] = band["mean"] * 100
                fig = px.bar(band, x="lvef_band", y="Death rate (%)", text=band["count"].map(lambda n: f"n={n}"),
                             title="6-month death rate by LVEF band")
                fig.update_layout(xaxis_title=None)
                show(fig)
        with c6:
            present = [c for c in COMORBIDITIES if c in df.columns]
            if present:
                prev = (df[present].mean() * 100).sort_values().reset_index()
                prev.columns = ["Comorbidity", "Prevalence (%)"]
                fig = px.bar(prev, x="Prevalence (%)", y="Comorbidity", orientation="h",
                             title="Comorbidity prevalence")
                show(fig)
            else:
                st.info("No comorbidity columns found.")

    # ---------------- Care pathway & outcomes ----------------
    with d_path:
        c1, c2, c3 = st.columns(3)
        with c1:
            if have("admission_way"):
                bar_counts(df["admission_way"], "Admission pathway", color="#4c78a8")
        with c2:
            if have("admission_ward"):
                bar_counts(df["admission_ward"], "Admission ward", color="#72b7b2")
        with c3:
            if have("destinationdischarge"):
                bar_counts(df["destinationdischarge"], "Discharge destination", color="#9d755d")

        c4, c5 = st.columns(2)
        with c4:
            if have("admission_way", "destinationdischarge"):
                ct = pd.crosstab(df["admission_way"], df["destinationdischarge"], normalize="index") * 100
                fig = px.imshow(ct, text_auto=".1f", color_continuous_scale="Blues",
                                title="Discharge destination by admission pathway (row %)",
                                aspect="auto")
                show(fig)
        with c5:
            labels = outcome_labels()
            if labels:
                rates = pd.DataFrame({
                    "Outcome": list(labels.values()),
                    "Rate (%)": [df[c].mean() * 100 for c in labels],
                })
                fig = px.bar(rates, x="Rate (%)", y="Outcome", orientation="h",
                             text=rates["Rate (%)"].map(lambda v: f"{v:.1f}%"),
                             title="Outcome rates")
                show(fig)

        c6, c7 = st.columns(2)
        with c6:
            if have("dischargeday"):
                fig = px.histogram(df, x="dischargeday", nbins=40, title="Length of stay (days)")
                fig.update_traces(marker_color="#54a24b")
                show(fig)
        with c7:
            if have("dischargeday", "admission_ward"):
                fig = px.box(df, x="admission_ward", y="dischargeday", title="Length of stay by admission ward")
                fig.update_layout(xaxis_title=None, yaxis_title="Days")
                show(fig)

    # ---------------- Labs, glucose & medications ----------------
    with d_lab:
        st.subheader("Lab / vital explorer")
        lab_choices = [c for c in LAB_OPTIONS if c in df.columns]
        out_choices = outcome_labels()
        if lab_choices and out_choices:
            e1, e2 = st.columns(2)
            with e1:
                lab = st.selectbox("Variable", lab_choices,
                                   index=lab_choices.index("brain_natriuretic_peptide")
                                   if "brain_natriuretic_peptide" in lab_choices else 0)
            with e2:
                out_key = st.selectbox("Outcome", list(out_choices.keys()),
                                       format_func=lambda k_: out_choices[k_],
                                       index=list(out_choices.keys()).index("death_within_6_months")
                                       if "death_within_6_months" in out_choices else 0)
            pf = pd.DataFrame({"Outcome": df[out_key].map({0: "No", 1: "Yes"}),
                               lab: df[lab]}).dropna()
            fig = px.box(pf, x="Outcome", y=lab, points="outliers",
                         title=f"{lab} by {out_choices[out_key]}")
            show(fig)
            a = pf.loc[pf["Outcome"] == "No", lab]
            b = pf.loc[pf["Outcome"] == "Yes", lab]
            if len(a) > 5 and len(b) > 5:
                _, p = stats.mannwhitneyu(a, b)
                st.caption(f"Median {a.median():.2f} (No) vs {b.median():.2f} (Yes) | "
                           f"Mann-Whitney U p-value = {p:.4f} "
                           f"({'statistically significant' if p < 0.05 else 'not significant'} at 0.05)")
        else:
            st.info("Lab explorer skipped - required columns not found.")

        st.subheader("Sodium and mortality")
        if have("sodium", "death_within_6_months"):
            s = df["sodium"]
            grp = pd.Series(np.where(s < 135, "Low sodium (<135)", "Normal sodium"),
                            index=df.index).where(s.notna())
            tab = (df.assign(group=grp).groupby("group")["death_within_6_months"]
                   .agg(["mean", "count"]).reset_index())
            tab["Death rate (%)"] = tab["mean"] * 100
            fig = px.bar(tab, x="group", y="Death rate (%)",
                         text=tab.apply(lambda r: f"{r['Death rate (%)']:.1f}% (n={int(r['count'])})", axis=1),
                         title="6-month death rate by admission sodium")
            fig.update_layout(xaxis_title=None)
            show(fig)

        st.subheader("Admission glucose in non-diabetic patients (stress hyperglycemia)")
        if have("glucose_blood_gas", "diabetes"):
            glu = df["glucose_blood_gas"]
            nd = df["diabetes"] == 0
            n_nd = int((nd & glu.notna()).sum())
            n_stress = int((nd & (glu > 7.8)).sum())
            n_undx = int((nd & (glu >= 11.1)).sum())
            m = st.columns(3)
            m[0].metric("Non-diabetics with glucose recorded", f"{n_nd:,}")
            m[1].metric("Glucose above 7.8 mmol/L", pct(n_stress / n_nd) if n_nd else "n/a")
            m[2].metric("Glucose 11.1 or above (possible undiagnosed diabetes)", pct(n_undx / n_nd) if n_nd else "n/a")
            grp = np.select(
                [df["diabetes"] == 1, nd & (glu > 7.8), nd & (glu <= 7.8)],
                ["Diagnosed diabetic", "Non-diabetic, stress hyperglycemia", "Non-diabetic, normal glucose"],
                default="Glucose not recorded")
            gdf = df.assign(glucose_group=grp)
            gdf = gdf[gdf["glucose_group"] != "Glucose not recorded"]
            cols_ = [c for c in ["death_within_6_months", "re_admission_within_6_months"] if c in df.columns]
            if cols_ and len(gdf):
                gg = (gdf.groupby("glucose_group")[cols_].mean() * 100).reset_index()
                gg = gg.melt(id_vars="glucose_group", var_name="Outcome", value_name="Rate (%)")
                gg["Outcome"] = gg["Outcome"].map(OUTCOMES)
                fig = px.bar(gg, x="glucose_group", y="Rate (%)", color="Outcome", barmode="group",
                             title="6-month outcomes by glucose / diabetes group")
                fig.update_layout(xaxis_title=None)
                show(fig)

        st.subheader("Medication usage")
        rx = [c for c in df.columns if c.startswith("rx_")]
        if rx:
            use = (df[rx].mean() * 100).sort_values().reset_index()
            use.columns = ["Medication", "Patients on drug (%)"]
            use["Medication"] = use["Medication"].str.replace("rx_", "", regex=False)
            fig = px.bar(use, x="Patients on drug (%)", y="Medication", orientation="h",
                         title="Share of patients on each medication")
            show(fig, height=650)

        st.subheader("Correlation between numeric variables")
        corr_opts = [c for c in LAB_OPTIONS if c in df.columns]
        sel = st.multiselect("Variables to include", corr_opts, default=corr_opts[:10])
        if len(sel) >= 2:
            corr = df[sel].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, title="Correlation matrix")
            show(fig, height=600)

# ==================================================================
# TAB 2: PRESCRIPTIVE
# ==================================================================
with tab_presc:
    st.header("Who should be flagged for action?")

    p_esc, p_gdmt, p_lact, p_trop, p_renal, p_score = st.tabs([
        "Escalation protocol", "Medication gap (HFrEF)", "Lactate hidden risk",
        "Troponin + BNP", "Cardiorenal check", "Follow-up risk score"])

    # ---------------- Escalation protocol ----------------
    with p_esc:
        st.subheader("Should severe patients be escalated automatically?")
        st.caption("Criteria: Killip grade 3-4, lactate above 2.15 mmol/L, reduced consciousness (GCS below 15). "
                   "A patient meeting two or more is treated as a candidate for ICU or step-down review.")
        esc = df["esc_flag"]
        missed = esc & ~df["in_icu"]
        m = st.columns(4)
        m[0].metric("Meet 2 or more criteria", f"{int(esc.sum()):,}")
        m[1].metric("Of these, in ICU", f"{int((esc & df['in_icu']).sum()):,}")
        m[2].metric("Not escalated", f"{int(missed.sum()):,}")
        m[3].metric("Death rate: not escalated vs rest",
                    f"{pct(rate(missed, 'death_within_6_months'))} vs {pct(rate(~missed, 'death_within_6_months'))}")

        c1, c2 = st.columns(2)
        with c1:
            if have("admission_ward") and esc.any():
                bar_counts(df.loc[esc, "admission_ward"], "Where the severe patients were placed", color="#e45756")
        with c2:
            if have("death_within_6_months"):
                cmp_ = pd.DataFrame({
                    "Group": ["Met 2+ criteria, not in ICU", "Everyone else"],
                    "Death rate (%)": [rate(missed, "death_within_6_months") * 100,
                                       rate(~missed, "death_within_6_months") * 100]})
                fig = px.bar(cmp_, x="Group", y="Death rate (%)",
                             text=cmp_["Death rate (%)"].map(lambda v: f"{v:.1f}%"),
                             title="6-month death rate")
                fig.update_traces(marker_color=["#e45756", "#4c78a8"])
                fig.update_layout(xaxis_title=None)
                show(fig)
        st.info("Recommendation: any patient with two or more of these criteria should trigger an automatic ICU or step-down review, "
                "whichever ward they were first routed to.")

    # ---------------- GDMT gap ----------------
    with p_gdmt:
        st.subheader("HFrEF patients missing guideline medicines")
        st.caption("HFrEF = LVEF 40% or below. The three guideline classes are a beta-blocker, an ACE-inhibitor/ARB and an MRA (spironolactone).")
        hf = df[df["hfref"]].copy()
        if len(hf) == 0:
            st.info("No HFrEF patients found (LVEF column missing or empty).")
        else:
            gap = hf[hf["gdmt_gap"]]
            full = hf[~hf["gdmt_gap"]]
            m = st.columns(4)
            m[0].metric("HFrEF patients", f"{len(hf):,}")
            m[1].metric("On all 3 classes", f"{len(full):,}")
            m[2].metric("Missing at least 1", f"{len(gap):,} ({pct(len(gap) / len(hf))})")
            m[3].metric("Readmission: gap vs full",
                        f"{pct(gap['re_admission_within_6_months'].mean() if 're_admission_within_6_months' in gap else np.nan)} vs "
                        f"{pct(full['re_admission_within_6_months'].mean() if 're_admission_within_6_months' in full else np.nan)}")

            c1, c2 = st.columns(2)
            with c1:
                cnt = hf["gdmt_count"].astype(int).value_counts().sort_index().reset_index()
                cnt.columns = ["Classes on (0-3)", "Patients"]
                cnt["Classes on (0-3)"] = cnt["Classes on (0-3)"].astype(str)
                fig = px.bar(cnt, x="Classes on (0-3)", y="Patients", text="Patients",
                             title="How many of the 3 classes each HFrEF patient is on")
                show(fig)
            with c2:
                oc = [c for c in ["death_within_6_months", "re_admission_within_6_months"] if c in hf.columns]
                if oc:
                    o = hf.assign(Group=np.where(hf["gdmt_gap"], "Missing 1+ class", "Full GDMT (3/3)"))
                    o = (o.groupby("Group")[oc].mean() * 100).reset_index().melt(
                        id_vars="Group", var_name="Outcome", value_name="Rate (%)")
                    o["Outcome"] = o["Outcome"].map(OUTCOMES)
                    fig = px.bar(o, x="Group", y="Rate (%)", color="Outcome", barmode="group",
                                 title="6-month outcomes by medication completeness")
                    fig.update_layout(xaxis_title=None)
                    show(fig)

            st.markdown("**Patients to review before discharge**")
            cols_ = [c for c in ["inpatient_number", "lvef", "on_bb", "on_acei_arb", "on_mra", "gdmt_count"] if c in gap.columns]
            gap_view = gap[cols_].sort_values("gdmt_count")
            st.dataframe(gap_view)
            st.download_button("Download this list (CSV)", gap_view.to_csv(index=False).encode("utf-8"),
                               file_name="hfref_medication_gap.csv", mime="text/csv")
            st.caption("Associations only: sicker patients may also be less able to tolerate every drug, so review each case clinically.")

    # ---------------- Lactate hidden risk ----------------
    with p_lact:
        st.subheader("Does lactate reveal risk that Killip grade misses?")
        if have("lactate", "killip_grade", "death_within_6_months"):
            rec = df["lactate"].notna()
            elev = df["lactate"] > 2
            n_rec = int(rec.sum())
            stable = df[rec & df["killip_grade"].isin([1, 2])].copy()
            stable["status"] = np.where(stable["lactate"] > 2, "Elevated lactate (>2)", "Normal lactate")
            d_el = stable.loc[stable["status"] == "Elevated lactate (>2)", "death_within_6_months"].mean()
            d_no = stable.loc[stable["status"] == "Normal lactate", "death_within_6_months"].mean()
            m = st.columns(4)
            m[0].metric("Patients with lactate recorded", f"{n_rec:,} ({pct(n_rec / len(df))})")
            m[1].metric("Elevated lactate", pct(elev.sum() / n_rec) if n_rec else "n/a")
            m[2].metric("Killip 1-2 with high lactate", f"{int((stable['status'] == 'Elevated lactate (>2)').sum()):,}")
            m[3].metric("Death rate in Killip 1-2: high vs normal lactate", f"{pct(d_el)} vs {pct(d_no)}")

            c1, c2 = st.columns(2)
            with c1:
                bk = (df[rec].groupby("killip_grade")["lactate"].apply(lambda s: (s > 2).mean() * 100)
                      .reset_index(name="Elevated lactate (%)"))
                bk["killip_grade"] = bk["killip_grade"].astype(int).astype(str)
                fig = px.bar(bk, x="killip_grade", y="Elevated lactate (%)",
                             title="Elevated lactate rate by Killip grade")
                fig.update_layout(xaxis_title="Killip grade")
                show(fig)
            with c2:
                tmp = df[rec].copy()
                tmp["status"] = np.where(tmp["lactate"] > 2, "Elevated lactate", "Normal lactate")
                ln = (tmp.groupby(["killip_grade", "status"])["death_within_6_months"].mean() * 100).reset_index()
                ln["killip_grade"] = ln["killip_grade"].astype(int)
                fig = px.line(ln, x="killip_grade", y="death_within_6_months", color="status", markers=True,
                              title="6-month death rate by Killip grade and lactate status")
                fig.update_layout(xaxis_title="Killip grade", yaxis_title="Death rate (%)")
                show(fig)
            st.caption("Lactate is missing for about half of patients, and the highest Killip grades have few patients, so read the right-hand end of the line with care.")
            st.info("Recommendation: check lactate in patients who look stable by Killip grade, because elevated lactate marks extra risk.")

    # ---------------- Troponin + BNP ----------------
    with p_trop:
        st.subheader("Possible acute ischemic trigger: high troponin and high BNP together")
        if have("high_sensitivity_troponin", "brain_natriuretic_peptide", "death_within_6_months"):
            q = st.slider("Percentile cut-off for 'high' (applied to both markers)", 0.50, 0.95, 0.75, 0.05)
            trop = df["high_sensitivity_troponin"]
            bnp = df["brain_natriuretic_peptide"]
            tq = df_all["high_sensitivity_troponin"].quantile(q)
            bq = df_all["brain_natriuretic_peptide"].quantile(q)
            flag = (trop > tq) & (bnp > bq)
            if "myocardial_infarction" in df.columns:
                no_mi = flag & (df["myocardial_infarction"] == 0)
            else:
                no_mi = flag
            m = st.columns(4)
            m[0].metric("Flagged patients", f"{int(flag.sum()):,} ({pct(flag.mean())})")
            m[1].metric("Flagged, no MI on record", f"{int(no_mi.sum()):,}")
            m[2].metric("Death rate: flagged vs rest",
                        f"{pct(rate(flag, 'death_within_6_months'))} vs {pct(rate(~flag, 'death_within_6_months'))}")
            m[3].metric("Readmission: flagged vs rest",
                        f"{pct(rate(flag, 're_admission_within_6_months'))} vs {pct(rate(~flag, 're_admission_within_6_months'))}")

            plot = df.assign(Group=np.where(flag, "Flagged (both high)", "Others"))
            plot = plot.dropna(subset=["high_sensitivity_troponin", "brain_natriuretic_peptide"])
            plot = plot[plot["high_sensitivity_troponin"] > 0]
            fig = px.scatter(plot, x="high_sensitivity_troponin", y="brain_natriuretic_peptide",
                             color="Group", opacity=0.55, log_x=True,
                             color_discrete_map={"Flagged (both high)": "#e45756", "Others": "#9aa5b1"},
                             title="Troponin (log scale) versus BNP")
            xmin, xmax = float(plot["high_sensitivity_troponin"].min()), float(plot["high_sensitivity_troponin"].max())
            ymax = float(plot["brain_natriuretic_peptide"].max())
            fig.add_trace(go.Scatter(x=[tq, tq], y=[0, ymax], mode="lines",
                                     line=dict(dash="dash", color="grey"), name="Troponin cut-off"))
            fig.add_trace(go.Scatter(x=[xmin, xmax], y=[bq, bq], mode="lines",
                                     line=dict(dash="dot", color="grey"), name="BNP cut-off"))
            fig.update_layout(xaxis_title="High-sensitivity troponin", yaxis_title="BNP")
            show(fig)
            st.caption("Each patient has a single admission troponin value. A true acute event needs a rising and falling pattern over repeated draws, "
                       "so treat this as a screening flag for cardiology review, not a diagnosis. Chronic kidney disease also raises troponin.")

    # ---------------- Cardiorenal ----------------
    with p_renal:
        st.subheader("Are diuretics adjusted for patients with poor kidney function?")
        if have("creatinine_enzymatic_method", "bun_creatinine_ratio"):
            ri = df["renal_impaired"]
            du = df["on_diuretic"]
            m = st.columns(4)
            m[0].metric("Renal function impaired", f"{int(ri.sum()):,} ({pct(ri.mean())})")
            m[1].metric("On a diuretic: impaired", pct(du[ri].mean()) if ri.any() else "n/a")
            m[2].metric("On a diuretic: normal", pct(du[~ri].mean()) if (~ri).any() else "n/a")
            q75 = df_all["bun_creatinine_ratio"].quantile(0.75)
            hi = ri & du & (df["bun_creatinine_ratio"] > q75)
            m[3].metric("Impaired + high BUN/Cr + on diuretic", f"{int(hi.sum()):,}")

            oc = [c for c in ["death_within_6_months", "re_admission_within_6_months"] if c in df.columns]
            if oc:
                sub = df[du].assign(Renal=np.where(df.loc[du, "renal_impaired"], "Renal impaired", "Renal normal"))
                o = (sub.groupby("Renal")[oc].mean() * 100).reset_index().melt(
                    id_vars="Renal", var_name="Outcome", value_name="Rate (%)")
                o["Outcome"] = o["Outcome"].map(OUTCOMES)
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(o, x="Renal", y="Rate (%)", color="Outcome", barmode="group",
                                 title="Outcomes among diuretic users, by renal function")
                    fig.update_layout(xaxis_title=None)
                    show(fig)
                with c2:
                    if "dischargeday" in df.columns:
                        los = sub.groupby("Renal")["dischargeday"].mean().reset_index()
                        fig = px.bar(los, x="Renal", y="dischargeday", text=los["dischargeday"].map(lambda v: f"{v:.1f}"),
                                     title="Average length of stay among diuretic users (days)")
                        fig.update_layout(xaxis_title=None, yaxis_title="Days")
                        show(fig)
            st.caption("Renal impaired = GFR below 60 or creatinine above 110 umol/L. Sicker patients get more diuretics and have worse kidneys, "
                       "so this is an association, not proof that dosing caused the outcomes.")
            st.info("Recommendation: flag the impaired-kidney, high BUN/creatinine patients on diuretics for closer dose titration and a repeat creatinine before discharge.")

    # ---------------- Follow-up risk score ----------------
    with p_score:
        st.subheader("Which patients need the enhanced follow-up program?")
        st.caption("One point each for: Killip 3-4, sodium below 135, lactate above 2, top-quartile comorbidity burden (CCI), "
                   "top-quartile BNP, and a missing guideline medicine in HFrEF. Score runs 0 to 6.")
        threshold = st.slider("Flag patients with a score of at least", 0, 6, 2)
        flagged = df["followup_score"] >= threshold
        deaths_total = g(df, "death_within_6_months").sum()
        captured = (df.loc[flagged, "death_within_6_months"].sum() / deaths_total
                    if ("death_within_6_months" in df.columns and deaths_total) else float("nan"))

        m = st.columns(4)
        m[0].metric("Patients flagged", f"{int(flagged.sum()):,} ({pct(flagged.mean())})")
        m[1].metric("6-month deaths captured", pct(captured))
        m[2].metric("Death rate: flagged vs not",
                    f"{pct(rate(flagged, 'death_within_6_months'))} vs {pct(rate(~flagged, 'death_within_6_months'))}")
        m[3].metric("Readmission: flagged vs not",
                    f"{pct(rate(flagged, 're_admission_within_6_months'))} vs {pct(rate(~flagged, 're_admission_within_6_months'))}")

        oc = [c for c in ["death_within_6_months", "re_admission_within_6_months"] if c in df.columns]
        c1, c2 = st.columns(2)
        with c1:
            if oc:
                by = (df.groupby("followup_score")[oc].mean() * 100).reset_index().melt(
                    id_vars="followup_score", var_name="Outcome", value_name="Rate (%)")
                by["Outcome"] = by["Outcome"].map(OUTCOMES)
                fig = px.line(by, x="followup_score", y="Rate (%)", color="Outcome", markers=True,
                              title="Outcome rate by follow-up score")
                fig.update_layout(xaxis_title="Follow-up score (0-6)")
                show(fig)
        with c2:
            if "death_within_6_months" in df.columns and deaths_total:
                srt = df.sort_values("followup_score", ascending=False).reset_index(drop=True)
                srt["cum_patients"] = (srt.index + 1) / len(srt) * 100
                srt["cum_deaths"] = srt["death_within_6_months"].cumsum() / deaths_total * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=srt["cum_patients"], y=srt["cum_deaths"], mode="lines",
                                         name="Risk score", line=dict(color="crimson")))
                fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random selection",
                                         line=dict(dash="dash", color="grey")))
                fig.add_trace(go.Scatter(x=[flagged.mean() * 100], y=[captured * 100], mode="markers",
                                         name="Your cut-off", marker=dict(size=12, color="black")))
                fig.update_layout(title="Share of deaths captured by flagging top-scoring patients",
                                  xaxis_title="% of patients flagged", yaxis_title="% of deaths captured")
                show(fig)

        st.markdown("**Flagged patient list**")
        want = ["inpatient_number", "followup_score", "killip_grade", "sodium", "lactate", "cci_score",
                "brain_natriuretic_peptide", "lvef", "death_within_6_months", "re_admission_within_6_months"]
        view = df.loc[flagged, [c for c in want if c in df.columns]].sort_values("followup_score", ascending=False)
        st.dataframe(view)
        st.download_button("Download flagged list (CSV)", view.to_csv(index=False).encode("utf-8"),
                           file_name="flagged_patients.csv", mime="text/csv")
# ==================================================================
# TAB 3: PREDICTIVE
# ==================================================================
with tab_pred:
    st.header("Can admission-day data predict 6-month readmission?")
    st.caption("Models are trained on all patients with a 75/25 split. The sidebar filters only change "
               "the patient group used in section 4.")

    with st.expander("Question, hypotheses and parameters", expanded=False):
        st.markdown("""
**Question:** using only information available at admission, can we predict whether a heart-failure patient is readmitted within 6 months?

**H0:** admission features carry no predictive signal for 6-month readmission (AUC = 0.5).
**H1:** admission features carry predictive signal (AUC above 0.5).

**Target:** `re_admission_within_6_months` (0 = not readmitted, 1 = readmitted)

**Preprocessing:** median imputation and scaling for numeric features, most-frequent imputation and one-hot encoding for categorical features, 75/25 stratified train/test split, random seed 42.
""")
        st.write("Numeric features:", ", ".join(NUM_FEATURES))
        st.write("Categorical features:", ", ".join(CAT_FEATURES))

    if have(TARGET, *NUM_FEATURES, *CAT_FEATURES):
        res = train_models(df_all, data_key)
        y_test = res["y_test"]
        yv = y_test.values
        base = float(yv.mean())
        model_names = list(res["pipes"].keys())
        metrics_df = pd.DataFrame(res["metrics"]).T[["Accuracy", "Precision", "Recall", "F1", "AUC"]]

        # ============ 1. COMPARE MODELS ============
        st.subheader("1. Compare the four models")
        st.caption(f"Trained on {res['n_train']:,} patients, evaluated on {res['n_test']:,} held-out patients. "
                   "The best value in each column is highlighted.")
        st.dataframe(metrics_df.style.highlight_max(color="lightgreen", axis=0).format("{:.3f}"))

        best_f1 = metrics_df["F1"].idxmax()
        best_rec = metrics_df["Recall"].idxmax()
        best_auc = metrics_df["AUC"].idxmax()
        st.markdown(
            f"- Best **F1**: {best_f1} ({metrics_df.loc[best_f1, 'F1']:.3f})\n"
            f"- Best **recall** (catches the most real readmissions): {best_rec} ({metrics_df.loc[best_rec, 'Recall']:.3f})\n"
            f"- Best **AUC** (overall ranking ability): {best_auc} ({metrics_df.loc[best_auc, 'AUC']:.3f})\n"
            "- Every model beats random guessing (AUC 0.5), so H0 is rejected, but AUC around 0.6 means only moderate signal."
        )

        c1, c2 = st.columns(2)
        with c1:
            mm = (metrics_df[["Accuracy", "Precision", "Recall", "F1"]].reset_index()
                  .rename(columns={"index": "Model"})
                  .melt(id_vars="Model", var_name="Metric", value_name="Score"))
            fig = px.bar(mm, x="Model", y="Score", color="Metric", barmode="group",
                         title="Accuracy, precision, recall and F1 by model")
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(xaxis_title=None)
            show(fig)
        with c2:
            rad = (metrics_df.reset_index().rename(columns={"index": "Model"})
                   .melt(id_vars="Model", var_name="Metric", value_name="Score"))
            fig = px.line_polar(rad, r="Score", theta="Metric", color="Model", line_close=True,
                                range_r=[0, 1], title="Metric profile of each model")
            show(fig)

        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure()
            for name, p_ in res["probas"].items():
                fpr, tpr, _ = roc_curve(y_test, p_)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"{name} (AUC {metrics_df.loc[name, 'AUC']:.3f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                     line=dict(dash="dash", color="grey")))
            fig.update_layout(title="ROC curves", xaxis_title="False positive rate",
                              yaxis_title="True positive rate")
            show(fig)
        with c4:
            fig = go.Figure()
            for name, p_ in res["probas"].items():
                prec_, rec_, _ = precision_recall_curve(y_test, p_)
                ap = average_precision_score(y_test, p_)
                fig.add_trace(go.Scatter(x=rec_, y=prec_, mode="lines", name=f"{name} (AP {ap:.3f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[base, base], mode="lines",
                                     name=f"No skill ({base:.2f})", line=dict(dash="dash", color="grey")))
            fig.update_layout(title="Precision-recall curves", xaxis_title="Recall", yaxis_title="Precision")
            show(fig)

        # ============ 2. INSPECT ONE MODEL ============
        st.subheader("2. Inspect one model in detail")
        chosen = st.selectbox("Model to inspect", model_names, index=0)
        p = res["probas"][chosen]

        c5, c6 = st.columns(2)
        with c5:
            cm = confusion_matrix(y_test, res["preds"][chosen])
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            x=["No readmit", "Readmit"], y=["No readmit", "Readmit"],
                            labels=dict(x="Predicted", y="Actual"),
                            title=f"Confusion matrix - {chosen}")
            show(fig)
        with c6:
            imp = importance_series(res["pipes"][chosen])
            imp = imp.reindex(imp.abs().sort_values(ascending=False).head(12).index).iloc[::-1]
            if hasattr(res["pipes"][chosen].named_steps["clf"], "coef_"):
                colors = np.where(imp.values > 0, "Raises risk", "Lowers risk")
                fig = px.bar(x=imp.values, y=imp.index, orientation="h", color=colors,
                             color_discrete_map={"Raises risk": "#e45756", "Lowers risk": "#4c78a8"},
                             labels={"x": "Coefficient", "y": "", "color": ""},
                             title="Top predictors (coefficients)")
            else:
                fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                             labels={"x": "Importance", "y": ""},
                             title="Top predictors (feature importance)")
            show(fig)

        c7, c8 = st.columns(2)
        with c7:
            dist = pd.DataFrame({"Predicted risk": p,
                                 "Actual": np.where(yv == 1, "Readmitted", "Not readmitted")})
            fig = px.histogram(dist, x="Predicted risk", color="Actual", barmode="overlay",
                               opacity=0.6, nbins=30, title="Predicted risk by actual outcome",
                               color_discrete_map={"Readmitted": "#e45756", "Not readmitted": "#4c78a8"})
            show(fig)
            st.caption("The more the two colours separate, the better the model tells the groups apart.")
        with c8:
            try:
                frac_pos, mean_pred = calibration_curve(yv, p, n_bins=8, strategy="quantile")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name=chosen))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                                         line=dict(dash="dash", color="grey")))
                fig.update_layout(title="Calibration: predicted vs observed readmission",
                                  xaxis_title="Mean predicted probability", yaxis_title="Observed readmission rate")
                show(fig)
                st.caption("Models trained with balanced class weights over-state probabilities, so use them to rank patients rather than as literal percentages.")
            except ValueError:
                st.info("Calibration curve could not be computed for this model.")

        c9, c10 = st.columns(2)
        with c9:
            dec = pd.DataFrame({"p": p, "y": yv})
            dec["decile"] = pd.qcut(dec["p"].rank(method="first"), 10, labels=False) + 1
            tab = dec.groupby("decile")["y"].agg(["mean", "count"]).reset_index()
            tab["Readmission rate (%)"] = tab["mean"] * 100
            fig = px.bar(tab, x="decile", y="Readmission rate (%)",
                         title="Actual readmission rate by predicted-risk decile (10 = highest risk)")
            fig.add_trace(go.Scatter(x=[1, 10], y=[base * 100, base * 100], mode="lines",
                                     name="Cohort average", line=dict(dash="dash", color="black")))
            fig.update_layout(xaxis_title="Risk decile")
            show(fig)
            top_rate = tab.loc[tab["decile"] == tab["decile"].max(), "Readmission rate (%)"].iloc[0]
            low_rate = tab.loc[tab["decile"] == tab["decile"].min(), "Readmission rate (%)"].iloc[0]
            st.caption(f"Highest-risk decile: {top_rate:.1f}% readmitted. Lowest-risk decile: {low_rate:.1f}%. "
                       f"Cohort average: {base * 100:.1f}%.")
        with c10:
            order = np.argsort(-p)
            ys = yv[order]
            if ys.sum() > 0:
                cum_pat = np.arange(1, len(ys) + 1) / len(ys) * 100
                cum_pos = np.cumsum(ys) / ys.sum() * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cum_pat, y=cum_pos, mode="lines", name=chosen,
                                         line=dict(color="crimson")))
                fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random selection",
                                         line=dict(dash="dash", color="grey")))
                fig.update_layout(title="Cumulative gain: readmissions found by flagging top-risk patients",
                                  xaxis_title="% of patients flagged (highest predicted risk first)",
                                  yaxis_title="% of readmissions captured")
                show(fig)
                k20 = max(int(0.2 * len(ys)), 1)
                st.caption(f"Flagging the top 20% of patients by predicted risk finds {cum_pos[k20 - 1]:.0f}% of the readmissions.")

        # ============ 3. THRESHOLD EXPLORER ============
        st.subheader("3. Choose the decision threshold")
        st.caption("A patient is flagged as high risk when the predicted probability is at or above the threshold. "
                   "Lowering it catches more readmissions (higher recall) but creates more false alarms (lower precision).")
        thr = st.slider("Probability threshold", 0.05, 0.95, 0.50, 0.05)
        pred_t = (p >= thr).astype(int)
        t = st.columns(5)
        t[0].metric("Patients flagged", f"{int(pred_t.sum()):,} ({pred_t.mean() * 100:.0f}%)")
        t[1].metric("Precision", f"{precision_score(yv, pred_t, zero_division=0):.3f}")
        t[2].metric("Recall", f"{recall_score(yv, pred_t, zero_division=0):.3f}")
        t[3].metric("F1", f"{f1_score(yv, pred_t, zero_division=0):.3f}")
        t[4].metric("Accuracy", f"{accuracy_score(yv, pred_t):.3f}")

        c11, c12 = st.columns(2)
        with c11:
            grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
            sweep = pd.DataFrame({
                "Threshold": grid,
                "Precision": [precision_score(yv, (p >= g_).astype(int), zero_division=0) for g_ in grid],
                "Recall": [recall_score(yv, (p >= g_).astype(int), zero_division=0) for g_ in grid],
                "F1": [f1_score(yv, (p >= g_).astype(int), zero_division=0) for g_ in grid],
            }).melt(id_vars="Threshold", var_name="Metric", value_name="Score")
            fig = px.line(sweep, x="Threshold", y="Score", color="Metric", markers=True,
                          title=f"Precision, recall and F1 across thresholds - {chosen}")
            fig.add_trace(go.Scatter(x=[thr, thr], y=[0, 1], mode="lines", name="Your threshold",
                                     line=dict(dash="dash", color="black")))
            fig.update_yaxes(range=[0, 1])
            show(fig)
        with c12:
            cm_t = confusion_matrix(yv, pred_t, labels=[0, 1])
            fig = px.imshow(cm_t, text_auto=True, color_continuous_scale="Oranges",
                            x=["No readmit", "Readmit"], y=["No readmit", "Readmit"],
                            labels=dict(x="Predicted", y="Actual"),
                            title=f"Confusion matrix at threshold {thr:.2f}")
            show(fig)

        # ============ 4. SELECTED PATIENT GROUP ============
        st.subheader("4. Performance in the selected patient group")
        in_view = y_test.index.isin(df.index)
        n_view = int(in_view.sum())
        st.caption(f"{n_view:,} of the {len(yv):,} held-out test patients match the sidebar filters "
                   f"(threshold {thr:.2f} from section 3).")
        sub_y, sub_p = yv[in_view], p[in_view]
        if n_view >= 30 and len(np.unique(sub_y)) == 2:
            sub_pred = (sub_p >= thr).astype(int)
            auc_g = roc_auc_score(sub_y, sub_p)
            gm = st.columns(5)
            gm[0].metric("AUC in this group", f"{auc_g:.3f}", delta=f"{auc_g - metrics_df.loc[chosen, 'AUC']:+.3f} vs all")
            gm[1].metric("Recall", f"{recall_score(sub_y, sub_pred, zero_division=0):.3f}")
            gm[2].metric("Precision", f"{precision_score(sub_y, sub_pred, zero_division=0):.3f}")
            gm[3].metric("Actual readmission rate", pct(sub_y.mean()))
            gm[4].metric("Share flagged", pct(sub_pred.mean()))
        else:
            st.info("Too few test patients (or only one outcome class) in this group to score the model. Widen the sidebar filters.")

        seg_opts = [c for c in ["gender", "agecat", "type_of_heart_failure",
                                "nyha_cardiac_function_classification", "admission_way", "bmi_category"]
                    if c in df_all.columns]
        if seg_opts:
            seg = st.selectbox("Compare predicted risk across", seg_opts,
                               format_func=lambda c_: c_.replace("_", " "))
            plot = pd.DataFrame({"segment": df_all.loc[y_test.index, seg].astype(str).values,
                                 "Predicted risk": p, "Actual": yv})
            seg_order = sorted(plot["segment"].unique(), key=age_key) if seg == "agecat" else sorted(plot["segment"].unique())
            c13, c14 = st.columns(2)
            with c13:
                fig = px.box(plot, x="segment", y="Predicted risk", title="Predicted risk by group (test patients)",
                             category_orders={"segment": seg_order})
                fig.update_layout(xaxis_title=None)
                show(fig)
            with c14:
                agg = plot.groupby("segment").agg(pred=("Predicted risk", "mean"),
                                                  actual=("Actual", "mean"), n=("Actual", "size")).reset_index()
                agg = agg[agg["n"] >= 10]
                long = agg.melt(id_vars=["segment", "n"], value_vars=["pred", "actual"],
                                var_name="Measure", value_name="Rate")
                long["Measure"] = long["Measure"].map({"pred": "Mean predicted risk",
                                                       "actual": "Actual readmission rate"})
                long["Rate (%)"] = long["Rate"] * 100
                fig = px.bar(long, x="segment", y="Rate (%)", color="Measure", barmode="group",
                             title="Predicted vs actual readmission by group",
                             category_orders={"segment": seg_order})
                fig.update_layout(xaxis_title=None)
                show(fig)
            st.caption("Groups with fewer than 10 test patients are hidden. A large gap between predicted and actual rates means the model is less reliable for that group.")

        # ============ 5. CROSS-VALIDATION ============
        st.subheader("5. Robustness: 5-fold cross-validation")
        if st.checkbox("Run 5-fold cross-validation (takes about 30 seconds)"):
            cv = cv_auc(df_all, data_key)
            cv_df = pd.DataFrame({"Model": list(cv.keys()),
                                  "Mean AUC": [float(v.mean()) for v in cv.values()],
                                  "Std": [float(v.std()) for v in cv.values()]})
            fig = px.bar(cv_df, x="Model", y="Mean AUC", error_y="Std",
                         text=cv_df["Mean AUC"].map(lambda v: f"{v:.3f}"),
                         title="Cross-validated AUC (mean and standard deviation over 5 folds)")
            fig.add_trace(go.Scatter(x=cv_df["Model"], y=[0.5] * len(cv_df), mode="lines",
                                     name="Chance (0.5)", line=dict(dash="dash", color="grey")))
            fig.update_yaxes(range=[0.4, 0.8])
            fig.update_layout(xaxis_title=None)
            show(fig)
            st.caption("Each bar averages 5 different train/test splits. Bars with overlapping error bars are not clearly different from each other.")

        # ============ 6. SINGLE-PATIENT CALCULATOR ============
        st.subheader("6. Try it: single-patient readmission risk")
        st.caption("Illustrative only. With AUC around 0.6 this model has moderate discrimination and is not for clinical use.")

        def num_input(col, label):
            s = df_all[col].dropna()
            lo, hi, med = float(s.quantile(0.01)), float(s.quantile(0.99)), float(s.median())
            if lo >= hi:
                return med
            return st.slider(label, lo, hi, med)

        r1, r2, r3 = st.columns(3)
        with r1:
            v_na = num_input("sodium", "Sodium (mmol/L)")
            v_bnp = num_input("brain_natriuretic_peptide", "BNP (pg/mL)")
        with r2:
            v_cr = num_input("creatinine_enzymatic_method", "Creatinine (umol/L)")
            v_ef = num_input("lvef", "LVEF (%)")
        with r3:
            v_cci = num_input("cci_score", "CCI score")
            v_rx = num_input("total_prescriptions_count", "Number of prescriptions")

        s1, s2, s3 = st.columns(3)
        with s1:
            v_way = st.selectbox("Admission pathway", sorted(df_all["admission_way"].dropna().unique()))
        with s2:
            v_hf = st.selectbox("Heart failure type", sorted(df_all["type_of_heart_failure"].dropna().unique()))
        with s3:
            v_nyha = st.selectbox("NYHA class", sorted(df_all["nyha_cardiac_function_classification"].dropna().unique()))

        row = {c: df_all[c].median() for c in NUM_FEATURES}
        row.update({c: df_all[c].mode().iloc[0] for c in CAT_FEATURES})
        row.update({"sodium": v_na, "brain_natriuretic_peptide": v_bnp,
                    "creatinine_enzymatic_method": v_cr, "lvef": v_ef,
                    "cci_score": v_cci, "total_prescriptions_count": v_rx,
                    "admission_way": v_way, "type_of_heart_failure": v_hf,
                    "nyha_cardiac_function_classification": v_nyha})
        row_df = pd.DataFrame([row])[NUM_FEATURES + CAT_FEATURES]
        prob = float(res["pipes"][chosen].predict_proba(row_df)[0, 1])
        q1, q2 = st.columns(2)
        q1.metric(f"Predicted 6-month readmission risk ({chosen})", f"{prob * 100:.1f}%")
        q2.metric("Cohort average", f"{df_all[TARGET].mean() * 100:.1f}%")

# ==================================================================
# TAB 4: SUMMARY  (descriptive + prescriptive + predictive in one view)
# ==================================================================
with tab_sum:
    st.header("Summary of the analysis")
    st.caption("A one-page overview: what the data shows (descriptive), what can be forecast (predictive) "
               "and what to do about it (prescriptive). Every number is calculated live from the data.")
    if len(df) < len(df_all):
        st.info(f"Sidebar filters are active: the descriptive and prescriptive numbers cover {len(df):,} of "
                f"{len(df_all):,} patients. The predictive models are always trained on all patients.")

    k = st.columns(5)
    k[0].metric("Patients", f"{len(df):,}")
    k[1].metric("Biventricular HF", pct((g(df, "type_of_heart_failure") == "Both").mean()))
    k[2].metric("NYHA class 3-4", pct(g(df, "nyha_cardiac_function_classification").isin([3, 4]).mean()))
    k[3].metric("6-month death", pct(g(df, "death_within_6_months").mean()))
    k[4].metric("6-month readmission", pct(g(df, "re_admission_within_6_months").mean()))

    # ============ 1. DESCRIPTIVE ============
    st.divider()
    st.subheader("1. Descriptive - what does the data show?")

    biv = (g(df, "type_of_heart_failure") == "Both").mean()
    nyha34 = g(df, "nyha_cardiac_function_classification").isin([3, 4]).mean()
    killip34 = (g(df, "killip_grade") >= 3).mean()
    emerg = (g(df, "admission_way") == "Emergency").mean()
    icu_rate = df["in_icu"].mean()
    home = (g(df, "destinationdischarge") == "Home").mean()
    facility = (g(df, "destinationdischarge") == "HealthcareFacility").mean()

    low_na = g(df, "sodium") < 135
    norm_na = g(df, "sodium") >= 135
    d_low = rate(low_na, "death_within_6_months")
    d_norm = rate(norm_na, "death_within_6_months")

    glu = g(df, "glucose_blood_gas")
    nd = g(df, "diabetes") == 0
    n_nd = int((nd & glu.notna()).sum())
    stress = (nd & (glu > 7.8)).sum() / n_nd if n_nd else float("nan")

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"""
- **Advanced disease from the start.** {pct(biv)} of patients have both-sided heart failure, {pct(nyha34)} are already markedly limited (NYHA 3-4), and {pct(killip34)} arrive with signs of fluid on the lungs or shock (Killip 3-4).
- **Care pathway.** {pct(emerg)} were emergency admissions, only {pct(icu_rate)} were admitted to the ICU, {pct(home)} went home and {pct(facility)} were transferred to another facility.
- **Sodium is a warning sign.** 6-month death rate is {pct(d_low)} with low sodium (below 135) versus {pct(d_norm)} with normal sodium ({fold(d_low, d_norm)}).
- **Hidden high glucose.** {pct(stress)} of non-diabetic patients with a glucose reading were above 7.8 mmol/L at admission (stress hyperglycemia).
""")
    with right:
        labels = outcome_labels()
        if labels:
            rates = pd.DataFrame({"Outcome": list(labels.values()),
                                  "Rate (%)": [df[c].mean() * 100 for c in labels]})
            fig = px.bar(rates, x="Rate (%)", y="Outcome", orientation="h",
                         text=rates["Rate (%)"].map(lambda v: f"{v:.1f}%"), title="Outcome rates")
            show(fig, height=340)

    # ============ 2. PREDICTIVE ============
    st.divider()
    st.subheader("2. Predictive - can admission-day data forecast 6-month readmission?")

    if have(TARGET, *NUM_FEATURES, *CAT_FEATURES):
        res = train_models(df_all, data_key)
        mdf = pd.DataFrame(res["metrics"]).T[["Accuracy", "Precision", "Recall", "F1", "AUC"]]
        b_auc, b_f1 = mdf["AUC"].idxmax(), mdf["F1"].idxmax()
        b_rec, b_acc = mdf["Recall"].idxmax(), mdf["Accuracy"].idxmax()

        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"""
- **Question.** Using only information known at admission (vitals, labs, comorbidity burden, heart-failure type, medications), can we predict who will be readmitted within 6 months?
- **Hypothesis result.** All four models beat chance (AUC 0.5), so we reject the null hypothesis: admission data carries real predictive signal. The best AUC is {mdf.loc[b_auc, 'AUC']:.3f} ({b_auc}), which is moderate, not strong.
- **Best model for catching readmissions.** {b_f1} has the best F1 ({mdf.loc[b_f1, 'F1']:.3f}) and recall ({mdf.loc[b_rec, 'Recall']:.3f} for {b_rec}), meaning it finds the largest share of patients who really do come back.
- **Accuracy can mislead.** {b_acc} has the highest accuracy ({mdf.loc[b_acc, 'Accuracy']:.3f}) but recall of only {mdf.loc[b_acc, 'Recall']:.3f}, so it misses most readmissions. For a prevention program, missing a high-risk patient costs more than a false alarm.
- **Honest limit.** Readmission also depends on what happens after discharge (medication adherence, follow-up access), which is not in admission data.
""")
            st.dataframe(mdf.style.highlight_max(color="lightgreen", axis=0).format("{:.3f}"))
        with right:
            mm = (mdf[["Accuracy", "Precision", "Recall", "F1"]].reset_index()
                  .rename(columns={"index": "Model"})
                  .melt(id_vars="Model", var_name="Metric", value_name="Score"))
            fig = px.bar(mm, x="Model", y="Score", color="Metric", barmode="group",
                         title="Model comparison")
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(xaxis_title=None)
            show(fig, height=380)

    # ============ 3. PRESCRIPTIVE ============
    st.divider()
    st.subheader("3. Prescriptive - what should we do about it?")

    esc = df["esc_flag"]
    missed = esc & ~df["in_icu"]
    d_missed = rate(missed, "death_within_6_months")
    d_rest = rate(~missed, "death_within_6_months")

    hf = df[df["hfref"]]
    gap = hf[hf["gdmt_gap"]]
    full = hf[~hf["gdmt_gap"]]
    gap_share = len(gap) / len(hf) if len(hf) else float("nan")

    rec_l = g(df, "lactate").notna()
    stable = df[rec_l & g(df, "killip_grade").isin([1, 2])]
    d_hi_l = col_mean(stable[stable["lactate"] > 2] if "lactate" in stable.columns else stable.iloc[0:0], "death_within_6_months")
    d_lo_l = col_mean(stable[stable["lactate"] <= 2] if "lactate" in stable.columns else stable.iloc[0:0], "death_within_6_months")

    trop = g(df, "high_sensitivity_troponin")
    bnp = g(df, "brain_natriuretic_peptide")
    trop_cut = g(df_all, "high_sensitivity_troponin").quantile(0.75)
    bnp_cut = g(df_all, "brain_natriuretic_peptide").quantile(0.75)
    tflag = (trop > trop_cut) & (bnp > bnp_cut)
    t_no_mi = int((tflag & (g(df, "myocardial_infarction") == 0)).sum())

    du, ri = df["on_diuretic"], df["renal_impaired"]
    rd_imp = col_mean(df[du & ri], "re_admission_within_6_months")
    rd_nor = col_mean(df[du & ~ri], "re_admission_within_6_months")

    fl = df["followup_score"] >= 2
    deaths_all = g(df, "death_within_6_months").sum()
    deaths_fl = df.loc[fl, "death_within_6_months"].sum() if "death_within_6_months" in df.columns else 0
    capture = deaths_fl / deaths_all if deaths_all else float("nan")

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"""
- **Escalate severe patients automatically.** {int(missed.sum())} patients met two or more objective severity criteria (Killip 3-4, lactate above 2.15, reduced consciousness) but were not in the ICU; their 6-month death rate was {pct(d_missed)} versus {pct(d_rest)} for everyone else ({fold(d_missed, d_rest)}).
- **Close the medication gap in weak-pump heart failure.** {pct(gap_share)} of HFrEF patients ({len(gap)} of {len(hf)}) are missing at least one guideline medicine, and their 6-month readmission rate is {pct(col_mean(gap, 're_admission_within_6_months'))} versus {pct(col_mean(full, 're_admission_within_6_months'))} on the full set.
- **Check lactate even when the patient looks stable.** Among Killip 1-2 patients, death rate is {pct(d_hi_l)} with elevated lactate versus {pct(d_lo_l)} with normal lactate.
- **Send high troponin + high BNP patients to cardiology.** {int(tflag.sum())} patients are in the top quarter of both markers, {t_no_mi} of them with no heart attack on record (a screening flag from a single measurement, not a diagnosis).
- **Titrate diuretics carefully when kidneys are weak.** Among diuretic users, readmission is {pct(rd_imp)} with impaired renal function versus {pct(rd_nor)} with normal function.
- **Target scarce follow-up.** A simple 0-6 admission score flags {int(fl.sum()):,} patients ({pct(fl.mean())}), yet those patients account for {pct(capture)} of all 6-month deaths.
""")
    with right:
        if "death_within_6_months" in df.columns and deaths_all:
            srt = df.sort_values("followup_score", ascending=False).reset_index(drop=True)
            srt["cum_patients"] = (srt.index + 1) / len(srt) * 100
            srt["cum_deaths"] = srt["death_within_6_months"].cumsum() / deaths_all * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=srt["cum_patients"], y=srt["cum_deaths"], mode="lines",
                                     name="Risk score", line=dict(color="crimson")))
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random selection",
                                     line=dict(dash="dash", color="grey")))
            fig.add_trace(go.Scatter(x=[fl.mean() * 100], y=[capture * 100], mode="markers",
                                     name="Score of 2 or more", marker=dict(size=12, color="black")))
            fig.update_layout(title="Deaths captured by flagging top-scoring patients",
                              xaxis_title="% of patients flagged", yaxis_title="% of deaths captured")
            show(fig, height=420)

    st.divider()
    st.success(
        f"Bottom line: not every heart-failure patient needs the same level of attention. About {pct(fl.mean())} of patients "
        f"carry {pct(capture)} of the 6-month deaths, and fixable gaps (missed escalation, missing medicines, low sodium, "
        f"raised lactate) are visible on the day of admission.")
    st.caption("Findings are associations in observational data, not proof of causation. See the other tabs for the detail behind each number.")


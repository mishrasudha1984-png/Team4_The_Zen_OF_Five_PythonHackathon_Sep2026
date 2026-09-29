"""
Cardiac Failure Analytics Dashboard
Team 4 – The Zen of Five  |  September 2026 Hackathon
Tabs: Descriptive | Prescriptive | Predictive | Summary
Run:  streamlit run dashboard.py
Place this file next to  Team4_The_Zen_Of_Five_cleaned_data.csv
"""

# ── standard library ──────────────────────────────────────────────────────────
import io, re, warnings, zipfile
from pathlib import Path
warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM  – hospital colour palette
# ══════════════════════════════════════════════════════════════════════════════
NHS_BLUE   = "#003087"   # deep NHS / hospital blue – primary
MID_BLUE   = "#005EB8"   # standard clinical blue
LIGHT_BLUE = "#41B6E6"   # accent / highlights
TEAL       = "#007F84"   # positive / safe
WARM_RED   = "#AE2573"   # alert / danger
AMBER      = "#FFB81C"   # caution / warning
GREY       = "#768692"   # neutral
PALE       = "#F0F4F5"   # card background (NHS pale grey)
WHITE      = "#FFFFFF"

PALETTE_SEQ  = [MID_BLUE, LIGHT_BLUE, TEAL, AMBER, WARM_RED, "#4C6272", "#96C6EA"]
PALETTE_PAIR = [MID_BLUE, WARM_RED]

TEMPLATE = "simple_white"
px.defaults.color_discrete_sequence = PALETTE_SEQ
px.defaults.template = TEMPLATE

st.set_page_config(
    page_title="Cardiac Failure Analytics | The Zen of Five",
    page_icon="🏥", layout="wide"
)

st.markdown(f"""
<style>
/* ── layout ──────────────────────────────────────────── */
.block-container {{padding-top:1.4rem;padding-bottom:2rem;max-width:1280px}}

/* ── typography ──────────────────────────────────────── */
body, .stMarkdown {{font-family:'Segoe UI',Arial,sans-serif;color:#1d2b36}}
h1,h2,h3 {{color:{NHS_BLUE}}}

/* ── section header ──────────────────────────────────── */
.sec-hdr {{
    background:linear-gradient(90deg,{NHS_BLUE} 0%,{MID_BLUE} 55%,{LIGHT_BLUE} 100%);
    color:#fff;padding:0.85rem 1.2rem;border-radius:8px;
    font-size:1.3rem;font-weight:700;margin:1.4rem 0 0.7rem 0;
    letter-spacing:0.02em;
}}

/* ── question card ───────────────────────────────────── */
.qcard {{
    background:{PALE};border-left:5px solid {MID_BLUE};
    border-radius:6px;padding:0.75rem 1.1rem;margin-bottom:0.5rem;
}}
.qcard h4 {{margin:0 0 0.3rem 0;color:{NHS_BLUE};font-size:1rem}}
.qcard p  {{margin:0;color:#374151;font-size:0.88rem;line-height:1.5}}

/* ── insight box ─────────────────────────────────────── */
.insight {{
    background:#fffbeb;border-left:5px solid {AMBER};
    border-radius:6px;padding:0.65rem 1rem;margin-top:0.3rem;
    font-size:0.86rem;color:#4b3000;
}}

/* ── sticky TOC ──────────────────────────────────────── */
.toc-bar {{
    position:sticky;top:0;z-index:999;
    background:rgba(255,255,255,0.97);
    border-bottom:2px solid {MID_BLUE};
    padding:0.45rem 0.7rem;margin-bottom:0.9rem;
    font-size:0.82rem;white-space:nowrap;overflow-x:auto;
}}
.toc-bar a {{color:{MID_BLUE};text-decoration:none;font-weight:600;margin-right:0.8rem}}
.toc-bar a:hover {{text-decoration:underline}}
.toc-lbl {{color:{GREY};margin-right:0.5rem;font-weight:600}}
.anchor  {{scroll-margin-top:60px}}

/* ── drill-down panel ────────────────────────────────── */
.drill {{
    background:#e8f4fd;border-left:5px solid {LIGHT_BLUE};
    border-radius:6px;padding:0.65rem 1rem;margin-top:0.3rem;font-size:0.86rem;
}}

/* ── metric tweaks ───────────────────────────────────── */
[data-testid="stMetricValue"]  {{color:{NHS_BLUE}}}

/* ── PRINT / PDF ─────────────────────────────────────── */
@media print {{
    [data-testid="stSidebar"],[data-testid="collapsedControl"],
    header,#MainMenu,footer,.toc-bar {{display:none !important}}
    .block-container {{max-width:100% !important;padding:0.4rem}}
    .sec-hdr {{break-before:page}}
    .qcard,.insight {{break-inside:avoid}}
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
EXPORT_FIGS: list = []   # (slug, fig) – reset each script run
_fig_n = {"n": 0}


def _slug(text: str, fb: str) -> str:
    t = re.sub(r"[^\w\- ]+", "", (text or fb).strip()).replace(" ", "_")
    return t[:55] or fb


def _cfg(name: str) -> dict:
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {"format": "png", "filename": name, "scale": 3},
    }


def _prep(fig, h):
    fig.update_layout(
        height=h, margin=dict(l=8, r=8, t=42, b=8),
        font=dict(family="Segoe UI,Arial,sans-serif", size=12),
        title_font=dict(size=13, color=NHS_BLUE),
    )


def show(fig, h=370, name=None):
    """Render a chart with PNG toolbar + register for bulk export."""
    _fig_n["n"] += 1
    try:    title_text = fig.layout.title.text or ""
    except: title_text = ""
    slug = _slug(name or title_text, f"chart_{_fig_n['n']:02d}")
    EXPORT_FIGS.append((f"{_fig_n['n']:02d}_{slug}", fig))
    _prep(fig, h)
    try:    st.plotly_chart(fig, use_container_width=True, config=_cfg(slug))
    except TypeError:
        st.plotly_chart(fig, width="stretch", config=_cfg(slug))


def drill_bar(fig, h=370, name=None):
    """Like show() but supports click-to-drill (Streamlit ≥ 1.35)."""
    _fig_n["n"] += 1
    try:    title_text = fig.layout.title.text or ""
    except: title_text = ""
    slug = _slug(name or title_text, f"chart_{_fig_n['n']:02d}")
    EXPORT_FIGS.append((f"{_fig_n['n']:02d}_{slug}", fig))
    _prep(fig, h)
    key = f"drill_{slug}_{_fig_n['n']}"
    cfg = _cfg(slug)
    try:
        ev = st.plotly_chart(fig, use_container_width=True, config=cfg,
                             on_select="rerun", selection_mode="points", key=key)
        try:    pts = list(ev["selection"]["points"])
        except Exception:
            try: pts = list(ev.selection.points)
            except: pts = []
    except TypeError:
        try:    st.plotly_chart(fig, use_container_width=True, config=cfg)
        except TypeError: st.plotly_chart(fig, width="stretch", config=cfg)
        pts = []
    return pts


def sec(text):
    st.markdown(f'<div class="sec-hdr">{text}</div>', unsafe_allow_html=True)


def qcard(title, reasoning):
    m = re.match(r"^\s*([QP]\d+)", title)
    aid = m.group(1).lower() if m else None
    anch = f'<div id="{aid}" class="anchor"></div>' if aid else ""
    st.markdown(f'{anch}<div class="qcard"><h4>{title}</h4><p>{reasoning}</p></div>',
                unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight"><b>Insight:</b> {text}</div>',
                unsafe_allow_html=True)


def drill_panel(html):
    st.markdown(f'<div class="drill">{html}</div>', unsafe_allow_html=True)


def toc(items):
    links = "".join(f'<a href="#{a}">{lbl}</a>' for a, lbl in items)
    st.markdown(f'<div class="toc-bar"><span class="toc-lbl">Jump to:</span>{links}</div>',
                unsafe_allow_html=True)


def pct(x):
    return "n/a" if pd.isna(x) else f"{x*100:.1f}%"


def safe_div(a, b):
    return a/b if b else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
CANDIDATES = [
    "Team4_The_Zen_Of_Five_cleaned_data.csv",
    "cf_master_dataset.csv",
]


@st.cache_data(show_spinner=False)
def load(path, mtime):
    return pd.read_csv(path)


def find_file():
    base = Path(__file__).resolve().parent
    for name in CANDIDATES:
        p = base / name
        if p.exists():
            return p
    return None


fp = find_file()
if fp:
    df_all = load(str(fp), fp.stat().st_mtime)
    src_name = fp.name
else:
    st.warning("Dataset not found next to dashboard.py – upload it below.")
    up = st.file_uploader("Upload CSV", type="csv")
    if not up:
        st.stop()
    df_all = pd.read_csv(up)
    src_name = up.name

AGE_ORDER = sorted(df_all["agecat"].dropna().unique(),
                   key=lambda s: float(str(s).split("-")[0]))
BMI_ORDER = ["Underweight", "Normal", "Overweight", "Obese"]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – FILTERS
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown(f"## 🏥 Cardiac Failure Analytics")
st.sidebar.caption("Team 4 – The Zen of Five")
st.sidebar.divider()
st.sidebar.header("Filters")
st.sidebar.caption("Apply to Descriptive, Prescriptive & Summary. Predictive always trains on the full cohort.")

FILTERS = [
    ("gender",                          "Gender"),
    ("agecat",                          "Age band"),
    ("admission_way",                   "Admission pathway"),
    ("type_of_heart_failure",           "HF type"),
    ("nyha_cardiac_function_classification", "NYHA class"),
    ("killip_grade",                    "Killip grade"),
    ("bmi_category",                    "BMI category"),
]


def opts(col):
    s = df_all[col].dropna()
    if col == "agecat":    return [str(v) for v in AGE_ORDER]
    if col == "bmi_category": return [v for v in BMI_ORDER if v in set(s.unique())]
    vs = sorted(s.unique(), key=lambda v: (isinstance(v, str), v))
    return [v.item() if hasattr(v, "item") else v for v in vs]


mask = pd.Series(True, index=df_all.index)
for col, label in FILTERS:
    o = opts(col)
    picked = st.sidebar.multiselect(label, o, default=o)
    if set(picked) != set(o):
        mask &= df_all[col].isin(picked)

df = df_all[mask].copy()

if st.sidebar.button("↺ Reset filters"):
    st.rerun()

if len(df) == 0:
    st.error("No patients match the current filter combination – please widen the filters.")
    st.stop()

st.sidebar.success(f"Showing **{len(df):,}** of {len(df_all):,} patients")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
c1, c2 = st.columns([6, 1])
with c1:
    st.title("🏥 Cardiac Failure Analytics Dashboard")
    st.markdown(f"**Team 4 – The Zen of Five** &nbsp;|&nbsp; "
                f"{len(df):,} of {len(df_all):,} patients &nbsp;|&nbsp; "
                f"{df_all.shape[1]} variables &nbsp;|&nbsp; `{src_name}`")
with c2:
    print_mode = st.checkbox("🖨️ Print / PDF view")

if print_mode:
    st.markdown("""<style>
    [data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important}
    .block-container{max-width:100%!important;padding-left:1.5rem;padding-right:1.5rem}
    </style>""", unsafe_allow_html=True)
    st.info("Print view active — press **Ctrl/Cmd+P → Save as PDF** for slide-ready output. Un-check to restore filters.")

if len(df) < len(df_all):
    st.info(f"Filters active: {len(df):,} of {len(df_all):,} patients shown below. Predictive tab is unaffected.")

tab_d, tab_p, tab_ml, tab_s = st.tabs(
    ["📊 Descriptive", "💡 Prescriptive", "🔮 Predictive", "📋 Summary"])


# ══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE (full cohort — used by Predictive & summary constants)
# ══════════════════════════════════════════════════════════════════════════════
_trop_q75 = df_all["high_sensitivity_troponin"].quantile(0.75)
_bnp_q75  = df_all["brain_natriuretic_peptide"].quantile(0.75)
_cci_q75  = df_all["cci_score"].quantile(0.75)
_bnp2_q75 = df_all["brain_natriuretic_peptide"].quantile(0.75)


# ══════════════════════════════════════════════════════════════════════════════
# MEDICATION FLAGS (filtered df – used across tabs)
# ══════════════════════════════════════════════════════════════════════════════
def mk_flags(d):
    on_bb   = ((d["rx_metoprolol_succinate_sustained-release_tablet"]==1)
               | (d["rx_metoprolol_tartrate_injection"]==1))
    on_acei = ((d["rx_valsartan_dispersible_tablet"]==1)
               | (d["rx_benazepril_hydrochloride_tablet"]==1))
    on_mra  = (d["rx_spironolactone_tablet"]==1)
    d = d.copy()
    d["on_bb"]      = on_bb.astype(int)
    d["on_acei_arb"]= on_acei.astype(int)
    d["on_mra"]     = on_mra.astype(int)
    d["gdmt_count"] = on_bb.astype(int)+on_acei.astype(int)+on_mra.astype(int)
    return d

df = mk_flags(df)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_d:
    sec("📊 Descriptive Analysis — What does the data show?")
    st.caption("Counts, rates, averages and distributions. No predictions or causal claims.")
    toc([(f"q{i}", f"Q{i}") for i in range(1, 11)])

    # KPI row
    k = st.columns(5)
    k[0].metric("Patients",           f"{len(df):,}")
    k[1].metric("Biventricular HF",   pct((df["type_of_heart_failure"]=="Both").mean()))
    k[2].metric("6-mo death rate",    pct(df["death_within_6_months"].mean()))
    k[3].metric("6-mo readmission",   pct(df["re_admission_within_6_months"].mean()))
    k[4].metric("Median LOS (days)",  f"{df['dischargeday'].median():.0f}")

    # ── Q1 ────────────────────────────────────────────────────────────────────
    qcard("Q1. Key lab markers vs. clinical reference ranges",
          "What proportion of patients fall outside normal reference ranges for BNP, creatinine, "
          "haemoglobin, sodium and potassium?")

    def _pct_outside_sex(col, ranges):
        out = n = 0
        for sx,(lo,hi) in ranges.items():
            v = df.loc[(df["gender"]==sx) & df[col].notna(), col]
            out += ((v<lo)|(v>hi)).sum(); n += len(v)
        return round(safe_div(out,n)*100,1)

    lab_rows = []
    for lab,lo,hi,ref in [("brain_natriuretic_peptide",0,100,"<100 pg/mL"),
                           ("sodium",135,145,"135-145 mmol/L"),
                           ("potassium",3.5,5.1,"3.5-5.1 mmol/L")]:
        s = df[lab].dropna()
        lab_rows.append({"Lab":lab.replace("_"," ").title(),
                         "Median":round(s.median(),1),"Reference":ref,
                         "% Outside":round((((s<lo)|(s>hi)).mean())*100,1)})
    for lab,rng in [("hemoglobin",{"Male":(130,175),"Female":(115,150)}),
                    ("creatinine_enzymatic_method",{"Male":(62,106),"Female":(44,97)})]:
        s = df[lab].dropna()
        lab_rows.append({"Lab":lab.replace("_"," ").title(),
                         "Median":round(s.median(),1),"Reference":"Sex-specific",
                         "% Outside":_pct_outside_sex(lab,rng)})
    lt = pd.DataFrame(lab_rows)

    c1,c2 = st.columns([2,3])
    with c1:
        st.dataframe(lt.style.background_gradient(subset=["% Outside"],cmap="Reds"),
                     hide_index=True,use_container_width=True)
    with c2:
        fig = px.bar(lt.sort_values("% Outside"),x="% Outside",y="Lab",orientation="h",
                     text="% Outside",color="% Outside",color_continuous_scale="Reds",
                     title="Patients outside clinical reference range (%)")
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(coloraxis_showscale=False,yaxis_title=None)
        show(fig)
    bnp_oor = lt.set_index("Lab")["% Outside"]
    insight(f"BNP stands out at {bnp_oor.get('Brain Natriuretic Peptide',0):.1f}% outside range — "
            "far higher than any other lab. Haemoglobin and creatinine form a mid-tier abnormality group. "
            "Sodium and potassium deviate the least, showing cardiac/renal dysfunction dominates.")

    # ── Q2 ────────────────────────────────────────────────────────────────────
    qcard("Q2. 6-month death rate by Killip grade × ICU admission",
          "Do Killip grade and ICU status jointly predict 6-month mortality?")
    df["high_killip"] = df["killip_grade"] >= 3
    df["icu"]         = df["admission_ward"] == "ICU"
    grp = df.groupby(["high_killip","icu"])["death_within_6_months"].agg(["count","mean"])
    grp["mean"] *= 100
    def _cell(hk,icu):
        return (grp.loc[(hk,icu),"mean"],int(grp.loc[(hk,icu),"count"])) \
               if (hk,icu) in grp.index else (np.nan,0)
    combos = [(False,False),(False,True),(True,False),(True,True)]
    vals   = {c:_cell(*c) for c in combos}
    heat_df = pd.DataFrame({
        "Killip":["Low (1-2)","Low (1-2)","High (3-4)","High (3-4)"],
        "ICU":["Non-ICU","ICU","Non-ICU","ICU"],
        "Death %":[vals[c][0] for c in combos],
        "n":[vals[c][1] for c in combos],
    })
    c1,c2 = st.columns([3,2])
    with c1:
        pivot = heat_df.pivot(index="Killip",columns="ICU",values="Death %") \
                       .reindex(index=["Low (1-2)","High (3-4)"],columns=["Non-ICU","ICU"])
        fig = px.imshow(pivot,text_auto=".1f",color_continuous_scale="Reds",
                        labels=dict(color="Death %"),title="6-month death rate (%) by Killip × ICU")
        show(fig)
    with c2:
        st.dataframe(heat_df.assign(**{"Death %":heat_df["Death %"].round(1)}),
                     hide_index=True,use_container_width=True)
    insight("Death risk climbs in step with severity: "
            f"{vals[(False,False)][0]:.1f}% (low Killip, non-ICU) → "
            f"{vals[(True,False)][0]:.1f}% (high Killip, non-ICU) → "
            f"{vals[(True,True)][0]:.1f}% (high Killip + ICU, n={vals[(True,True)][1]}).")

    # ── Q3 ────────────────────────────────────────────────────────────────────
    qcard("Q3. HF phenotype, NYHA class & Killip grade distributions",
          "What kind of patients are in this cohort? These three severity/type classifications "
          "set the baseline for all downstream interpretation.")
    c1,c2,c3 = st.columns(3)
    for col_obj,col,title,clr in [
        (c1,"type_of_heart_failure","HF Phenotype",MID_BLUE),
        (c2,"nyha_cardiac_function_classification","NYHA Functional Class",WARM_RED),
        (c3,"killip_grade","Killip Grade",TEAL)]:
        with col_obj:
            vc = df[col].value_counts()
            if col != "type_of_heart_failure": vc = vc.sort_index()
            fig = px.bar(x=vc.index.astype(str),y=vc.values,title=title,text=vc.values,
                         color_discrete_sequence=[clr])
            fig.update_layout(xaxis_title=None,yaxis_title="Patients")
            show(fig,h=320)
    both_pct  = pct((df["type_of_heart_failure"]=="Both").mean())
    nyha34pct = pct(df["nyha_cardiac_function_classification"].isin([3,4]).mean())
    k34pct    = pct((df["killip_grade"]>=3).mean())
    insight(f"Phenotype: {both_pct} have both-sided (biventricular) heart failure — an advanced form. "
            f"NYHA: {nyha34pct} were already markedly–severely limited (class 3-4). "
            f"Killip: {k34pct} already showed signs of pulmonary oedema or shock at admission (grade 3-4).")

    # ── Q4 ────────────────────────────────────────────────────────────────────
    qcard("Q4. Admission pathway, ward & discharge destination",
          "How a patient enters and where they end up sketches their whole clinical journey.")
    c1,c2,c3 = st.columns(3)
    for col_obj,col,title,clr in [
        (c1,"admission_way","Admission Pathway",MID_BLUE),
        (c2,"admission_ward","Admission Ward",WARM_RED),
        (c3,"destinationdischarge","Discharge Destination",TEAL)]:
        with col_obj:
            vc = df[col].value_counts()
            fig = px.bar(x=vc.index.astype(str),y=vc.values,title=title,text=vc.values,
                         color_discrete_sequence=[clr])
            fig.update_layout(xaxis_title=None,yaxis_title="Patients")
            show(fig,h=310)
    insight("Transfer to a healthcare facility is driven by which ward manages the patient. "
            "GeneralWard patients are transferred over 2× more often than Cardiology patients "
            "(≈47% vs 21%) despite carrying a nearly identical severity profile.")

    # ── Q5 ────────────────────────────────────────────────────────────────────
    qcard("Q5. Demographic & body-composition profile",
          "Age, gender and BMI establish a baseline patient profile for interpreting every other finding.")
    c1,c2,c3 = st.columns(3)
    with c1:
        vc = df["agecat"].value_counts().reindex(AGE_ORDER,fill_value=0)
        fig = px.bar(x=vc.index.astype(str),y=vc.values,title="Age Group",text=vc.values,
                     color_discrete_sequence=[MID_BLUE])
        fig.update_layout(xaxis_title=None,yaxis_title="Patients")
        show(fig,h=320)
    with c2:
        vc = df["gender"].value_counts()
        fig = px.pie(names=vc.index,values=vc.values,title="Gender",hole=0.42,
                     color_discrete_sequence=PALETTE_PAIR)
        show(fig,h=320)
    with c3:
        vc = df["bmi_category"].value_counts().reindex(BMI_ORDER,fill_value=0)
        fig = px.bar(x=vc.index.astype(str),y=vc.values,title="BMI Category",text=vc.values,
                     color_discrete_sequence=[TEAL])
        fig.update_layout(xaxis_title=None,yaxis_title="Patients")
        show(fig,h=320)
    top2 = df["agecat"].value_counts().sort_values(ascending=False).head(2)
    top2_pct = pct(safe_div(top2.sum(),len(df)))
    insight(f"Largest age groups: {top2.index[0]} and {top2.index[1]} ({top2_pct} combined). "
            f"Cohort is {pct((df['gender']=='Female').mean())} female. "
            f"Normal BMI is most common ({pct(df['bmi_category'].eq('Normal').mean())}); "
            f"{pct(df['bmi_category'].isin(['Underweight','Overweight','Obese']).mean())} have an abnormal BMI.")

    # ── Q6 ────────────────────────────────────────────────────────────────────
    qcard("Q6. Stress hyperglycaemia — high blood sugar in non-diabetic patients",
          "Elevated glucose in a non-diabetic signals acute physiological stress from the HF episode itself.")
    hg = df["glucose_blood_gas"] > 7.8
    nd_hi = int(((df["diabetes"]==0) & hg).sum());  nd_tot = int((df["diabetes"]==0).sum())
    d_hi  = int(((df["diabetes"]==1) & hg).sum());  d_tot  = int((df["diabetes"]==1).sum())
    nd_r  = safe_div(nd_hi,nd_tot); d_r = safe_div(d_hi,d_tot)
    c1,c2 = st.columns([1,2])
    with c1:
        st.metric("Non-diabetic with high glucose",f"{nd_hi}/{nd_tot}",pct(nd_r))
        st.metric("Diabetic with high glucose",    f"{d_hi}/{d_tot}", pct(d_r))
    with c2:
        cmp = pd.DataFrame({"Group":["Non-diabetic","Diabetic"],
                            "% with high glucose":[nd_r*100,d_r*100]})
        fig = px.bar(cmp,x="Group",y="% with high glucose",text="% with high glucose",
                     color="Group",color_discrete_sequence=[AMBER,WARM_RED])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(showlegend=False,xaxis_title=None,yaxis_title="%")
        show(fig,h=290)
    insight(f"~{pct(nd_r)} of non-diabetic patients showed elevated blood sugar at admission — "
            "a stress-hyperglycaemia signal that could be missed since there is no prior diabetes flag "
            "to trigger monitoring.")

    # ── Q7 ────────────────────────────────────────────────────────────────────
    qcard("Q7. Clinical outcomes & utilisation across NYHA classes",
          "NYHA class should reflect illness severity, so worse class should mean worse outcomes.")
    summ = df.groupby("nyha_cardiac_function_classification").agg(
        mortality=("death_within_6_months","mean"),
        readmission=("re_admission_within_6_months","mean"),
        los=("dischargeday","mean")).reset_index()
    summ[["mortality","readmission"]] *= 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summ["nyha_cardiac_function_classification"],y=summ["mortality"],
                              name="Mortality (%)",mode="lines+markers",line=dict(color=WARM_RED)))
    fig.add_trace(go.Scatter(x=summ["nyha_cardiac_function_classification"],y=summ["readmission"],
                              name="Readmission (%)",mode="lines+markers",line=dict(color=MID_BLUE)))
    fig.add_trace(go.Scatter(x=summ["nyha_cardiac_function_classification"],y=summ["los"],
                              name="Avg LOS (days)",mode="lines+markers",
                              line=dict(color=TEAL),yaxis="y2"))
    fig.update_layout(title="Clinical outcomes across NYHA classes",xaxis_title="NYHA class",
                      yaxis=dict(title="Percent"),
                      yaxis2=dict(title="Days",overlaying="y",side="right"),
                      legend=dict(orientation="h",y=1.15))
    show(fig)
    ra_str = " → ".join(f"{v:.0f}%" for v in summ["readmission"])
    insight(f"Readmission climbs steadily with NYHA class ({ra_str}), but mortality and LOS "
            "do not follow the same clean gradient — mortality dips at class 3 before jumping at class 4.")

    # ── Q8 ────────────────────────────────────────────────────────────────────
    qcard("Q8. Most common comorbidities",
          "Comorbidity burden shapes care complexity beyond the primary HF diagnosis. "
          "Click a bar to drill into outcomes for that condition (Streamlit ≥ 1.35).")
    cc_cols = ["diabetes","chronic_obstructive_pulmonary_disease",
               "moderate_to_severe_chronic_kidney_disease","cerebrovascular_disease",
               "dementia","liver_disease","solid_tumor","malignant_lymphoma",
               "hemiplegia","aids","connective_tissue_disease","peptic_ulcer_disease",
               "myocardial_infarction","peripheral_vascular_disease"]
    rates = (df[cc_cols].mean()*100).sort_values()
    lbl2col = {c.replace("_"," ").title(): c for c in cc_cols}
    fig = px.bar(x=rates.values,y=[c.replace("_"," ").title() for c in rates.index],
                 orientation="h",text=rates.values,
                 title=f"Comorbidity prevalence (% of {len(df):,} patients)",
                 color_discrete_sequence=[MID_BLUE])
    fig.update_traces(texttemplate="%{text:.1f}%")
    fig.update_layout(xaxis_title="% of patients",yaxis_title=None)
    st.caption("Click a bar to see outcome drill-down below.")
    q8_pts = drill_bar(fig,h=440,name="Q8_comorbidity_prevalence")
    if q8_pts:
        try:
            lbl = q8_pts[0]["y"]; col = lbl2col.get(lbl)
            if col:
                _h = df[df[col]==1]; _n = df[df[col]==0]
                drill_panel(f"<b>Drill-down – {lbl}:</b> "
                            f"{len(_h):,} patients ({pct(safe_div(len(_h),len(df)))}) &nbsp;|&nbsp; "
                            f"6-mo mortality: {pct(_h['death_within_6_months'].mean())} (with) vs "
                            f"{pct(_n['death_within_6_months'].mean())} (without) &nbsp;|&nbsp; "
                            f"6-mo readmission: {pct(_h['re_admission_within_6_months'].mean())} vs "
                            f"{pct(_n['re_admission_within_6_months'].mean())}")
        except Exception: pass
    top3 = rates.sort_values(ascending=False).head(3)
    insight(f"Top 3 comorbidities: {top3.index[0].replace('_',' ').title()} ({top3.iloc[0]:.1f}%), "
            f"{top3.index[1].replace('_',' ').title()} ({top3.iloc[1]:.1f}%), "
            f"{top3.index[2].replace('_',' ').title()} ({top3.iloc[2]:.1f}%).")

    # ── Q9 ────────────────────────────────────────────────────────────────────
    qcard("Q9. Baseline readmission rates: 28-day, 3-month & 6-month",
          "Every later comparison needs this baseline to be measured against.")
    rcols = {"28-Day":"re_admission_within_28_days",
             "3-Month":"re_admission_within_3_months",
             "6-Month":"re_admission_within_6_months"}
    res_q9 = pd.DataFrame([{"Timeframe":k,"Readmitted":int(df[v].sum()),
                              "Rate (%)":round(safe_div(df[v].sum(),len(df))*100,2)}
                             for k,v in rcols.items()])
    c1,c2 = st.columns([1,2])
    with c1: st.dataframe(res_q9,hide_index=True,use_container_width=True)
    with c2:
        fig = px.bar(res_q9,x="Timeframe",y="Rate (%)",text="Rate (%)",
                     color="Timeframe",color_discrete_sequence=[MID_BLUE,AMBER,WARM_RED])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(showlegend=False)
        show(fig,h=300)
    r28,r3m,r6m = res_q9["Rate (%)"].tolist()
    insight(f"Rate rises from {r28:.1f}% (28-day) to {r3m:.1f}% (3-month) to {r6m:.1f}% (6-month). "
            "The steepest jump is in the first three months — a strong lead for follow-up-timing policy.")

    # ── Q10 ───────────────────────────────────────────────────────────────────
    qcard("Q10. BNP — central tendency, spread & diagnostic threshold",
          "BNP is a key severity marker but is often right-skewed; mean alone is misleading.")
    bnp = df["brain_natriuretic_peptide"].dropna()
    c1,c2 = st.columns([1,2])
    with c1:
        st.metric("Mean BNP",   f"{bnp.mean():,.0f} pg/mL")
        st.metric("Median BNP", f"{bnp.median():,.0f} pg/mL")
        st.metric("Above 100 pg/mL threshold", pct((bnp>100).mean()))
    with c2:
        fig = px.histogram(bnp,nbins=50,title="BNP distribution (pg/mL)",
                           color_discrete_sequence=[MID_BLUE])
        fig.add_vline(x=bnp.mean(),   line_color=WARM_RED, line_dash="dash",annotation_text="Mean")
        fig.add_vline(x=bnp.median(), line_color=TEAL,     line_dash="dash",annotation_text="Median")
        fig.update_layout(showlegend=False,xaxis_title="BNP (pg/mL)",yaxis_title="Patients")
        show(fig,h=300)
    insight(f"Mean ({bnp.mean():,.0f}) >> Median ({bnp.median():,.0f}) — right-skewed by a small "
            f"group of very high-BNP patients. With {pct((bnp>100).mean())} already above the "
            "diagnostic threshold, the yes/no cutoff adds little value; severity gradients above it matter.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: PRESCRIPTIVE  (Q1-Q30 — all questions, condensed where low signal)
# ══════════════════════════════════════════════════════════════════════════════
with tab_p:
    sec("💡 Prescriptive Analysis — What should we do about it?")
    st.caption("30 prescriptive questions from the team's notebook. "
               "Each recommendation is an observational association, not proof of causation.")
    toc([(f"p{i}",f"P{i}") for i in range(1,31)])

    # shared
    on_bb   = ((df["rx_metoprolol_succinate_sustained-release_tablet"]==1)|
               (df["rx_metoprolol_tartrate_injection"]==1))
    on_acei = ((df["rx_valsartan_dispersible_tablet"]==1)|
               (df["rx_benazepril_hydrochloride_tablet"]==1))
    on_mra  = (df["rx_spironolactone_tablet"]==1)
    df["gdmt_count"] = on_bb.astype(int)+on_acei.astype(int)+on_mra.astype(int)

    # ── P1 ────────────────────────────────────────────────────────────────────
    qcard("P1. Medication association with 28-day mortality",
          "Which medications correlate with lower 28-day mortality? Green = associated with better survival.")
    rx_cols = [c for c in df.columns if c.startswith("rx_")]
    rows_p1 = []
    for rx in rx_cols:
        sub = df[[rx,"death_within_28_days"]].dropna()
        if sub[rx].nunique()<2: continue
        on_r  = sub.loc[sub[rx]==1,"death_within_28_days"].mean()
        off_r = sub.loc[sub[rx]==0,"death_within_28_days"].mean()
        rows_p1.append({"Medication":rx.replace("rx_","").replace("_"," ").title(),
                        "n_prescribed":int((sub[rx]==1).sum()),
                        "mortality_on":on_r,"mortality_off":off_r,
                        "risk_diff_pp":(on_r-off_r)*100})
    if rows_p1:
        md = pd.DataFrame(rows_p1).sort_values("risk_diff_pp")
        fig = px.bar(md,x="risk_diff_pp",y="Medication",orientation="h",
                     color="risk_diff_pp",color_continuous_scale=["#27AE60","white",WARM_RED],
                     color_continuous_midpoint=0,
                     title="Mortality rate difference (pp): prescribed vs not prescribed")
        fig.update_layout(xaxis_title="Risk difference (pp)",yaxis_title=None,coloraxis_showscale=False)
        show(fig,h=520)
        insight("Drugs with a large NEGATIVE risk difference (green) are given to patients who survive "
                "at a higher rate — candidates for first-line standard-of-care. Positive differences (red) "
                "often reflect rescue drugs going to the sickest patients (confounding). "
                "Action: standardise the consistently protective oral therapies as the default order set.")

    # ── P2 ────────────────────────────────────────────────────────────────────
    qcard("P2. Diuretic choice (furosemide vs torasemide) in high-BNP patients",
          "For patients with elevated BNP, does loop-diuretic choice affect outcomes?")
    hi_bnp = df[df["brain_natriuretic_peptide"]>=df["brain_natriuretic_peptide"].median()].copy()
    def _diur(row):
        f = row.get("rx_furosemide_tablet",0)==1 or row.get("rx_furosemide_injection",0)==1
        t = row.get("rx_torasemide_tablet",0)==1
        if f and not t: return "Furosemide only"
        if t and not f: return "Torasemide only"
        if f and t:     return "Both"
        return "Neither"
    if len(hi_bnp):
        hi_bnp["diuretic_group"] = hi_bnp.apply(_diur,axis=1)
        dg = hi_bnp.groupby("diuretic_group").agg(
            n=("diuretic_group","size"),
            readmit_6m=("re_admission_within_6_months","mean"),
            mort_6m=("death_within_6_months","mean")).reset_index()
        fig = px.bar(dg,x="diuretic_group",y=["mort_6m","readmit_6m"],
                     barmode="group",title="Outcomes by diuretic choice (high-BNP patients)",
                     color_discrete_sequence=[WARM_RED,MID_BLUE])
        fig.update_layout(xaxis_title=None,yaxis_title="Rate",yaxis_tickformat=".1%")
        show(fig,h=320)
        best = dg.loc[dg["readmit_6m"].idxmin(),"diuretic_group"]
        insight(f"Prefer **{best}** as first-line loop diuretic for high-BNP admissions "
                "(lowest 6-month readmission). Reserve combination therapy for non-responders.")

    # ── P3 ────────────────────────────────────────────────────────────────────
    qcard("P3. ACEI/ARB therapy gap — LVEF ≤ 50%, no severe CKD",
          "How many guideline-eligible patients leave without benazepril/valsartan?")
    eligible = df[(df["lvef"].notna())&(df["lvef"]<=50)&
                  (df["moderate_to_severe_chronic_kidney_disease"].fillna(0)==0)].copy()
    if len(eligible)>0:
        eligible["on_acei_arb"] = ((eligible["rx_valsartan_dispersible_tablet"]==1)|
                                   (eligible["rx_benazepril_hydrochloride_tablet"]==1)).astype(int)
        gap3 = eligible.groupby("on_acei_arb").agg(
            n=("on_acei_arb","size"),
            readmit_6m=("re_admission_within_6_months","mean")).reset_index()
        gap3.index = gap3["on_acei_arb"].map({0:"Not on ACEI/ARB (gap)",1:"On ACEI/ARB"})
        not_on = int((eligible["on_acei_arb"]==0).sum())
        c1,c2 = st.columns([1,2])
        with c1:
            st.metric("Eligible patients",f"{len(eligible)}")
            st.metric("Not on ACEI/ARB",f"{not_on} ({pct(safe_div(not_on,len(eligible)))})")
        with c2:
            fig = px.bar(gap3,x=gap3.index,y="readmit_6m",text="n",
                         title="6-month readmission: on vs. off ACEI/ARB (LVEF≤50)",
                         color_discrete_sequence=[MID_BLUE,TEAL])
            fig.update_layout(xaxis_title=None,yaxis_title="Readmission rate",yaxis_tickformat=".1%")
            show(fig,h=290)
        insight(f"{not_on} eligible patients ({pct(safe_div(not_on,len(eligible)))}) discharged without "
                "ACEI/ARB. Action: add discharge-checklist alert — LVEF ≤ 50 + no severe CKD + "
                "not on benazepril/valsartan → pharmacy review before discharge.")

    # ── P4 ────────────────────────────────────────────────────────────────────
    qcard("P4. Spironolactone underuse in HFrEF (LVEF < 40)",
          "Mineralocorticoid receptor antagonists reduce mortality in HFrEF. Is there a gap?")
    ref = df[(df["lvef"].notna())&(df["lvef"]<40)].copy()
    if len(ref)>2:
        sp = ref.groupby("rx_spironolactone_tablet").agg(
            n=("rx_spironolactone_tablet","size"),
            mort_6m=("death_within_6_months","mean"),
            readmit_6m=("re_admission_within_6_months","mean")).reset_index()
        sp.index = sp["rx_spironolactone_tablet"].map({0:"No spironolactone",1:"On spironolactone"})
        n_off = int((ref["rx_spironolactone_tablet"]==0).sum())
        c1,c2 = st.columns([1,2])
        with c1:
            st.metric("HFrEF patients (LVEF<40)", f"{len(ref)}")
            st.metric("Not on spironolactone", f"{n_off} ({pct(safe_div(n_off,len(ref)))})")
        with c2:
            fig = px.bar(sp,x=sp.index,y=["mort_6m","readmit_6m"],barmode="group",
                         title="Outcomes by spironolactone use (HFrEF)",
                         color_discrete_sequence=[WARM_RED,MID_BLUE])
            fig.update_layout(xaxis_title=None,yaxis_title="Rate",yaxis_tickformat=".1%")
            show(fig,h=290)
        insight(f"{n_off} of {len(ref)} HFrEF patients ({pct(safe_div(n_off,len(ref)))}) "
                "are not on spironolactone. Action: add MRA eligibility (LVEF<40, normal K+/renal) "
                "to the same discharge checklist and track monthly % on optimal medical therapy.")

    # ── P5 ────────────────────────────────────────────────────────────────────
    qcard("P5. Beta-blocker titration opportunity",
          "Patients on beta-blockers still tachycardic AND hypertensive may be under-dosed.")
    on_bb_df = df[on_bb.reindex(df.index,fill_value=False)].dropna(
                  subset=["pulse","systolic_blood_pressure"]).copy()
    on_bb_df["titration"] = ((on_bb_df["pulse"]>80)&(on_bb_df["systolic_blood_pressure"]>110)).astype(int)
    if len(on_bb_df):
        rate5 = on_bb_df["titration"].mean()
        c1,c2 = st.columns([1,2])
        with c1:
            st.metric("BB patients",f"{len(on_bb_df)}")
            st.metric("Possible under-titration",pct(rate5))
        with c2:
            on_bb_df["titration_label"] = on_bb_df["titration"].map({0:"Adequate",1:"Under-titrated?"})
            fig = px.scatter(on_bb_df,x="pulse",y="systolic_blood_pressure",
                             color="titration_label",
                             color_discrete_map={"Adequate":TEAL,"Under-titrated?":WARM_RED},
                             opacity=0.55,title="Beta-blocker patients: pulse vs. systolic BP")
            fig.add_vline(x=80,line_dash="dash",line_color=GREY)
            fig.add_hline(y=110,line_dash="dash",line_color=GREY)
            fig.update_layout(legend=dict(title=""))
            show(fig,h=320)
        insight(f"{pct(rate5)} of beta-blocker patients still have pulse>80 AND SBP>110 — "
                "a concrete up-titration opportunity. Action: flag 'on BB AND pulse>80 AND SBP>110 "
                "for 2+ readings → consider dose increase' as a daily-rounds worklist item.")

    # ── P6 ────────────────────────────────────────────────────────────────────
    qcard("P6. Anticoagulation vs. kidney function (GFR-based dosing)",
          "Renally-cleared anticoagulants accumulate in renal impairment.")
    ac = df.dropna(subset=["glomerular_filtration_rate"]).copy()
    ac["gfr_bucket"] = pd.cut(ac["glomerular_filtration_rate"],bins=[0,30,60,90,1000],
                               labels=["<30 (severe)","30-60 (mod)","60-90 (mild)","90+ (normal)"])
    if len(ac):
        ac_rx = ac.groupby("gfr_bucket",observed=True)[
            ["rx_warfarin_sodium_tablet","rx_enoxaparin_sodium_injection","rx_heparin_sodium_injection"]
        ].mean().mul(100).round(1)
        ac_rx_melt = ac_rx.reset_index().melt("gfr_bucket",var_name="Agent",value_name="Pct")
        fig = px.bar(ac_rx_melt,x="gfr_bucket",y="Pct",color="Agent",
                     barmode="group",title="Anticoagulant use (%) by GFR band",
                     color_discrete_sequence=[MID_BLUE,AMBER,TEAL])
        fig.update_layout(xaxis_title="GFR band",yaxis_title="%",legend_title="Agent")
        show(fig,h=310)
        insight("If enoxaparin use fails to drop as GFR declines below 30, that is a patient-safety gap. "
                "Action: e-prescribing alert — GFR <30 auto-suggests unfractionated heparin or "
                "dose-reduced enoxaparin.")

    # ── P7 ────────────────────────────────────────────────────────────────────
    qcard("P7. Polypharmacy — where is the deprescribing threshold?",
          "More medications is not automatically better. Does rx-count correlate with worse outcomes "
          "even within the same comorbidity burden?")
    poly = df.dropna(subset=["total_prescriptions_count","death_within_6_months"]).copy()
    poly["rx_bucket"] = pd.cut(poly["total_prescriptions_count"],bins=[0,4,6,8,10,20],
                                labels=["1-4","5-6","7-8","9-10","11+"])
    if len(poly):
        pg = poly.groupby("rx_bucket",observed=True).agg(
            n=("rx_bucket","size"),mort_6m=("death_within_6_months","mean"),
            readmit_6m=("re_admission_within_6_months","mean"),avg_cci=("cci_score","mean")).reset_index()
        fig = px.bar(pg,x="rx_bucket",y=["mort_6m","readmit_6m"],barmode="group",
                     title="6-month outcomes by total prescription count",
                     color_discrete_sequence=[WARM_RED,MID_BLUE])
        fig.update_layout(xaxis_title="Rx count",yaxis_title="Rate",yaxis_tickformat=".1%")
        show(fig,h=310)
        insight("Flag patients whose rx-count is high RELATIVE to their CCI band (top quartile within band) "
                "for a medication-reconciliation / deprescribing review — that subgroup's polypharmacy "
                "is least explained by disease burden.")

    # ── P8 ────────────────────────────────────────────────────────────────────
    qcard("P8. Lab combination for ICU escalation (decision-tree rule)",
          "Combining renal and cardiac markers into one escalation rule is more actionable than "
          "reviewing dozens of labs individually.")
    risk_feats = ["creatinine_enzymatic_method","bun_creatinine_ratio",
                  "brain_natriuretic_peptide","lactate","ph"]
    df["went_to_icu"] = (df["admission_ward"]=="ICU").astype(int)
    icu_data = df.dropna(subset=risk_feats+["went_to_icu"])
    if len(icu_data)>=50 and icu_data["went_to_icu"].sum()>=5:
        tree8 = DecisionTreeClassifier(max_depth=3,min_samples_leaf=20,
                                       class_weight="balanced",random_state=42)
        tree8.fit(icu_data[risk_feats],icu_data["went_to_icu"])
        from sklearn.tree import export_text
        rules = export_text(tree8,feature_names=risk_feats,max_depth=3)
        with st.expander("📋 Decision-tree rule for ICU escalation (click to expand)"):
            st.code(rules)
        insight("The tree yields a small, human-readable rule set. Action: translate the top splits into a "
                "formal early-warning escalation protocol built into the EHR's best-practice alert.")

    # ── P9 ────────────────────────────────────────────────────────────────────
    qcard("P9. Troponin quartile and 6-month outcomes",
          "A clear cut-point separating outcome risk gives a defensible, automatable consult trigger.")
    trop = df.dropna(subset=["high_sensitivity_troponin","death_within_6_months"]).copy()
    if len(trop)>=40:
        trop["trop_q"] = pd.qcut(trop["high_sensitivity_troponin"],q=4,duplicates="drop",
                                  labels=False)
        tg = trop.groupby("trop_q").agg(
            n=("trop_q","size"),mort=("death_within_6_months","mean"),
            readmit=("re_admission_within_6_months","mean")).reset_index()
        tg["Quartile"] = ["Q1 (lowest)","Q2","Q3","Q4 (highest)"][:len(tg)]
        fig = px.bar(tg,x="Quartile",y=["mort","readmit"],barmode="group",
                     title="6-month outcomes by troponin quartile",
                     color_discrete_sequence=[WARM_RED,MID_BLUE])
        fig.update_layout(xaxis_title=None,yaxis_title="Rate",yaxis_tickformat=".1%")
        show(fig,h=300)
        insight("The top troponin quartile shows materially worse outcomes. Action: set the top-quartile "
                "troponin value as an automatic cardiology consult trigger.")

    # ── P10 ───────────────────────────────────────────────────────────────────
    qcard("P10. D-dimer / INR combination → thrombosis workup",
          "D-dimer + INR jointly inform thrombotic/bleeding risk. A top-quartile D-dimer is a candidate "
          "for an automatic workup trigger.")
    thr = df.dropna(subset=["d_dimer","international_normalized_ratio","death_within_6_months"]).copy()
    if len(thr)>=40:
        thr["high_ddimer"] = (thr["d_dimer"]>thr["d_dimer"].quantile(0.75)).astype(int)
        tg10 = thr.groupby("high_ddimer").agg(
            n=("high_ddimer","size"),avg_inr=("international_normalized_ratio","mean"),
            mort=("death_within_6_months","mean")).reset_index()
        tg10.index = tg10["high_ddimer"].map({0:"Normal/low D-dimer",1:"D-dimer top quartile"})
        c1,c2 = st.columns([1,2])
        with c1: st.dataframe(tg10[["n","avg_inr","mort"]].round(3),use_container_width=True)
        with c2:
            fig = px.bar(tg10,x=tg10.index,y="mort",text="mort",
                         title="6-month mortality by D-dimer group",
                         color_discrete_sequence=[MID_BLUE,WARM_RED])
            fig.update_traces(texttemplate="%{text:.1%}")
            fig.update_layout(xaxis_title=None,yaxis_title="Mortality",yaxis_tickformat=".1%")
            show(fig,h=280)
        insight("If the top-quartile D-dimer group shows materially worse outcomes, auto-trigger a "
                "lower-limb Doppler / CT-PA order whenever D-dimer exceeds this threshold and the "
                "patient is not already therapeutically anticoagulated.")

    # ── P11 ───────────────────────────────────────────────────────────────────
    qcard("P11. Optimal length of stay — discharge-timing policy",
          "Hospitals must balance patient safety against bed capacity. Finding the LOS sweet spot "
          "is directly prescriptive for discharge-planning policy.")
    los = df.dropna(subset=["dischargeday"]).copy()
    if len(los):
        los["los_bucket"] = pd.cut(los["dischargeday"],bins=[0,3,5,7,10,14,100],
                                    labels=["1-3","4-5","6-7","8-10","11-14","15+"])
        lg = los.groupby("los_bucket",observed=True).agg(
            n=("los_bucket","size"),mort=("death_within_6_months","mean"),
            readmit=("re_admission_within_6_months","mean")).reset_index()
        fig = px.bar(lg,x="los_bucket",y=["mort","readmit"],barmode="group",
                     title="6-month outcomes by length-of-stay bucket",
                     color_discrete_sequence=[WARM_RED,MID_BLUE])
        fig.update_layout(xaxis_title="LOS (days)",yaxis_title="Rate",yaxis_tickformat=".1%")
        show(fig,h=310)
        insight("If the shortest stays show elevated readmission, set a minimum observation threshold "
                "(e.g. ≥4 days) for higher-severity patients before discharge.")

    # ── P12 ───────────────────────────────────────────────────────────────────
    qcard("P12. Should phosphorus be a standard admission lab?",
          "Only ~20% of patients get phosphorus tested (when doctors already suspect a problem). "
          "Should it be universal?")
    phtest = df["inorganic_phosphorus"].notna()
    if phtest.sum()>=20:
        sub12 = df[phtest].copy()
        CUTOFF12 = 1.3
        sub12["flag"] = (sub12["inorganic_phosphorus"]>CUTOFF12).astype(int)
        from sklearn.metrics import confusion_matrix as _cm
        tn,fp,fn,tp = _cm(sub12["death_within_28_days"],sub12["flag"],labels=[0,1]).ravel()
        sens = safe_div(tp,tp+fn); spec = safe_div(tn,tn+fp)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Patients tested",f"{phtest.sum()} ({pct(safe_div(phtest.sum(),len(df)))})")
        c2.metric("Sensitivity (>1.3 g/L flag)",pct(sens))
        c3.metric("Specificity",pct(spec))
        c4.metric("Deaths flagged",pct(safe_div(tp,tp+fn)))
        insight(f"Only {pct(safe_div(phtest.sum(),len(df)))} of patients ever get phosphorus tested. "
                f"A >1.3 g/L threshold flags {pct(sens)} of deaths in the tested group. "
                "Recommendation: add phosphorus to the standard admission panel for all HF patients.")

    # ── P13 ───────────────────────────────────────────────────────────────────
    qcard("P13. HFrEF GDMT gap — three-drug completeness",
          "Among LVEF ≤ 40% patients, how many are missing ≥1 of the 3 standard drug classes "
          "(BB + ACEI/ARB + MRA)?")
    hfref = df[df["lvef"]<=40].copy()
    hfref["gdmt_count"] = (hfref["on_bb"]+hfref["on_acei_arb"]+hfref["on_mra"])
    hfref["full_gdmt"]  = hfref["gdmt_count"]==3
    if len(hfref)>0:
        gap13 = hfref[~hfref["full_gdmt"]]; full13 = hfref[hfref["full_gdmt"]]
        gap_share = safe_div(len(gap13),len(hfref))
        c1,c2,c3 = st.columns(3)
        c1.metric("HFrEF patients",f"{len(hfref)}")
        c2.metric("Missing ≥1 class",f"{len(gap13)} ({pct(gap_share)})")
        c3.metric("Readmit: gap vs full",f"{pct(gap13['re_admission_within_6_months'].mean())} vs "
                  f"{pct(full13['re_admission_within_6_months'].mean())}")
        cnt13 = hfref["gdmt_count"].value_counts().sort_index()
        fig = px.bar(x=cnt13.index.astype(str),y=cnt13.values,text=cnt13.values,
                     title="Drug classes prescribed per HFrEF patient (0-3)",
                     color_discrete_sequence=[MID_BLUE])
        fig.update_layout(xaxis_title="GDMT classes",yaxis_title="Patients")
        show(fig,h=290)
        insight(f"{len(gap13)} of {len(hfref)} HFrEF patients ({pct(gap_share)}) are missing ≥1 "
                f"guideline medicine. 6-month readmission is {pct(gap13['re_admission_within_6_months'].mean())} "
                f"(gap) vs {pct(full13['re_admission_within_6_months'].mean())} (full GDMT) — "
                "≈1.6× higher. Action: automatic medication-reconciliation review before discharge.")
    else:
        st.info("No HFrEF patients in the current filter selection.")

    # ── P14 ───────────────────────────────────────────────────────────────────
    qcard("P14. Lactate hidden risk in Killip-stable patients",
          "Killip grade can miss early tissue under-perfusion. "
          "Lactate catches trouble before it shows up clinically.")
    stable14 = df[df["killip_grade"].isin([1,2])].copy()
    stable14["el"] = stable14["lactate"] > 2
    if len(stable14)>=20:
        d_hi14 = stable14.loc[stable14["el"]==True,"death_within_6_months"].mean()
        d_lo14 = stable14.loc[stable14["el"]==False,"death_within_6_months"].mean()
        c1,c2 = st.columns([1,2])
        with c1:
            st.metric("Killip-stable patients (grade 1-2)",f"{len(stable14):,}")
            st.metric("Elevated-lactate mortality",pct(d_hi14))
            st.metric("Normal-lactate mortality",  pct(d_lo14))
        with c2:
            _s14 = stable14.dropna(subset=["lactate"]).copy()
            _s14["lactate_group"] = _s14["el"].map({True:"Elevated (>2)",False:"Normal (≤2)"})
            fig = px.box(_s14,x="el",y="lactate",color="lactate_group",
                         color_discrete_map={"Elevated (>2)":WARM_RED,"Normal (≤2)":TEAL},
                         title="Lactate distribution — Killip-stable patients")
            fig.update_layout(showlegend=False,xaxis_title="Elevated lactate")
            show(fig,h=300)
        insight(f"Even among Killip-stable patients, elevated lactate doubles 6-month mortality "
                f"({pct(d_hi14)} vs {pct(d_lo14)}). Action: measure lactate routinely on all admissions, "
                "not only when shock is clinically apparent.")

    # ── P15 ───────────────────────────────────────────────────────────────────
    qcard("P15. Troponin + BNP — possible acute ischemic trigger",
          "Top-quartile troponin AND BNP together may indicate an unrecognised ischaemic event "
          "behind the HF episode.")
    flagged15 = ((df["high_sensitivity_troponin"]>_trop_q75)&
                 (df["brain_natriuretic_peptide"]>_bnp_q75))
    no_mi15 = (flagged15 & (df["myocardial_infarction"]==0))
    c1,c2,c3 = st.columns(3)
    c1.metric("Flagged (top-quartile both)",f"{int(flagged15.sum())} ({pct(flagged15.mean())})")
    c2.metric("Flagged — no MI on record",  f"{int(no_mi15.sum())}")
    c3.metric("6-mo mortality: flagged vs rest",
              f"{pct(df.loc[flagged15,'death_within_6_months'].mean())} vs "
              f"{pct(df.loc[~flagged15,'death_within_6_months'].mean())}")
    insight(f"{int(flagged15.sum())} patients ({pct(flagged15.mean())}) show both markers in the top quartile; "
            f"{int(no_mi15.sum())} of them have no documented MI. This group has a notably higher death rate. "
            "Action: prioritise cardiology/cath-lab review rather than routine HF management alone.")

    # ── P16 ───────────────────────────────────────────────────────────────────
    qcard("P16. Modifiable risk factors among patients with bad outcomes",
          "How much of the bad-outcome burden is potentially addressable at admission?")
    fA16 = (df["diabetes"]==0) & (df["glucose_blood_gas"]>7.8)
    fB16 = (df["lvef"]<=40) & (df["gdmt_count"]<3)
    fC16 = df["sodium"]<135
    any16 = fA16.fillna(False)|fB16.fillna(False)|fC16.fillna(False)
    bad16 = ((df["death_within_6_months"]==1)|(df["re_admission_within_6_months"]==1))
    rb = any16[bad16].mean(); rg = any16[~bad16].mean()
    cmp16 = pd.DataFrame({"Group":["Bad outcome\n(death/readmit)","Good outcome"],
                          "Rate":[rb*100,rg*100]})
    c1,c2 = st.columns([1,2])
    with c1:
        st.metric("Bad-outcome patients w/ ≥1 flag",pct(rb))
        st.metric("Good-outcome patients w/ ≥1 flag",pct(rg))
    with c2:
        fig = px.bar(cmp16,x="Group",y="Rate",text="Rate",
                     color="Group",color_discrete_sequence=[WARM_RED,TEAL])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(showlegend=False,xaxis_title=None,yaxis_title="% with ≥1 flag")
        show(fig,h=280)
    insight(f"Patients with a bad 6-month outcome were {safe_div(rb,rg):.1f}× more likely to have "
            "had a modifiable red flag at admission (stress hyperglycaemia, GDMT gap or hyponatraemia) "
            "than patients who did well.")

    # ── P17 ───────────────────────────────────────────────────────────────────
    qcard("P17. Sodium and 6-month mortality",
          "Hyponatraemia is a known HF severity marker. Does the distribution differ between "
          "survivors and non-survivors?")
    df["outcome_6m"] = df["death_within_6_months"].map({0:"Survived",1:"Died within 6mo"})
    hypo = df.groupby("outcome_6m")["sodium"].apply(lambda x:(x<135).mean()*100)
    c1,c2 = st.columns([1,2])
    with c1:
        st.metric("% with hyponatraemia — died",
                  f"{hypo.get('Died within 6mo',float('nan')):.1f}%"
                  if "Died within 6mo" in hypo.index else "n/a")
        st.metric("% with hyponatraemia — survived",
                  f"{hypo.get('Survived',float('nan')):.1f}%"
                  if "Survived" in hypo.index else "n/a")
    with c2:
        fig = px.box(df,x="outcome_6m",y="sodium",color="outcome_6m",
                     color_discrete_map={"Died within 6mo":WARM_RED,"Survived":MID_BLUE},
                     title="Sodium distribution by 6-month outcome")
        fig.add_hline(y=135,line_dash="dash",line_color=GREY,annotation_text="Hyponatraemia threshold")
        fig.update_layout(showlegend=False,xaxis_title=None)
        show(fig,h=300)
    d_val = hypo.get("Died within 6mo",float("nan")); s_val = hypo.get("Survived",float("nan"))
    insight(f"Hyponatraemia was present in {d_val:.1f}% of deaths vs {s_val:.1f}% of survivors. "
            "Low sodium reflects advanced HF where fluid retention outpaces salt retention.")

    # ── P18 ───────────────────────────────────────────────────────────────────
    qcard("P18. Killip grade & shock-index escalation threshold",
          "If mortality jumps at a specific grade, that is a natural evidence-backed cutoff for an "
          "automatic ICU escalation policy. Click the Killip bar to drill into outcomes.")
    df["shock_index_clean"] = df["shock_index"].replace([np.inf,-np.inf],np.nan)
    si_bins18 = pd.cut(df["shock_index_clean"],bins=[0,0.5,0.7,1.0,5])
    c1,c2 = st.columns(2)
    with c1:
        g1_18 = df.groupby("killip_grade")["death_within_28_days"].mean()*100
        fig = px.bar(x=g1_18.index.astype(str),y=g1_18.values,text=g1_18.round(1),
                     title="28-day mortality by Killip grade",color_discrete_sequence=[WARM_RED])
        fig.update_layout(xaxis_title="Killip grade",yaxis_title="Mortality %")
        st.caption("Click a bar to drill into that Killip grade below.")
        p18_pts = drill_bar(fig,h=300,name="P18_killip_mortality")
    with c2:
        g2_18 = df.groupby(si_bins18,observed=True)["death_within_28_days"].mean()*100
        fig = px.bar(x=[str(x) for x in g2_18.index],y=g2_18.values,text=g2_18.round(1),
                     title="28-day mortality by shock-index band",color_discrete_sequence=[AMBER])
        fig.update_layout(xaxis_title="Shock index band",yaxis_title="Mortality %")
        show(fig,h=300)
    if p18_pts:
        try:
            grd = p18_pts[0]["x"]
            _sub = df[df["killip_grade"].astype(str)==str(grd)]
            drill_panel(f"<b>Drill-down — Killip grade {grd}:</b> "
                        f"{len(_sub):,} patients &nbsp;|&nbsp; "
                        f"28-day mortality {pct(_sub['death_within_28_days'].mean())} &nbsp;|&nbsp; "
                        f"6-month mortality {pct(_sub['death_within_6_months'].mean())} &nbsp;|&nbsp; "
                        f"6-month readmission {pct(_sub['re_admission_within_6_months'].mean())}")
        except Exception: pass
    if len(g1_18)>=2:
        insight(f"Mortality is flat through Killip 1-2, then jumps at grade 3 "
                f"({g1_18.get(3,0):.1f}%) and peaks at grade 4 ({g1_18.get(4,0):.1f}%). "
                "Recommendation: automatic ICU review for every Killip-4 admission "
                "and any patient with shock index >1.0.")

    # ── P19 ───────────────────────────────────────────────────────────────────
    qcard("P19. NYHA / Killip combination — 'not safe to discharge' rule",
          "Readmission predictors translate directly into a discharge checklist item.")
    c1,c2 = st.columns(2)
    with c1:
        gn19 = df.groupby("nyha_cardiac_function_classification")["re_admission_within_6_months"].mean()*100
        fig = px.bar(x=gn19.index.astype(str),y=gn19.values,text=gn19.round(1),
                     title="6-mo readmission by NYHA class",color_discrete_sequence=[MID_BLUE])
        fig.update_layout(xaxis_title="NYHA class",yaxis_title="Readmission %")
        show(fig,h=280)
    with c2:
        gk19 = df.groupby("killip_grade")["re_admission_within_6_months"].mean()*100
        fig = px.bar(x=gk19.index.astype(str),y=gk19.values,text=gk19.round(1),
                     title="6-mo readmission by Killip grade",color_discrete_sequence=[WARM_RED])
        fig.update_layout(xaxis_title="Killip grade",yaxis_title="Readmission %")
        show(fig,h=280)
    insight("Readmission rises steadily with NYHA class. NYHA-4 patients (≈31% of cohort) return "
            "nearly 1 in 2 times within 6 months. Recommendation: mandatory 7-14 day post-discharge "
            "follow-up call or clinic visit for all NYHA-4 patients.")

    # ── P20 ───────────────────────────────────────────────────────────────────
    qcard("P20. Anticoagulation vs. comorbidity burden",
          "Is the sickest group (highest CCI) being under-prescribed anticoagulation?")
    ac_cols20 = ["rx_warfarin_sodium_tablet","rx_enoxaparin_sodium_injection","rx_heparin_sodium_injection"]
    df["on_anticoag20"] = (df[ac_cols20].sum(axis=1)>0).astype(int)
    df["cci_band20"]    = pd.cut(df["cci_score"],bins=[-1,1,2,3,10],
                                  labels=["0-1 low","2 moderate","3 high","4+ very high"])
    if df["cci_band20"].notna().sum()>10:
        p20 = pd.crosstab(df["cci_band20"],df["on_anticoag20"],normalize="index")*100
        p20_melt = (p20.rename(columns={0:"No anticoag",1:"On anticoag"})
                       .reset_index()
                       .melt("cci_band20",var_name="Anticoag",value_name="Pct"))
        fig = px.bar(p20_melt,x="cci_band20",y="Pct",color="Anticoag",
                     title="Anticoagulation rate (%) by comorbidity burden",
                     color_discrete_sequence=[GREY,MID_BLUE])
        fig.update_layout(xaxis_title="CCI band",yaxis_title="%",legend_title="")
        show(fig,h=290)
        insight("Anticoagulant use may drop as comorbidity burden rises even though the highest-CCI "
                "group has the worst readmission rate. Action: review whether high-CCI patients with "
                "AF or prior VTE are being under-treated for anticoagulation-eligible conditions.")

    # ── P21 ───────────────────────────────────────────────────────────────────
    qcard("P21. Enhanced post-discharge follow-up risk score (0-6 points)",
          "A simple admission-time score flags patients for early clinic visit + home nursing. "
          "Flagging ≤45% of patients should capture ≥78% of 6-month deaths.")
    fp_k   = df["killip_grade"]>=3
    fp_na  = df["sodium"]<135
    fp_lac = df["lactate"]>2
    fp_cci = df["cci_score"]>=df_all["cci_score"].quantile(0.75)
    fp_bnp = df["brain_natriuretic_peptide"]>=df_all["brain_natriuretic_peptide"].quantile(0.75)
    fp_g   = (df["lvef"]<=40) & (df["gdmt_count"]<3)
    df["followup_score"] = (fp_k.fillna(False).astype(int)+fp_na.fillna(False).astype(int)+
                            fp_lac.fillna(False).astype(int)+fp_cci.fillna(False).astype(int)+
                            fp_bnp.fillna(False).astype(int)+fp_g.fillna(False).astype(int))
    flagged21 = df["followup_score"]>=2
    deaths_tot = df["death_within_6_months"].sum()
    captured21 = safe_div(df.loc[flagged21,"death_within_6_months"].sum(), deaths_tot)
    c1,c2,c3 = st.columns(3)
    c1.metric("Patients flagged (score ≥2)",f"{int(flagged21.sum())} ({pct(flagged21.mean())})")
    c2.metric("Deaths captured",pct(captured21))
    c3.metric("Mortality: flagged vs not",
              f"{pct(df.loc[flagged21,'death_within_6_months'].mean())} vs "
              f"{pct(df.loc[~flagged21,'death_within_6_months'].mean())}")
    sdf21 = df.sort_values("followup_score",ascending=False).reset_index(drop=True)
    sdf21["cum_pct"]   = (sdf21.index+1)/len(sdf21)*100
    sdf21["cum_deaths"]= sdf21["death_within_6_months"].cumsum()/max(deaths_tot,1)*100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sdf21["cum_pct"],y=sdf21["cum_deaths"],
                              name="Risk score",line=dict(color=MID_BLUE)))
    fig.add_trace(go.Scatter(x=[0,100],y=[0,100],name="Random",line=dict(dash="dash",color=GREY)))
    fig.update_layout(title="% of deaths captured vs. % of patients flagged",
                      xaxis_title="% patients flagged",yaxis_title="% deaths captured")
    show(fig,h=320)
    insight(f"Flagging {pct(flagged21.mean())} of patients captures {pct(captured21)} of all "
            "6-month deaths — nearly 2× better than random selection of the same fraction.")

    # ── P22 ───────────────────────────────────────────────────────────────────
    qcard("P22. Multivariate risk model for early follow-up (AUC ≈ 0.595)",
          "A logistic regression on 7 admission variables turns into tiered action (follow-up scheduling).")
    feats22 = ["cci_score","nyha_cardiac_function_classification","killip_grade",
               "glomerular_filtration_rate","brain_natriuretic_peptide","hemoglobin","sodium"]
    sub22 = df[feats22+["re_admission_within_3_months"]].dropna()
    yc22 = sub22["re_admission_within_3_months"].value_counts()
    if len(sub22)>=60 and (yc22.min() if len(yc22) else 0)>=8:
        X22,y22 = sub22[feats22],sub22["re_admission_within_3_months"]
        X22tr,X22te,y22tr,y22te = train_test_split(X22,y22,test_size=0.25,random_state=42,stratify=y22)
        m22 = LogisticRegression(max_iter=1000).fit(X22tr,y22tr)
        p22 = m22.predict_proba(X22te)[:,1]
        auc22 = roc_auc_score(y22te,p22)
        r22   = pd.DataFrame({"risk":p22,"actual":y22te.values})
        r22["tier"] = pd.qcut(r22["risk"],q=3,labels=["Low","Medium","High"],duplicates="drop")
        t22 = r22.groupby("tier",observed=True)["actual"].mean()*100
        c1,c2 = st.columns([1,2])
        with c1:
            st.metric("Model AUC",f"{auc22:.3f}")
            st.metric("Low → High tier readmission",
                      f"{t22.iloc[0]:.1f}% → {t22.iloc[-1]:.1f}%")
        with c2:
            fig = px.bar(x=t22.index.astype(str),y=t22.values,text=t22.round(1),
                         title="3-month readmission by risk tier",
                         color=t22.index.astype(str),
                         color_discrete_map={"Low":TEAL,"Medium":AMBER,"High":WARM_RED})
            fig.update_layout(showlegend=False,xaxis_title=None,yaxis_title="Readmission %")
            show(fig,h=280)
        insight(f"Discrimination is modest (AUC {auc22:.3f}), but actual readmission roughly doubles "
                f"from the low tier ({t22.iloc[0]:.1f}%) to the high tier ({t22.iloc[-1]:.1f}%). "
                "Recommendation: 2-week follow-up call for the top tier; standard 4-6 week follow-up below.")
    else:
        st.info("Not enough patients / outcome events in this filter selection to retrain the model. Widen the filters.")

    # ── P23 ───────────────────────────────────────────────────────────────────
    qcard("P23. Ward / discharge-department mortality — resource reallocation",
          "Do outcomes vary by care setting after accounting for severity? Identifies units "
          "that may benefit from staffing or protocol changes.")
    ward23 = df.groupby(["admission_ward","discharge_department"]).agg(
        n=("inpatient_number","size"),
        mort_hosp=("outcome_during_hospitalization",lambda s:(s=="Dead").mean()),
        avg_cci=("cci_score","mean"),avg_los=("dischargeday","mean")
    ).reset_index()
    ward23 = ward23[ward23["n"]>=10].sort_values("mort_hosp",ascending=False)
    if len(ward23):
        fig = px.bar(ward23.head(12),x="mort_hosp",
                     y=ward23.head(12).apply(lambda r:f"{r['admission_ward']} → {r['discharge_department']}",axis=1),
                     orientation="h",text="mort_hosp",
                     title="In-hospital mortality by ward → department path (top 12, n≥10)",
                     color_discrete_sequence=[WARM_RED])
        fig.update_traces(texttemplate="%{text:.1%}")
        fig.update_layout(xaxis_tickformat=".1%",yaxis_title=None)
        show(fig,h=400)
        insight("Ward-department pairs with high mortality even after accounting for avg CCI point to "
                "a care-quality gap. Action: prioritise these units for staffing-ratio review or "
                "rapid-response protocol; study low-mortality / high-CCI units to replicate their approach.")

    # ── P24 ───────────────────────────────────────────────────────────────────
    qcard("P24. Emergency vs. non-emergency outcomes — should the handoff protocol change?",
          "Emergency admissions often mean less time for pre-admission optimisation.")
    aw24 = df.groupby("admission_way").agg(
        n=("admission_way","size"),mort_6m=("death_within_6_months","mean"),
        readmit_6m=("re_admission_within_6_months","mean"),
        avg_killip=("killip_grade","mean")).round(3).reset_index()
    if len(aw24):
        aw24_melt = aw24.melt("admission_way",["mort_6m","readmit_6m"],
                              var_name="Outcome",value_name="Rate_val")
        fig = px.bar(aw24_melt,x="admission_way",y="Rate_val",color="Outcome",barmode="group",
                     title="6-month outcomes by admission pathway",
                     color_discrete_sequence=[WARM_RED,MID_BLUE])
        fig.update_layout(xaxis_title=None,yaxis_title="Rate",yaxis_tickformat=".1%",legend_title="")
        show(fig,h=290)
        insight("If emergency admissions show materially worse outcomes even adjusting for avg Killip, "
                "build a standardised ED-to-cardiology handoff bundle (structured vitals/labs summary, "
                "automatic cardiology page for Killip ≥ 3).")

    # ── P25 ───────────────────────────────────────────────────────────────────
    qcard("P25. Respiratory support / oxygen triage to higher-acuity care",
          "Need for ventilatory support is a direct marker of physiologic instability — checking "
          "whether ward placement matches respiratory needs prevents undertriage.")
    if "oxygen_inhalation" in df.columns and "admission_ward" in df.columns:
        rs25 = df.groupby(["oxygen_inhalation","admission_ward"]).agg(
            n=("admission_ward","size"),mort=("death_within_6_months","mean")).reset_index()
        rs25 = rs25[rs25["n"]>=10]
        if len(rs25):
            fig = px.bar(rs25,x="oxygen_inhalation",y="mort",color="admission_ward",
                         barmode="group",title="6-month mortality by oxygen status × ward",
                         color_discrete_sequence=PALETTE_SEQ)
            fig.update_layout(xaxis_title="Oxygen status",yaxis_title="Mortality",
                               yaxis_tickformat=".1%",legend_title="Ward")
            show(fig,h=300)
            insight("If OxygenTherapy patients are often placed in GeneralWard and show elevated mortality, "
                    "add a hard rule: any patient requiring supplemental oxygen at admission → "
                    "telemetry-monitored bed minimum, with automatic escalation for NIMV/IMV.")

    # ── P26 ───────────────────────────────────────────────────────────────────
    qcard("P26. Albumin cutoff for nutrition consult",
          "A statistically-derived albumin threshold turns a continuous lab into a binary action — "
          "'consult or no consult'.")
    from sklearn.metrics import roc_curve as _roc_curve
    sub26_all = df_all[["albumin","death_within_6_months"]].dropna()
    if len(sub26_all)>=50 and sub26_all["death_within_6_months"].sum()>=5:
        fpr26,tpr26,th26 = _roc_curve(sub26_all["death_within_6_months"],-sub26_all["albumin"])
        cutoff26 = -th26[int(np.argmax(tpr26-fpr26))]
        sub26 = df[["albumin","death_within_6_months"]].dropna()
        bl26 = sub26["albumin"] < cutoff26
        db26 = sub26.loc[bl26,"death_within_6_months"].mean()
        da26 = sub26.loc[~bl26,"death_within_6_months"].mean()
        c1,c2 = st.columns([1,2])
        with c1:
            st.metric("Derived cutoff",f"{cutoff26:.1f} g/L")
            st.metric("Mortality below cutoff",pct(db26))
            st.metric("Mortality above cutoff",pct(da26))
        with c2:
            fig = px.bar(x=["Above cutoff\n(adequate nutrition)","Below cutoff\n(malnourished)"],
                         y=[da26*100,db26*100],text=[f"{da26*100:.1f}%",f"{db26*100:.1f}%"],
                         title=f"6-month mortality by albumin cutoff ({cutoff26:.1f} g/L)",
                         color_discrete_sequence=[TEAL,WARM_RED])
            fig.update_layout(showlegend=False,xaxis_title=None,yaxis_title="Mortality %")
            show(fig,h=280)
        insight(f"Patients with albumin < {cutoff26:.1f} g/L have {safe_div(db26,da26):.1f}× the "
                "6-month mortality of those above it. Action: automatic dietitian/nutrition consult for "
                f"any patient with albumin < {cutoff26:.1f} g/L at admission.")

    # ── P27 ───────────────────────────────────────────────────────────────────
    qcard("P27. Statin intensification gap for poorly-controlled LDL",
          "Are patients with LDL > 2.6 mmol/L actually being prescribed statin therapy?")
    sub27 = df[["low_density_lipoprotein_cholesterol","rx_atorvastatin_calcium_tablet",
                "death_within_6_months"]].dropna()
    if len(sub27)>=30:
        sub27["ldl_ctrl"] = pd.cut(sub27["low_density_lipoprotein_cholesterol"],
                                    bins=[0,1.8,2.6,10],labels=["<1.8\ncontrolled",
                                    "1.8-2.6\nborderline",">2.6\npoorly controlled"])
        sp27 = sub27.groupby("ldl_ctrl",observed=True)["rx_atorvastatin_calcium_tablet"].mean()*100
        poor27 = sub27[sub27["ldl_ctrl"]==">2.6\npoorly controlled"]
        nos27 = int((poor27["rx_atorvastatin_calcium_tablet"]==0).sum())
        nos27_r = safe_div(nos27,len(poor27))
        mp27 = poor27.groupby("rx_atorvastatin_calcium_tablet")["death_within_6_months"].mean()
        c1,c2,c3 = st.columns(3)
        c1.metric("Poorly-controlled LDL",f"{len(poor27)}")
        c2.metric("Not on statin",f"{nos27} ({pct(nos27_r)})")
        c3.metric("Mortality: no statin vs statin",
                  f"{pct(mp27.get(0.0,float('nan')))} vs {pct(mp27.get(1.0,float('nan')))}")
        fig = px.bar(x=sp27.index.astype(str),y=sp27.values,text=sp27.round(1),
                     title="% on statin therapy by LDL control level",
                     color_discrete_sequence=[MID_BLUE])
        fig.update_layout(xaxis_title=None,yaxis_title="On statin %")
        show(fig,h=280)
        insight(f"{nos27} of {len(poor27)} patients ({pct(nos27_r)}) with poorly controlled LDL "
                "are not on statin therapy — coverage barely rises with worsening LDL. "
                "Action: flag every patient with LDL >2.6 mmol/L and no statin order for a mandatory "
                "pharmacy/cardiology review before discharge.")

    # ── P28 ───────────────────────────────────────────────────────────────────
    qcard("P28. WBC threshold for infection workup",
          "Infection is a common HF decompensation precipitant. Leukocytosis is an easy, "
          "already-available signal to systematise a workup.")
    wbc28 = df.dropna(subset=["white_blood_cell","death_within_6_months"]).copy()
    if len(wbc28)>=40:
        wbc28["wbc_b"] = pd.cut(wbc28["white_blood_cell"],bins=[0,4,10,15,100],
                                  labels=["<4 (low)","4-10 (normal)","10-15 (elevated)","15+ (high)"])
        wg28 = wbc28.groupby("wbc_b",observed=True)["death_within_6_months"].mean()*100
        if len(wg28)>=2:
            fig = px.bar(x=wg28.index.astype(str),y=wg28.values,text=wg28.round(1),
                         title="6-month mortality by WBC band",color_discrete_sequence=[MID_BLUE])
            fig.update_layout(xaxis_title="WBC (×10⁹/L)",yaxis_title="Mortality %")
            show(fig,h=290)
            insight(f"Mortality climbs with WBC: " +
                    " → ".join(f"{v:.1f}% ({lbl})" for lbl,v in wg28.items()) +
                    ". Action: auto-trigger blood/urine cultures + chest imaging for WBC >10 ×10⁹/L "
                    "at admission if no infection source is already documented.")
        else:
            st.info("Not enough WBC variety in this filter selection.")

    # ── P29 ───────────────────────────────────────────────────────────────────
    qcard("P29. Anion gap — metabolic derangement subgroup",
          "An elevated anion gap signals an unmeasured acid load. Does it predict worse outcomes?")
    ag29 = pd.cut(df["anion_gap"],bins=[0,8,16,50],
                   labels=["<8 low","8-16 normal",">16 high (acidosis)"])
    df["ag_band"] = ag29
    ag_mort = df.groupby("ag_band",observed=True)["death_within_28_days"].mean()*100
    ag_lac  = df.groupby("ag_band",observed=True)["lactate"].mean()
    c1,c2 = st.columns(2)
    with c1:
        fig = px.bar(x=ag_mort.index.astype(str),y=ag_mort.values,text=ag_mort.round(1),
                     title="28-day mortality by anion-gap band",color_discrete_sequence=[WARM_RED])
        fig.update_layout(xaxis_title=None,yaxis_title="Mortality %")
        show(fig,h=290)
    with c2:
        fig = px.bar(x=ag_lac.index.astype(str),y=ag_lac.values,text=ag_lac.round(2),
                     title="Mean lactate by anion-gap band (construct validity)",
                     color_discrete_sequence=[AMBER])
        fig.update_layout(xaxis_title=None,yaxis_title="Lactate (mmol/L)")
        show(fig,h=290)
    n_high_ag = int((df["ag_band"]==">16 high (acidosis)").sum())
    if ">16 high (acidosis)" in ag_mort.index and "8-16 normal" in ag_mort.index:
        fold29 = safe_div(ag_mort[">16 high (acidosis)"],ag_mort["8-16 normal"])
        insight(f"High anion-gap group (n={n_high_ag}) has {fold29:.1f}× the 28-day mortality of the "
                "normal band, and their lactate is nearly double — confirming real metabolic distress. "
                "Action: flag anion gap >16 for aggressive metabolic workup.")

    # ── P30 ───────────────────────────────────────────────────────────────────
    qcard("P30. BMI category & the obesity paradox",
          "Both underweight ('cardiac cachexia') and obesity carry distinct HF risks. "
          "Confirming the pattern directs nutrition referrals precisely rather than broadly.")
    sub30 = df[df["bmi_category"].isin(BMI_ORDER)].copy()
    sub30["bmi_category"] = pd.Categorical(sub30["bmi_category"],categories=BMI_ORDER,ordered=True)
    if len(sub30)>=20:
        g30 = sub30.groupby("bmi_category",observed=True)[
            ["death_within_6_months","re_admission_within_6_months"]].mean()*100
        c1,c2 = st.columns(2)
        with c1:
            fig = px.bar(x=g30.index.astype(str),y=g30["death_within_6_months"],
                         text=g30["death_within_6_months"].round(1),
                         title="6-month mortality by BMI category",color_discrete_sequence=[WARM_RED])
            fig.update_layout(xaxis_title=None,yaxis_title="Mortality %")
            show(fig,h=280)
        with c2:
            fig = px.bar(x=g30.index.astype(str),y=g30["re_admission_within_6_months"],
                         text=g30["re_admission_within_6_months"].round(1),
                         title="6-month readmission by BMI category",color_discrete_sequence=[MID_BLUE])
            fig.update_layout(xaxis_title=None,yaxis_title="Readmission %")
            show(fig,h=280)
        if g30["death_within_6_months"].notna().sum()>=2:
            best30  = g30["death_within_6_months"].idxmin()
            worst30 = g30["death_within_6_months"].idxmax()
            insight(f"{best30} patients have the lowest 6-month mortality "
                    f"({g30.loc[best30,'death_within_6_months']:.1f}%); "
                    f"{worst30} the highest ({g30.loc[worst30,'death_within_6_months']:.1f}%) — "
                    "consistent with the 'obesity paradox' in heart failure. "
                    "Recommendation: target Underweight patients specifically for nutrition support; "
                    "do NOT use high BMI as the mortality flag in this cohort.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: PREDICTIVE
# ══════════════════════════════════════════════════════════════════════════════
NUM_F = ["cci_score","systolic_blood_pressure","diastolic_blood_pressure","pulse",
         "brain_natriuretic_peptide","creatinine_enzymatic_method","sodium","potassium",
         "hemoglobin","white_blood_cell","platelet","lvef","total_prescriptions_count"]
CAT_F = ["gender","bmi_category","admission_way","type_of_heart_failure",
         "nyha_cardiac_function_classification"]
TARGET = "re_admission_within_6_months"


@st.cache_resource(show_spinner="Training prediction models on full cohort…")
def train_models(_df, key):
    X = _df[NUM_F+CAT_F].copy()
    y = _df[TARGET]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)
    pre = ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),NUM_F),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                         ("ohe",OneHotEncoder(handle_unknown="ignore"))]),CAT_F),
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000,class_weight="balanced"),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5,class_weight="balanced",random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
    }
    pipes,metrics,probas,preds = {},{},{},{}
    for nm,mdl in models.items():
        pipe = Pipeline([("pre",pre),("clf",mdl)])
        pipe.fit(Xtr,ytr)
        p = pipe.predict_proba(Xte)[:,1]; yh = pipe.predict(Xte)
        pipes[nm]=pipe; probas[nm]=p; preds[nm]=yh
        metrics[nm]={"Accuracy":accuracy_score(yte,yh),"Precision":precision_score(yte,yh,zero_division=0),
                     "Recall":recall_score(yte,yh,zero_division=0),"F1":f1_score(yte,yh,zero_division=0),
                     "AUC":roc_auc_score(yte,p)}
    return dict(pipes=pipes,metrics=metrics,probas=probas,preds=preds,y_test=yte,
                n_train=len(Xtr),n_test=len(Xte))


with tab_ml:
    sec("🔮 Predictive Analysis — What is likely to happen?")
    st.caption("Flagship model: 6-month readmission prediction using admission-day data only. "
               "All 7 predictive questions from the notebook are covered (5 as live models, "
               "2 as methodology/summary cards).")

    with st.expander("Hypotheses, parameters & target variable", expanded=False):
        st.markdown("""
**H₀:** Admission-time clinical features have no predictive relationship with 6-month readmission (AUC = 0.5).  
**H₁:** They carry real predictive signal (AUC > 0.5).

**Target:** `re_admission_within_6_months` (binary: 0 = not readmitted, 1 = readmitted)

| Feature | Type | Clinical connection |
|---|---|---|
| `cci_score` | Numeric | Competing chronic conditions |
| `systolic/diastolic_blood_pressure` | Numeric | Haemodynamic stability |
| `pulse` | Numeric | Cardiac output proxy |
| `brain_natriuretic_peptide` | Numeric | HF severity |
| `creatinine`, `sodium`, `potassium`, `haemoglobin` | Numeric | Renal/electrolyte status |
| `white_blood_cell`, `platelet` | Numeric | Infection / haematologic burden |
| `lvef` | Numeric | Systolic function |
| `total_prescriptions_count` | Numeric | Polypharmacy burden |
| `gender`, `bmi_category`, `admission_way`, `HF type`, `NYHA class` | Categorical | Demographics & severity |
""")

    res = train_models(df_all, src_name)
    mdf = pd.DataFrame(res["metrics"]).T[["Accuracy","Precision","Recall","F1","AUC"]]

    st.subheader("Model comparison — 6-month readmission")
    st.caption(f"Trained on {res['n_train']:,} patients | Evaluated on {res['n_test']:,} held-out patients "
               "(full cohort, unaffected by sidebar filters)")
    st.dataframe(mdf.style.highlight_max(color="#d4edda",axis=0).format("{:.3f}"),
                 use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        mm = (mdf[["Accuracy","Precision","Recall","F1"]].reset_index()
              .rename(columns={"index":"Model"}).melt("Model",var_name="Metric",value_name="Score"))
        fig = px.bar(mm,x="Model",y="Score",color="Metric",barmode="group",
                     title="Metric comparison across models",color_discrete_sequence=PALETTE_SEQ)
        fig.update_yaxes(range=[0,1])
        show(fig)
    with c2:
        fig = go.Figure()
        for nm,p in res["probas"].items():
            fpr,tpr,_ = roc_curve(res["y_test"],p)
            fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",
                                      name=f"{nm} (AUC {mdf.loc[nm,'AUC']:.3f})"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
                                  line=dict(dash="dash",color=GREY)))
        fig.update_layout(title="ROC curves — all four models",
                          xaxis_title="False positive rate",yaxis_title="True positive rate")
        show(fig)

    best_auc = mdf["AUC"].idxmax(); best_f1 = mdf["F1"].idxmax()
    insight(f"All four models beat AUC 0.5 → **H₀ rejected** — admission data carries real predictive "
            f"signal. {best_auc} has the highest AUC ({mdf.loc[best_auc,'AUC']:.3f}); "
            f"{best_f1} has the best recall/F1 ({mdf.loc[best_f1,'F1']:.3f}) — preferred for a "
            "preventive-care programme where missing a high-risk patient costs more than a false alarm.")

    st.subheader("Final model: Logistic Regression — confusion matrix & feature importance")
    fp_name = "Logistic Regression"
    fp_pipe = res["pipes"][fp_name]
    fp_pred = res["preds"][fp_name]
    fp_prob = res["probas"][fp_name]
    c1,c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(res["y_test"],fp_pred)
        fig = px.imshow(cm,text_auto=True,color_continuous_scale="Blues",
                        x=["Predicted: No readmit","Predicted: Readmit"],
                        y=["Actual: No readmit","Actual: Readmit"],
                        title=f"Confusion matrix — {fp_name}")
        show(fig)
    with c2:
        feat_n  = fp_pipe.named_steps["pre"].get_feature_names_out()
        coefs   = fp_pipe.named_steps["clf"].coef_[0]
        imp_ser = pd.Series(coefs,index=feat_n).sort_values(key=abs,ascending=False).head(12)
        imp_ser = imp_ser.iloc[::-1]
        clrs    = [WARM_RED if v>0 else TEAL for v in imp_ser.values]
        fig = go.Figure(go.Bar(x=imp_ser.values,
                               y=[n.split("__")[-1] for n in imp_ser.index],
                               orientation="h",marker_color=clrs))
        fig.update_layout(title="Top predictors (red = raises risk, green = lowers risk)",
                          xaxis_title="Coefficient")
        show(fig)

    st.subheader("Risk tiers from predicted probabilities")
    ro = pd.DataFrame({"prob":fp_prob,"actual":res["y_test"].values})
    ro["tier"] = pd.qcut(ro["prob"],q=3,labels=["Low","Medium","High"])
    tr = ro.groupby("tier",observed=True)["actual"].mean()*100
    fig = px.bar(x=tr.index.astype(str),y=tr.values,text=tr.round(1),
                 title="Actual 6-month readmission rate by model-predicted risk tier",
                 color=tr.index.astype(str),
                 color_discrete_map={"Low":TEAL,"Medium":AMBER,"High":WARM_RED})
    fig.update_layout(showlegend=False,xaxis_title=None,yaxis_title="Readmission %")
    show(fig,h=300)
    insight(f"Actual readmission rises from {tr.iloc[0]:.1f}% (Low) to {tr.iloc[-1]:.1f}% (High). "
            "The model's ranking is directionally useful for prioritising follow-up even at moderate AUC.")

    st.divider()
    st.subheader("All 7 predictive questions — team's notebook summary")
    other = pd.DataFrame([
        {"#":"Q1","Question":"6-month readmission prediction",
         "Best AUC/result":"0.640 (Gradient Boosting)","Final model":"Logistic Regression (best recall)",
         "Key insight":"H₀ rejected. Modest signal — useful for risk-tier prioritisation."},
        {"#":"Q2","Question":"Discharge destination (home vs. facility)",
         "Best AUC/result":"0.650","Final model":"Logistic Regression",
         "Key insight":"Real signal (p=0.02) but below accuracy bar for individual-patient action on day 1. "
                        "Triggers early social-work consult."},
        {"#":"Q3","Question":"HF type (Left / Right / Both) — multiclass",
         "Best AUC/result":"Macro F1 best metric","Final model":"Random Forest",
         "Key insight":"COPD history and occupation are the only significant demographic predictors. "
                        "Right-sided class n=51 — interpret with caution."},
        {"#":"Q4","Question":"Need for supplemental oxygen at admission",
         "Best AUC/result":"AUC-ranked","Final model":"Random Forest",
         "Key insight":"Age category and COPD history are strongest predictors; "
                        "useful for pre-allocating respiratory equipment."},
        {"#":"Q5","Question":"Which outcome is most predictable from admission data?",
         "Best AUC/result":"AUC 80-98% (diagnoses + 28-day death)","Final model":"Logistic Regression",
         "Key insight":"Diagnoses and short-term mortality predict well; "
                        "readmission/ED-return AUC only 67-68% — utilisation forecasting needs "
                        "post-discharge data."},
        {"#":"Q6","Question":"Length of stay — continuous vs. binary",
         "Best AUC/result":"R²=0.16 continuous; AUC=0.752 binary (long-stay flag)","Final model":"Random Forest",
         "Key insight":"Exact LOS is hard to forecast; binary 'long stay' flag works meaningfully better "
                        "and should drive bed-planning tools."},
        {"#":"Q7","Question":"Need for IV inotropic support",
         "Best AUC/result":"~0.73 (Random Forest)","Final model":"Random Forest",
         "Key insight":"BNP, urea, troponin, creatinine and GFR top predictors. "
                        "Useful as a triage aid — not a clinical decision tool alone."},
    ])
    st.dataframe(other,hide_index=True,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_s:
    sec("📋 Executive Summary")
    st.caption("Headline findings from all three analysis sections — the closing slide.")

    biv   = (df["type_of_heart_failure"]=="Both").mean()
    ny34  = df["nyha_cardiac_function_classification"].isin([3,4]).mean()
    k34   = (df["killip_grade"]>=3).mean()
    icu_r = (df["admission_ward"]=="ICU").mean()

    st.subheader("1. Who are these patients? (Descriptive)")
    k = st.columns(4)
    k[0].metric("Biventricular HF",    pct(biv))
    k[1].metric("NYHA class 3-4",      pct(ny34))
    k[2].metric("Killip grade 3-4",    pct(k34))
    k[3].metric("ICU admission rate",  pct(icu_r))
    hypo_d = hypo.get("Died within 6mo",float("nan")); hypo_s = hypo.get("Survived",float("nan"))
    st.markdown(f"- **{pct(biv)} have biventricular (both-sided) heart failure** — an advanced form, "
                f"not an early-stage cohort.\n"
                f"- **{pct(ny34)} were NYHA class 3-4** at admission — already markedly–severely limited.\n"
                f"- Sodium is an early warning sign: **{hypo_d:.0f}% of patients who died** were "
                f"hyponatraemic vs **{hypo_s:.0f}% of survivors**.")

    st.divider()
    st.subheader("2. What can we forecast? (Predictive)")
    k = st.columns(3)
    k[0].metric("Best AUC — 6-mo readmission", f"{mdf['AUC'].max():.3f}")
    k[1].metric("Best F1",                      f"{mdf['F1'].max():.3f}")
    k[2].metric("H₀ rejected",                  "Yes — real signal")
    st.markdown("- Admission-day data carries **real but moderate** predictive signal for 6-month "
                "readmission — better than chance, not yet strong enough to act on alone.\n"
                "- Diagnoses and 28-day mortality predict well (AUC 80-98%); readmission and "
                "ED-return are harder (AUC ~67-68%) — AI effort should focus on diagnosis-support "
                "and risk-flagging rather than utilisation forecasting.")

    st.divider()
    st.subheader("3. What should we do? (Prescriptive — top 5 actions)")
    n_miss21 = int(flagged21.sum()) if "flagged21" in dir() else 0
    hfref_s = df[df["lvef"]<=40]; gap_s = hfref_s[hfref_s["gdmt_count"]<3]
    hfref_gap_share = safe_div(len(gap_s),len(hfref_s)) if len(hfref_s) else float("nan")
    capt21 = captured21 if "captured21" in dir() else float("nan")
    k = st.columns(3)
    k[0].metric("Under-triaged patients (10× death rate)", f"~133")
    k[1].metric("HFrEF missing GDMT",f"{len(gap_s)} ({pct(hfref_gap_share)})")
    k[2].metric("Deaths captured by flagging ≤45% of patients", pct(capt21))
    st.markdown(
        "1. 🔴 **ICU escalation protocol** — ~133 patients met ≥2 criteria (Killip ≥3, lactate >2.15, "
        "reduced consciousness) but were not escalated. Their death rate was ~10× higher.\n"
        f"2. 🔴 **GDMT medication gap** — {pct(hfref_gap_share)} of HFrEF patients are missing ≥1 of "
        "the 3 guideline medicines; readmission is ~1.6× higher.\n"
        "3. 🟡 **Follow-up risk score** — a simple 6-point admission score captures "
        f"{pct(capt21)} of 6-month deaths by flagging only ~45% of patients.\n"
        "4. 🟡 **Albumin nutrition consult** — albumin below the derived cutoff predicts ~2× "
        "higher 6-month mortality; automatic dietitian referral costs nothing extra.\n"
        "5. 🟡 **Statin audit** — ~53% of patients with poorly-controlled LDL are not on statin; "
        "those on statin have 3.5× lower mortality in that group.")

    st.divider()
    st.success("**Bottom line — one sentence:** This is an advanced-disease cohort with clear, "
               "fixable gaps at the point of admission; identifying them automatically with "
               "existing data (labs, meds, severity scores) can redirect limited follow-up resources "
               "to the patients most likely to benefit.")
    st.caption("All findings are observational associations. Predictive models trained on full cohort. "
               "Sidebar filters affect Descriptive, Prescriptive and Summary figures.")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — EXPORT FOR PPT  (placed last so EXPORT_FIGS is fully populated)
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.divider()
st.sidebar.subheader("📦 Export for PPT")
st.sidebar.caption(
    f"Each chart already has a **camera icon** in its toolbar (hover the chart to reveal it) — "
    f"click it to download that chart as a high-resolution PNG. "
    f"**{len(EXPORT_FIGS)} charts** are registered this view."
)
st.sidebar.caption("Bulk PNG export also needs: `pip install -U kaleido`")

if st.sidebar.button("Generate PNG export pack (all charts)"):
    with st.spinner(f"Rendering {len(EXPORT_FIGS)} charts to PNG — this may take 30-60 s…"):
        _buf = io.BytesIO()
        ok_n = fail_n = 0
        with zipfile.ZipFile(_buf,"w",zipfile.ZIP_DEFLATED) as _zf:
            for _fn,_fg in EXPORT_FIGS:
                try:
                    _zf.writestr(f"{_fn}.png",_fg.to_image(format="png",scale=3,width=1000,height=620))
                    ok_n += 1
                except Exception:
                    fail_n += 1
        _buf.seek(0)
    if ok_n>0:
        st.sidebar.success(f"✅ {ok_n} charts rendered{f' ({fail_n} failed)' if fail_n else ''}.")
        st.sidebar.download_button("⬇️ Download ZIP of all charts",data=_buf.getvalue(),
                                    file_name="cardiac_failure_charts.zip",mime="application/zip")
    else:
        st.sidebar.error("Could not render any charts. Install kaleido: `pip install -U kaleido`")

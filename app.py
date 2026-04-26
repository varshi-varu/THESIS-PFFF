"""
PFFF v12 — Streamlit Web App
==============================
WHAT THIS APP DOES:
  Computes the Fragility Index (FI%) for Indian highway DPR projects.
  FI = probability that the project's IRR falls below its hurdle rate
       under realistic uncertainty (Monte Carlo, 5,000 iterations).

FIXES IN v12 vs previous versions:
  1. LA% direction CORRECT: higher LA% → lower FI (less delay, less LA cost overrun)
  2. FIRR now properly more fragile than EIRR (cost sensitivity fixed)
  3. Landing page shows project selector, not raw CPRR data
  4. Zero-stress proof clears previous charts
  5. Switching values shown prominently
  6. Removed clutter (Stage 2 logic text, SCN deep-dive, monitoring protocol)
  7. Charts are full-size and readable

HOW TO RUN:
  streamlit run app_v12.py
  (pfff_engine_v12.py must be in the same folder as pfff_engine.py)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import brentq
import io, json, warnings
warnings.filterwarnings("ignore")

# ── Engine ─────────────────────────────────────────────────────────────
try:
    from pfff_engine import (
        PROJECTS, MODES, HURDLES,
        compute_scn, run_mcs, simulate_mode,
        spearman_tornado, rcf_acid_test, eirr_iter,
        fi_color, verdict, compute_switching_values, compute_sv_both,
    )
except ImportError as e:
    st.error(f"Cannot find pfff_engine.py in the same folder.\nError: {e}")
    st.stop()

# ── Page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PFFF v12 — NHAI DPR Fragility Auditor",
    page_icon="🏛️", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { padding-top: 0.5rem; }
.fi-badge { border-radius:10px; padding:16px 20px; text-align:center; }
.fi-big   { font-size:3rem; font-weight:700; line-height:1.1; }
.fi-sub   { font-size:0.95rem; margin-top:4px; }
.kpi-box  { background:#f8f9fa; border-radius:8px; padding:12px 10px;
            border-left:4px solid #dee2e6; text-align:center; }
.kpi-val  { font-size:1.7rem; font-weight:700; }
.kpi-lbl  { font-size:0.76rem; color:#6c757d; }
.sv-ok    { color:#198754; font-weight:700; }
.sv-warn  { color:#856404; font-weight:700; }
.sv-crit  { color:#842029; font-weight:700; }
.note     { background:#e8f4fd; border-left:4px solid #0d6efd; border-radius:6px;
            padding:10px 14px; font-size:0.86rem; color:#0c3c60; margin:6px 0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────
def fc(fi):
    return "#198754" if fi<25 else "#856404" if fi<50 else "#842029"

def bg(fi):
    return "#D1E7DD" if fi<25 else "#FFF3CD" if fi<50 else "#F8D7DA"

def vt(fi):
    return "GREEN — Approve" if fi<25 else "AMBER — Conditional Approve" if fi<50 else "RED — Return DPR"


# ── Cache ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=None)
def _sim(pj, mode, n):
    p    = json.loads(pj)
    scn  = compute_scn(p)
    samp = run_mcs(p, scn, n)
    res  = simulate_mode(p, scn, samp, mode, n)
    torn = spearman_tornado(p, scn, samp, res["eirr_arr"])
    rcf  = rcf_acid_test(p, scn, samp, res["fi_p"])
    svs  = compute_switching_values(p, scn)
    return res, scn, samp, torn, rcf, svs


@st.cache_data(show_spinner=False, ttl=None)
def _zs(pj):
    p   = json.loads(pj)
    scn = compute_scn(p)
    return eirr_iter(p, scn, p["civil_cr"], 0.0,
                     p["yr1_aadt"], p["growth"], 1.0, 1.0) * 100


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏛️ PFFF v12")
    st.caption("Probabilistic Feasibility Fragility Framework\nM.BEM Thesis | SPA Delhi 2024 | Varshni M S")
    st.divider()

    # ── Landing: project selector ─────────────────────────────────────
    st.markdown("**Select a Project Template**")
    proj_key = st.selectbox(
        "Project",
        list(PROJECTS.keys()),
        format_func=lambda k: f"{k} — {PROJECTS[k]['name'][:35]}",
        key="proj_sel"
    )
    if st.button("📂 Load Project", use_container_width=True, type="primary"):
        st.session_state["p"] = dict(PROJECTS[proj_key])

    if "p" not in st.session_state:
        st.session_state["p"] = dict(PROJECTS["P2"])

    p = st.session_state["p"]
    st.divider()

    # ── Simulation settings ───────────────────────────────────────────
    n_iter   = st.select_slider("Monte Carlo Iterations", [1000,2000,5000,10000], value=5000)
    sim_mode = st.selectbox("Procurement Mode", MODES,
                            index=MODES.index(p.get("dpr_mode","EPC")))
    st.divider()

    # ── Editable inputs ───────────────────────────────────────────────
    st.markdown("**✏️ Edit DPR Inputs**")

    with st.expander("Economic Parameters", expanded=True):
        p["dpr_eirr"]  = st.number_input("DPR EIRR (%)", value=float(p["dpr_eirr"]), step=0.1,
                                          help="Consultant's stated EIRR from DPR")
        p["cost_sens"] = st.number_input("Cost Sensitivity (pp/1%)", value=float(p.get("cost_sens",0.15)), step=0.01,
                                          help="From DPR sensitivity table. (EIRR_base − EIRR_at_+15%cost)/15")
        p["traf_sens"] = st.number_input("Traffic Sensitivity (pp/1%)", value=float(p.get("traf_sens",0.20)), step=0.01,
                                          help="From DPR sensitivity table. (EIRR_base − EIRR_at_−15%traf)/15")

        st.markdown("**FIRR & Equity (HAM/BOT only)**")
        has_firr = st.checkbox("Project has FIRR", value=(p.get("dpr_firr") not in (None, 0)))
        p["dpr_firr"] = st.number_input("DPR FIRR (%)", value=float(p.get("dpr_firr") or 12.0),
                                         step=0.1) if has_firr else None
        has_eq = st.checkbox("Project has Equity IRR", value=(p.get("dpr_eq") not in (None, 0)))
        p["dpr_eq"] = st.number_input("DPR Equity IRR (%)", value=float(p.get("dpr_eq") or 15.0),
                                       step=0.1) if has_eq else None

    with st.expander("Costs & Traffic", expanded=False):
        p["civil_cr"]  = st.number_input("Civil Cost (₹ Cr)", value=float(p["civil_cr"]), step=10.0)
        p["la_cr"]     = st.number_input("LA Cost (₹ Cr)", value=float(p["la_cr"]), step=10.0)
        p["om_cr"]     = st.number_input("O&M Yr1 (₹ Cr)", value=float(p.get("om_cr",20.0)))
        p["scale_cr"]  = p["civil_cr"]
        p["base_aadt"] = st.number_input("Base Year AADT", value=int(p["base_aadt"]))
        p["yr1_aadt"]  = st.number_input("Year-1 AADT (DPR forecast)", value=int(p["yr1_aadt"]))
        p["growth"]    = st.number_input("Growth Rate", value=float(p.get("growth",0.065)), step=0.005)
        p["dpr_yr"]    = st.number_input("DPR Year", value=int(p.get("dpr_yr",2020)), step=1, min_value=1990, max_value=2030)
        p["survey_yr"] = st.number_input("Survey Year", value=int(p.get("survey_yr",2019)), step=1, min_value=1990, max_value=2030)
        p["survey_indep"] = st.checkbox("Independent Survey", value=bool(p.get("survey_indep",False)))

    with st.expander("🏗️ Risk Conditioners", expanded=True):
        st.caption("These shift the MCS distributions BEFORE simulation runs.")

        p["la_pct"] = st.slider(
            "LA% Complete at DPR",
            0, 100, int(p.get("la_pct",50)),
            help="Higher LA% → lower delay risk (p_stall) AND lower LA cost overrun (v06_mult). "
                 "Increasing this REDUCES FI_FIRR and FI_Equity significantly. "
                 "EIRR changes only via delay (LA is excluded from EIRR per IRC SP:30)."
        )
        p["geotech"]    = st.select_slider("Geotech Quality",
                                            ["DESKTOP","PARTIAL","COMPLETE"], value=p.get("geotech","PARTIAL"))
        p["contractor"] = st.select_slider("Contractor Capability",
                                            ["STRESSED","ADEQUATE","STRONG"], value=p.get("contractor","ADEQUATE"))
        p["community"]  = st.select_slider("Community / R&R Risk",
                                            ["LOW","LOW_MEDIUM","MEDIUM","HIGH","EXTREME"],
                                            value=p.get("community","MEDIUM"))
        p["terrain"]    = st.selectbox("Terrain",
                                        ["PLAIN","ROLLING","COASTAL_ROLLING","HILLY","MIXED_MOUNTAIN","MOUNTAIN"],
                                        index=["PLAIN","ROLLING","COASTAL_ROLLING","HILLY","MIXED_MOUNTAIN","MOUNTAIN"].index(p.get("terrain","PLAIN")))
        p["forest_clr"] = st.selectbox("Forest Clearance Status",
                                        ["NONE","CLEARED","EIA_PENDING","NOT_APPLIED","PENDING","STAGE_II","BLOCKED"],
                                        index=["NONE","CLEARED","EIA_PENDING","NOT_APPLIED","PENDING","STAGE_II","BLOCKED"].index(p.get("forest_clr","NONE")))
        p["crossings"]  = st.selectbox("Major Crossings",
                                        ["LOW","MODERATE","HIGH","VERY_HIGH"],
                                        index=["LOW","MODERATE","HIGH","VERY_HIGH"].index(p.get("crossings","LOW")))
        p["network"]    = st.selectbox("Network Type",
                                        ["STANDALONE","FEEDER","CORRIDOR_LINK"],
                                        index=["STANDALONE","FEEDER","CORRIDOR_LINK"].index(p.get("network","FEEDER")))
        p["proj_type"]  = st.selectbox("Project Type",["GREENFIELD","BROWNFIELD"],
                                        index=["GREENFIELD","BROWNFIELD"].index(p.get("proj_type","GREENFIELD")))
        p["forest_pct"] = st.number_input("Forest Area (%)", value=float(p.get("forest_pct",0.0)))

    st.session_state["p"] = p


# ══════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════

pj = json.dumps(p, default=str)

with st.spinner(f"Running {n_iter:,} Monte Carlo iterations…"):
    res, scn, samp, tornado, rcf, svs = _sim(pj, sim_mode, n_iter)

ep   = res["eirr_arr"] * 100
fi   = res["fi_p"]
p10  = np.percentile(ep, 10)
p20  = np.percentile(ep, 20)
p50  = np.percentile(ep, 50)
p80  = np.percentile(ep, 80)
p90  = np.percentile(ep, 90)


# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════

col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown(f"## {p['name']}")
    st.caption(f"Mode: **{sim_mode}** (DPR chosen: **{p['dpr_mode']}**) · "
               f"Survey age at DPR: **{scn['survey_age']}yr** · "
               f"DPR EIRR: **{p['dpr_eirr']:.2f}%** · "
               f"{n_iter:,} iterations")

with col_badge:
    st.markdown(f"""
    <div class='fi-badge' style='background:{bg(fi)}; border:2px solid {fc(fi)}'>
      <div class='fi-big' style='color:{fc(fi)}'>{fi:.1f}%</div>
      <div class='fi-sub' style='color:{fc(fi)}'>{vt(fi)}</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Zero-stress proof (inline, clears on toggle) ──────────────────────
if st.toggle("🟢 Show Zero-Stress Calibration Proof"):
    zs = _zs(pj)
    d  = abs(zs - p["dpr_eirr"])
    status = "✅ PASS" if d < 0.05 else f"⚠️ Δ={d:.2f}pp"
    st.markdown(f"""<div class='note'>
    <b>Zero-Stress Proof:</b> When fed exactly DPR values (no overrun, no delay, traffic = DPR forecast) →
    Simulated EIRR = <b>{zs:.2f}%</b> | DPR Stated = <b>{p['dpr_eirr']:.2f}%</b> | <b>{status}</b><br>
    <small>This confirms the model is a stress-tester, not a rejection machine.
    The FI ({fi:.1f}%) reflects what happens under realistic input variation.</small>
    </div>""", unsafe_allow_html=True)


# ── KPI Row ────────────────────────────────────────────────────────────
st.markdown("### Key Outputs")
kc1,kc2,kc3,kc4,kc5,kc6 = st.columns(6)

def kpi(col, val, lbl, col_hex, sub=""):
    col.markdown(f"""<div class='kpi-box' style='border-left-color:{col_hex}'>
    <div class='kpi-val' style='color:{col_hex}'>{val}</div>
    <div class='kpi-lbl'>{lbl}</div>
    <div class='kpi-lbl'>{sub}</div>
    </div>""", unsafe_allow_html=True)

kpi(kc1, f"{fi:.1f}%",   "FI Primary",   fc(fi),    vt(fi).split("—")[0].strip())
kpi(kc2, f"{res['fi_eirr']:.1f}%", "FI EIRR",  fc(res['fi_eirr']), "Hurdle 12%")

fi_f_str = f"{res['fi_firr']:.1f}%" if not np.isnan(res['fi_firr']) else "N/A"
fc_f = fc(res['fi_firr']) if not np.isnan(res['fi_firr']) else "#6c757d"
kpi(kc3, fi_f_str, "FI FIRR", fc_f, "Hurdle 10%")

fi_e_str = f"{res['fi_eq']:.1f}%" if not np.isnan(res['fi_eq']) else "N/A"
fc_e = fc(res['fi_eq']) if not np.isnan(res['fi_eq']) else "#6c757d"
eq_h = res.get('hurdle_eq')
kpi(kc4, fi_e_str, "FI Equity", fc_e, f"Hurdle {eq_h*100:.0f}%" if eq_h else "N/A")

kpi(kc5, f"{p50:.2f}%",  "P50 EIRR",    "#198754" if p50>=12 else "#842029", f"DPR: {p['dpr_eirr']:.2f}%")
kpi(kc6, f"{p50-12:+.2f}pp", "Safety Margin", "#198754" if p50>=12 else "#842029", "P50 − 12% hurdle")

st.divider()


# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════

tab_irr, tab_drivers, tab_sv, tab_batch, tab_export = st.tabs([
    "📊 IRR Distributions",
    "🎯 Fragility Drivers",
    "🔑 Switching Values",
    "📋 All Projects",
    "💾 Export"
])


# ─────────────────────────────────────────────────────────────────────
# TAB 1 — IRR DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────
with tab_irr:
    def hist_chart(arr, hurdle_val, color, title, dpr_val=None):
        valid = arr[~np.isnan(arr)] * 100 if arr is not None else np.array([])
        if len(valid) < 10:
            fig = go.Figure()
            fig.add_annotation(text=f"{title}<br><br>Not applicable for {sim_mode}",
                               xref="paper",yref="paper",x=0.5,y=0.5,
                               showarrow=False,font=dict(size=13,color="#6c757d"))
            fig.update_layout(height=380,plot_bgcolor="#FAFAFA",paper_bgcolor="white",
                              xaxis_visible=False,yaxis_visible=False)
            return fig
        fi_v = np.sum(valid < hurdle_val*100)/len(valid)*100
        fig = go.Figure()
        # fail zone
        fig.add_vrect(x0=min(valid)-3, x1=hurdle_val*100,
                      fillcolor="rgba(220,53,69,0.08)", line_width=0,
                      annotation_text="Below hurdle",annotation_position="top left",
                      annotation_font_color="#842029",annotation_font_size=9)
        fig.add_trace(go.Histogram(x=valid,nbinsx=55,name="Simulated",
                                    marker_color=color,marker_line=dict(color="white",width=0.4),
                                    opacity=0.85))
        fig.add_vline(x=hurdle_val*100,line_dash="dash",line_color="#DC3545",line_width=2.5,
                      annotation_text=f"Hurdle {hurdle_val*100:.0f}%",
                      annotation_font_color="#DC3545")
        if dpr_val:
            fig.add_vline(x=dpr_val,line_dash="dot",line_color="#212529",line_width=2,
                          annotation_text=f"DPR {dpr_val:.1f}%",annotation_position="top right")
        for pv,pn,pc in [(np.percentile(valid,20),"P20","#FFC107"),
                          (np.percentile(valid,50),"P50","#0D6EFD"),
                          (np.percentile(valid,80),"P80","#198754")]:
            fig.add_vline(x=pv,line_dash="longdash",line_color=pc,line_width=1.2)
        fig.add_annotation(text=f"<b>FI = {fi_v:.1f}%</b><br>{vt(fi_v).split('—')[0].strip()}",
                           xref="paper",yref="paper",x=0.02,y=0.96,showarrow=False,
                           bgcolor=bg(fi_v),bordercolor=fc(fi_v),borderwidth=1.5,borderpad=5,
                           font=dict(size=11,color=fc(fi_v)))
        fig.update_layout(
            title=f"<b>{title}</b>",height=400,
            plot_bgcolor="#FAFAFA",paper_bgcolor="white",bargap=0.04,showlegend=False,
            xaxis=dict(title="IRR (%)",gridcolor="#EEEEEE"),
            yaxis=dict(title="Frequency",gridcolor="#EEEEEE"),
            margin=dict(l=50,r=50,t=50,b=40)
        )
        return fig

    c1,c2,c3 = st.columns(3)
    with c1:
        st.plotly_chart(hist_chart(res["eirr_arr"], HURDLES["EIRR"], "#17A589",
                                   "Economic IRR (EIRR) | Society's View",
                                   dpr_val=p["dpr_eirr"]),
                        use_container_width=True)
    with c2:
        st.plotly_chart(hist_chart(
            res["firr_arr"] if not np.all(np.isnan(res["firr_arr"])) else None,
            HURDLES["FIRR"], "#8E44AD",
            "Financial IRR (FIRR) | Lender's View",
            dpr_val=p.get("dpr_firr")),
            use_container_width=True)
    with c3:
        eq_h_val = res.get("hurdle_eq") or HURDLES["EQ_BOT"]
        st.plotly_chart(hist_chart(
            res["eq_arr"] if not np.all(np.isnan(res["eq_arr"])) else None,
            eq_h_val, "#2471A3",
            f"Equity IRR | Concessionaire's View",
            dpr_val=p.get("dpr_eq")),
            use_container_width=True)

    st.markdown("""<div class='note'>
    <b>Reading the charts:</b> Each bar shows how many of the 5,000 simulations produced that IRR.
    The <b>red dashed line</b> is the hurdle rate. The shaded zone to the left = failure region.
    <b>FI% = area of histogram to the left of the hurdle.</b>
    </div>""", unsafe_allow_html=True)

    # Percentile table
    st.markdown("#### EIRR Percentile Summary")
    df_pct = pd.DataFrame({
        "Percentile": ["P10","P20","P50 (central)","P80","P90","DPR Stated"],
        "EIRR (%)": [round(x,2) for x in [p10,p20,p50,p80,p90,p["dpr_eirr"]]],
        "vs 12% Hurdle": [f"{x-12:+.2f}pp" for x in [p10,p20,p50,p80,p90,p["dpr_eirr"]]],
        "Meaning": [
            "10% chance EIRR is this low",
            "20% chance EIRR is this low",
            "Realistic central outcome",
            "80% chance EIRR is this low",
            "90% chance EIRR is this low",
            "Consultant's stated estimate"
        ]
    })
    st.dataframe(df_pct, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 2 — FRAGILITY DRIVERS
# ─────────────────────────────────────────────────────────────────────
with tab_drivers:
    col_t, col_m = st.columns([3,2])

    with col_t:
        st.markdown("#### Spearman Rank Tornado")
        st.caption("Shows which input variable has the highest rank-correlation with EIRR outcomes. "
                   "Red = higher value → lower EIRR (risk). Blue = higher value → higher EIRR (opportunity).")
        names = [t[0] for t in tornado[:7]][::-1]
        rhos  = [t[1] for t in tornado[:7]][::-1]
        colors_bar = ["#DC3545" if r<0 else "#0D6EFD" for r in rhos]
        fig_tor = go.Figure(go.Bar(
            x=rhos, y=names, orientation="h",
            marker_color=colors_bar, opacity=0.85,
            text=[f"{r:+.3f}" for r in rhos], textposition="outside"
        ))
        fig_tor.add_vline(x=0, line_color="#212529", line_width=1)
        fig_tor.update_layout(
            title=f"<b>Primary Driver: {tornado[0][0]}</b>  (ρ={tornado[0][1]:+.3f})",
            height=380, plot_bgcolor="#FAFAFA", paper_bgcolor="white",
            xaxis=dict(title="Spearman ρ with EIRR",gridcolor="#EEEEEE"),
            margin=dict(l=10,r=80,t=50,b=40), showlegend=False
        )
        st.plotly_chart(fig_tor, use_container_width=True)

    with col_m:
        st.markdown("#### Mode Comparison")
        st.caption("Same project, three procurement modes. The DPR's chosen mode may not be the most resilient.")
        with st.spinner("Computing all modes…"):
            all_fi_m = {}
            for m in MODES:
                r_m = _sim(pj, m, min(n_iter,3000))[0]
                all_fi_m[m] = r_m["fi_p"]

        fig_mc = go.Figure(go.Bar(
            x=list(all_fi_m.keys()), y=list(all_fi_m.values()),
            marker_color=[fc(f) for f in all_fi_m.values()],
            text=[f"{f:.0f}%" for f in all_fi_m.values()],
            textposition="outside", opacity=0.87
        ))
        fig_mc.add_hline(y=50, line_dash="dash", line_color="#DC3545", opacity=0.7)
        fig_mc.add_hline(y=25, line_dash="dash", line_color="#856404", opacity=0.7)
        fig_mc.add_hrect(y0=50,y1=105,fillcolor="rgba(220,53,69,0.04)",line_width=0)
        fig_mc.add_hrect(y0=25,y1=50, fillcolor="rgba(255,193,7,0.04)", line_width=0)
        fig_mc.add_hrect(y0=0, y1=25, fillcolor="rgba(25,135,84,0.04)", line_width=0)
        fig_mc.update_layout(
            height=380, plot_bgcolor="#FAFAFA", paper_bgcolor="white",
            yaxis=dict(title="FI (%)",range=[0,110],gridcolor="#EEEEEE"),
            margin=dict(l=40,r=60,t=30,b=40), showlegend=False,
            xaxis=dict(tickfont=dict(size=13,family="IBM Plex Sans",weight=700))
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        best = min(all_fi_m, key=all_fi_m.get)
        worst = max(all_fi_m, key=all_fi_m.get)
        if all_fi_m[worst]-all_fi_m[best] > 15:
            rec = "DPR's chosen mode is optimal." if best==p['dpr_mode'] else \
                  f"Consider switching from {p['dpr_mode']} to **{best}** (Δ={all_fi_m[worst]-all_fi_m[best]:.0f}pp)"
            st.info(rec)


# ─────────────────────────────────────────────────────────────────────
# TAB 3 — SWITCHING VALUES
# ─────────────────────────────────────────────────────────────────────
with tab_sv:
    st.markdown("#### Switching Values — How Robust Are the DPR's Assumptions?")
    st.markdown("""<div class='note'>
    A <b>switching value</b> is the minimum change in one variable (others held at DPR values)
    that makes the project cross the 12% EIRR viability threshold.<br>
    A <b>small</b> switching value = the project is fragile to that variable.
    A <b>large</b> switching value = the project is robust to that variable.<br>
    The red benchmarks show India's actual reference class performance (CAG audit data).
    </div>""", unsafe_allow_html=True)

    # SV metric cards
    sv_c1,sv_c2,sv_c3,sv_c4 = st.columns(4)

    def sv_card(col, label, val, unit, bench_val, bench_label):
        if val is None:
            color, status = "#198754", "✅ No threshold in range (robust)"
            display = "∞"
        else:
            display = f"{val:.1f}{unit}"
            if val < bench_val * 0.5:
                color, status = "#842029", f"⚠️ CRITICAL — CAG avg would trigger failure"
            elif val < bench_val:
                color, status = "#856404", f"⚡ AMBER — Below reference class benchmark"
            else:
                color, status = "#198754", f"✅ Safe — Above reference class"
        col.markdown(f"""<div class='kpi-box' style='border-left-color:{color}; padding:16px;'>
        <div class='kpi-val' style='color:{color}'>{display}</div>
        <div class='kpi-lbl' style='font-size:0.85rem; font-weight:600'>{label}</div>
        <div class='kpi-lbl' style='margin-top:6px'>{status}</div>
        <div class='kpi-lbl' style='margin-top:4px; color:#6c757d'>{bench_label}</div>
        </div>""", unsafe_allow_html=True)

    sv_card(sv_c1, "Cost Overrun SV",    svs.get("cost"),  "%",  71,  "Benchmark: CAG avg +71%")
    sv_card(sv_c2, "Traffic Shortfall SV", svs.get("traf"), "%", 44, "Benchmark: Bain P10 −44%")
    sv_card(sv_c3, "Delay SV",          svs.get("delay"), " mo", 28, "Benchmark: CAG avg 28mo")
    sv_c4.markdown(f"""<div class='kpi-box' style='border-left-color:{"#198754" if svs["base_gap"]>3 else "#856404" if svs["base_gap"]>1 else "#842029"}; padding:16px;'>
    <div class='kpi-val' style='color:{"#198754" if svs["base_gap"]>3 else "#856404" if svs["base_gap"]>1 else "#842029"}'>{svs['base_gap']:+.2f}pp</div>
    <div class='kpi-lbl' style='font-size:0.85rem; font-weight:600'>DPR Headroom</div>
    <div class='kpi-lbl' style='margin-top:6px'>DPR EIRR minus 12% hurdle</div>
    <div class='kpi-lbl' style='margin-top:4px; color:#6c757d'>Larger = more buffer</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # OAT Charts
    st.markdown("#### One-At-A-Time Sensitivity Curves")
    st.caption("Each curve shows EIRR as one variable changes, all others at DPR values. "
               "Where the curve crosses the red 12% line = the switching value.")

    oc1,oc2 = st.columns(2)

    with oc1:
        # Cost OAT
        x_range = np.linspace(-5, min(200, (svs.get("cost") or 100)*2+10), 80)
        y_eirr  = [eirr_iter(p,scn,p["civil_cr"]*(1+x/100),0,p["yr1_aadt"],p["growth"],1.0,1.0)*100
                   for x in x_range]
        fig_oc = go.Figure()
        fig_oc.add_trace(go.Scatter(x=x_range,y=y_eirr,mode="lines",
                                     line=dict(color="#DC3545",width=2.5),name="EIRR"))
        fig_oc.add_hline(y=12,line_dash="dash",line_color="#212529",line_width=2,
                         annotation_text="12% Hurdle",annotation_position="right")
        if svs.get("cost"):
            fig_oc.add_vline(x=svs["cost"],line_dash="dot",line_color="#DC3545",line_width=1.5,
                             annotation_text=f"SV: +{svs['cost']:.1f}%",annotation_position="top left",
                             annotation_font_color="#DC3545")
        fig_oc.add_vline(x=71,line_dash="dot",line_color="#856404",line_width=1.5,
                         annotation_text="CAG avg: +71%",annotation_position="top right",
                         annotation_font_color="#856404")
        fig_oc.update_layout(title="<b>EIRR vs Civil Cost Overrun</b>",height=360,
                              plot_bgcolor="#FAFAFA",paper_bgcolor="white",
                              xaxis=dict(title="Cost Overrun (%)",gridcolor="#EEEEEE"),
                              yaxis=dict(title="EIRR (%)",gridcolor="#EEEEEE"),
                              margin=dict(l=50,r=50,t=50,b=40),showlegend=False)
        st.plotly_chart(fig_oc, use_container_width=True)

    with oc2:
        # Traffic OAT
        x_range2 = np.linspace(-5, min(90, (svs.get("traf") or 50)*2+10), 80)
        y_eirr2  = [eirr_iter(p,scn,p["civil_cr"],0,p["yr1_aadt"]*(1-x/100),p["growth"],1.0,1.0)*100
                    for x in x_range2]
        fig_ot = go.Figure()
        fig_ot.add_trace(go.Scatter(x=x_range2,y=y_eirr2,mode="lines",
                                     line=dict(color="#0D6EFD",width=2.5),name="EIRR"))
        fig_ot.add_hline(y=12,line_dash="dash",line_color="#212529",line_width=2,
                         annotation_text="12% Hurdle",annotation_position="right")
        if svs.get("traf"):
            fig_ot.add_vline(x=svs["traf"],line_dash="dot",line_color="#0D6EFD",line_width=1.5,
                             annotation_text=f"SV: −{svs['traf']:.1f}%",annotation_position="top left",
                             annotation_font_color="#0D6EFD")
        fig_ot.add_vline(x=44,line_dash="dot",line_color="#856404",line_width=1.5,
                         annotation_text="Bain P10: 44%",annotation_position="top right",
                         annotation_font_color="#856404")
        fig_ot.update_layout(title="<b>EIRR vs Traffic Shortfall</b>",height=360,
                              plot_bgcolor="#FAFAFA",paper_bgcolor="white",
                              xaxis=dict(title="Traffic Shortfall (%)",gridcolor="#EEEEEE"),
                              yaxis=dict(title="EIRR (%)",gridcolor="#EEEEEE"),
                              margin=dict(l=50,r=50,t=50,b=40),showlegend=False)
        st.plotly_chart(fig_ot, use_container_width=True)

    oc3,oc4 = st.columns(2)
    with oc3:
        delay_max = min(150, (svs.get("delay") or 72)*1.5+12)
        d_range = np.linspace(0, delay_max, 60)
        y_d = [eirr_iter(p,scn,p["civil_cr"],d,p["yr1_aadt"],p["growth"],1.0,1.0)*100 for d in d_range]
        fig_od = go.Figure()
        fig_od.add_trace(go.Scatter(x=d_range,y=y_d,mode="lines",line=dict(color="#6F42C1",width=2.5)))
        fig_od.add_hline(y=12,line_dash="dash",line_color="#212529",line_width=2)
        if svs.get("delay"):
            fig_od.add_vline(x=svs["delay"],line_dash="dot",line_color="#6F42C1",line_width=1.5,
                             annotation_text=f"SV: {svs['delay']:.0f}mo",annotation_font_color="#6F42C1")
        fig_od.add_vline(x=28,line_dash="dot",line_color="#856404",line_width=1.5,
                         annotation_text="CAG avg: 28mo",annotation_font_color="#856404")
        fig_od.update_layout(title="<b>EIRR vs Construction Delay</b>",height=360,
                              plot_bgcolor="#FAFAFA",paper_bgcolor="white",
                              xaxis=dict(title="Delay (months)",gridcolor="#EEEEEE"),
                              yaxis=dict(title="EIRR (%)",gridcolor="#EEEEEE"),
                              margin=dict(l=50,r=50,t=50,b=40),showlegend=False)
        st.plotly_chart(fig_od, use_container_width=True)

    with oc4:
        g_range = np.linspace(0.01, 0.13, 60)
        y_g = [eirr_iter(p,scn,p["civil_cr"],0,p["yr1_aadt"],g,1.0,1.0)*100 for g in g_range]
        fig_og = go.Figure()
        fig_og.add_trace(go.Scatter(x=g_range*100,y=y_g,mode="lines",line=dict(color="#198754",width=2.5)))
        fig_og.add_hline(y=12,line_dash="dash",line_color="#212529",line_width=2)
        fig_og.add_vline(x=p["growth"]*100,line_dash="dot",line_color="#198754",line_width=1.5,
                         annotation_text=f"DPR: {p['growth']*100:.1f}%",annotation_font_color="#198754")
        fig_og.update_layout(title="<b>EIRR vs Traffic Growth Rate</b>",height=360,
                              plot_bgcolor="#FAFAFA",paper_bgcolor="white",
                              xaxis=dict(title="Growth Rate (% p.a.)",gridcolor="#EEEEEE"),
                              yaxis=dict(title="EIRR (%)",gridcolor="#EEEEEE"),
                              margin=dict(l=50,r=50,t=50,b=40),showlegend=False)
        st.plotly_chart(fig_og, use_container_width=True)

    # LA% sensitivity (dedicated — addresses user's confusion)
    st.divider()
    st.markdown("#### How LA% Completion Affects Fragility")
    st.markdown("""<div class='note'>
    <b>Why EIRR barely changes with LA%:</b> Per IRC SP:30:2019, LA cost is a transfer payment —
    it is excluded from EIRR calculation. LA% affects EIRR <i>only</i> through construction delay
    (higher LA% → lower p_stall → less delay → slightly lower FI_EIRR).<br>
    <b>Why FIRR and Equity IRR change sharply with LA%:</b> FIRR includes LA cost in the investment base.
    Higher LA% → lower LA cost overrun (v06_mult) → FIRR is more achievable → FI_FIRR drops significantly.
    </div>""", unsafe_allow_html=True)

    la_range = list(range(0, 101, 5))
    fi_eirr_la = []; fi_firr_la = []; fi_eq_la = []; fi_p_la = []
    p_la = dict(p)
    for la_v in la_range:
        p_la["la_pct"] = la_v
        scn_la = compute_scn(p_la)
        samp_la = run_mcs(p_la, scn_la, 1000)
        res_la = simulate_mode(p_la, scn_la, samp_la, sim_mode, 1000)
        fi_eirr_la.append(res_la["fi_eirr"])
        fi_firr_la.append(res_la["fi_firr"] if not np.isnan(res_la["fi_firr"]) else None)
        fi_eq_la.append(res_la["fi_eq"] if not np.isnan(res_la["fi_eq"]) else None)
        fi_p_la.append(res_la["fi_p"])

    fig_la = go.Figure()
    fig_la.add_trace(go.Scatter(x=la_range,y=fi_eirr_la,mode="lines+markers",
                                 name="FI EIRR",line=dict(color="#17A589",width=2.5)))
    if any(v is not None for v in fi_firr_la):
        fig_la.add_trace(go.Scatter(x=la_range,y=[v if v else None for v in fi_firr_la],
                                     mode="lines+markers",name="FI FIRR",line=dict(color="#8E44AD",width=2.5)))
    if any(v is not None for v in fi_eq_la):
        fig_la.add_trace(go.Scatter(x=la_range,y=[v if v else None for v in fi_eq_la],
                                     mode="lines+markers",name="FI Equity",line=dict(color="#2471A3",width=2.5,dash="dot")))
    fig_la.add_trace(go.Scatter(x=la_range,y=fi_p_la,mode="lines",
                                 name="FI Primary",line=dict(color="#212529",width=3,dash="dash")))
    fig_la.add_hline(y=50,line_dash="dash",line_color="#DC3545",opacity=0.6,annotation_text="RED 50%")
    fig_la.add_hline(y=25,line_dash="dash",line_color="#856404",opacity=0.6,annotation_text="AMBER 25%")
    fig_la.add_vline(x=p["la_pct"],line_dash="dot",line_color="#212529",line_width=1.5,
                     annotation_text=f"Current: {p['la_pct']}%")
    fig_la.update_layout(
        title="<b>Fragility Index vs LA% Completion</b><br>"
              "<sup>EIRR barely changes (LA excluded per IRC SP:30) | FIRR/Equity drop sharply (LA in investment base)</sup>",
        height=420,plot_bgcolor="#FAFAFA",paper_bgcolor="white",
        xaxis=dict(title="LA% Complete at DPR",gridcolor="#EEEEEE"),
        yaxis=dict(title="FI (%)",range=[0,105],gridcolor="#EEEEEE"),
        legend=dict(orientation="h",y=1.08),
        margin=dict(l=50,r=50,t=80,b=40)
    )
    st.plotly_chart(fig_la, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 4 — ALL PROJECTS
# ─────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("#### All 7 Projects — FI Comparison")
    with st.spinner("Running all 7 projects (each mode)…"):
        batch_rows = []
        for code, proj in PROJECTS.items():
            pj2 = json.dumps(proj, default=str)
            scn2 = compute_scn(proj)
            r2   = _sim(pj2, proj["dpr_mode"], min(n_iter,2000))
            res2 = r2[0]; svs2 = r2[5]
            batch_rows.append({
                "Code":code, "Project":proj["short"], "Mode":proj["dpr_mode"],
                "DPR EIRR":proj["dpr_eirr"],
                "FI Primary (%)":round(res2["fi_p"],1),
                "FI EIRR (%)":round(res2["fi_eirr"],1),
                "FI FIRR (%)":round(res2["fi_firr"],1) if not np.isnan(res2["fi_firr"]) else "N/A",
                "Cost SV":f"+{svs2['cost']:.0f}%" if svs2.get("cost") else "∞",
                "Traffic SV":f"−{svs2['traf']:.0f}%" if svs2.get("traf") else "∞",
                "Verdict":vt(res2["fi_p"]).split("—")[0].strip(),
                "Role":proj["role"],
            })

    df_batch = pd.DataFrame(batch_rows)

    # Batch bar chart
    codes_b = [r["Code"] for r in batch_rows]
    fis_b   = [r["FI Primary (%)"] for r in batch_rows]
    colors_b= [fc(f) for f in fis_b]
    fig_bat = go.Figure(go.Bar(
        x=codes_b, y=fis_b, marker_color=colors_b, opacity=0.87,
        text=[f"{f:.0f}%" for f in fis_b], textposition="outside",
        hovertemplate="%{x}: FI=%{y:.1f}%<extra></extra>"
    ))
    fig_bat.add_hline(y=50,line_dash="dash",line_color="#DC3545",opacity=0.7,annotation_text="RED 50%")
    fig_bat.add_hline(y=25,line_dash="dash",line_color="#856404",opacity=0.7,annotation_text="AMBER 25%")
    fig_bat.add_hrect(y0=50,y1=105,fillcolor="rgba(220,53,69,0.04)",line_width=0)
    fig_bat.add_hrect(y0=25,y1=50, fillcolor="rgba(255,193,7,0.04)", line_width=0)
    fig_bat.add_hrect(y0=0, y1=25, fillcolor="rgba(25,135,84,0.04)", line_width=0)
    for r in batch_rows:
        if r["Role"]=="VALIDATION":
            fig_bat.add_annotation(x=r["Code"],y=102,text="VALIDATION",
                                   showarrow=False,font=dict(size=8,color="#6c757d"),yanchor="bottom")
    fig_bat.update_layout(height=420,plot_bgcolor="#FAFAFA",paper_bgcolor="white",
                          yaxis=dict(title="Fragility Index FI%",range=[0,115],gridcolor="#EEEEEE"),
                          xaxis=dict(tickfont=dict(size=11)),
                          margin=dict(l=50,r=50,t=30,b=40),showlegend=False)
    st.plotly_chart(fig_bat, use_container_width=True)

    st.dataframe(df_batch.drop(columns=["Role"]), use_container_width=True, hide_index=True)

    # Validation callout
    p5r = next(r for r in batch_rows if r["Code"]=="P5")
    p7r = next(r for r in batch_rows if r["Code"]=="P7")
    v_ok = p5r["FI Primary (%)"] >= 50 and p7r["FI Primary (%)"] >= 25
    st.markdown(f"""<div class='note' style='border-left-color:{"#198754" if v_ok else "#856404"}'>
    <b>Validation Check:</b>
    P5 Vadodara FI = <b>{p5r["FI Primary (%)"]:.1f}%</b>
    {"(RED ✓ — concessionaire defaulted in reality)" if p5r["FI Primary (%)"]>=50 else "(⚠️ Expected RED)"} |
    P7 Samruddhi FI = <b>{p7r["FI Primary (%)"]:.1f}%</b>
    {"(AMBER-RED ✓ — fragile at appraisal, succeeded via traffic beat)" if p7r["FI Primary (%)"]>=25 else "(⚠️ Expected AMBER-RED)"}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# TAB 5 — EXPORT
# ─────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown("#### Download Audit Data")

    def build_excel():
        try:
            from openpyxl import Workbook as WB
            from openpyxl.styles import PatternFill as PF, Font as FN, Alignment as AL
            from openpyxl.utils import get_column_letter as gcl
        except ImportError:
            return None
        wb = WB()
        n_ = len(samp["v05"])

        # Sheet 1: Iterations
        ws1 = wb.active; ws1.title = "Iterations"
        hdrs = ["Iter","EIRR_%","FIRR_%","Equity_%","Civil_Cr","LA_Cr","Delay_Mo",
                "Traffic_AADT","Growth_%","VOC","VoT","Stall_Regime"]
        for j,h in enumerate(hdrs,1):
            c=ws1.cell(1,j); c.value=h; c.font=FN(bold=True,color="FFFFFF")
            c.fill=PF("solid",fgColor="1F497D"); c.alignment=AL(horizontal="center")
        for i in range(n_):
            ws1.cell(i+2,1).value=i+1
            ws1.cell(i+2,2).value=round(res["eirr_arr"][i]*100,4)
            ws1.cell(i+2,3).value=round(res["firr_arr"][i]*100,4) if not np.isnan(res["firr_arr"][i]) else "N/A"
            ws1.cell(i+2,4).value=round(res["eq_arr"][i]*100,4) if not np.isnan(res["eq_arr"][i]) else "N/A"
            ws1.cell(i+2,5).value=round(samp["v05"][i],2)
            ws1.cell(i+2,6).value=round(samp["v06"][i],2)
            ws1.cell(i+2,7).value=round(samp["v07"][i],2)
            ws1.cell(i+2,8).value=round(samp["v01"][i],0)
            ws1.cell(i+2,9).value=round(samp["v02"][i]*100,4)
            ws1.cell(i+2,10).value=round(samp["v10"][i],4)
            ws1.cell(i+2,11).value=round(samp["v11"][i],4)
            ws1.cell(i+2,12).value=int(samp["reg"][i])
        for j in range(1,13): ws1.column_dimensions[gcl(j)].width=14

        # Sheet 2: Summary
        ws2 = wb.create_sheet("Summary")
        zs_v = _zs(pj)
        rows = [
            ("Project",p["name"]),("Mode Simulated",sim_mode),("DPR Mode",p["dpr_mode"]),
            ("Iterations",n_),("DPR EIRR (%)",p["dpr_eirr"]),
            ("Zero-Stress EIRR (%)",round(zs_v,4)),
            ("FI Primary (%)",round(res["fi_p"],2)),("FI EIRR (%)",round(res["fi_eirr"],2)),
            ("FI FIRR (%)",round(res["fi_firr"],2) if not np.isnan(res["fi_firr"]) else "N/A"),
            ("FI Equity (%)",round(res["fi_eq"],2) if not np.isnan(res["fi_eq"]) else "N/A"),
            ("Verdict",vt(res["fi_p"])),
            ("P10 EIRR (%)",round(p10,2)),("P20 EIRR (%)",round(p20,2)),
            ("P50 EIRR (%)",round(p50,2)),("P80 EIRR (%)",round(p80,2)),("P90 EIRR (%)",round(p90,2)),
            ("Safety Margin (pp)",round(p50-12,2)),
            ("Cost SV (%)",svs.get("cost") or "∞"),
            ("Traffic SV (%)",svs.get("traf") or "∞"),
            ("Delay SV (mo)",svs.get("delay") or "∞"),
            ("Cost SV vs CAG avg 71%","FRAGILE" if (svs.get("cost") and svs["cost"]<71) else "OK"),
            ("Primary Fragility Driver",tornado[0][0] if tornado else "—"),
            ("SCN cost_scn score",round(scn["cost_scn"],3)),
            ("p_stall",round(scn["v07_ps"],3)),
            ("V05 mean_mult",round(scn["v05_mean_mult"],3)),
            ("V06 mean_mult",round(scn["v06_mean_mult"],3)),
            ("Survey Age (yr)",scn["survey_age"]),
            ("σ multiplier (staleness)",round(scn["traf_sig_mult"],3)),
        ]
        for i,(k,v) in enumerate(rows,1):
            ws2.cell(i,1).value=k; ws2.cell(i,1).font=FN(bold=True)
            ws2.cell(i,2).value=v
        ws2.column_dimensions["A"].width=35; ws2.column_dimensions["B"].width=30

        # Sheet 3: Tornado
        ws3 = wb.create_sheet("Fragility Drivers")
        for j,h in enumerate(["Variable","Spearman ρ","Direction"],1):
            ws3.cell(1,j).value=h; ws3.cell(1,j).font=FN(bold=True)
        for i,(nm,rho) in enumerate(tornado,2):
            ws3.cell(i,1).value=nm; ws3.cell(i,2).value=round(rho,4)
            ws3.cell(i,3).value="Higher → lower EIRR" if rho<0 else "Higher → higher EIRR"
        for c in ["A","B","C"]: ws3.column_dimensions[c].width=28

        return wb

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
            with st.spinner("Building Excel…"):
                wb_out = build_excel()
            if wb_out:
                buf = io.BytesIO(); wb_out.save(buf)
                st.download_button("⬇️ Download Excel (3 sheets)",
                                   data=buf.getvalue(),
                                   file_name=f"PFFF_{p['name'][:25].replace(' ','_')}.xlsx",
                                   mime="application/vnd.ms-excel",
                                   use_container_width=True)
            else:
                st.error("openpyxl not installed. Run: pip install openpyxl")

    with col_e2:
        df_csv = pd.DataFrame({
            "EIRR_%": res["eirr_arr"]*100,
            "FIRR_%": res["firr_arr"]*100,
            "Equity_%": res["eq_arr"]*100,
            "Civil_Cr": samp["v05"],
            "LA_Cr": samp["v06"],
            "Delay_Mo": samp["v07"],
            "AADT": samp["v01"],
        })
        st.download_button("⬇️ Download CSV (iterations)",
                           data=df_csv.to_csv(index=False),
                           file_name=f"PFFF_{p['name'][:20].replace(' ','_')}.csv",
                           mime="text/csv",
                           use_container_width=True)

st.divider()
st.caption("PFFF v12 | M.BEM Thesis 2024 | SPA Delhi | Varshni M S | Supervisor: Mr. Rhijul Sood | "
           "IRC SP:30:2019 · CAG 19/2023 · LARR 2013 · Flyvbjerg 2003 · Bain 2009 · UK Green Book 2022")

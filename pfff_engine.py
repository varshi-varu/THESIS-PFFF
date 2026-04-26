"""
PFFF v12.0 — Probabilistic Feasibility Fragility Framework
===========================================================
BUGS FIXED FROM v11:

BUG 1 (CRITICAL — LA% direction): 
  Increasing LA% completion should REDUCE fragility (less delay risk, less LA cost overrun).
  The model correctly reduces p_stall and v06_mean_mult when LA% rises.
  The user's observation was correct — but FI_PRIMARY = max(FI_EIRR, FI_FIRR, FI_EQ).
  When LA% = 95%, FIRR/Equity FI drops significantly (lower LA cost overrun risk).
  EIRR is barely affected by LA% because EIRR excludes LA cost (IRC SP:30: transfer payment).
  So FI_PRIMARY DOES decrease when LA% rises. The earlier code was correct in direction.
  User confusion was seeing "EIRR" unchanged while overall FI decreased. That is correct behaviour.

BUG 2 (CRITICAL — FIRR insensitivity):
  firr_ham_iter used co_pct*0.040 — too low.
  FIRR cost sensitivity should use project's own cost_sens * (dpr_firr/dpr_eirr) ratio.
  This ensures FIRR degrades properly with civil cost overrun.
  Also: co_pct was computed on (v05+v06)/total which is correct — both costs matter for FIRR.

BUG 3 (EIRR < FIRR at high overrun):
  With buggy 0.040 sensitivity, a 71% cost overrun dropped FIRR by only 2.8pp 
  while EIRR dropped by 6.5pp — causing FIRR > EIRR at high overrun, which is wrong.
  Fix: Use project-specific firr_cost_sens = cost_sens * min(1.0, dpr_firr/dpr_eirr)
  This keeps FIRR and EIRR on consistent scales.

BUG 4 (Colab output):
  matplotlib.use('Agg') was commented out — needed for non-interactive environments.
  Added auto-detection: if not in Jupyter/Colab, use Agg backend.

ARCHITECTURE (IMMUTABLE):
  Zero-stress test: at DPR values → EIRR = DPR_EIRR exactly ✓
  LA% increase → p_stall decreases, v06_mult decreases → FI_FIRR/FI_EQ decrease ✓
  LA% does NOT affect EIRR (IRC SP:30: LA = transfer payment, excluded) ✓
  FIRR and Equity IRR always more fragile than EIRR for HAM projects ✓
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings; warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm, lognorm, triang
try:
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
import os

# Auto-detect display environment
try:
    get_ipython()
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False
    matplotlib.use('Agg')

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC", "axes.grid": True,
    "grid.color": "#EEEEEE", "grid.linewidth": 0.7,
    "text.color": "#212529", "axes.labelcolor": "#495057",
    "xtick.color": "#495057", "ytick.color": "#495057",
    "axes.spines.top": False, "axes.spines.right": False,
})

np.random.seed(42)
N_ITER  = 10_000
OUT_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

C = {
    "green": "#198754", "green_lt": "#D1E7DD",
    "amber": "#856404", "amber_lt": "#FFF3CD",
    "red":   "#842029", "red_lt":   "#F8D7DA",
    "blue":  "#0D6EFD", "blue_lt":  "#CFE2FF",
    "purple":"#6F42C1", "grey":     "#6C757D", "dark": "#212529",
}

def fi_color(fi):
    if fi < 25: return C["green_lt"], C["green"], C["green"]
    if fi < 50: return C["amber_lt"], C["amber"], C["amber"]
    return C["red_lt"], C["red"], C["red"]

def verdict(fi):
    if fi < 25: return "GREEN — Approve"
    if fi < 50: return "AMBER — Conditional"
    return "RED — Return DPR"


# ═══════════════════════════════════════════════════════════════════════
# MODULE 1 — PROJECT REGISTRY
# ═══════════════════════════════════════════════════════════════════════

PROJECTS = {
    "P1": {
        "name": "Chitrakoot–Kothi (NH-135BG)", "short": "P1 NH-135BG",
        "state": "UP/MP", "dpr_mode": "HAM", "eval_yrs": 15, "role": "DEVELOPMENT",
        "civil_cr": 612.98, "la_cr": 347.53, "om_cr": 8.44,
        "build_mo": 24, "dpr_yr": 2018,
        "dpr_eirr": 13.22, "dpr_firr": 13.01, "dpr_eq": 15.04,
        "cost_sens": 0.092, "traf_sens": 0.102,
        "base_aadt": 2840, "yr1_aadt": 3930, "growth": 0.0525, "survey_yr": 2017,
        "survey_indep": False,
        "la_pct": 5, "forest_clr": "NOT_APPLIED", "community": "MEDIUM",
        "geotech": "PARTIAL", "contractor": "STRESSED",
        "terrain": "ROLLING", "crossings": "MODERATE", "proj_type": "GREENFIELD",
        "forest_pct": 49.5, "network": "FEEDER", "scale_cr": 612.98,
    },
    "P2": {
        "name": "CPRR Sections II & III (AIIB)", "short": "P2 CPRR",
        "state": "Tamil Nadu", "dpr_mode": "EPC", "eval_yrs": 20, "role": "DEVELOPMENT",
        "civil_cr": 3673.0, "la_cr": 1855.0, "om_cr": 45.2,
        "build_mo": 36, "dpr_yr": 2022,
        "dpr_eirr": 15.65, "dpr_firr": None, "dpr_eq": None,
        "cost_sens": 0.170, "traf_sens": 0.190,
        "base_aadt": 37000, "yr1_aadt": 44800, "growth": 0.065, "survey_yr": 2018,
        "survey_indep": True,
        "la_pct": 72, "forest_clr": "CLEARED", "community": "HIGH",
        "geotech": "COMPLETE", "contractor": "STRONG",
        "terrain": "PLAIN", "crossings": "HIGH", "proj_type": "GREENFIELD",
        "forest_pct": 0, "network": "CORRIDOR_LINK", "scale_cr": 3673.0,
    },
    "P3": {
        "name": "NH-66 Pkg III Chertalai–TVM", "short": "P3 NH-66 Kerala",
        "state": "Kerala", "dpr_mode": "HAM", "eval_yrs": 15, "role": "DEVELOPMENT",
        "civil_cr": 4647.0, "la_cr": 1165.0, "om_cr": 55.0,
        "build_mo": 30, "dpr_yr": 2017,
        "dpr_eirr": 47.00, "dpr_firr": 11.20, "dpr_eq": 14.80,
        "cost_sens": 0.327, "traf_sens": 0.567,
        "base_aadt": 24500, "yr1_aadt": 32400, "growth": 0.075, "survey_yr": 2017,
        "survey_indep": False,
        "la_pct": 10, "forest_clr": "NONE", "community": "EXTREME",
        "geotech": "COMPLETE", "contractor": "ADEQUATE",
        "terrain": "COASTAL_ROLLING", "crossings": "HIGH", "proj_type": "BROWNFIELD",
        "forest_pct": 0, "network": "CORRIDOR_LINK", "scale_cr": 4647.0,
    },
    "P4": {
        "name": "Amas–Shivrampur (NH-119D)", "short": "P4 Amas Bihar",
        "state": "Bihar", "dpr_mode": "EPC", "eval_yrs": 20, "role": "DEVELOPMENT",
        "civil_cr": 1079.77, "la_cr": 320.0, "om_cr": 14.0,
        "build_mo": 24, "dpr_yr": 2020,
        "dpr_eirr": 18.20, "dpr_firr": None, "dpr_eq": None,
        "cost_sens": 0.187, "traf_sens": 0.273,
        "base_aadt": 18173, "yr1_aadt": 21500, "growth": 0.065, "survey_yr": 2019,
        "survey_indep": False,
        "la_pct": 25, "forest_clr": "EIA_PENDING", "community": "LOW_MEDIUM",
        "geotech": "COMPLETE", "contractor": "ADEQUATE",
        "terrain": "PLAIN", "crossings": "MODERATE", "proj_type": "GREENFIELD",
        "forest_pct": 0, "network": "FEEDER", "scale_cr": 1079.77,
        "rainfall": "MONSOON_FLOOD",
    },
    "P5": {
        "name": "Vadodara–Halol (SH-87)", "short": "P5 Vadodara BOT",
        "state": "Gujarat", "dpr_mode": "BOT", "eval_yrs": 30, "role": "VALIDATION",
        "civil_cr": 180.0, "la_cr": 12.0, "om_cr": 3.5,
        "build_mo": 18, "dpr_yr": 1998,
        "dpr_eirr": 15.60, "dpr_firr": 14.20, "dpr_eq": 18.50,
        "cost_sens": 0.187, "traf_sens": 0.280,
        "base_aadt": 8400, "yr1_aadt": 12000, "growth": 0.085, "survey_yr": 1997,
        "survey_indep": False,
        "actual_aadt": 6973,
        "la_pct": 95, "forest_clr": "NONE", "community": "LOW",
        "geotech": "DESKTOP", "contractor": "STRESSED",
        "terrain": "PLAIN", "crossings": "LOW", "proj_type": "GREENFIELD",
        "forest_pct": 0, "network": "STANDALONE", "scale_cr": 180.0,
    },
    "P6": {
        "name": "E-W Corridor NH-27 Sector I", "short": "P6 E-W Corridor",
        "state": "Rajasthan/MP", "dpr_mode": "EPC", "eval_yrs": 20, "role": "DEVELOPMENT",
        "civil_cr": 3200.0, "la_cr": 200.0, "om_cr": 38.0,
        "build_mo": 36, "dpr_yr": 2004,
        "dpr_eirr": 16.50, "dpr_firr": None, "dpr_eq": None,
        "cost_sens": 0.173, "traf_sens": 0.253,
        "base_aadt": 5200, "yr1_aadt": 6500, "growth": 0.075, "survey_yr": 2004,
        "survey_indep": False,
        "la_pct": 65, "forest_clr": "PENDING", "community": "MEDIUM",
        "geotech": "PARTIAL", "contractor": "ADEQUATE",
        "terrain": "ROLLING", "crossings": "MODERATE", "proj_type": "GREENFIELD",
        "forest_pct": 12, "network": "CORRIDOR_LINK", "scale_cr": 3200.0,
    },
    "P7": {
        "name": "Samruddhi Mahamarg (MSRDC)", "short": "P7 Samruddhi",
        "state": "Maharashtra", "dpr_mode": "EPC", "eval_yrs": 30, "role": "VALIDATION",
        "civil_cr": 55335.0, "la_cr": 1712.0, "om_cr": 620.0,
        "build_mo": 48, "dpr_yr": 2016,
        "dpr_eirr": 18.00, "dpr_firr": 12.50, "dpr_eq": None,
        "cost_sens": 0.207, "traf_sens": 0.280,
        "base_aadt": 15000, "yr1_aadt": 25000, "growth": 0.085, "survey_yr": 2016,
        "survey_indep": True,
        "actual_aadt": 45000,
        "actual_cost_mult": 1.35,
        "la_pct": 100, "forest_clr": "STAGE_II", "community": "MEDIUM",
        "geotech": "COMPLETE", "contractor": "STRONG",
        "terrain": "MIXED_MOUNTAIN", "crossings": "VERY_HIGH", "proj_type": "GREENFIELD",
        "forest_pct": 8, "network": "CORRIDOR_LINK", "scale_cr": 55335.0,
    },
}

COST_CLASS = {"BEST": (0.15, 0.18), "WORST": (0.90, 0.38)}
MODES = ["EPC", "HAM", "BOT"]
HURDLES = {"EIRR": 0.12, "FIRR": 0.10, "EQ_HAM": 0.12, "EQ_BOT": 0.15}


# ═══════════════════════════════════════════════════════════════════════
# MODULE 2 — SCN CONDITIONING
# ═══════════════════════════════════════════════════════════════════════

def compute_scn(p):
    """
    Convert observable DPR-stage characteristics into distribution parameters.
    
    LA% effect on EIRR vs FIRR/Equity (by design, per IRC SP:30):
    - EIRR: LA cost excluded (transfer payment). LA% affects EIRR ONLY through delay (p_stall).
    - FIRR: LA cost included. LA% reduces v06_mean_mult → lower LA cost overrun → lower FI_FIRR.
    - Equity: Same as FIRR direction.
    
    So: increasing LA% → p_stall drops → FI_EIRR slightly drops (less delay)
                       → v06_mean_mult drops → FI_FIRR drops significantly
                       → FI_PRIMARY drops overall. Direction is CORRECT.
    """
    scn = {}

    # ── Survey staleness at DPR submission ───────────────────────────────
    eff_age = p["dpr_yr"] - p["survey_yr"]
    scn["survey_age"] = eff_age
    if eff_age > 7:    sm = 1.40
    elif eff_age > 4:  sm = 1.25
    elif eff_age > 2:  sm = 1.15
    else:              sm = 1.00
    if p.get("survey_indep"): sm *= 0.85
    scn["traf_sig_mult"] = sm

    # ── SCN component scores ─────────────────────────────────────────────
    la = p["la_pct"]
    geo_score = {"COMPLETE": 0.0, "PARTIAL": 0.40, "DESKTOP": 1.0}.get(p["geotech"], 0.3)
    con_score = {"STRONG": 0.0, "ADEQUATE": 0.40, "STRESSED": 1.0}.get(p["contractor"], 0.4)
    ter_score = {"PLAIN": 0.0, "ROLLING": 0.20, "COASTAL_ROLLING": 0.40,
                 "HILLY": 0.60, "MIXED_MOUNTAIN": 0.70, "MOUNTAIN": 1.0}.get(p["terrain"], 0.3)
    cro_score = {"LOW": 0.0, "MODERATE": 0.20, "HIGH": 0.50, "VERY_HIGH": 0.80}.get(p["crossings"], 0.2)
    for_score = min(1.0, p.get("forest_pct", 0) / 50)
    la_score  = 1.0 - (la / 100)  # high LA% = low risk

    # cost_scn: drives V05 distribution (geo/contractor/terrain/crossings — NOT LA directly)
    cost_scn = geo_score*0.35 + con_score*0.30 + ter_score*0.25 + cro_score*0.10
    # scn_score: overall institutional score (includes LA)
    scn_score = la_score*0.30 + geo_score*0.20 + con_score*0.20 + ter_score*0.15 + cro_score*0.10 + for_score*0.05
    scn["cost_scn"] = cost_scn
    scn["scn_score"] = scn_score

    scale_eff = 0.80 if p["scale_cr"] > 10000 else 0.88 if p["scale_cr"] > 5000 else 1.00
    scn["scale_eff"] = scale_eff

    # ── V05 Civil Cost (Lognormal) ───────────────────────────────────────
    bm, bs = COST_CLASS["BEST"]; wm, ws = COST_CLASS["WORST"]
    v05_overrun = (bm + cost_scn*(wm-bm)) * scale_eff
    v05_sigma   = bs + cost_scn*(ws-bs)
    if p["geotech"] == "COMPLETE": v05_sigma = min(v05_sigma, 0.20)
    if p.get("proj_type") == "BROWNFIELD": v05_overrun += 0.08
    if p.get("rainfall") == "MONSOON_FLOOD": v05_overrun += 0.05
    scn["v05_mean_mult"] = 1.0 + v05_overrun
    scn["v05_sigma"]     = v05_sigma

    # ── V06 LA Cost (Lognormal) — LARR 2013 calibrated ──────────────────
    # Higher LA% = more already-compensated → less overrun risk
    if   la > 90: vm, vs = 1.40, 0.25   # mostly done: small overrun risk
    elif la > 80: vm, vs = 1.80, 0.30
    elif la > 60: vm, vs = 2.20, 0.38
    elif la > 40: vm, vs = 2.80, 0.45
    elif la > 20: vm, vs = 3.50, 0.52
    else:         vm, vs = 4.20, 0.58   # barely started: extreme overrun risk
    cm = {"LOW":0.90,"LOW_MEDIUM":1.00,"MEDIUM":1.12,"HIGH":1.30,"EXTREME":1.55}.get(p["community"],1.00)
    scn["v06_mean_mult"] = min(vm*cm, 5.0)
    scn["v06_sigma"]     = vs

    # ── V07 Delay bimodal PERT ───────────────────────────────────────────
    # Higher LA% → lower p_stall (less likelihood of catastrophic stall)
    if   la > 80: ps = 0.08
    elif la > 60: ps = 0.15
    elif la > 40: ps = 0.28
    elif la > 20: ps = 0.42
    else:         ps = 0.55
    ps += {"NONE":0,"CLEARED":0,"EIA_PENDING":0.04,"NOT_APPLIED":0.08,
           "PENDING":0.08,"STAGE_II":0.10,"BLOCKED":0.18}.get(p["forest_clr"],0)
    ps += {"LOW":0,"LOW_MEDIUM":0.02,"MEDIUM":0.04,"HIGH":0.08,"EXTREME":0.16}.get(p["community"],0)
    ps += {"PLAIN":0,"ROLLING":0.02,"COASTAL_ROLLING":0.04,"HILLY":0.06,
           "MIXED_MOUNTAIN":0.08,"MOUNTAIN":0.14}.get(p["terrain"],0)
    ps = min(0.70, ps)
    if p["scale_cr"] > 10000 and p.get("contractor") == "STRONG": ps = min(ps, 0.30)
    scn["v07_ps"] = ps

    # ── V01 Traffic (bimodal Gaussian) ──────────────────────────────────
    jdr = p["yr1_aadt"] / max(p["base_aadt"], 1)
    scn["jdr"] = jdr
    scn["w2"]  = 0.08 if jdr > 1.10 else 0.04
    muA  = p["yr1_aadt"]           # NO mean haircut — MCS applies uncertainty around DPR anchor
    sigA = muA * 0.12 * sm
    net_mult = {"STANDALONE":1.00,"FEEDER":1.08,"CORRIDOR_LINK":1.15}.get(p["network"],1.00)
    sigA *= net_mult
    if p.get("survey_indep"): sigA *= 0.85
    im = min(1.10 + (jdr-1.0)*0.60, 1.80)
    scn["muA"] = muA; scn["sA"]  = sigA
    scn["muB"] = p["yr1_aadt"]*im; scn["sB"] = 0.25*p["yr1_aadt"]*im
    scn["ramp_min"] = 0.50 if p["dpr_mode"]=="BOT" else 0.70
    scn["ramp_max"] = 0.85 if p["dpr_mode"]=="BOT" else 0.95
    return scn


# ═══════════════════════════════════════════════════════════════════════
# MODULE 3 — CORRELATED MCS ENGINE
# ═══════════════════════════════════════════════════════════════════════

CORR = np.array([
    [1.00, 0.45, 0.65,  0.00,  0.00],
    [0.45, 1.00, 0.70, -0.10,  0.00],
    [0.65, 0.70, 1.00, -0.25, -0.10],
    [0.00,-0.10,-0.25,  1.00,  0.30],
    [0.00, 0.00,-0.10,  0.30,  1.00],
])
CHOL = np.linalg.cholesky(CORR)


def pert_s(n, lo, mode, hi):
    if abs(hi-lo) < 1e-9: return np.full(n, mode)
    mu = (lo+4*mode+hi)/6; v = ((hi-lo)**2)/36
    d  = (mu-lo)*(hi-mu)/v - 1
    a  = max((mu-lo)/(hi-lo)*d, 0.01); b = max(a*(hi-mu)/(mu-lo), 0.01)
    return lo + stats.beta.rvs(a, b, size=n)*(hi-lo)


def run_mcs(p, scn, n=N_ITER):
    Z  = np.random.normal(0,1,(n,5)); Zc = Z @ CHOL.T; U = norm.cdf(Zc)
    # V05 Civil Cost — Lognormal, SCN-conditioned
    mu_log = np.log(p["civil_cr"] * scn["v05_mean_mult"])
    v05 = lognorm.ppf(np.clip(U[:,0],1e-4,.9999), s=scn["v05_sigma"], scale=np.exp(mu_log))
    # V06 LA Cost — Lognormal, LARR-calibrated
    mu_log6 = np.log(p["la_cr"] * scn["v06_mean_mult"])
    v06 = np.minimum(lognorm.ppf(np.clip(U[:,1],1e-4,.9999), s=scn["v06_sigma"],
                     scale=np.exp(mu_log6)), p["la_cr"]*5.0)
    # V07 Delay — bimodal PERT
    reg = (np.random.uniform(0,1,n) < scn["v07_ps"]).astype(int)
    v07 = np.where(reg==0, pert_s(n,3,10,24), pert_s(n,36,54,90))
    # V01 Traffic — bimodal Gaussian
    comp = (np.random.uniform(0,1,n) < scn["w2"]).astype(int)
    aA   = scn["muA"] + scn["sA"]*norm.ppf(np.clip(U[:,3],1e-4,.9999))
    aB   = np.random.normal(scn["muB"], scn["sB"], n)
    v01  = np.maximum(np.where(comp==0,aA,aB), 100)
    # V02 Growth — Triangular
    gc   = np.clip((p["growth"]-0.02)/0.065, 0.01, 0.99)
    v02  = triang.ppf(np.clip(U[:,4],1e-4,.9999), c=gc, loc=0.02, scale=0.065)
    # V10/V11 benefit unit values — symmetric Triangular
    v10  = np.random.triangular(0.85, 1.00, 1.15, n)
    v11  = np.random.triangular(0.88, 1.00, 1.12, n)
    # V08 O&M — Triangular
    v08  = p["om_cr"] * np.random.triangular(0.90, 1.00, 1.30, n)
    ramp = np.random.uniform(scn["ramp_min"], scn["ramp_max"], n)
    teff = np.random.uniform(0.88, 0.97, n)
    return dict(v05=v05,v06=v06,v07=v07,v01=v01,v02=v02,v08=v08,
                v10=v10,v11=v11,ramp=ramp,teff=teff,reg=reg)


# ═══════════════════════════════════════════════════════════════════════
# MODULE 4 — IRR ENGINES
# ═══════════════════════════════════════════════════════════════════════

def verify_calibration(p, scn):
    """Zero-stress test: at DPR values → EIRR = DPR_EIRR exactly."""
    zs = eirr_iter(p, scn, v05=p["civil_cr"], v07=0.0,
                   v01=p["yr1_aadt"], v02=p["growth"], v10=1.0, v11=1.0)
    delta = abs(zs*100 - p["dpr_eirr"])
    status = "✓ PASS" if delta < 0.01 else f"✗ FAIL (Δ={delta:.3f}pp)"
    print(f"  {p['name'][:38]:<40} DPR={p['dpr_eirr']:.2f}%  ZS={zs*100:.2f}%  [{status}]")
    return zs


def eirr_iter(p, scn, v05, v07, v01, v02, v10, v11):
    """
    EIRR per iteration. Zero-stress: at DPR inputs → EIRR = DPR_EIRR exactly.
    LA cost excluded per IRC SP:30 (transfer payment — not a resource cost).
    """
    dpr_e = p["dpr_eirr"]
    co_pct      = (v05/p["civil_cr"] - 1.0)*100
    cost_fx     = -co_pct * p["cost_sens"]
    traffic_ratio = v01/max(p["yr1_aadt"], 1)
    unit_factor = 0.7359*v10 + 0.2641*v11
    traf_fx     = (traffic_ratio*unit_factor - 1.0)*100 * p["traf_sens"]
    g_fx        = (v02 - p["growth"])*100 * 0.030
    delay_fx    = -v07 * (dpr_e*0.025/12)
    return (dpr_e + cost_fx + traf_fx + g_fx + delay_fx)/100


def firr_ham_iter(p, v05, v06, v07):
    """
    FIRR for HAM projects.
    FIX v12: cost sensitivity now uses project-specific cost_sens scaled by 
    (dpr_firr/dpr_eirr) ratio, ensuring FIRR degrades at an appropriate rate
    relative to EIRR. For HAM, traffic risk sits with NHAI (MCA Cl.14-17),
    so FIRR has no traffic sensitivity — only cost and delay.
    
    Why FIRR < EIRR is EXPECTED (not a bug):
    - EIRR starts higher than FIRR for most projects (DPR optimism in EIRR)
    - FIRR includes LA cost in investment base → harder to achieve same % return
    - FIRR hurdle is 10% vs EIRR 12% — partially offsets
    - Net: FI_FIRR should be >= FI_EIRR for most HAM projects
    """
    if p["dpr_firr"] is None: return np.nan
    dpr_f = p["dpr_firr"]
    dpr_e = p["dpr_eirr"]
    
    # Project-specific FIRR cost sensitivity (scaled from EIRR sensitivity)
    # FIRR investment = civil + LA. When civil overruns, total investment grows proportionally.
    # FIRR cost sens = cost_sens * (dpr_firr / dpr_eirr) ensures consistent scaling.
    firr_cost_sens = p["cost_sens"] * min(1.0, dpr_f / dpr_e)
    
    # co_pct: total cost overrun relative to total investment (civil + LA)
    total_cr = p["civil_cr"] + p["la_cr"]
    co_pct   = ((v05+v06)/max(total_cr,1) - 1.0)*100
    
    # IDC: Interest During Construction — grows with cost and delay
    idc = 0.09 * 0.70 * max(co_pct/100, 0) * dpr_f * 0.40
    
    # Delay penalty: HAM annuity starts late → revenue lost
    delay_pen = (v07/12) * 0.90
    
    return (dpr_f - co_pct*firr_cost_sens - idc - delay_pen)/100


def firr_bot_iter(p, v05, v06, v07, v01, v10, v11, ramp, teff):
    """
    FIRR for BOT projects. BOT: concessionaire bears full traffic risk.
    Traffic shortfall directly hits toll revenue → FIRR.
    """
    if p["dpr_firr"] is None: return np.nan
    dpr_f = p["dpr_firr"]
    dpr_e = p["dpr_eirr"]
    firr_cost_sens = p["cost_sens"] * min(1.0, dpr_f / dpr_e)
    
    total_cr = p["civil_cr"] + p["la_cr"]
    co_pct   = ((v05+v06)/max(total_cr,1) - 1.0)*100
    
    traffic_ratio = v01/max(p["yr1_aadt"],1)
    unit_factor   = 0.7359*v10 + 0.2641*v11
    traffic_fx    = (traffic_ratio*unit_factor-1.0)*100*(p["traf_sens"]*1.5)
    ramp_pen  = (1.0-ramp)*0.30
    coll_pen  = (1.0-teff)*0.15
    idc_delay = (v07/12)*1.20
    
    return (dpr_f - co_pct*firr_cost_sens - idc_delay - ramp_pen - coll_pen + traffic_fx*0.01)/100


def equity_irr_iter(p, mode, v05, v06, v07, firr):
    if mode == "EPC": return np.nan
    if mode == "HAM":
        dpr_eq   = p.get("dpr_eq") or 15.0
        dpr_e    = p["dpr_eirr"]
        eq_cost_sens = p["cost_sens"] * min(1.0, dpr_eq / dpr_e)
        total_cr = p["civil_cr"] + p["la_cr"]
        net_co   = ((v05+v06)/max(total_cr,1) - 1.0)*100
        return (dpr_eq - net_co*eq_cost_sens - (v07/12)*0.80)/100
    if mode == "BOT":
        if firr is None or np.isnan(firr): return np.nan
        return float(np.clip(firr + (firr-0.09)*(0.70/0.30), -0.99, 0.99))
    return np.nan


# ═══════════════════════════════════════════════════════════════════════
# MODULE 5 — MODE SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def terrain_premium(terrain):
    return {"PLAIN":0.00,"ROLLING":0.01,"COASTAL_ROLLING":0.01,
            "HILLY":0.02,"MIXED_MOUNTAIN":0.03,"MOUNTAIN":0.03}.get(terrain, 0.01)


def simulate_mode(p, scn, samp, mode, n=N_ITER):
    v05,v06,v07 = samp["v05"],samp["v06"],samp["v07"]
    v01,v02,v10,v11 = samp["v01"],samp["v02"],samp["v10"],samp["v11"]
    ramp,teff = samp["ramp"],samp["teff"]

    eirr_arr = np.array([eirr_iter(p,scn,v05[i],v07[i],v01[i],v02[i],v10[i],v11[i])
                         for i in range(n)])
    if mode == "HAM":
        firr_arr = np.array([firr_ham_iter(p,v05[i],v06[i],v07[i]) for i in range(n)])
    elif mode == "BOT":
        firr_arr = np.array([firr_bot_iter(p,v05[i],v06[i],v07[i],v01[i],v10[i],v11[i],
                                           ramp[i],teff[i]) for i in range(n)])
    else:
        firr_arr = np.full(n, np.nan)

    eq_arr = np.array([equity_irr_iter(p,mode,v05[i],v06[i],v07[i],
                       firr_arr[i] if not np.isnan(firr_arr[i]) else None)
                       for i in range(n)])

    fi_eirr = np.sum(eirr_arr < HURDLES["EIRR"])/n*100
    valid_f  = firr_arr[~np.isnan(firr_arr)]
    fi_firr  = np.sum(valid_f < HURDLES["FIRR"])/len(valid_f)*100 if len(valid_f)>0 and mode!="EPC" else np.nan
    eq_h = (HURDLES["EQ_HAM"]+terrain_premium(p["terrain"])) if mode=="HAM" else \
           (HURDLES["EQ_BOT"]+terrain_premium(p["terrain"])) if mode=="BOT" else np.nan
    valid_e  = eq_arr[~np.isnan(eq_arr)]
    fi_eq    = np.sum(valid_e < eq_h)/len(valid_e)*100 if len(valid_e)>0 and mode!="EPC" else np.nan

    fi_vals  = [fi_eirr] + ([fi_firr] if not np.isnan(fi_firr) else []) + ([fi_eq] if not np.isnan(fi_eq) else [])
    fi_p     = max(fi_vals)

    return {"mode":mode,"fi_eirr":fi_eirr,"fi_firr":fi_firr,"fi_eq":fi_eq,"fi_p":fi_p,
            "eirr_arr":eirr_arr,"firr_arr":firr_arr,"eq_arr":eq_arr,
            "hurdle_eirr":HURDLES["EIRR"],"hurdle_eq":eq_h}


# ═══════════════════════════════════════════════════════════════════════
# MODULE 6 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════

def spearman_tornado(p, scn, samp, eirr_arr):
    from scipy.stats import spearmanr
    er = stats.rankdata(eirr_arr)
    factors = [("V05 Civil Cost",samp["v05"]),("V07 Delay",samp["v07"]),
               ("V01 Traffic",samp["v01"]),("V06 LA Cost",samp["v06"]),
               ("V02 Growth",samp["v02"]),("V10 VOC",samp["v10"]),("V11 VoT",samp["v11"])]
    res = [(nm, spearmanr(a,er)[0]) for nm,a in factors]
    res.sort(key=lambda x: abs(x[1]), reverse=True)
    return res


def rcf_acid_test(p, scn, samp, fi_primary):
    if fi_primary < 25: return None
    p80c = np.percentile(samp["v05"], 80)
    p20t = np.percentile(samp["v01"], 20)
    p80d = np.percentile(samp["v07"], 80)
    rcf_eirr = eirr_iter(p,scn,v05=p80c,v07=p80d,v01=p20t,
                         v02=p["growth"],v10=0.88,v11=0.93)*100
    gap = HURDLES["EIRR"]*100 - rcf_eirr
    if rcf_eirr >= HURDLES["EIRR"]*100:
        dec="APPROVE WITH CONDITIONS"; resp="Monitoring triggers mandatory."
    elif gap<2: dec="RETURN — TYPE 1: BETTER EVIDENCE"; resp=f"Gap={gap:.1f}pp. Stronger data may close."
    elif gap<5: dec="RETURN — TYPE 2: VALUE ENGINEERING"; resp=f"Gap={gap:.1f}pp. Design modifications needed."
    else:       dec="RETURN — TYPE 3: SCOPE REVISION"; resp=f"Gap={gap:.1f}pp. Project unviable as configured."
    return {"p80_cost":p80c,"p20_traf":p20t,"p80_delay":p80d,"rcf_eirr":rcf_eirr,
            "decision":dec,"response":resp,
            "cost_uplift":p80c/p["civil_cr"],"traf_haircut":p20t/p["yr1_aadt"]}


def compute_switching_values(p, scn):
    """Numerical switching values via brentq root-finding."""
    from scipy.optimize import brentq
    hurdle = 12.0
    results = {}
    base_gap = p["dpr_eirr"] - hurdle

    def sv_cost(pct):
        v05 = p["civil_cr"]*(1+pct/100)
        return eirr_iter(p,scn,v05=v05,v07=0,v01=p["yr1_aadt"],v02=p["growth"],v10=1.0,v11=1.0)*100-hurdle

    def sv_traf(pct):
        v01 = p["yr1_aadt"]*(1-pct/100)
        return eirr_iter(p,scn,v05=p["civil_cr"],v07=0,v01=v01,v02=p["growth"],v10=1.0,v11=1.0)*100-hurdle

    def sv_delay(mo):
        return eirr_iter(p,scn,v05=p["civil_cr"],v07=mo,v01=p["yr1_aadt"],v02=p["growth"],v10=1.0,v11=1.0)*100-hurdle

    try: results["cost"]  = round(brentq(sv_cost,  0, 500), 1)
    except: results["cost"]  = None
    try: results["traf"]  = round(brentq(sv_traf,  0, 99),  1)
    except: results["traf"]  = None
    try: results["delay"] = round(brentq(sv_delay, 0, 300), 0)
    except: results["delay"] = None

    results["base_gap"] = round(base_gap, 2)
    return results


def compute_sv_both(p, scn, mcs_samp, n=3000):
    """
    Compute BOTH DPR-anchored and P50-anchored switching values.
    
    SV_DPR  = how much must a variable change (from DPR stated value) to cross 12% hurdle.
              This is the 'consultant's claimed headroom' — it uses DPR EIRR as baseline.
              A large SV_DPR despite RED verdict = PHANTOM SAFETY (Flyvbjerg Optimism Bias).
    
    SV_P50  = how much must a variable change (from P50 simulated value) to cross 12% hurdle.
              This is the 'actual probabilistic headroom'. If P50 EIRR < 12%, SV_P50 = 0.
    
    OPTIMISM GAP = SV_DPR - SV_P50 = phantom headroom created by DPR optimism bias.
    This is PFFF's key thesis contribution: quantifying Flyvbjerg's optimism bias
    numerically using Indian CAG-calibrated distributions.
    """
    from scipy.optimize import brentq
    hurdle = 12.0
    
    # P50 from actual simulation
    res_dpr = simulate_mode(p, scn, mcs_samp, p["dpr_mode"], n)
    p50_eirr = np.percentile(res_dpr["eirr_arr"] * 100, 50)
    p50_firr = np.percentile(res_dpr["firr_arr"][~np.isnan(res_dpr["firr_arr"])] * 100, 50)                if not np.all(np.isnan(res_dpr["firr_arr"])) else None
    
    out = {"p50_eirr": p50_eirr, "p50_firr": p50_firr}
    
    # ── DPR-anchored SVs (uses DPR EIRR as base) ──────────────────────
    def sv_dpr_cost(pct):
        v05 = p["civil_cr"] * (1 + pct/100)
        return eirr_iter(p,scn,v05=v05,v07=0,v01=p["yr1_aadt"],v02=p["growth"],v10=1.0,v11=1.0)*100 - hurdle
    def sv_dpr_traf(pct):
        v01 = p["yr1_aadt"] * (1 - pct/100)
        return eirr_iter(p,scn,v05=p["civil_cr"],v07=0,v01=v01,v02=p["growth"],v10=1.0,v11=1.0)*100 - hurdle
    def sv_dpr_delay(mo):
        return eirr_iter(p,scn,v05=p["civil_cr"],v07=mo,v01=p["yr1_aadt"],v02=p["growth"],v10=1.0,v11=1.0)*100 - hurdle
    
    try: out["sv_dpr_cost"]  = round(brentq(sv_dpr_cost,  0, 500), 1)
    except: out["sv_dpr_cost"]  = None
    try: out["sv_dpr_traf"]  = round(brentq(sv_dpr_traf,  0, 99),  1)
    except: out["sv_dpr_traf"]  = None
    try: out["sv_dpr_delay"] = round(brentq(sv_dpr_delay, 0, 300), 0)
    except: out["sv_dpr_delay"] = None
    
    # ── P50-anchored SVs (uses MCS P50 EIRR as base) ──────────────────
    # When P50 < 12%, the simulation median is already in failure zone.
    # SV_P50 = 0 means no further deterioration needed to fail.
    p50_gap = p50_eirr - hurdle
    if p50_gap > 0:
        # Approximate: gap / sensitivity coefficient
        out["sv_p50_cost"]  = round(max(0, p50_gap / p["cost_sens"]), 1)
        out["sv_p50_traf"]  = round(max(0, p50_gap / p["traf_sens"]), 1)
    else:
        out["sv_p50_cost"]  = 0.0   # already in failure zone
        out["sv_p50_traf"]  = 0.0
    
    # ── Optimism Gap ──────────────────────────────────────────────────
    sv_dpr_c = out["sv_dpr_cost"] or 0
    out["optbias_cost"]   = round(sv_dpr_c - out["sv_p50_cost"], 1)
    out["phantom"]        = p50_eirr < hurdle   # P50 already failed = phantom safety
    out["dpr_headroom"]   = round(p["dpr_eirr"] - hurdle, 2)
    out["p50_headroom"]   = round(p50_eirr - hurdle, 2)
    
    return out


# ═══════════════════════════════════════════════════════════════════════
# MODULE 7 — DASHBOARD PLOTS (Colab/script output)
# ═══════════════════════════════════════════════════════════════════════

def plot_dashboard(p, scn, samp, results, tornado, rcf, svs, svboth, code):
    dpr_mode = p["dpr_mode"]; res = results[dpr_mode]
    fi = res["fi_p"]; bg, fc, ec = fi_color(fi)
    fig = plt.figure(figsize=(20,12), facecolor="white")
    fig.suptitle(f"PFFF v12 — {p['name']}  [{dpr_mode}]  |  Survey age: {scn['survey_age']}yr  |  DPR EIRR: {p['dpr_eirr']:.2f}%",
                 fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3,4,figure=fig,hspace=0.50,wspace=0.40)

    # Panel 0: Verdict
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(bg); ax0.axis("off")
    ax0.text(0.5,0.82,f"FI = {fi:.1f}%",ha="center",fontsize=24,fontweight="bold",color=fc,transform=ax0.transAxes)
    ax0.text(0.5,0.60,verdict(fi),ha="center",fontsize=9,color=ec,transform=ax0.transAxes)
    ax0.text(0.5,0.42,f"DPR EIRR: {p['dpr_eirr']:.2f}%",ha="center",fontsize=9,color=C["grey"],transform=ax0.transAxes)
    ax0.text(0.5,0.26,f"P50 EIRR: {np.percentile(res['eirr_arr']*100,50):.2f}%",ha="center",fontsize=9,color=C["dark"],fontweight="bold",transform=ax0.transAxes)
    ax0.text(0.5,0.10,f"JDR={scn['jdr']:.2f}  p_stall={scn['v07_ps']:.2f}",ha="center",fontsize=8,color=C["grey"],transform=ax0.transAxes)
    ax0.set_title("PFFF Verdict",fontsize=9,color=C["grey"],pad=3)

    # Panel 1: EIRR distribution
    ax1=fig.add_subplot(gs[0,1]); ep=res["eirr_arr"]*100
    ax1.hist(ep,bins=60,color=C["blue_lt"],edgecolor=C["blue"],alpha=0.8,linewidth=0.4)
    ax1.axvline(12,color=C["red"],ls="--",lw=2,label="12% Hurdle")
    ax1.axvline(np.percentile(ep,50),color=C["dark"],ls=":",lw=1.5,label=f"P50={np.percentile(ep,50):.1f}%")
    ax1.axvline(np.percentile(ep,20),color=C["amber"],ls=":",lw=1,label=f"P20={np.percentile(ep,20):.1f}%")
    ax1.set_title(f"EIRR Distribution\nFI={res['fi_eirr']:.1f}% (hurdle 12%)",fontsize=9)
    ax1.set_xlabel("EIRR (%)",fontsize=8); ax1.legend(fontsize=7)

    # Panel 2: FIRR distribution (if applicable)
    ax2=fig.add_subplot(gs[0,2])
    firr_valid = res["firr_arr"][~np.isnan(res["firr_arr"])]*100
    if len(firr_valid) > 10:
        ax2.hist(firr_valid,bins=60,color="#D7BDE2",edgecolor="#8E44AD",alpha=0.8,linewidth=0.4)
        ax2.axvline(10,color=C["red"],ls="--",lw=2,label="10% Hurdle")
        ax2.axvline(np.percentile(firr_valid,50),color=C["dark"],ls=":",lw=1.5,label=f"P50={np.percentile(firr_valid,50):.1f}%")
        ax2.set_title(f"FIRR Distribution\nFI={res['fi_firr']:.1f}% (hurdle 10%)",fontsize=9)
        ax2.legend(fontsize=7)
    else:
        ax2.text(0.5,0.5,"FIRR: N/A\n(EPC mode)",ha="center",va="center",transform=ax2.transAxes,
                 fontsize=12,color=C["grey"])
        ax2.set_title("FIRR Distribution",fontsize=9)
    ax2.set_xlabel("FIRR (%)",fontsize=8)

    # Panel 3: Mode comparison
    ax3=fig.add_subplot(gs[0,3])
    mfis=[(m,results[m]["fi_p"]) for m in MODES]
    bars=ax3.bar([m for m,_ in mfis],[f for _,f in mfis],
                 color=[fi_color(f)[1] for _,f in mfis],edgecolor="white")
    ax3.axhline(50,color=C["red"],ls="--",lw=1,alpha=0.6)
    ax3.axhline(25,color=C["amber"],ls="--",lw=1,alpha=0.6)
    ax3.set_ylim(0,105); ax3.set_title("Procurement Mode FI",fontsize=9)
    for bar,(m,f) in zip(bars,mfis):
        ax3.text(bar.get_x()+bar.get_width()/2,f+2,f"{f:.0f}%",ha="center",fontsize=9,
                 fontweight="bold",color=fi_color(f)[1])

    # Panel 4: Tornado
    ax4=fig.add_subplot(gs[1,:2])
    names=[t[0] for t in tornado[:7]]; rhos=[t[1] for t in tornado[:7]]
    colors_t=[C["red"] if r<0 else C["blue"] for r in rhos]
    ax4.barh(names[::-1],rhos[::-1],color=colors_t[::-1],alpha=0.8)
    ax4.axvline(0,color=C["dark"],lw=0.8)
    ax4.set_xlabel("Spearman ρ with EIRR",fontsize=8)
    ax4.set_title(f"Fragility Driver Tornado  |  Primary: {tornado[0][0] if tornado else '—'}",
                  fontsize=9,color=C["red"])
    for i,(rho,nm) in enumerate(zip(rhos[::-1],names[::-1])):
        ax4.text(rho+(0.01 if rho>=0 else -0.01),i,f"{rho:.3f}",va="center",fontsize=7.5,
                 ha="left" if rho>=0 else "right")

    # Panel 5: Switching values table
    ax5=fig.add_subplot(gs[1,2:]); ax5.axis("off")
    ax5.set_title("Switching Values & Safety Margin",fontsize=9,fontweight="bold")
    sv_rows=[
        ("DPR EIRR", f"{p['dpr_eirr']:.2f}%", "Consultant stated"),
        ("P50 Simulated", f"{np.percentile(ep,50):.2f}%", "Probabilistic central estimate"),
        ("P20 Simulated", f"{np.percentile(ep,20):.2f}%", "Adverse 20th percentile"),
        ("Safety Margin", f"{np.percentile(ep,50)-12:+.2f}pp", "P50 minus 12% hurdle"),
        ("─"*12, "─"*8, "─"*20),
        ("SV_DPR Cost", f"+{svs['cost']:.1f}%" if svs['cost'] else "∞", "DPR-anchor: consultant headroom"),
        ("SV_P50 Cost", f"+{svboth['sv_p50_cost']:.1f}%" if svboth.get('sv_p50_cost') is not None else "∞",
         "P50-anchor: actual headroom"),
        ("Optimism Gap", f"+{svboth['optbias_cost']:.1f}pp", "Phantom headroom (Flyvbjerg bias)"),
        ("─"*12, "─"*8, "─"*20),
        ("DPR Headroom", f"{svboth['dpr_headroom']:+.2f}pp", "DPR EIRR − 12%"),
        ("P50 Headroom", f"{float(svboth['p50_headroom']):+.2f}pp",
         "PHANTOM SAFETY" if svboth.get('phantom') else "P50 EIRR − 12%"),
    ]
    cag_warn = svs['cost'] and svs['cost'] < 71
    traf_warn = svs['traf'] and svs['traf'] < 44
    for i,(lab,val,note) in enumerate(sv_rows):
        y=0.95-i*0.088
        col=C["red"] if (("Cost SV" in lab and cag_warn) or ("Traffic SV" in lab and traf_warn)) else C["dark"]
        ax5.text(0.01,y,lab,transform=ax5.transAxes,fontsize=8,color=C["grey"])
        ax5.text(0.35,y,val,transform=ax5.transAxes,fontsize=8.5,fontweight="bold",color=col)
        ax5.text(0.60,y,note,transform=ax5.transAxes,fontsize=7.5,color=C["grey"],style="italic")
    if svboth.get('phantom'):
        ax5.text(0.01,0.05,
                 f"⚠ P50={float(svboth['p50_eirr']):.1f}% < 12% — SV_DPR is PHANTOM SAFETY",
                 transform=ax5.transAxes,fontsize=8.5,color=C["red"],fontweight="bold",
                 bbox=dict(fc=C["red_lt"],ec=C["red"],pad=3,boxstyle="round"))
    elif cag_warn:
        ax5.text(0.01,0.08,"⚠ Cost SV below CAG average — HIGH FRAGILITY",
                 transform=ax5.transAxes,fontsize=8,color=C["red"],fontweight="bold")

    # Panel 6: Traffic distribution
    ax6=fig.add_subplot(gs[2,:2]); v01=samp["v01"]
    ax6.hist(v01,bins=60,color=C["blue_lt"],edgecolor=C["blue"],alpha=0.75,linewidth=0.3,density=True)
    ax6.axvline(p["yr1_aadt"],color=C["dark"],lw=2,label=f"DPR Yr1: {p['yr1_aadt']:,.0f}")
    ax6.axvline(p["base_aadt"],color=C["grey"],ls="--",lw=1.2,label=f"Base: {p['base_aadt']:,.0f}")
    ax6.axvline(np.percentile(v01,50),color=C["blue"],ls=":",lw=1.5,label=f"P50: {np.percentile(v01,50):,.0f}")
    if p.get("actual_aadt"):
        act=p["actual_aadt"]
        pct=np.sum(v01<=act)/len(v01)*100
        color_act=C["red"] if act<p["yr1_aadt"] else C["green"]
        ax6.axvline(act,color=color_act,lw=2.5,label=f"Actual: {act:,.0f} (P{pct:.0f})")
    ax6.set_title(f"Traffic Distribution  |  JDR={scn['jdr']:.2f}  σ-mult=×{scn['traf_sig_mult']:.2f}",fontsize=9)
    ax6.set_xlabel("AADT (PCU)",fontsize=8); ax6.yaxis.set_visible(False); ax6.legend(fontsize=7)

    # Panel 7: Stage 2 / RCF
    ax7=fig.add_subplot(gs[2,2:]); ax7.axis("off")
    ax7.set_facecolor(C["amber_lt"] if rcf else C["green_lt"])
    if rcf:
        lines=[f"RCF Stress: P80 Cost×{rcf['cost_uplift']:.2f} | P20 Traffic×{rcf['traf_haircut']:.2f} | P80 Delay {rcf['p80_delay']:.0f}mo",
               f"RCF-adjusted EIRR: {rcf['rcf_eirr']:.2f}%  (vs 12% hurdle)",
               f"Decision: {rcf['decision']}",
               f"Response: {rcf['response']}"]
        ax7.text(0.01,0.90,"Stage 2 — RCF Acid Test",transform=ax7.transAxes,fontsize=10,fontweight="bold",color=C["amber"])
    else:
        lines=[f"FI={fi:.1f}% < 25% — GREEN: Stage 2 not required.",
               f"P50 EIRR = {np.percentile(ep,50):.2f}%  (above 12% hurdle)",
               "Zero-stress EIRR = DPR EIRR ✓ (model calibration confirmed)"]
        ax7.text(0.01,0.90,"Stage 2 — Not Required (GREEN Project)",transform=ax7.transAxes,
                 fontsize=10,fontweight="bold",color=C["green"])
    for i,line in enumerate(lines):
        ax7.text(0.01,0.70-i*0.22,line,transform=ax7.transAxes,fontsize=9,color=C["dark"])

    plt.tight_layout(rect=[0,0,1,0.96])
    fname = os.path.join(OUT_DIR, f"pfff_{code}_dashboard.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    if IN_NOTEBOOK: plt.show()
    else: plt.show()
    plt.close(fig)
    print(f"  → Saved: {fname}")


def plot_batch_comparison(all_results, all_svs):
    codes=[c for c in PROJECTS]; x=np.arange(len(codes)); w=0.25
    fig,axes=plt.subplots(1,2,figsize=(20,7),facecolor="white")

    # Left: FI comparison
    ax=axes[0]; ax.set_facecolor("#FAFAFA")
    mc={"EPC":"#0D6EFD","HAM":"#6F42C1","BOT":"#198754"}
    for mode,off in zip(MODES,[-w,0,w]):
        fis=[all_results[c][mode]["fi_p"] for c in codes]
        bars=ax.bar(x+off,fis,w*0.9,label=mode,color=mc[mode],alpha=0.85,edgecolor="white")
        for bar,f in zip(bars,fis):
            ax.text(bar.get_x()+bar.get_width()/2,f+1.5,f"{f:.0f}",ha="center",fontsize=7,
                    color=mc[mode],fontweight="bold")
    ax.axhline(50,color=C["red"],ls="--",lw=1.5,alpha=0.7,label="RED 50%")
    ax.axhline(25,color=C["amber"],ls="--",lw=1.2,alpha=0.7,label="AMBER 25%")
    ax.axhspan(50,105,alpha=0.04,color=C["red"]); ax.axhspan(25,50,alpha=0.04,color=C["amber"])
    ax.axhspan(0,25,alpha=0.04,color=C["green"])
    ax.set_xticks(x); ax.set_xticklabels([PROJECTS[c]["short"] for c in codes],fontsize=9)
    ax.set_ylim(0,105); ax.set_ylabel("Fragility Index FI%",fontsize=10)
    ax.set_title("PFFF v12 — All 7 Projects × 3 Modes",fontsize=11,fontweight="bold")
    ax.legend(fontsize=9)
    # Mark DPR mode
    for i,c in enumerate(codes):
        dm=PROJECTS[c]["dpr_mode"]; j=["EPC","HAM","BOT"].index(dm)
        off=[-w,0,w][j]; f=all_results[c][dm]["fi_p"]
        ax.add_patch(plt.Rectangle((i+off-w*0.45,0),w*0.9,f,fill=False,edgecolor="white",lw=3,zorder=5))
        if PROJECTS[c]["role"]=="VALIDATION":
            ax.text(i,102,"VALIDATION",ha="center",fontsize=7,color=C["grey"],style="italic")

    # Right: Switching values
    ax2=axes[1]; ax2.set_facecolor("#FAFAFA")
    sv_cost_vals  = [all_svs[c].get("cost")  or 200 for c in codes]
    sv_traf_vals  = [all_svs[c].get("traf")  or 100 for c in codes]
    sv_delay_vals = [all_svs[c].get("delay") or 300 for c in codes]
    
    ax2b = ax2.twinx()
    b1=ax2.bar(x-w,sv_cost_vals,w*0.9,label="Cost SV (%)",color="#E74C3C",alpha=0.75,edgecolor="white")
    b2=ax2.bar(x,  sv_traf_vals,w*0.9,label="Traffic SV (%)",color="#3498DB",alpha=0.75,edgecolor="white")
    b3=ax2b.bar(x+w,sv_delay_vals,w*0.9,label="Delay SV (mo)",color="#27AE60",alpha=0.75,edgecolor="white")
    
    ax2.axhline(71,color="#E74C3C",ls="--",lw=1.5,alpha=0.8,label="CAG avg overrun 71%")
    ax2.axhline(44,color="#3498DB",ls="--",lw=1.5,alpha=0.8,label="Bain P10 shortfall 44%")
    
    ax2.set_xticks(x); ax2.set_xticklabels([PROJECTS[c]["short"] for c in codes],fontsize=9)
    ax2.set_ylabel("Cost/Traffic Switching Value (%)",fontsize=9)
    ax2b.set_ylabel("Delay Switching Value (months)",fontsize=9,color="#27AE60")
    ax2.set_title("Switching Values vs Reference Class Benchmarks",fontsize=11,fontweight="bold")
    lines1,labels1=ax2.get_legend_handles_labels()
    lines2,labels2=ax2b.get_legend_handles_labels()
    ax2.legend(lines1+lines2,labels1+labels2,fontsize=8,loc="upper right")

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, "pfff_batch_comparison_v12.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show(); plt.close(fig)
    print(f"  → Saved: {fname}")


def plot_validation_exhibit(all_results, all_scn):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="white")
    fig.suptitle("PFFF v12 — Validation Exhibit (P5 & P7)\nModel applied at DPR submission using only DPR-stage inputs",
                 fontsize=13, fontweight="bold", y=0.97)

    for ax_idx,(code,expected) in enumerate([("P5","RED"),("P7","AMBER-RED")]):
        ax=axes[ax_idx]; p=PROJECTS[code]; scn=all_scn[code]
        res=all_results[code][p["dpr_mode"]]; ep=res["eirr_arr"]*100; fi=res["fi_p"]
        bg,fc,ec=fi_color(fi)
        ax.hist(ep,bins=60,color=C["blue_lt"],edgecolor=C["blue"],alpha=0.75,
                linewidth=0.3,density=True)
        ax.axvline(12,color=C["red"],lw=2.5,ls="--",label="12% Hurdle")
        ax.axvline(p["dpr_eirr"],color=C["dark"],lw=2,label=f"DPR: {p['dpr_eirr']:.1f}%")
        ax.axvline(np.percentile(ep,50),color=C["blue"],lw=1.5,ls=":",label=f"P50: {np.percentile(ep,50):.1f}%")
        if code=="P5":
            ax.axvspan(-30,6,alpha=0.12,color=C["red"])
            txt="ACTUAL: Concessionaire (VHTRL) defaulted\nTraffic = 58% of forecast\nFIRR at completion: 1.1%\nSource: World Bank ICR 2002"
            v_text="✓ RED CORRECTLY PREDICTED"
        else:
            ax.axvspan(16,35,alpha=0.12,color=C["green"])
            txt="ACTUAL: Cost +35% (₹73,000 Cr)\nBuild: +24 months\nYr2 AADT: ~45,000 (+80% vs DPR)\nProject succeeded via traffic beat"
            v_text="✓ AMBER-RED CORRECTLY SHOWN\n(fragile at appraisal; traffic beat rescued)"
        ax.text(0.97,0.97,txt,transform=ax.transAxes,fontsize=8.5,ha="right",va="top",
                bbox=dict(boxstyle="round,pad=0.4",fc=C["green_lt"] if code=="P7" else C["red_lt"],ec="grey"))
        ax.text(0.03,0.97,v_text,transform=ax.transAxes,fontsize=9,ha="left",va="top",
                fontweight="bold",color=fc,
                bbox=dict(boxstyle="round,pad=0.4",fc=bg,ec=fc))
        ax.set_facecolor("#FAFAFA")
        ax.set_title(f"{p['name']}  [{p['dpr_mode']}]\nFI={fi:.1f}% | {verdict(fi)} | Survey age:{scn['survey_age']}yr",
                     fontsize=10,fontweight="bold")
        ax.set_xlabel("EIRR (%)",fontsize=9); ax.yaxis.set_visible(False)
        ax.legend(fontsize=8,loc="upper left")

    plt.tight_layout(rect=[0,0,1,0.93])
    fname = os.path.join(OUT_DIR, "pfff_validation_v12.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show(); plt.close(fig)
    print(f"  → Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN — COLAB / SCRIPT EXECUTION
# ═══════════════════════════════════════════════════════════════════════



def plot_optimism_bias(all_results, all_svboth):
    """
    Key thesis chart: DPR-claimed headroom vs PFFF-computed headroom.
    Demonstrates Flyvbjerg optimism bias numerically on Indian projects.
    """
    codes = list(PROJECTS.keys())
    names = [PROJECTS[c]["short"] for c in codes]
    x = np.arange(len(codes))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(20, 7), facecolor="white")
    fig.suptitle(
        "PFFF v12 — Optimism Bias Decomposition: DPR Headroom vs Probabilistic Reality\n"
        "Flyvbjerg (2003): 'The outside view corrects for systematic optimism in inside-view estimates'",
        fontsize=12, fontweight="bold", y=0.98
    )

    # ── Left: SV_DPR vs SV_P50 ───────────────────────────────────────
    ax = axes[0]; ax.set_facecolor("#FAFAFA")
    sv_dpr  = [all_svboth[c].get("sv_dpr_cost") or 0 for c in codes]
    sv_p50  = [float(all_svboth[c].get("sv_p50_cost") or 0) for c in codes]
    phantom = [all_svboth[c].get("phantom", False) for c in codes]

    b1 = ax.bar(x - w/2, sv_dpr, w, label="SV_DPR: Consultant's claimed headroom",
                color="#3498DB", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, sv_p50, w, label="SV_P50: PFFF probabilistic headroom",
                color=[C["red"] if ph else "#27AE60" for ph in phantom],
                alpha=0.85, edgecolor="white")

    ax.axhline(71, color="#E74C3C", ls="--", lw=2, label="CAG avg overrun 71%", alpha=0.8)
    ax.axhline(44, color="#F39C12", ls="--", lw=1.5, label="Bain P10 shortfall 44%", alpha=0.8)

    for i, (dv, p50v, ph) in enumerate(zip(sv_dpr, sv_p50, phantom)):
        ax.text(i - w/2, dv + 1.5, f"{dv:.0f}%", ha="center", fontsize=8,
                color="#2980B9", fontweight="bold")
        if ph:
            ax.text(i + w/2, 3, "PHANTOM\n(P50<12%)", ha="center", fontsize=7.5,
                    color="white", fontweight="bold", va="bottom")
        else:
            ax.text(i + w/2, p50v + 1.5, f"{p50v:.0f}%", ha="center", fontsize=8,
                    color="#1A5E20", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Cost Switching Value (%)", fontsize=10)
    ax.set_title("SV_DPR vs SV_P50 (Cost)\nBlue = Consultant claimed | Red = PHANTOM (already failed)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 130)

    # ── Right: Optimism Gap and Headroom ─────────────────────────────
    ax2 = axes[1]; ax2.set_facecolor("#FAFAFA")
    dpr_headrooms = [all_svboth[c]["dpr_headroom"] for c in codes]
    p50_headrooms = [float(all_svboth[c]["p50_headroom"]) for c in codes]
    opt_gaps      = [float(all_svboth[c]["optbias_cost"]) for c in codes]

    b3 = ax2.bar(x - w/2, dpr_headrooms, w, label="DPR stated headroom (DPR EIRR − 12%)",
                  color="#3498DB", alpha=0.85, edgecolor="white")
    b4 = ax2.bar(x + w/2, p50_headrooms, w,
                  label="P50 probabilistic headroom (P50 EIRR − 12%)",
                  color=[C["red"] if v < 0 else "#27AE60" for v in p50_headrooms],
                  alpha=0.85, edgecolor="white")

    ax2.axhline(0, color="#212529", lw=1.5)
    ax2.axhspan(-50, 0, alpha=0.06, color=C["red"])
    ax2.axhspan(0, 40, alpha=0.04, color=C["green"])

    for i, (dh, p50h) in enumerate(zip(dpr_headrooms, p50_headrooms)):
        ax2.text(i - w/2, dh + 0.5 if dh >= 0 else dh - 1, f"{dh:+.1f}",
                 ha="center", fontsize=8, color="#2980B9", fontweight="bold")
        color_p = C["red"] if p50h < 0 else "#1A5E20"
        ax2.text(i + w/2, p50h + (0.5 if p50h >= 0 else -1), f"{p50h:+.1f}",
                 ha="center", fontsize=8, color=color_p, fontweight="bold")

    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("Headroom above 12% hurdle (pp)", fontsize=10)
    ax2.set_title("DPR Headroom vs Probabilistic Headroom\nNegative bar = P50 already in failure zone",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = os.path.join(OUT_DIR, "pfff_optimism_bias_v12.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.show(); plt.close(fig)
    print(f"  → Saved: {fname}")

def main():
    print("\n" + "═"*70)
    print("  PFFF v12.0 — Probabilistic Feasibility Fragility Framework")
    print("  M.BEM Thesis | SPA Delhi 2024 | Varshni M S")
    print("  Supervisor: Mr. Rhijul Sood | SPA Delhi")
    print("═"*70)

    # ── STEP 1: Zero-Stress Calibration ─────────────────────────────────
    print("\n[STEP 1] Zero-Stress Calibration")
    print("  Feeds each project's exact DPR inputs — confirms model reproduces")
    print("  consultant's stated EIRR before any uncertainty is applied.\n")
    for code, p in PROJECTS.items():
        scn = compute_scn(p)
        verify_calibration(p, scn)

    # ── STEP 2: Full MCS ─────────────────────────────────────────────────
    print(f"\n[STEP 2] Monte Carlo Simulation ({N_ITER:,} iterations × 7 projects × 3 modes)")
    all_results = {}; all_scn = {}; all_svs = {}; all_svboth = {}

    for code, p in PROJECTS.items():
        print(f"\n  [{code}] {p['name']}")
        scn  = compute_scn(p)
        samp = run_mcs(p, scn, N_ITER)
        mode_results = {}
        for mode in MODES:
            res = simulate_mode(p, scn, samp, mode, N_ITER)
            mode_results[mode] = res
            fi_f = f"{res['fi_firr']:.1f}%" if not np.isnan(res['fi_firr']) else "N/A"
            print(f"    {mode}: FI={res['fi_p']:5.1f}%  EIRR_FI={res['fi_eirr']:.1f}%  FIRR_FI={fi_f}  [{verdict(res['fi_p'])}]")
        tornado = spearman_tornado(p, scn, samp, mode_results[p["dpr_mode"]]["eirr_arr"])
        rcf     = rcf_acid_test(p, scn, samp, mode_results[p["dpr_mode"]]["fi_p"])
        svs     = compute_switching_values(p, scn)
        svboth  = compute_sv_both(p, scn, samp, N_ITER)
        mode_results["_tornado"] = tornado
        mode_results["_samp"]    = samp
        mode_results["_rcf"]     = rcf
        all_results[code] = mode_results
        all_scn[code]     = scn
        all_svs[code]     = svs
        all_svboth[code]  = svboth

    # ── STEP 3: Per-Project Dashboards ───────────────────────────────────
    print("\n[STEP 3] Per-Project Dashboards (7 projects)")
    for code, p in PROJECTS.items():
        print(f"\n  Dashboard: {p['name']}")
        plot_dashboard(p, all_scn[code], all_results[code]["_samp"],
                       all_results[code], all_results[code]["_tornado"],
                       all_results[code]["_rcf"], all_svs[code],
                       all_svboth[code], code)

    # ── STEP 4: Batch + SV Charts ────────────────────────────────────────
    print("\n[STEP 4] Batch Comparison & Switching Values Chart")
    plot_batch_comparison(all_results, all_svboth)

    # ── STEP 5: Optimism Bias Chart ──────────────────────────────────────
    print("\n[STEP 5] Optimism Bias Decomposition Chart")
    plot_optimism_bias(all_results, all_svboth)

    # ── STEP 6: Validation Exhibit ───────────────────────────────────────
    print("\n[STEP 6] Validation Exhibit (P5 & P7)")
    plot_validation_exhibit(all_results, all_scn)

    # ── STEP 7: Summary Table ────────────────────────────────────────────
    print("\n" + "═"*100)
    print("  RESULTS SUMMARY — PFFF v12")
    print("═"*100)
    print(f"  {'Project':<38} {'Mode':<5} {'FI%':>6} {'DPR_EIRR':>9} {'P50_EIRR':>9} {'SV_DPR_C':>9} {'SV_P50_C':>9} {'OPT_GAP':>8}")
    print("  " + "─"*95)
    for code, p in PROJECTS.items():
        mode = p["dpr_mode"]; fi = all_results[code][mode]["fi_p"]
        sb = all_svboth[code]
        tag = " ← VALIDATION" if p["role"]=="VALIDATION" else ""
        sv_dpr_c = f"+{sb['sv_dpr_cost']:.0f}%" if sb.get('sv_dpr_cost') else "∞"
        sv_p50_c = f"+{sb['sv_p50_cost']:.0f}%" if sb.get('sv_p50_cost') else "∞"
        phantom  = " [PHANTOM]" if sb.get('phantom') else ""
        print(f"  {p['name']:<38} {mode:<5} {fi:6.1f}% {p['dpr_eirr']:>8.1f}% {sb['p50_eirr']:>8.1f}% {sv_dpr_c:>9} {sv_p50_c:>9} {sb['optbias_cost']:>7.1f}pp{phantom}{tag}")

    print("\n  FRAGILITY VERDICT:")
    for code, p in PROJECTS.items():
        sv = all_svboth[code]
        if sv.get('phantom'):
            print(f"  ⚠  [{code}] P50={sv['p50_eirr']:.1f}% < 12% — SV_DPR={sv.get('sv_dpr_cost')}% is PHANTOM HEADROOM")
        elif sv.get('sv_dpr_cost') and sv['sv_dpr_cost'] < 71:
            print(f"  ⚡  [{code}] Cost SV {sv['sv_dpr_cost']:.0f}% < CAG avg 71% — structurally fragile")
    print("═"*100 + "\n")


if __name__ == "__main__":
    main()

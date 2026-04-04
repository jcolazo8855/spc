# ═══════════════════════════════════════════════════════════════════════════════
#  📊 Statistical Process Control — X̄ and R Charts
#  Shewhart / Sargent coefficients · 3-sigma control limits · White theme
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPC Control Charts",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  SHEWHART / SARGENT COEFFICIENTS
#  Source: ASTM STP 15D; derived via d2, d3 constants from the
#  Studentized-range distribution.
#  A2 = 3/(d2·√n)   D4 = 1 + 3·(d3/d2)   D3 = max(0, 1 − 3·(d3/d2))
# ═══════════════════════════════════════════════════════════════════════════════
COEFS = {
    2:  dict(d2=1.128, d3=0.853, A2=1.880, D3=0.000, D4=3.267),
    3:  dict(d2=1.693, d3=0.888, A2=1.023, D3=0.000, D4=2.574),
    4:  dict(d2=2.059, d3=0.880, A2=0.729, D3=0.000, D4=2.282),
    5:  dict(d2=2.326, d3=0.864, A2=0.577, D3=0.000, D4=2.114),
    6:  dict(d2=2.534, d3=0.848, A2=0.483, D3=0.000, D4=2.004),
    7:  dict(d2=2.704, d3=0.833, A2=0.419, D3=0.076, D4=1.924),
    8:  dict(d2=2.847, d3=0.820, A2=0.373, D3=0.136, D4=1.864),
    9:  dict(d2=2.970, d3=0.808, A2=0.337, D3=0.184, D4=1.816),
    10: dict(d2=3.078, d3=0.797, A2=0.308, D3=0.223, D4=1.777),
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp { background: #ffffff; color: #0f172a; }

section[data-testid="stSidebar"] {
    background: #f8f9fc !important;
    border-right: 1px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"] * { color: #374151 !important; }
section[data-testid="stSidebar"] label {
    font-size: 12px !important;
    font-weight: 500 !important;
}

/* Page title */
.page-title {
    font-size: 28px; font-weight: 800; letter-spacing: -0.8px;
    color: #0f172a; margin-bottom: 2px;
}
.page-sub {
    font-size: 13px; color: #94a3b8; font-weight: 400; margin-bottom: 22px;
}

/* Section header */
.section-hdr {
    font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
    color: #94a3b8; text-transform: uppercase;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 7px; margin-bottom: 14px;
}

/* Metric cards */
.kpi-row { display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 110px;
    background: #f8f9fc; border: 1px solid #e2e8f0;
    border-top: 3px solid; border-radius: 6px;
    padding: 12px 14px;
}
.kpi-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    color: #94a3b8; text-transform: uppercase;
}
.kpi-value {
    font-size: 22px; font-weight: 700; letter-spacing: -0.5px;
    margin-top: 4px;
}

/* Coefficient table */
.coef-table {
    width: 100%; border-collapse: collapse;
    font-size: 13px;
}
.coef-table th {
    background: #f1f5f9; text-align: center;
    padding: 9px 14px; font-weight: 600;
    color: #374151; border: 1px solid #e2e8f0;
    font-size: 12px; letter-spacing: 0.3px;
}
.coef-table td {
    text-align: center; padding: 8px 14px;
    border: 1px solid #e2e8f0; color: #374151;
    font-variant-numeric: tabular-nums;
}
.coef-table tr:nth-child(even) td { background: #f8f9fc; }
.coef-table td.hl { font-weight: 700; }

/* OOC summary */
.ooc-box {
    border: 1px solid; border-radius: 6px;
    padding: 12px 16px; font-size: 13px;
    line-height: 1.7; margin-top: 12px;
}

/* Sidebar sub-section */
.sb-hdr {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #e2e8f0 !important;
    padding-bottom: 5px !important;
    margin-top: 14px !important;
    margin-bottom: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE (white-theme SPC)
# ═══════════════════════════════════════════════════════════════════════════════
C_UCL      = "#dc2626"        # red — control limits
C_CL_X     = "#1e40af"        # blue — X-bar centre line
C_CL_R     = "#0369a1"        # sky-blue — R centre line
C_IN       = "#374151"        # dark-gray — in-control points
C_OOC      = "#dc2626"        # red — out-of-control points
C_WARN2    = "rgba(251,191,36,0.12)"   # 2-sigma zone fill
C_WARN1    = "rgba(34,197,94,0.07)"    # 1-sigma zone fill
C_GRID     = "#f1f5f9"
C_LINE_BG  = "#ffffff"

FONT = dict(family="Inter, sans-serif", color="#374151")

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def base_layout(height=340, **extra):
    return dict(
        plot_bgcolor=C_LINE_BG,
        paper_bgcolor="rgba(0,0,0,0)",
        font=FONT,
        height=height,
        margin=dict(t=14, b=52, l=62, r=24),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0", borderwidth=1,
            font=dict(family="Inter, sans-serif", size=12, color="#374151"),
        ),
        xaxis=dict(
            gridcolor=C_GRID, linecolor="#e2e8f0",
            zerolinecolor="#e2e8f0", tickfont=dict(size=12),
            title_font=dict(size=13),
        ),
        yaxis=dict(
            gridcolor=C_GRID, linecolor="#e2e8f0",
            zerolinecolor="#e2e8f0", tickfont=dict(size=12),
            title_font=dict(size=13),
        ),
        **extra,
    )


def _add_limit_line(fig, y, label, color, dash="solid", row=1):
    fig.add_hline(
        y=y, line_color=color, line_width=1.6,
        line_dash=dash, row=row, col=1,
        annotation_text=f" {label} = {y:.4f}",
        annotation_position="right",
        annotation_font=dict(family="Inter, sans-serif", size=11, color=color),
    )


def _add_zone_band(fig, y_lo, y_hi, fill_color, row=1):
    """Horizontal filled band between two y values."""
    fig.add_hrect(y0=y_lo, y1=y_hi, fillcolor=fill_color,
                  line_width=0, row=row, col=1)


def kpi(label, value, color):
    return (
        f'<div class="kpi-card" style="border-top-color:{color};">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color};">{value}</div>'
        f'</div>'
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def generate_data(
    k: int,          # number of samples
    n: int,          # sample size
    mu: float,       # process mean
    sigma: float,    # within-sample std dev
    # X-bar OOC injection
    ooc_x_indices: list,
    ooc_x_shift: float,      # shift in σ units (positive or negative)
    ooc_x_direction: str,    # "positive" | "negative" | "alternating"
    # R OOC injection
    ooc_r_indices: list,
    ooc_r_mult: float,       # multiply σ by this for those samples
    seed: int = 42,
) -> np.ndarray:
    """Return array shape (k, n)."""
    rng  = np.random.RandomState(seed)
    data = rng.normal(mu, sigma, (k, n))

    # Inject mean shifts (X-bar OOC)
    for pos, idx in enumerate(ooc_x_indices):
        if 0 <= idx < k:
            sign = (1 if ooc_x_direction == "positive"
                    else -1 if ooc_x_direction == "negative"
                    else (1 if pos % 2 == 0 else -1))
            data[idx] += sign * ooc_x_shift * sigma

    # Inject variability inflation (R OOC) — add extra noise to sample
    for idx in ooc_r_indices:
        if 0 <= idx < k:
            data[idx] += rng.normal(0, (ooc_r_mult - 1) * sigma, n)

    return data


def compute_spc(data: np.ndarray, n: int):
    """Compute X-bar, R, and control limits."""
    k       = data.shape[0]
    xbar    = data.mean(axis=1)
    ranges  = data.max(axis=1) - data.min(axis=1)
    xbarbar = xbar.mean()
    rbar    = ranges.mean()

    c = COEFS[n]
    A2, D3, D4 = c["A2"], c["D3"], c["D4"]

    UCL_x = xbarbar + A2 * rbar
    LCL_x = xbarbar - A2 * rbar
    UCL_R = D4 * rbar
    LCL_R = D3 * rbar

    # Sigma estimate from R-bar
    sigma_est = rbar / c["d2"]

    # 1-sigma and 2-sigma zones for X-bar
    sigma_x = A2 * rbar / 3          # ≡ σ/√n in terms of control-chart scale
    z1_upper_x = xbarbar + sigma_x
    z1_lower_x = xbarbar - sigma_x
    z2_upper_x = xbarbar + 2 * sigma_x
    z2_lower_x = xbarbar - 2 * sigma_x

    return dict(
        k=k, xbar=xbar, ranges=ranges,
        xbarbar=xbarbar, rbar=rbar,
        UCL_x=UCL_x, LCL_x=LCL_x,
        UCL_R=UCL_R, LCL_R=LCL_R,
        sigma_est=sigma_est,
        z1_upper_x=z1_upper_x, z1_lower_x=z1_lower_x,
        z2_upper_x=z2_upper_x, z2_lower_x=z2_lower_x,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════
def xbar_chart(spc: dict, show_zones: bool) -> go.Figure:
    k       = spc["k"]
    xbar    = spc["xbar"]
    UCL_x   = spc["UCL_x"]
    LCL_x   = spc["LCL_x"]
    xbarbar = spc["xbarbar"]
    samples = list(range(1, k + 1))

    ooc_mask = (xbar > UCL_x) | (xbar < LCL_x)
    ic_mask  = ~ooc_mask

    fig = go.Figure()

    # Zone bands
    if show_zones:
        _add_zone_band(fig, spc["z1_lower_x"], spc["z1_upper_x"], C_WARN1)
        _add_zone_band(fig, spc["z2_lower_x"], spc["z1_lower_x"], C_WARN2)
        _add_zone_band(fig, spc["z1_upper_x"], spc["z2_upper_x"], C_WARN2)

    # Control-limit lines
    _add_limit_line(fig, UCL_x,   "UCL", C_UCL, "solid")
    _add_limit_line(fig, xbarbar, "X̄̄",  C_CL_X, "dash")
    _add_limit_line(fig, LCL_x,   "LCL", C_UCL, "solid")

    # Connecting line
    fig.add_trace(go.Scatter(
        x=samples, y=xbar,
        mode="lines",
        line=dict(color="#94a3b8", width=1.2),
        showlegend=False,
        hoverinfo="skip",
    ))

    # In-control points
    if ic_mask.any():
        fig.add_trace(go.Scatter(
            x=[s for s, m in zip(samples, ic_mask) if m],
            y=xbar[ic_mask],
            mode="markers",
            name="In control",
            marker=dict(color=C_IN, size=8, symbol="circle",
                        line=dict(color="#ffffff", width=1.5)),
            hovertemplate="Subgroup %{x}<br>X̄ = %{y:.4f}<extra></extra>",
        ))

    # Out-of-control points
    if ooc_mask.any():
        fig.add_trace(go.Scatter(
            x=[s for s, m in zip(samples, ooc_mask) if m],
            y=xbar[ooc_mask],
            mode="markers+text",
            name="Out of control",
            marker=dict(color=C_OOC, size=10, symbol="circle",
                        line=dict(color="#ffffff", width=1.5)),
            text=["▲" if v > UCL_x else "▼"
                  for v in xbar[ooc_mask]],
            textposition="top center",
            textfont=dict(color=C_OOC, size=13),
            hovertemplate="Subgroup %{x}<br>X̄ = %{y:.4f} ⚠ OOC<extra></extra>",
        ))

    fig.update_layout(
        **base_layout(height=340,
                      xaxis_title="Subgroup",
                      yaxis_title="Subgroup mean (X̄)"),
    )
    fig.update_xaxes(range=[0.2, k + 0.8], dtick=1 if k <= 25 else 5)
    return fig


def r_chart(spc: dict) -> go.Figure:
    k       = spc["k"]
    ranges  = spc["ranges"]
    UCL_R   = spc["UCL_R"]
    LCL_R   = spc["LCL_R"]
    rbar    = spc["rbar"]
    samples = list(range(1, k + 1))

    ooc_mask = (ranges > UCL_R) | (ranges < LCL_R)
    ic_mask  = ~ooc_mask

    fig = go.Figure()

    _add_limit_line(fig, UCL_R, "UCL_R", C_UCL, "solid")
    _add_limit_line(fig, rbar,  "R̄",    C_CL_R, "dash")
    if spc["LCL_R"] > 0:
        _add_limit_line(fig, LCL_R, "LCL_R", C_UCL, "solid")

    # Connecting line
    fig.add_trace(go.Scatter(
        x=samples, y=ranges,
        mode="lines",
        line=dict(color="#94a3b8", width=1.2),
        showlegend=False,
        hoverinfo="skip",
    ))

    if ic_mask.any():
        fig.add_trace(go.Scatter(
            x=[s for s, m in zip(samples, ic_mask) if m],
            y=ranges[ic_mask],
            mode="markers",
            name="In control",
            marker=dict(color="#0369a1", size=8, symbol="diamond",
                        line=dict(color="#ffffff", width=1.5)),
            hovertemplate="Subgroup %{x}<br>R = %{y:.4f}<extra></extra>",
        ))

    if ooc_mask.any():
        fig.add_trace(go.Scatter(
            x=[s for s, m in zip(samples, ooc_mask) if m],
            y=ranges[ooc_mask],
            mode="markers+text",
            name="Out of control",
            marker=dict(color=C_OOC, size=10, symbol="diamond",
                        line=dict(color="#ffffff", width=1.5)),
            text=["▲" if v > UCL_R else "▼" for v in ranges[ooc_mask]],
            textposition="top center",
            textfont=dict(color=C_OOC, size=13),
            hovertemplate="Subgroup %{x}<br>R = %{y:.4f} ⚠ OOC<extra></extra>",
        ))

    fig.update_layout(
        **base_layout(height=300,
                      xaxis_title="Subgroup",
                      yaxis_title="Subgroup range (R)"),
    )
    fig.update_xaxes(range=[0.2, k + 0.8], dtick=1 if k <= 25 else 5)
    fig.update_yaxes(rangemode="tozero")
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 SPC Control Charts")
    st.markdown("---")

    # ── Process parameters ───────────────────────────────────────────────────
    st.markdown('<p class="sb-hdr">Process parameters</p>', unsafe_allow_html=True)
    n_sample   = st.slider("Sample size (n)", 2, 10, 5,
                           help="Number of individual measurements per sample")
    n_samples = st.slider("Number of samples (k)", 10, 40, 25)
    mu          = st.number_input("Process mean (μ)", value=100.0, step=1.0)
    sigma       = st.number_input("Std dev within sample (σ)", value=2.0,
                                  min_value=0.01, step=0.1)
    seed        = st.number_input("Random seed", value=42, step=1,
                                  help="Change for a different random sample")

    # ── Auto-populated coefficients ──────────────────────────────────────────
    c = COEFS[n_sample]
    st.markdown('<p class="sb-hdr">Sargent coefficients (auto)</p>',
                unsafe_allow_html=True)
    col_c1, col_c2, col_c3 = st.columns(3)
    for col_w, label, val in [(col_c1, "A₂", c["A2"]),
                               (col_c2, "D₃", c["D3"]),
                               (col_c3, "D₄", c["D4"])]:
        col_w.markdown(
            f'<div style="text-align:center;padding:6px 2px;">'
            f'<div style="font-size:10px;color:#94a3b8;font-weight:600;'
            f'letter-spacing:1px;text-transform:uppercase;">{label}</div>'
            f'<div style="font-size:14px;font-weight:700;color:#1e40af;'
            f'margin-top:2px;">{val:.2f}</div></div>',
            unsafe_allow_html=True,
        )

    # ── X-bar OOC injection ──────────────────────────────────────────────────
    st.markdown('<p class="sb-hdr">X-bar out-of-control injection</p>',
                unsafe_allow_html=True)
    n_ooc_x     = st.slider("Number of OOC samples (X̄)", 0, 6, 2)
    ooc_x_shift = st.slider("Mean shift magnitude (σ units)", 1.0, 6.0, 3.5, 0.1,
                             disabled=(n_ooc_x == 0))
    ooc_x_dir   = st.selectbox("Shift direction",
                                ["positive", "negative", "alternating"],
                                disabled=(n_ooc_x == 0))

    # ── R OOC injection ──────────────────────────────────────────────────────
    st.markdown('<p class="sb-hdr">R chart out-of-control injection</p>',
                unsafe_allow_html=True)
    n_ooc_r    = st.slider("Number of OOC samples (R)", 0, 4, 1)
    ooc_r_mult = st.slider("Range inflation multiplier", 2.0, 8.0, 4.0, 0.5,
                            help="σ multiplied by this for the spiked samples",
                            disabled=(n_ooc_r == 0))

    # ── Display options ──────────────────────────────────────────────────────
    st.markdown('<p class="sb-hdr">Display options</p>', unsafe_allow_html=True)
    show_zones   = st.checkbox("Show sigma zones (X̄ chart)", value=True)
    show_data_tbl= st.checkbox("Show sample data table",   value=False)

# ── Auto-place OOC sample indices evenly across the k samples ─────────────
def auto_place(n_ooc: int, k: int, default_start: int) -> list:
    """Return n_ooc evenly-spaced 0-based indices within [0, k)."""
    if n_ooc == 0:
        return []
    step = max(1, k // (n_ooc + 1))
    return [(default_start + i * step) % k for i in range(n_ooc)]

ooc_x_idx = auto_place(n_ooc_x, n_samples, default_start=4)
ooc_r_idx  = auto_place(n_ooc_r, n_samples, default_start=9)

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPUTE
# ═══════════════════════════════════════════════════════════════════════════════
data = generate_data(
    k=n_samples, n=n_sample, mu=mu, sigma=sigma,
    ooc_x_indices=ooc_x_idx, ooc_x_shift=ooc_x_shift,
    ooc_x_direction=ooc_x_dir,
    ooc_r_indices=ooc_r_idx,  ooc_r_mult=(ooc_r_mult if n_ooc_r > 0 else 1.0),
    seed=int(seed),
)

spc = compute_spc(data, n_sample)
n_ooc_x_detected = int(((spc["xbar"] > spc["UCL_x"]) |
                          (spc["xbar"] < spc["LCL_x"])).sum())
n_ooc_r_detected = int(((spc["ranges"] > spc["UCL_R"]) |
                          (spc["ranges"] < spc["LCL_R"])).sum())

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="page-title">BAT 3301 – Colazo – SPC Demo</div>'
    '<div class="page-sub">'
    'X̄ and R charts · Sargent (Shewhart) coefficients · 3σ control limits'
    '</div>',
    unsafe_allow_html=True,
)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
status_color = "#dc2626" if (n_ooc_x_detected + n_ooc_r_detected) > 0 else "#16a34a"
status_text  = "⚠ OOC signals" if (n_ooc_x_detected + n_ooc_r_detected) > 0 else "✓ In control"

with k1: st.markdown(kpi("Process mean (μ)", f"{spc['xbarbar']:.3f}", "#1e40af"), unsafe_allow_html=True)
with k2: st.markdown(kpi("Mean range (R̄)",  f"{spc['rbar']:.3f}",    "#0369a1"), unsafe_allow_html=True)
with k3: st.markdown(kpi("σ̂ (from R̄)",      f"{spc['sigma_est']:.3f}", "#374151"), unsafe_allow_html=True)
with k4: st.markdown(kpi("X̄ OOC points",    str(n_ooc_x_detected),  "#dc2626" if n_ooc_x_detected else "#16a34a"), unsafe_allow_html=True)
with k5: st.markdown(kpi("R OOC points",     str(n_ooc_r_detected),  "#dc2626" if n_ooc_r_detected else "#16a34a"), unsafe_allow_html=True)
with k6: st.markdown(kpi("Status",           status_text,            status_color), unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── Control limits summary ────────────────────────────────────────────────────
col_cl1, col_cl2 = st.columns(2)
with col_cl1:
    st.markdown(f"""
<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;
            padding:12px 16px;font-size:13px;line-height:2;">
<strong style="color:#1e40af;">X̄ Chart</strong><br>
UCL = X̄̄ + A₂·R̄ = {spc['xbarbar']:.3f} + {c['A2']}·{spc['rbar']:.3f}
    = <strong style="color:#dc2626;">{spc['UCL_x']:.4f}</strong><br>
CL &nbsp;= X̄̄ = <strong style="color:#1e40af;">{spc['xbarbar']:.4f}</strong><br>
LCL = X̄̄ − A₂·R̄ = <strong style="color:#dc2626;">{spc['LCL_x']:.4f}</strong>
</div>
""", unsafe_allow_html=True)

with col_cl2:
    lcl_r_expr = (f"<strong style='color:#dc2626;'>{spc['LCL_R']:.4f}</strong>"
                  if c["D3"] > 0 else
                  "<strong style='color:#94a3b8;'>0 (D₃ = 0 for n ≤ 6)</strong>")
    st.markdown(f"""
<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:6px;
            padding:12px 16px;font-size:13px;line-height:2;">
<strong style="color:#0369a1;">R Chart</strong><br>
UCL = D₄·R̄ = {c['D4']}·{spc['rbar']:.3f}
    = <strong style="color:#dc2626;">{spc['UCL_R']:.4f}</strong><br>
CL &nbsp;= R̄ = <strong style="color:#0369a1;">{spc['rbar']:.4f}</strong><br>
LCL = D₃·R̄ = {lcl_r_expr}
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">X̄ Chart (Subgroup Means)</div>',
            unsafe_allow_html=True)
st.plotly_chart(xbar_chart(spc, show_zones),
                use_container_width=True, config={"displayModeBar": False})
if n_ooc_x > 0:
    shift_units = ooc_x_shift * sigma
    cl_half     = c["A2"] * spc["rbar"]        # half-width of 3σ control limits
    st.caption(
        f"X̄ OOC injection: {n_ooc_x} sample(s) at position(s) "
        f"{[i+1 for i in ooc_x_idx]}  ·  "
        f"shift = {ooc_x_shift:.1f}σ {ooc_x_dir}  "
        f"({ooc_x_shift:.1f} × σ = {ooc_x_shift:.1f} × {sigma:.2f} = {shift_units:.2f} units).  "
        f"The 3σ control limits span ±{cl_half:.2f} units from X̄̄, "
        f"so a {shift_units:.2f}-unit shift "
        f"{'reliably exceeds' if shift_units > cl_half else 'may not exceed'} the limits."
    )

st.markdown('<div class="section-hdr">R Chart (Subgroup Ranges)</div>',
            unsafe_allow_html=True)
st.plotly_chart(r_chart(spc),
                use_container_width=True, config={"displayModeBar": False})
if n_ooc_r > 0:
    st.caption(
        f"R OOC injection: {n_ooc_r} sample(s) at position(s) "
        f"{[i+1 for i in ooc_r_idx]}  ·  "
        f"range multiplier = {ooc_r_mult:.1f}×"
    )

# ── OOC summary ───────────────────────────────────────────────────────────────
total_ooc = n_ooc_x_detected + n_ooc_r_detected
if total_ooc > 0:
    ooc_x_list = [i+1 for i, v in enumerate(spc["xbar"])
                  if v > spc["UCL_x"] or v < spc["LCL_x"]]
    ooc_r_list = [i+1 for i, v in enumerate(spc["ranges"])
                  if v > spc["UCL_R"] or v < spc["LCL_R"]]
    st.markdown(f"""
<div class="ooc-box" style="border-color:#fca5a5;background:#fef2f2;color:#7f1d1d;">
<strong>⚠ Out-of-Control Signals Detected</strong><br>
{"• X̄ chart: samples " + str(ooc_x_list) + " are beyond control limits<br>" if ooc_x_list else ""}
{"• R chart: samples " + str(ooc_r_list) + " are beyond control limits" if ooc_r_list else ""}
</div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div class="ooc-box" style="border-color:#86efac;background:#f0fdf4;color:#14532d;">
<strong>✓ Process appears to be in statistical control</strong> —
all sample means and ranges fall within the 3σ control limits.
</div>
""", unsafe_allow_html=True)

# ── Subgroup data table ───────────────────────────────────────────────────────
if show_data_tbl:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">Subgroup Data</div>', unsafe_allow_html=True)
    col_labels = [f"x{j+1}" for j in range(n_sample)]
    df_data = pd.DataFrame(data, columns=col_labels)
    df_data.insert(0, "Subgroup", range(1, n_samples + 1))
    df_data["X̄"]  = spc["xbar"].round(4)
    df_data["R"]   = spc["ranges"].round(4)
    df_data["OOC"] = [
        ("X̄" if (x > spc["UCL_x"] or x < spc["LCL_x"]) else "") +
        (" R" if (r > spc["UCL_R"] or r < spc["LCL_R"]) else "")
        for x, r in zip(spc["xbar"], spc["ranges"])
    ]
    df_data = df_data.round(4)

    def style_ooc(row):
        if row["OOC"].strip():
            return ["background-color:#fef2f2; color:#dc2626; font-weight:600;"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_data.style.apply(style_ooc, axis=1),
        use_container_width=True, height=300,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  COEFFICIENTS TABLE (n = 4 to 8)  — rendered as styled st.dataframe()
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
st.markdown(
    '<div class="section-hdr">Sargent / Shewhart Coefficients Reference Table'
    ' &nbsp;—&nbsp; Sample sizes n = 4 to 8</div>',
    unsafe_allow_html=True,
)

# Build DataFrame
coef_rows = []
for n in range(4, 9):
    cc = COEFS[n]
    coef_rows.append({
        "n":              n,
        "d₂":             cc["d2"],
        "d₃":             cc["d3"],
        "A₂":             cc["A2"],
        "D₃":             cc["D3"],
        "D₄":             cc["D4"],
        "X̄ UCL/LCL  (±A₂·R̄)": cc["A2"],
        "R UCL  (D₄·R̄)":        cc["D4"],
        "R LCL  (D₃·R̄)":        cc["D3"],
    })

df_coef = pd.DataFrame(coef_rows).set_index("n")

# Column format: 3 decimal places
fmt = {col: "{:.3f}" for col in df_coef.columns}

def style_coef_table(styler):
    # Highlight current n row in blue tint
    def row_style(row):
        if row.name == n_sample:
            return [
                "background-color:#dbeafe; color:#1e40af; font-weight:700;"
            ] * len(row)
        return [""] * len(row)

    # Column-level colours: D₃, D₄ and the limit columns in red
    red_cols = ["D₃", "D₄", "R UCL  (D₄·R̄)", "R LCL  (D₃·R̄)"]

    def col_color(col):
        # col is a pandas Series; use col.name to get the column label
        if col.name in red_cols:
            return [
                "" if idx == n_sample else "color:#dc2626;"
                for idx in df_coef.index
            ]
        return [""] * len(df_coef)

    styler = (
        styler
        .apply(row_style, axis=1)
        .apply(col_color, axis=0)
        .format(fmt)
        .set_table_styles([
            # Header
            {"selector": "thead th",
             "props": [("background-color", "#f1f5f9"),
                       ("color", "#374151"),
                       ("font-weight", "600"),
                       ("font-size", "13px"),
                       ("text-align", "center"),
                       ("padding", "10px 14px"),
                       ("border", "1px solid #e2e8f0")]},
            # Cells
            {"selector": "tbody td",
             "props": [("text-align", "center"),
                       ("padding", "9px 14px"),
                       ("border", "1px solid #e2e8f0"),
                       ("font-size", "13px")]},
            # Index column
            {"selector": "tbody th",
             "props": [("text-align", "center"),
                       ("padding", "9px 14px"),
                       ("border", "1px solid #e2e8f0"),
                       ("font-size", "13px"),
                       ("font-weight", "600"),
                       ("background-color", "#f8f9fc")]},
            # Alternate row shading
            {"selector": "tbody tr:nth-child(even) td",
             "props": [("background-color", "#f8f9fc")]},
        ])
    )
    return styler

st.dataframe(
    style_coef_table(df_coef.style),
    use_container_width=True,
    height=215,          # fits 5 rows without scroll
)

# Caption below the table
st.caption(
    f"▶ Row highlighted in blue = current selection (n = {n_sample}).  "
    "D₃ = 0 for n ≤ 6 (no lower control limit on the R chart)."
)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# Formula reference (plain markdown — renders reliably everywhere)
st.markdown("""
**Formulas** &nbsp;&nbsp;
UCL(X̄) = X̄̄ + A₂·R̄ &nbsp;·&nbsp;
LCL(X̄) = X̄̄ − A₂·R̄ &nbsp;·&nbsp;
UCL(R) = D₄·R̄ &nbsp;·&nbsp;
LCL(R) = D₃·R̄ &nbsp;&nbsp;
""")

"""Shared Streamlit layout: theme, header, banner, sidebar controls, cached data/detector loading."""
from __future__ import annotations
import html
import json
from dataclasses import dataclass
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from pipeline import anonymise, review, run, score
from pipeline.config import ROOT, load_config

NAVY, BLUE, LBLUE, ORANGE = "#1F3864", "#0070C0", "#DEEAF6", "#C55A11"
SEV_COLOR = {"High": "#D9480F", "Medium": "#E8A317", "Low": "#8896A8"}
DET_LABEL = {"cost": "Cost outlier", "duplicate": "Near-duplicate", "stalled": "Stalled work", "agency": "Agency / district pattern", "delay": "Process delay"}
pio.templates["lens"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", color="#1F2937"), colorway=[BLUE, ORANGE, NAVY, "#8896A8", "#E8A317", "#2E8B57"],
    paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#E9EEF4", linecolor="#CBD5E1"), yaxis=dict(gridcolor="#E9EEF4", linecolor="#CBD5E1"), title=dict(font=dict(size=15, color=NAVY))))
pio.templates.default = "lens"


@dataclass
class Ctx:
    works: pd.DataFrame
    dets: pd.DataFrame
    scored: pd.DataFrame
    cfg: dict
    anon: bool
    meta: dict
    maps: dict
    weights: dict

    @property
    def synthetic(self) -> bool:
        return self.meta["source"] == "synthetic"

    def show(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply anonymisation to a frame for display."""
        return anonymise.apply(df, self.maps, self.anon)


@st.cache_data(show_spinner="Preparing data…")
def _works(sig: str) -> tuple[pd.DataFrame, dict]:
    return run.load_works()


@st.cache_data(show_spinner="Running the five detectors (first load only)…")
def _dets(sig: str, _works: pd.DataFrame, cfg_json: str) -> pd.DataFrame:
    return run.run_all(_works, json.loads(cfg_json))


@st.cache_data
def _wide(sig: str, cfg_json: str, _dets: pd.DataFrame, _ids: pd.Series):
    return score.wide(_dets, _ids)


@st.cache_data
def _maps(sig: str, _works: pd.DataFrame) -> dict:
    return anonymise.build_maps(_works)


def _cfg(meta: dict) -> dict:
    cfg = load_config()
    cfg["as_of"] = meta["as_of"]
    return cfg


def pipeline_key(meta: dict) -> str:
    """Cache key for detector results (dataset signature + non-weight config)."""
    cfg = _cfg(meta)
    cfg_json = json.dumps({k: v for k, v in cfg.items() if k not in ("weights", "severity", "flag_threshold", "multi_detector_bonus")}, default=str)
    return f"{run.signature()}|{cfg_json}"


def setup(title: str, icon: str = "🔎") -> Ctx:
    """First call on every page: page config, CSS, header, banner, sidebar; returns the data context."""
    st.set_page_config(page_title=f"{title} · MPLADS-Lens", page_icon=icon, layout="wide", initial_sidebar_state="expanded")
    st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text()}</style>", unsafe_allow_html=True)
    sig = run.signature()
    works, meta = _works(sig)
    cfg = _cfg(meta)
    key = pipeline_key(meta)
    cfg_json = key.split("|", 3)[3]
    ov = st.session_state.get("dets_override")
    dets = ov[1] if ov and ov[0] == key else _dets(key, works, cfg_json)
    scores, reasons = _wide(key, cfg_json, dets, works["work_id"])
    synthetic = meta["source"] == "synthetic"
    badge = '<span class="badge">Synthetic demo data</span>' if synthetic else '<span class="badge user">User-supplied data</span>'
    st.markdown(f'<div class="lens-header"><div><span class="brand">MPLADS-Lens</span><span class="sub">SIH26102 - MoSPI</span></div>{badge}</div>'
                '<div class="lens-banner">Outliers for review, not findings of fraud.</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Controls")
        anon = st.toggle("Anonymised view", value=True, key="anon", disabled=synthetic,
                         help="Always on for synthetic data. Can be turned off only for your own local data.")
        anon = True if synthetic else anon
        with st.expander("Detector weights (live)", expanded=False):
            def _reset():
                for d, v in cfg["weights"].items():
                    st.session_state[f"w_{d}"] = float(v)
            weights = {d: st.slider(DET_LABEL[d], 0.0, 1.0, float(v), 0.05, key=f"w_{d}") for d, v in cfg["weights"].items()}
            st.caption("Weights are relative: the largest one can reach 100 alone.")
            st.button("Reset weights", on_click=_reset)
        st.caption(f"{len(works):,} works · as of {meta['as_of']}")
    scored = score.combine(scores, reasons, weights, cfg)
    return Ctx(works, dets, scored, cfg, anon, meta, _maps(sig, works), weights)


def footer(ctx: Ctx) -> None:
    badge = "Synthetic demo data" if ctx.synthetic else "User-supplied data"
    st.markdown(f'<div class="lens-footer"><span>Scores are review prompts based on statistical outliers. They are not findings about any work, agency or person. '
                f'Legitimate reasons often explain unusual values.</span><span class="badge">{badge}</span></div>', unsafe_allow_html=True)


def chip(sev: str) -> str:
    return f'<span class="chip {sev}">{sev}</span>'


def kpi(label: str, value: str) -> str:
    return f'<div class="card kpi"><div class="l">{html.escape(label)}</div><div class="v">{html.escape(value)}</div></div>'


def inr(x: float) -> str:
    """Compact rupee format: lakh / crore."""
    if x >= 1e7:
        return f"₹{x / 1e7:,.2f} Cr"
    return f"₹{x / 1e5:,.1f} L"


def flagged_table(ctx: Ctx) -> pd.DataFrame:
    """Scored works joined to descriptive columns (display-anonymised), with review flag."""
    w = ctx.works[["work_id", "work_category", "state", "district", "block_or_village", "implementing_agency", "constituency", "mp_ref",
                   "work_title", "sanctioned_amount", "released_amount", "status"]]
    d = ctx.scored.merge(w, on="work_id", how="left")
    d = ctx.show(d)
    d["reviewed"] = d["work_id"].isin(review.reviewed_ids())
    return d

import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import ui
from detectors.agency import FEATURES
from pipeline import peer_groups, review

ctx = ui.setup("Work Detail", "🔍")
st.markdown("## Work detail")
d = ui.flagged_table(ctx)
cand = d[d.score > 0].sort_values("score", ascending=False)
if cand.empty:
    st.info("No works have any signal yet. Open the Data page to load data.")
    st.stop()
ids = cand.work_id.head(500).tolist()
sel = st.session_state.get("selected_work", ids[0])
if sel not in ids:
    ids = [sel] + ids
wid = st.selectbox("Work", ids, index=ids.index(sel), help="Pick from the top-scoring works, or click a row on the Flagged Works page.")
st.session_state["selected_work"] = wid
row = d[d.work_id == wid].iloc[0]
W = ctx.works.set_index("work_id")
raw = W.loc[wid]
dd = ctx.dets[ctx.dets.work_id == wid].set_index("detector")

st.markdown(f'{ui.chip(row.severity)} &nbsp; **{wid}** · {row.work_category} · {row.district} · {ui.inr(row.sanctioned_amount)} · score **{row.score:.0f}**', unsafe_allow_html=True)
st.markdown(f'<div class="reason">{html.escape(row.reason or "No signal.")}</div>', unsafe_allow_html=True)
c1, c2 = st.columns([1.2, 1])
with c1:
    labels = [ui.DET_LABEL[k] for k in ui.DET_LABEL]
    contrib = [row[f"c_{k}"] for k in ui.DET_LABEL]
    fig = go.Figure(go.Bar(x=contrib, y=labels, orientation="h", marker_color=ui.BLUE, text=[f"raw {row[f's_{k}']:.0f}" for k in ui.DET_LABEL], textposition="outside"))
    fig.update_layout(title="Score breakdown (weighted contribution)", height=280, yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 115]))
    st.plotly_chart(fig, width="stretch")
with c2:
    st.markdown('<div class="legit"><b>Legitimate reasons to check</b><ul style="margin:6px 0 0 16px">'
                '<li><b>Terrain and remoteness</b>: hill, desert and tribal areas cost more to build in.</li>'
                '<li><b>Materials and specification</b>: higher grade or larger structures.</li>'
                '<li><b>Revised scope</b>: an approved scope change or an earlier phase.</li>'
                '<li><b>Data entry</b>: a typo, wrong unit or a missing update.</li></ul></div>', unsafe_allow_html=True)


def _evidence_header(t: str) -> None:
    st.markdown(f"#### {t}")


if "cost" in dd.index:
    ev = dd.loc["cost", "evidence"]
    _evidence_header("Cost outlier evidence")
    st.caption(f"Peer group: **{ev['label']}** · defined by {ev['level_name']} · {ev['n']} works" + (" · fallback level used" if ev["level"] > 0 else " · no fallback needed"))
    peers = peer_groups.members(ctx.works, ev["level"], ev["key"])
    fig = go.Figure()
    fig.add_trace(go.Box(x=peers.sanctioned_amount / 1e5, name="Peers", boxpoints="outliers", marker_color=ui.BLUE, line_color=ui.NAVY, orientation="h"))
    fig.add_trace(go.Scatter(x=[raw.sanctioned_amount / 1e5], y=["Peers"], mode="markers", name="This work", marker=dict(size=16, color=ui.ORANGE, symbol="diamond", line=dict(width=1, color="white"))))
    fig.add_vline(x=ev["median"] / 1e5, line_dash="dash", line_color=ui.NAVY, annotation_text="peer median")
    fig.update_layout(height=260, xaxis_title="Sanctioned amount (₹ lakh)", xaxis_type="log")
    st.plotly_chart(fig, width="stretch")
    near = peers[peers.work_id != wid].assign(gap=lambda x: (np.log(x.sanctioned_amount) - np.log(raw.sanctioned_amount)).abs()).sort_values("gap").head(10)
    near = ctx.show(near)
    st.markdown("**10 nearest peers by amount**")
    st.dataframe(near.assign(**{"Amount (₹ lakh)": (near.sanctioned_amount / 1e5).round(1)})[["work_id", "work_category", "district", "implementing_agency", "Amount (₹ lakh)", "status"]],
                 hide_index=True, width="stretch")

if "duplicate" in dd.index:
    ev = dd.loc["duplicate", "evidence"]
    _evidence_header("Near-duplicate evidence")
    other = ev["pair_work_id"]
    st.caption(f"Text similarity {ev['cosine']:.2f} · fuzzy match {ev['fuzz']:.0f}/100 · sanctioned {ev['days_apart']} days apart")
    pair = ctx.show(ctx.works[ctx.works.work_id.isin([wid, other])].set_index("work_id"))
    fields = ["work_description", "work_category", "block_or_village", "district", "implementing_agency", "recommended_date", "sanctioned_date", "sanctioned_amount", "status"]
    rows = ""
    for f in fields:
        a, b = (str(pair.loc[wid, f]).split(" 00:")[0], str(pair.loc[other, f]).split(" 00:")[0])
        cls = ' class="diff"' if a != b else ""
        rows += f"<tr><th>{f.replace('_', ' ')}</th><td><span{cls}>{html.escape(a)}</span></td><td><span{cls}>{html.escape(b)}</span></td></tr>"
    st.markdown(f'<table class="cmp"><tr><th></th><th>This work ({wid})</th><th>Earlier match ({other})</th></tr>{rows}</table>', unsafe_allow_html=True)

if "stalled" in dd.index:
    ev = dd.loc["stalled", "evidence"]
    _evidence_header("Stalled work timeline")
    pts = [("Recommended", ev["recommended"]), (f"Sanctioned · ₹{ev['released'] / 1e5:.1f} L released", ev["sanctioned"]), ("Last progress", ev["last_progress"]), ("As of", ev["as_of"])]
    pts = [(n, t) for n, t in pts if pd.notna(t)]
    fig = go.Figure(go.Scatter(x=[t for _, t in pts], y=[0] * len(pts), mode="lines+markers+text", text=[n for n, _ in pts], textposition=["top center", "bottom center"] * 2,
                               marker=dict(size=14, color=[ui.BLUE, ui.NAVY, ui.ORANGE, "#8896A8"][:len(pts)]), line=dict(color="#CBD5E1", width=3)))
    fig.update_layout(height=230, yaxis=dict(visible=False, range=[-1, 1]), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("No update was ever recorded after sanction." if pd.isna(ev["last_progress"]) else f"{ev['years_stalled']:.1f} years since the last recorded progress. The data has one release amount and no release date, so release is shown at sanction.")

if "agency" in dd.index:
    ev = dd.loc["agency", "evidence"]
    _evidence_header(f"{ev['kind'].title()} pattern evidence")
    fig = make_subplots(rows=1, cols=5, subplot_titles=[FEATURES[k][0].replace(" ", "<br>", 1) for k in FEATURES])
    for i, k in enumerate(FEATURES, 1):
        fig.add_trace(go.Bar(x=["This " + ev["kind"], "Peer median"], y=[ev["features"][k], ev["peer_median"][k]], marker_color=[ui.ORANGE if k in ev["top"] else ui.BLUE, "#8896A8"],
                             text=[FEATURES[k][2](ev["features"][k]), FEATURES[k][2](ev["peer_median"][k])], textposition="outside"), row=1, col=i)
    fig.update_layout(height=320, showlegend=False)
    fig.update_annotations(font_size=11)
    st.plotly_chart(fig, width="stretch")
    st.caption("Highlighted bars are the two features that deviate most from the peer median.")

if "delay" in dd.index:
    ev = dd.loc["delay", "evidence"]
    _evidence_header("Process delay evidence")
    m = st.columns(3)
    m[0].metric("Sanction delay (days)", "n/a" if ev["sanction_delay_days"] is None else f"{ev['sanction_delay_days']:.0f}", f"guideline threshold {ev['threshold']}", delta_color="off")
    m[1].metric("Completion time (days)", "n/a" if ev["completion_days"] is None else f"{ev['completion_days']:.0f}")
    m[2].metric("Peer 90th percentile (days)", "n/a" if ev["peer_p90"] is None else f"{ev['peer_p90']:.0f}")

st.markdown("#### Review")
with st.form("review_form", clear_on_submit=True):
    note = st.text_area("Reviewer note", placeholder="What did you check? What did you conclude?")
    if st.form_submit_button("Mark as reviewed", type="primary"):
        review.add(wid, note or "(no note)")
        st.success("Saved to the local review log. It now shows in the Flagged Works table.")
for ts, n in review.history(wid):
    st.markdown(f"- `{ts}` {html.escape(n)}")
ui.footer(ctx)

import pandas as pd
import plotly.express as px
import streamlit as st
import ui

ctx = ui.setup("Overview", "📊")
st.markdown("## Overview")
d = ui.flagged_table(ctx)
fl = d[d.flagged]
if d.empty:
    st.warning("No data loaded. Open the Data page to generate or upload works.")
    st.stop()
amt = fl["sanctioned_amount"].sum()
q = ctx.works["quality_score"].mean()
cols = st.columns(5)
for col, (l, v) in zip(cols, [("Works analysed", f"{len(d):,}"), ("Works flagged", f"{len(fl):,}"), ("Flagged share", f"{len(fl) / len(d) * 100:.1f}%"),
                              ("Amount in flagged works", ui.inr(amt)), ("Data-quality score", f"{q:.0f}/100")]):
    col.markdown(ui.kpi(l, v), unsafe_allow_html=True)
st.markdown("&nbsp;")
c1, c2 = st.columns(2)
cnt = pd.DataFrame({"Detector": [ui.DET_LABEL[k] for k in ui.DET_LABEL], "Works": [int((d[f"s_{k}"] > 0).sum()) for k in ui.DET_LABEL]})
f1 = px.bar(cnt, x="Works", y="Detector", orientation="h", title="Works firing each detector (before weighting)", color_discrete_sequence=[ui.BLUE])
f1.update_layout(yaxis=dict(autorange="reversed", title=""), height=320)
c1.plotly_chart(f1, width="stretch")
f2 = px.histogram(d[d.score > 0], x="score", nbins=25, title="Score distribution (works with any signal)", color_discrete_sequence=[ui.NAVY])
f2.update_layout(height=320, xaxis_title="Final score", yaxis_title="Works")
f2.add_vline(x=ctx.cfg["flag_threshold"], line_dash="dash", line_color=ui.ORANGE, annotation_text="flag threshold")
c2.plotly_chart(f2, width="stretch")
by = fl.groupby("district").size().sort_values(ascending=False).head(15).reset_index(name="Flagged works")
f3 = px.bar(by, x="district", y="Flagged works", title="Flagged works by district (top 15)", color_discrete_sequence=[ui.ORANGE])
f3.update_layout(height=340, xaxis_title="", xaxis_tickangle=-40)
st.plotly_chart(f3, width="stretch")
st.markdown("#### Ten highest scores")
top = d.sort_values(["score", "sanctioned_amount"], ascending=False).head(10)
rows = "".join(f"<tr><td>{r.work_id}</td><td>{r.work_category}</td><td>{r.district}</td><td>{ui.inr(r.sanctioned_amount)}</td><td>{r.score:.0f}</td><td>{ui.chip(r.severity)}</td><td>{r.reason}</td></tr>"
               for r in top.itertuples())
st.markdown(f'<div style="overflow-x:auto"><table class="cmp"><tr><th>Work</th><th>Category</th><th>District</th><th>Amount</th><th>Score</th><th>Severity</th><th style="width:40%">Reason</th></tr>{rows}</table></div>', unsafe_allow_html=True)
st.page_link("pages/2_Flagged_Works.py", label="See all flagged works →")
ui.footer(ctx)

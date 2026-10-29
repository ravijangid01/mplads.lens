import json
import plotly.express as px
import streamlit as st
import ui
from detectors import agency
from pipeline import anonymise, run

ctx = ui.setup("Agencies and Districts", "🏛")
st.markdown("## Agencies and districts")


@st.cache_data(show_spinner="Computing aggregates…")
def feats(sig: str, by: str, _works, cfg_json: str):
    return agency.score_entities(agency.entity_features(_works, ctx.cfg, by), ctx.cfg)[0]


kind = st.radio("View", ["Agencies", "Districts"], horizontal=True)
by = "implementing_agency" if kind == "Agencies" else "district"
f = feats(run.signature(), by, ctx.works, json.dumps(ctx.cfg["agency"])).reset_index()
f[by] = f[by].astype(str).map(ctx.maps[by]) if ctx.anon else f[by]
f["Pattern"] = f["flag"].map({True: "Anomalous pattern", False: "Typical"})
fig = px.scatter(f, x="completion_rate", y="mean_delay", size="n_works", color="Pattern", hover_name=by,
                 color_discrete_map={"Anomalous pattern": ui.ORANGE, "Typical": ui.BLUE}, title=f"Completion rate vs average sanction delay ({kind.lower()})")
fig.update_layout(xaxis_title="Completion rate", yaxis_title="Mean sanction delay (days)", xaxis_tickformat=".0%", height=430)
st.plotly_chart(fig, width="stretch")
st.caption("Anomalous means the IsolationForest isolates it and at least one feature is far from the peer median. It is a prompt to look, not a conclusion.")
t = f.sort_values(["score", "max_dev"], ascending=False)
st.dataframe(t[[by, "n_works", "completion_rate", "share_stalled", "mean_delay", "spend_ratio", "mean_cost_z", "score", "explain"]].rename(columns={by: kind[:-1], "explain": "Why flagged"}),
             hide_index=True, width="stretch",
             column_config={"completion_rate": st.column_config.NumberColumn("Completion", format="percent"), "share_stalled": st.column_config.NumberColumn("Stalled", format="percent"),
                            "spend_ratio": st.column_config.NumberColumn("Spend/release", format="percent"), "mean_delay": st.column_config.NumberColumn("Mean delay (d)", format="%.0f"),
                            "mean_cost_z": st.column_config.NumberColumn("Mean cost z", format="%.2f"), "score": st.column_config.NumberColumn("Score", format="%.0f")})
ui.footer(ctx)

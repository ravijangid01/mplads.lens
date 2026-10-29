import json
import pandas as pd
import plotly.express as px
import streamlit as st
import ui
import yaml
from pipeline import clean, ingest, run
from pipeline.config import ROOT, USER_DATA, load_config

ctx = ui.setup("Data", "🗂")
st.markdown("## Data")
w = ctx.works
c = st.columns(4)
for col, (l, v) in zip(c, [("Source", "Synthetic demo" if ctx.synthetic else "User-supplied"), ("Rows", f"{len(w):,}"), ("States / districts", f"{w.state.nunique()} / {w.district.nunique()}"),
                           ("Mean data quality", f"{w.quality_score.mean():.0f}/100")]):
    col.markdown(ui.kpi(l, v), unsafe_allow_html=True)
st.markdown("&nbsp;")
a, b = st.columns(2)
fields = [x for x in clean.KEY_FIELDS if x in w]
comp = pd.DataFrame({"Field": fields, "Share present": [float((w[x].notna() & (w[x].astype(str).str.strip() != "")).mean()) for x in fields]})
fig = px.bar(comp, x="Share present", y="Field", orientation="h", title="Field completeness", color_discrete_sequence=[ui.BLUE])
fig.update_layout(height=340, xaxis_tickformat=".0%", yaxis=dict(autorange="reversed", title=""))
a.plotly_chart(fig, width="stretch")
sc = w.status.value_counts().rename_axis("Status").reset_index(name="Works")
b.plotly_chart(px.pie(sc, names="Status", values="Works", hole=.5, title="Works by status", color_discrete_sequence=[ui.BLUE, ui.ORANGE, "#8896A8"]).update_layout(height=340), width="stretch")

st.markdown("### Actions")
x1, x2 = st.columns(2)
with x1:
    if st.button("Rerun the whole pipeline"):
        bar = st.progress(0.0, text="Starting…")
        st.cache_data.clear()
        dets = run.run_all(ctx.works, ctx.cfg, lambda p, t: bar.progress(min(p, 1.0), text=t))
        st.session_state["dets_override"] = (ui.pipeline_key(ctx.meta), dets)
        st.success(f"Pipeline finished: {len(dets):,} detector flags.")
        st.rerun()
with x2:
    if st.button("Regenerate synthetic data"):
        with st.spinner("Generating 20,000 synthetic works…"):
            run.ensure_synthetic(force=True)
            run.set_active("synthetic", load_config()["as_of"])
            st.cache_data.clear()
            st.session_state.pop("dets_override", None)
        st.rerun()
if not ctx.synthetic and st.button("Switch back to synthetic demo data"):
    run.set_active("synthetic", load_config()["as_of"]); st.cache_data.clear(); st.session_state.pop("dets_override", None); st.rerun()

st.markdown("### Upload your own data (CSV or Excel)")
st.caption("Your file stays on this machine or server. It is never sent anywhere else. Real work-level MPLADS exports are not published on the public portal, so bring your own authorised extract.")
up = st.file_uploader("Works file", type=["csv", "xlsx", "xls"])
if up is not None:
    try:
        raw = ingest.read_table(up, up.name)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read that file: {e}")
        st.stop()
    st.write(f"Read **{len(raw):,}** rows and **{len(raw.columns)}** columns.")
    st.dataframe(raw.head(5), width="stretch")
    ym = st.file_uploader("Optional: column-map YAML (see config/column_map.example.yaml)", type=["yaml", "yml"], key="ymap")
    guess, unit0 = ingest.suggest_map(list(raw.columns)), "rupees"
    if ym is not None:
        g2, unit0 = ingest.load_map_yaml(ym.getvalue().decode())
        guess.update({k: v for k, v in g2.items() if v in raw.columns})
    st.markdown("**Column mapping**. Required columns are marked with *.")
    cols = ["— none —"] + list(raw.columns)
    mapping, grid = {}, st.columns(3)
    for i, cn in enumerate(ingest.CANONICAL):
        g = guess.get(cn)
        pick = grid[i % 3].selectbox(cn + (" *" if cn in ingest.REQUIRED else ""), cols, index=cols.index(g) if g in cols else 0, key=f"map_{cn}")
        if pick != "— none —":
            mapping[cn] = pick
    u1, u2 = st.columns(2)
    unit = u1.selectbox("Unit of the amount columns", ["rupees", "lakh", "crore"], index=["rupees", "lakh", "crore"].index(unit0 if unit0 in ("rupees", "lakh", "crore") else "rupees"))
    asof = u2.date_input("Reference date (as-of) for ageing", value=pd.Timestamp.today().date())
    st.download_button("Download this mapping as YAML", yaml.safe_dump({"amount_unit": unit, "columns": mapping}).encode(), "column_map.yaml")
    if st.button("Apply mapping and run", type="primary"):
        d, rep = ingest.apply_map(raw, mapping)
        if rep["missing_required"]:
            st.error("Please map the required columns: " + ", ".join(rep["missing_required"]))
        elif d.empty:
            st.error("No usable rows after validation. " + json.dumps(rep["dropped"]))
        else:
            d = clean.clean(d, unit)
            USER_DATA.parent.mkdir(parents=True, exist_ok=True)
            d.to_csv(USER_DATA, index=False, compression="gzip")
            run.set_active("user", str(asof))
            st.cache_data.clear(); st.session_state.pop("dets_override", None)
            st.session_state["upload_report"] = rep
            st.rerun()
if "upload_report" in st.session_state:
    r = st.session_state["upload_report"]
    st.success(f"Loaded {r['rows_out']:,} of {r['rows_in']:,} rows.")
    dropped = {k: v for k, v in r["dropped"].items() if v}
    if dropped:
        st.warning("Rows dropped: " + "; ".join(f"{v:,} ({k})" for k, v in dropped.items()))
    if r["missing_optional"]:
        st.caption("Optional columns not mapped (detectors that need them will see fewer signals): " + ", ".join(r["missing_optional"]))
ui.footer(ctx)

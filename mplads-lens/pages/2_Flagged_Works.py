import streamlit as st
import ui

ctx = ui.setup("Flagged Works", "🚩")
st.markdown("## Flagged works")
d = ui.flagged_table(ctx)
d = d[d.score > 0]
with st.container(border=True):
    r1 = st.columns(4)
    states = r1[0].multiselect("State", sorted(d.state.dropna().unique()))
    dists = r1[1].multiselect("District", sorted(d.district.dropna().unique()))
    cats = r1[2].multiselect("Category", sorted(d.work_category.dropna().unique()))
    dets = r1[3].multiselect("Detector", list(ui.DET_LABEL), format_func=ui.DET_LABEL.get)
    r2 = st.columns([1, 1, 1.4, 1.6])
    sev = r2[0].multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium"])
    topn = r2[1].selectbox("Show top", [10, 25, 50, 100, 200], index=0)
    lo, hi = r2[2].slider("Score range", 0, 100, (int(ctx.cfg["flag_threshold"]), 100))
    q = r2[3].text_input("Search", placeholder="work id, category, reason…")
m = d.score.between(lo, hi)
for col, sel in [("state", states), ("district", dists), ("work_category", cats), ("severity", sev)]:
    if sel:
        m &= d[col].isin(sel)
for k in dets:
    m &= d[f"s_{k}"] > 0
if q:
    blob = d[["work_id", "work_category", "district", "reason", "work_title"]].astype(str).agg(" ".join, axis=1).str.lower()
    m &= blob.str.contains(q.lower(), regex=False)
f = d[m].sort_values(["score", "sanctioned_amount"], ascending=False)
st.caption(f"{len(f):,} works match the filters. Showing the top {min(topn, len(f))}. Click a row to open its evidence.")
if f.empty:
    st.info("No works match these filters. Try widening the score range or clearing a filter.")
else:
    t = f.head(topn).reset_index(drop=True)
    view = t.assign(Amount=(t.sanctioned_amount / 1e5).round(1), Severity=t.severity.map(lambda s: {"High": "🟠 High", "Medium": "🟡 Medium", "Low": "⚪ Low"}[s]),
                    Reviewed=t.reviewed.map({True: "✔ reviewed", False: ""}))[
        ["work_id", "work_category", "district", "Amount", "score", "Severity", "reason", "Reviewed"]].rename(
        columns={"work_id": "Work ID", "work_category": "Category", "district": "District", "score": "Score", "reason": "Reason"})
    ev = st.dataframe(view, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="flag_tbl",
                      column_config={"Amount": st.column_config.NumberColumn("Amount (₹ lakh)", format="%.1f"), "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                                     "Reason": st.column_config.TextColumn(width="large")})
    if ev.selection.rows:
        st.session_state["selected_work"] = t.loc[ev.selection.rows[0], "work_id"]
        st.switch_page("pages/3_Work_Detail.py")
    export = f.drop(columns=[c for c in f.columns if c.startswith("c_")])
    st.download_button("Download filtered results (CSV)", export.to_csv(index=False).encode(), "mplads_lens_flagged.csv", "text/csv")
ui.footer(ctx)

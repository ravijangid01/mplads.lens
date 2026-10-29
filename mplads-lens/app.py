"""MPLADS-Lens landing page."""
import streamlit as st
import ui

ctx = ui.setup("Home", "🔎")
s = ctx.scored
st.markdown("## Explainable outlier detection for MPLADS works data")
st.write("MPLADS-Lens looks across works data for **unusual costs, near-duplicate works, stalled works, unusual agency or district patterns, and process delays**. "
         "Every flag comes with a plain-language reason and the evidence behind it, so a reviewer can decide quickly whether it needs a closer look.")
c = st.columns(4)
fl = s[s.flagged]
for col, (l, v) in zip(c, [("Works analysed", f"{len(s):,}"), ("Works flagged", f"{len(fl):,}"), ("High severity", f"{(s.severity == 'High').sum():,}"),
                           ("Data", "Synthetic demo" if ctx.synthetic else "User-supplied")]):
    col.markdown(ui.kpi(l, v), unsafe_allow_html=True)
st.markdown("&nbsp;")
a, b, c3 = st.columns(3)
with a:
    st.markdown('<div class="card"><b>Start with the overview</b><br>Headline numbers, flags by detector and the ten highest scores.</div>', unsafe_allow_html=True)
    st.page_link("pages/1_Overview.py", label="Open Overview →")
with b:
    st.markdown('<div class="card"><b>Review flagged works</b><br>Filter, read the reason, export CSV and open the evidence.</div>', unsafe_allow_html=True)
    st.page_link("pages/2_Flagged_Works.py", label="Open Flagged Works →")
with c3:
    st.markdown('<div class="card"><b>Bring your own data</b><br>Upload a CSV or Excel export and map its columns.</div>', unsafe_allow_html=True)
    st.page_link("pages/5_Data.py", label="Open Data →")
st.markdown("&nbsp;")
st.info("This is outlier detection for review. Costs vary legitimately with terrain, remoteness, materials and revised scope; see **Method and Limits**.")
ui.footer(ctx)

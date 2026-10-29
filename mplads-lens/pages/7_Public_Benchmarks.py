import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ui

ctx = ui.setup("Public Benchmarks", "📌")
st.markdown("## Public benchmarks")
st.write("The public MPLADS dashboard shows only aggregate figures and per-member allocated limits, not work-level records. "
         "These are the headline totals, captured from the portal on 28 Oct 2026. Verify against the live portal before quoting them. Nothing here uses member names.")
B = pd.DataFrame({"Lok Sabha": [8402.42, 4.06, 114831, 6101.60, 85902, 4534.91, 37918, 1870.00, 3073.88],
                  "Rajya Sabha": [3374.43, 10.45, 26899, 2360.01, 20979, 1825.14, 10590, 829.35, 1325.07]},
                 index=["Allocated limit (₹ Cr)", "Consented for calamity (₹ Cr)", "Works recommended (no.)", "Recommended (₹ Cr)", "Works sanctioned (no.)", "Sanctioned (₹ Cr)",
                        "Works completed (no.)", "Completed (₹ Cr)", "Expenditure on completed and on-going (₹ Cr)"])
st.dataframe(B.style.format("{:,.2f}"), width="stretch")
r = pd.DataFrame({"Lok Sabha": [B.loc["Works sanctioned (no.)", "Lok Sabha"] / B.loc["Works recommended (no.)", "Lok Sabha"], B.loc["Works completed (no.)", "Lok Sabha"] / B.loc["Works sanctioned (no.)", "Lok Sabha"],
                                B.loc["Expenditure on completed and on-going (₹ Cr)", "Lok Sabha"] / B.loc["Allocated limit (₹ Cr)", "Lok Sabha"]],
                  "Rajya Sabha": [B.loc["Works sanctioned (no.)", "Rajya Sabha"] / B.loc["Works recommended (no.)", "Rajya Sabha"], B.loc["Works completed (no.)", "Rajya Sabha"] / B.loc["Works sanctioned (no.)", "Rajya Sabha"],
                                  B.loc["Expenditure on completed and on-going (₹ Cr)", "Rajya Sabha"] / B.loc["Allocated limit (₹ Cr)", "Rajya Sabha"]]},
                 index=["Sanctioned / recommended", "Completed / sanctioned", "Expenditure / allocated limit"])
fig = go.Figure([go.Bar(name=c, x=r.index, y=r[c], text=[f"{v:.0%}" for v in r[c]], textposition="outside", marker_color=col) for c, col in [("Lok Sabha", ui.NAVY), ("Rajya Sabha", ui.BLUE)]])
fig.update_layout(barmode="group", title="Funnel ratios derived from the totals", yaxis_tickformat=".0%", yaxis_range=[0, 1], height=380)
st.plotly_chart(fig, width="stretch")
st.caption("The portal states it displays works recommended online on or after 1 April 2023. The synthetic data follows the same date range. Ratios compare counts or amounts as published and say nothing about any single work.")
ui.footer(ctx)

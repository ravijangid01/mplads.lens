import pandas as pd
import streamlit as st
import ui

ctx = ui.setup("Method and Limits", "📖")
cfg = ctx.cfg
st.markdown("## Method and limits")
st.markdown("**What this is.** MPLADS-Lens is outlier detection for review. We do NOT claim fraud or wrongdoing by any work, agency or person. "
            "No confirmed labels exist, so every flag is a statistical prompt that a human reviewer must interpret.")
st.markdown("### The five detectors")
st.markdown(f"""
1. **Cost outlier.** Each work is compared with *peers*: the same category, district class (plain, hill, desert, tribal, coastal) and sanction year. If that group has fewer than {cfg['cost']['min_peer_size']} works, the peer group widens to category × state, then to category only, and the level used is shown. We take log(cost) and a robust z-score (median and MAD). Upward outliers above **{cfg['cost']['z_threshold']}** are flagged.
2. **Near-duplicate.** Descriptions are lowercased and stripped of numbers, punctuation and common words. Two works in the same block and category are matched when TF-IDF cosine ≥ **{cfg['duplicate']['cosine_min']}**, fuzzy token-set ratio ≥ **{cfg['duplicate']['fuzz_min']}**, and sanction dates are within **{cfg['duplicate']['window_days']}** days. The later work is flagged.
3. **Stalled work.** Not completed, sanctioned more than **{cfg['stalled']['min_sanction_age_days']}** days ago, funds released, and no progress update for more than **{cfg['stalled']['min_no_progress_days']}** days (or none ever).
4. **Agency or district pattern.** Per agency and per district we compute completion rate, share stalled, mean sanction delay, spend-to-release ratio and mean cost z. An IsolationForest (contamination {cfg['agency']['contamination']}, fixed seed) finds unusual entities; we only flag those with a feature at least {cfg['agency']['min_dev_z']} robust deviations from the peer median, and name the top two features.
5. **Process delay.** Sanction later than **{cfg['delay']['sanction_delay_days']}** days after recommendation, or completion time above {cfg['delay']['overrun_factor']}× the peer {cfg['delay']['percentile']}th percentile. *The 75-day figure is a placeholder to verify against the current MPLADS Guidelines.*
""")
st.markdown("### Scoring")
st.write(f"Each detector scores flagged works 0-100. Weights are divided by the largest weight so the heaviest detector alone can reach 100, the scores are summed, "
         f"{cfg['multi_detector_bonus']} points are added when two or more detectors fire, and the result is capped at 100. Severity: High ≥ {cfg['severity']['high']}, Medium {cfg['severity']['medium']}-{cfg['severity']['high'] - 1}, Low below. "
         f"Flagged means a final score of at least {cfg['flag_threshold']}. Current weights: " + ", ".join(f"{ui.DET_LABEL[k]} {v:.2f}" for k, v in ctx.weights.items()) + ".")
st.markdown("### Limitations")
st.markdown("""
- **Legitimate variation is real.** Terrain, remoteness, materials, revised scope and inflation all change costs. Peer groups reduce, but cannot remove, this.
- **Data gaps.** Missing dates or amounts reduce what can be detected. The data has no release date, so stalled timelines show release at sanction.
- **No ground truth.** There are no confirmed outcomes, so thresholds cannot be tuned against real findings.
- **Public data is aggregate.** The public portal publishes totals and per-member limits, not work-level records. This demo runs on synthetic data shaped like the e-SAKSHI schema.
- **Free text.** Anonymisation covers constituency, member, agency, district and location columns, not names that appear inside user-supplied work descriptions.
""")
st.markdown("### Validation against planted anomalies")
if "_planted_type" in ctx.works and ctx.works["_planted_type"].ne("none").any():
    pl = ctx.works.set_index("work_id")["_planted_type"]
    rows = []
    for t, k in [("cost_outlier", "cost"), ("duplicate", "duplicate"), ("stalled", "stalled"), ("agency_pattern", "agency"), ("delay", "delay")]:
        truth, hit = set(pl[pl == t].index), set(ctx.dets[ctx.dets.detector == k].work_id)
        rows.append({"Detector": ui.DET_LABEL[k], "Planted": len(truth), "Flagged": len(hit), "Recall": len(truth & hit) / max(len(truth), 1), "Precision": len(truth & hit) / max(len(hit), 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={"Recall": st.column_config.NumberColumn(format="percent"), "Precision": st.column_config.NumberColumn(format="percent")})
    st.caption("Computed live against hidden labels used only here and in tests. Low precision for stalled and delay is expected: ordinary works also meet those rules, and planted agency works legitimately look stalled too. "
               "On real data there are no such labels.")
else:
    st.info("Validation needs the synthetic data, which carries planted labels. Switch back to it on the Data page.")
ui.footer(ctx)

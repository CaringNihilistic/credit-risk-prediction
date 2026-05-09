import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import joblib
import pandas as pd
import numpy as np

# ── Load model ────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_credit_risk_final.pkl")
model = joblib.load(MODEL_PATH)

# ── Prediction logic ──────────────────────────────────────────
def predict(ext1, ext2, ext3,
            amt_credit, amt_income, amt_annuity, amt_goods,
            age_years, years_employed, cnt_family):

    # Convert human-readable → model format
    days_birth    = -int(age_years * 365)
    days_employed = -int(years_employed * 365) if years_employed > 0 else 365243

    data = {
        "EXT_SOURCE_1":     ext1,        "EXT_SOURCE_2":      ext2,
        "EXT_SOURCE_3":     ext3,        "AMT_CREDIT":        amt_credit,
        "AMT_INCOME_TOTAL": amt_income,  "AMT_ANNUITY":       amt_annuity,
        "DAYS_BIRTH":       days_birth,  "DAYS_EMPLOYED":     days_employed,
        "AMT_GOODS_PRICE":  amt_goods,   "CNT_FAM_MEMBERS":   cnt_family,
    }
    data["CREDIT_INCOME_RATIO"]     = data["AMT_CREDIT"]       / (data["AMT_INCOME_TOTAL"] + 1)
    data["ANNUITY_INCOME_RATIO"]    = data["AMT_ANNUITY"]       / (data["AMT_INCOME_TOTAL"] + 1)
    data["CREDIT_TERM"]             = data["AMT_ANNUITY"]       / (data["AMT_CREDIT"] + 1)
    data["INCOME_PER_PERSON"]       = data["AMT_INCOME_TOTAL"]  / (data["CNT_FAM_MEMBERS"] + 1)
    data["GOODS_PRICE_CREDIT_DIFF"] = data["AMT_CREDIT"]        - data["AMT_GOODS_PRICE"]
    data["GOODS_TO_CREDIT_RATIO"]   = data["AMT_GOODS_PRICE"]   / (data["AMT_CREDIT"] + 1)
    data["AGE_YEARS"]               = age_years
    data["EMPLOYMENT_TO_AGE_RATIO"] = days_employed             / (days_birth + 1)
    data["EXT_SOURCE_MEAN"]         = np.mean([ext1, ext2, ext3])
    data["EXT_SOURCE_MIN"]          = np.min ([ext1, ext2, ext3])
    data["EXT_SOURCE_MAX"]          = np.max ([ext1, ext2, ext3])
    data["EXT_SOURCE_PROD"]         = ext1 * ext2 * ext3
    data["EXT_SOURCE_STD"]          = np.std ([ext1, ext2, ext3])

    df = pd.DataFrame([data])
    df = df.reindex(columns=model.get_booster().feature_names, fill_value=0)
    proba = model.predict_proba(df)[0, 1]

    if proba >= 0.677:
        level, icon, color, msg = "HIGH RISK",   "▲", "#f25c6e", "Applicant is likely to default. Loan approval not recommended."
    elif proba >= 0.40:
        level, icon, color, msg = "MEDIUM RISK", "◆", "#f59e0b", "Borderline applicant. Additional verification is advised."
    else:
        level, icon, color, msg = "LOW RISK",    "●", "#2dd4a0", "Applicant is likely to repay. Loan can be considered."

    bar_w = int(proba * 100)
    return f"""
    <div class="rc">
      <div class="rc-top">
        <div class="rc-left">
          <div class="rc-sublabel">DEFAULT PROBABILITY</div>
          <div class="rc-pct" style="color:{color}">{proba*100:.1f}<span class="rc-unit">%</span></div>
        </div>
        <div class="rc-sep"></div>
        <div class="rc-right">
          <div class="rc-sublabel">ASSESSMENT</div>
          <div class="rc-badge" style="color:{color}">{icon} {level}</div>
          <div class="rc-msg">{msg}</div>
        </div>
      </div>
      <div class="rc-bar-bg">
        <div class="rc-bar-fill" style="width:{bar_w}%;background:{color};"></div>
        <div class="rc-bar-tick" style="left:40%"></div>
        <div class="rc-bar-tick" style="left:67.7%"></div>
      </div>
      <div class="rc-bar-labels">
        <span>0%</span>
        <span style="position:absolute;left:40%">40%</span>
        <span style="position:absolute;left:67.7%">67.7% ▲</span>
        <span style="margin-left:auto">100%</span>
      </div>
      <div class="rc-footer">
        Threshold 0.677 &nbsp;·&nbsp; Kaggle AUC 0.782 &nbsp;·&nbsp;
        XGBoost + Optuna (100 trials × 5-fold CV) &nbsp;·&nbsp;
        Age {int(age_years)} yrs · Employed {int(years_employed)} yrs
      </div>
    </div>
    """


# ── Custom theme (kills Gradio orange) ────────────────────────
theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#fdf8ee", c100="#f5e8c2", c200="#e9cc84", c300="#ddb046",
        c400="#c9a84c", c500="#b8882a", c600="#9c7422", c700="#7a5a18",
        c800="#5c420f", c900="#3d2c08", c950="#1e1604",
    ),
    secondary_hue=gr.themes.Color(
        c50="#f0f4fa", c100="#d8e2f0", c200="#b0c4e0", c300="#88a6d0",
        c400="#6088c0", c500="#4070b0", c600="#3058a0", c700="#204090",
        c800="#102880", c900="#081870", c950="#040c38",
    ),
    neutral_hue=gr.themes.Color(
        c50="#e8edf5", c100="#ccd4e0", c200="#a0aec0", c300="#8fa3c0",
        c400="#6b7fa0", c500="#4a5a72", c600="#2a3a52", c700="#1f2b3e",
        c800="#161c27", c900="#111620", c950="#0a0d12",
    ),
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("DM Mono"), "monospace"],
).set(
    body_background_fill="#0a0d12",
    body_background_fill_dark="#0a0d12",
    body_text_color="#e8edf5",
    body_text_color_dark="#e8edf5",
    block_background_fill="#111620",
    block_background_fill_dark="#111620",
    block_border_color="#1f2b3e",
    block_border_color_dark="#1f2b3e",
    block_label_text_color="#8fa3c0",
    block_label_text_color_dark="#8fa3c0",
    input_background_fill="#0d1018",
    input_background_fill_dark="#0d1018",
    input_border_color="#1f2b3e",
    input_border_color_dark="#1f2b3e",
    input_border_color_focus="#9c7422",
    input_border_color_focus_dark="#9c7422",
    slider_color="#c9a84c",
    slider_color_dark="#c9a84c",
    button_primary_background_fill="linear-gradient(135deg,#b8882a,#e2b84a,#b8882a)",
    button_primary_background_fill_dark="linear-gradient(135deg,#b8882a,#e2b84a,#b8882a)",
    button_primary_background_fill_hover="linear-gradient(135deg,#c9a84c,#f0cc60,#c9a84c)",
    button_primary_text_color="#0a0d12",
    button_primary_text_color_dark="#0a0d12",
    shadow_drop="none",
    shadow_drop_lg="none",
    border_color_primary="#1f2b3e",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

.gradio-container {
  max-width: 980px !important;
  margin: 0 auto !important;
  padding: 0 16px 60px !important;
  background: #0a0d12 !important;
}

/* Section labels */
.sec-label > .prose p {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.67rem !important;
  letter-spacing: 0.16em !important;
  text-transform: uppercase !important;
  color: #c9a84c !important;
  margin: 24px 0 6px 0 !important;
}

/* Input cards */
.icard {
  background: #111620 !important;
  border: 1px solid #1f2b3e !important;
  border-radius: 10px !important;
  padding: 18px !important;
  gap: 14px !important;
}

/* Field labels — short, no-wrap */
.icard label > span,
.icard .gradio-slider label > span,
.icard .gradio-number label > span {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: #c8d4e8 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  display: block !important;
}

/* Info/hint text under inputs */
.icard .gradio-slider .info,
.icard .gradio-number .info,
.icard .info {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.74rem !important;
  color: #4a5a72 !important;
  margin-top: 3px !important;
  line-height: 1.4 !important;
}

/* Number inputs */
.icard input[type="number"] {
  background: #0d1018 !important;
  border: 1px solid #1f2b3e !important;
  border-radius: 7px !important;
  color: #e8edf5 !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 0.92rem !important;
  padding: 9px 12px !important;
  transition: border-color 0.2s !important;
}
.icard input[type="number"]:focus {
  border-color: #9c7422 !important;
  box-shadow: 0 0 0 3px rgba(201,168,76,0.10) !important;
  outline: none !important;
}

/* Sliders — force gold */
input[type="range"] { accent-color: #c9a84c !important; }
input[type="range"]::-webkit-slider-thumb {
  background: #c9a84c !important;
  border: 2px solid #0a0d12 !important;
  box-shadow: 0 0 0 2px #c9a84c !important;
}
input[type="range"]::-moz-range-thumb {
  background: #c9a84c !important;
  border: 2px solid #0a0d12 !important;
}

/* Run button */
#run-btn button {
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.10em !important;
  font-size: 0.88rem !important;
  height: 50px !important;
  border-radius: 9px !important;
  box-shadow: 0 4px 20px rgba(201,168,76,0.20) !important;
  border: none !important;
  transition: opacity 0.2s, transform 0.15s !important;
}
#run-btn button:hover  { opacity: 0.85 !important; transform: translateY(-1px) !important; }
#run-btn button:active { transform: translateY(0) !important; }

/* Result card */
.rc {
  background: #111620;
  border: 1px solid #1f2b3e;
  border-radius: 12px;
  padding: 28px 30px 22px;
  animation: fadeUp .3s ease;
}
@keyframes fadeUp {
  from { opacity:0; transform:translateY(8px) }
  to   { opacity:1; transform:translateY(0) }
}
.rc-top { display:flex; align-items:flex-start; gap:28px; margin-bottom:22px; }
.rc-left { flex-shrink:0; }
.rc-sublabel {
  font-family:'DM Mono',monospace; font-size:0.62rem;
  letter-spacing:0.14em; text-transform:uppercase;
  color:#4a5a72; margin-bottom:6px;
}
.rc-pct {
  font-family:'Syne',sans-serif;
  font-size:3rem; font-weight:800; line-height:1;
}
.rc-unit {
  font-size:1.4rem; font-weight:700;
  vertical-align:top; margin-top:6px; display:inline-block;
}
.rc-sep { width:1px; height:72px; background:#1f2b3e; flex-shrink:0; margin-top:18px; }
.rc-right { flex:1; }
.rc-badge {
  font-family:'Syne',sans-serif;
  font-size:1.1rem; font-weight:700;
  letter-spacing:0.04em; margin-bottom:8px;
}
.rc-msg { font-family:'DM Sans',sans-serif; font-size:0.86rem; color:#8fa3c0; line-height:1.5; }
.rc-bar-bg {
  position:relative; height:6px;
  background:#1a2030; border-radius:4px;
  overflow:visible; margin-bottom:6px;
}
.rc-bar-fill { height:100%; border-radius:4px; transition:width .5s ease; }
.rc-bar-tick {
  position:absolute; top:-3px;
  width:1px; height:12px;
  background:#3a4a62;
}
.rc-bar-labels {
  position:relative; display:flex;
  font-family:'DM Mono',monospace;
  font-size:0.60rem; color:#4a5a72;
  margin-bottom:16px; letter-spacing:0.04em;
  height:14px;
}
.rc-footer {
  font-family:'DM Mono',monospace; font-size:0.66rem;
  color:#3a4a62; letter-spacing:0.04em;
  border-top:1px solid #1a2030;
  padding-top:14px; margin-top:4px;
}

footer, .show-api, .built-with { display:none !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0a0d12; }
::-webkit-scrollbar-thumb { background:#1f2b3e; border-radius:3px; }
"""

HEADER = """
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<div style="padding:28px 0 20px; border-bottom:1px solid #1f2b3e; margin-bottom:2px;">
  <div style="display:flex; align-items:center; gap:16px; margin-bottom:14px;">
    <div style="width:46px;height:46px;border-radius:10px;flex-shrink:0;
      background:linear-gradient(145deg,#b8882a 0%,#e2b84a 100%);
      display:flex;align-items:center;justify-content:center;
      font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;color:#0a0d12;">CR</div>
    <div>
      <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
        color:#e8edf5;letter-spacing:-0.02em;line-height:1;">Credit Risk Prediction</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.70rem;color:#4a5a72;
        letter-spacing:0.12em;margin-top:5px;">
        HOME CREDIT DEFAULT RISK &nbsp;·&nbsp; XGBOOST &nbsp;·&nbsp; OPTUNA</div>
    </div>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;">
    <span style="background:#14190a;border:1px solid #3a3010;border-radius:6px;
      padding:3px 11px;font-size:0.70rem;font-family:'DM Mono',monospace;
      color:#c9a84c;letter-spacing:0.06em;">✦ Kaggle AUC 0.782</span>
    <span style="background:#111620;border:1px solid #1f2b3e;border-radius:6px;
      padding:3px 11px;font-size:0.70rem;font-family:'DM Mono',monospace;
      color:#6b7fa0;letter-spacing:0.06em;">307,511 applicants</span>
    <span style="background:#111620;border:1px solid #1f2b3e;border-radius:6px;
      padding:3px 11px;font-size:0.70rem;font-family:'DM Mono',monospace;
      color:#6b7fa0;letter-spacing:0.06em;">157 features</span>
    <span style="background:#111620;border:1px solid #1f2b3e;border-radius:6px;
      padding:3px 11px;font-size:0.70rem;font-family:'DM Mono',monospace;
      color:#6b7fa0;letter-spacing:0.06em;">Threshold 0.677</span>
  </div>
</div>
"""

PLACEHOLDER = """
<div style="background:#111620;border:1px dashed #1f2b3e;border-radius:12px;
  padding:32px;text-align:center;font-family:'DM Sans',sans-serif;
  color:#3a4a62;font-size:0.88rem;letter-spacing:0.02em;">
  <div style="font-size:1.6rem;margin-bottom:10px;opacity:0.3;">◈</div>
  Set applicant parameters above and click
  <span style="color:#9c7422;font-weight:500;">RUN CREDIT ASSESSMENT</span>
</div>
"""

FOOTER = """
<div style="margin-top:40px;padding-top:20px;border-top:1px solid #1a2030;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;
    font-family:'DM Mono',monospace;font-size:0.74rem;margin-bottom:16px;">
    <div style="padding:8px 0;border-bottom:1px solid #1a2030;">
      <div style="color:#3a4a62;margin-bottom:2px;">VALIDATION AUC</div>
      <div style="color:#e8edf5;font-weight:500;">0.786</div>
    </div>
    <div style="padding:8px 12px;border-bottom:1px solid #1a2030;">
      <div style="color:#3a4a62;margin-bottom:2px;">KAGGLE PUBLIC</div>
      <div style="color:#e8edf5;font-weight:500;">0.782</div>
    </div>
    <div style="padding:8px 12px;border-bottom:1px solid #1a2030;">
      <div style="color:#3a4a62;margin-bottom:2px;">KAGGLE PRIVATE</div>
      <div style="color:#e8edf5;font-weight:500;">0.778</div>
    </div>
    <div style="padding:8px 0;">
      <div style="color:#3a4a62;margin-bottom:2px;">TRAINING SAMPLES</div>
      <div style="color:#e8edf5;font-weight:500;">307,511</div>
    </div>
    <div style="padding:8px 12px;">
      <div style="color:#3a4a62;margin-bottom:2px;">OPTUNA TRIALS</div>
      <div style="color:#e8edf5;font-weight:500;">100 × 5-fold CV</div>
    </div>
    <div style="padding:8px 12px;">
      <div style="color:#3a4a62;margin-bottom:2px;">CLASS IMBALANCE</div>
      <div style="color:#e8edf5;font-weight:500;">scale_pos_weight 11.39</div>
    </div>
  </div>
  <div style="font-family:'DM Sans',sans-serif;font-size:0.76rem;color:#3a4a62;">
    Built by
    <a href="https://github.com/CaringNihilistic" style="color:#9c7422;text-decoration:none;font-weight:500;">Ayush Yadav</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/CaringNihilistic/credit-risk-prediction" style="color:#9c7422;text-decoration:none;">GitHub ↗</a>
    &nbsp;·&nbsp; B.Tech CS · HBTU Kanpur · 2023–27
  </div>
</div>
"""

# ── Build app ──────────────────────────────────────────────────
with gr.Blocks(theme=theme, css=CSS, title="Credit Risk Prediction") as demo:

    gr.HTML(HEADER)

    # ── Section 1: External Credit Scores ────────────────────
    gr.Markdown("External Credit Scores", elem_classes=["sec-label"])
    with gr.Group(elem_classes=["icard"]):
        with gr.Row():
            ext1 = gr.Slider(0.0, 1.0, value=0.50, step=0.01,
                label="External Credit Score 1  (EXT_SOURCE_1)",
                info="Normalized score from a third-party credit bureau (e.g. CIBIL/Experian). "
                     "0 = very high default risk · 1 = very low risk. Source identity not disclosed by Home Credit.")
            ext2 = gr.Slider(0.0, 1.0, value=0.55, step=0.01,
                label="External Credit Score 2  (EXT_SOURCE_2)",
                info="Score from a second external agency. The single most predictive feature in this model — "
                     "small changes here have the largest impact on the prediction.")
            ext3 = gr.Slider(0.0, 1.0, value=0.60, step=0.01,
                label="External Credit Score 3  (EXT_SOURCE_3)",
                info="Score from a third external source. The model combines all three scores (mean, min, max, "
                     "product, std) — together they account for the top 5 features by importance.")

    # ── Section 2: Financial Details ─────────────────────────
    gr.Markdown("Financial Details", elem_classes=["sec-label"])
    with gr.Group(elem_classes=["icard"]):
        with gr.Row():
            amt_credit  = gr.Number(value=500_000, label="Loan Amount",
                info="Total credit amount applied for (₹)")
            amt_income  = gr.Number(value=150_000, label="Annual Income",
                info="Applicant's total annual income (₹)")
            amt_annuity = gr.Number(value=25_000,  label="Annual Annuity",
                info="Yearly loan repayment amount (₹)")
            amt_goods   = gr.Number(value=450_000, label="Goods Price",
                info="Price of the goods the loan is financing (₹)")

    # ── Section 3: Personal Details ──────────────────────────
    gr.Markdown("Personal Details", elem_classes=["sec-label"])
    with gr.Group(elem_classes=["icard"]):
        with gr.Row():
            age_years      = gr.Number(value=33,  label="Age",
                info="Applicant's age in years (18–70)")
            years_employed = gr.Number(value=5,   label="Years Employed",
                info="How many years at current job. Enter 0 if unemployed.")
            cnt_family     = gr.Number(value=2,   label="Family Members",
                info="Total number of family members in household")

    # ── Run button ───────────────────────────────────────────
    btn = gr.Button("RUN CREDIT ASSESSMENT", variant="primary", elem_id="run-btn")

    # ── Result ───────────────────────────────────────────────
    gr.Markdown("Assessment Result", elem_classes=["sec-label"])
    result = gr.HTML(value=PLACEHOLDER)

    btn.click(
        fn=predict,
        inputs=[ext1, ext2, ext3, amt_credit, amt_income,
                amt_annuity, amt_goods, age_years,
                years_employed, cnt_family],
        outputs=result,
    )

    gr.HTML(FOOTER)

demo.launch()

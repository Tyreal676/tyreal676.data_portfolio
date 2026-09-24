
# Predicting Loneliness from BRFSS Survey Data

A machine learning project exploring whether self-reported life-circumstances and health
data from the CDC's Behavioral Risk Factor Surveillance System (BRFSS) can predict
feelings of social isolation. This is framed as a loneliness outreach screening detector, 
meant for strategic interception of high risk individuals. 


## Problem Statement

This project began as an attempt to test whether Americans are becoming more socially isolated
using GSS trend data (2004 onward). I found no measurable increase in the measures examined, so I
pivoted to a different question: which factors predict feeling socially isolated, using
BRFSS 2022.

Loneliness and social isolation are associated with meaningful negative health outcomes.
This project asks: can demographic, health-status, and behavioral survey responses predict
self-reported social isolation, and which factors matter most?

This is a portfolio/research exercise built around a hypothetical loneliness screener, not a
validated clinical or public-health tool. Because it uses post-survey data, it illustrates the
approach rather than something that could be run on people who haven't been surveyed.

## Data Source

This project uses the 2022 CDC Behavioral Risk Factor Surveillance System (BRFSS) survey data.

- Source: CDC BRFSS Annual Survey Data (https://www.cdc.gov/brfss/annual_data/annual_2022.html)
- Files used: `LLCP22V1.XPT`, `LLCP22V2.XPT`, `LLCP22V3.XPT`
- Codebook: `USCODE22_LLCP_102523.HTML` (linked on the same page)

**To reproduce:** download the files above into a `Data/` folder and update the
`folder_path` variable in the first cell if needed.

## Setup

pip install -r requirements.txt

## Target Variable

`SDHISOLT` ("How often do you feel socially isolated from others?") was recoded to binary:

- **1 (isolated):** Always / Usually / Sometimes
- **0 (not isolated):** Rarely / Never

This cutoff follows the response scale ("Always/Usually/Sometimes" indicate some isolation,
"Rarely/Never" do not). An exploratory t-SNE of the feature space showed only a weak tendency
for values 1-3 to sit closer together than 4-5, with heavy overlap, so it is consistent with the
cutoff but is not evidence for it. Since the same features are later used for prediction, it
should not be read as independent validation.

## Data Cleaning

BRFSS encodes missing/non-response data as in-band sentinel values (e.g., `7`/`9` or
`77`/`99` for "don't know"/"refused") rather than true nulls, and some columns use `88`
to mean a legitimate zero (e.g., "0 drinks," "no children"). Both were handled explicitly:

- Sentinel codes (`7`/`9`, `77`/`99`, `7777`/`9999`) → `NaN`
- `88`-as-zero codes (`PHYSHLTH`, `POORHLTH`, `CHILDREN`, `AVEDRNK3`) → `0`
- `WEIGHT2`/`HEIGHT3` mix imperial and metric encodings in the same column; metric-coded
  responses (~0.5–0.9% of respondents) were treated as missing rather than unit-converted,
  given the small share affected

An earlier attempt at listwise deletion (dropping any row with a sentinel code across ~48
columns) reduced the dataset from 433K to ~12K rows, due to BRFSS's skip-pattern design
(most respondents are structurally exempt from at least one module question). Per-column
sentinel replacement was used instead, preserving the rows with valid targets and letting XGBoost's
native missing-value handling do the rest.

## Feature Selection

Columns were dropped for two distinct reasons:

**Sparsity** (<15% non-null): `CDHELP`, `CDSOCIAL`, `CRGVLNG1`, `CRGVHRS1`, `HADSEX`,
`CNCRTYP2`, `FIREARM5`, `RCSRLTN2`, `RCSGEND1`

**Structural/content reasons:**
- `PREGNANT` — skip-logic gated to a subset of respondents by sex/age
- `HADHYST2` — missingness pattern is strongly entangled with `SEXVAR` (only asked of
  women), risking the appearance of independent signal that's really riding on sex
- `LSATISFY`, `EMTSUPRT`, `SDHSTRE1` — confirmed via the CDC codebook to share the same
  "Social Determinants and Health Equity" module as the target variable `SDHISOLT`;
  these measure related facets of psychosocial wellbeing rather than independent
  predictors, and their outsized SHAP importance in early modeling was consistent with
  construct overlap rather than genuine independent signal

Nominal (non-ordinal) categorical variables — `MARITAL`, `EMPLOY1`, `RENTHOM1`,
`PRIMINSR`, `DIABETE4`, `COVIDPOS`, `CAREGIV1`, `ACEDIVRC`, `RRCLASS3`, `SOMALE`,
`SOFEMALE` — were one-hot encoded rather than left as raw integers, since their category
codes carry no meaningful order.

## Model

An XGBoost classifier (`scale_pos_weight=1.5`) was trained via a `Pipeline` with
`ColumnTransformer`-based preprocessing (one-hot encoding + passthrough), evaluated with
a standard train/test split.

**Results:**

30% base rate for comparison, 20% held out for testing

| Metric | XGBoost | Logistic Regression (baseline) |
|---|---|---|
| Accuracy | 0.75 | 0.71 |
| ROC-AUC | 0.77 | 0.76 |
| Precision | 0.60 | 0.51 |
| Recall | 0.52 | 0.66 |

XGBoost did not perform meaningfully better than logistic regression on ROC-AUC (0.77 vs.
0.76). The precision/recall difference between the two reflects mismatched
imbalance-correction strength (`scale_pos_weight=1.5` vs. `class_weight='balanced'`), not
a difference in the models themselves. XGBoost was retained because it can be tuned for a
desired precision/recall operating point via `scale_pos_weight`, and because it can handle
missing survey responses natively, without requiring imputation.

### Why precision-heavy?

A precision-heavy operating point was chosen under the assumption that a false positive
(flagging someone as isolated who isn't) is costly — either because outreach capacity is
limited, or because misidentifying someone as lonely carries its own downside. This comes
at a real cost: at `scale_pos_weight=1.5`, the model misses roughly 48% of true positive
cases. A recall-heavy approach would be more appropriate if the downstream intervention
were low-cost and low-stigma (e.g., an advertising campaign), rather than resource-intensive.

The specific operating point (precision 0.60, recall 0.52) was a judgment call about how many
true cases to catch versus how many false alarms to accept, not derived from a quantified cost.

## Interpretability

SHAP was used to investigate feature contributions. SHAP describes what the model relies on, not causal effects.
 Top features included `MENTHLTH`(mentally unhealthy days), `ADDEPEV3` (depression diagnosis), `MARITAL` status,
`GENHLTH` (general health), and `DECIDE` (cognitive/decision difficulty).

SHAP scatter plots on binary/one-hot features (e.g., `MARITAL_1.0`) revealed that a
feature's effect isn't uniform across individuals — logistic regression's single
coefficient per category would average over this variation, while SHAP surfaces it
directly. Identifying the *specific* driver of that variation, however, requires
interaction-level analysis beyond single-feature attribution — see Next Steps.

## Limitations

- Self-reported survey data; both features and target are subject to recall and
  social-desirability bias
- `INCOME3` is ordinal but not equal-interval — bracket widths increase at higher incomes,
  so effects should be read as step changes across brackets, not a linear per-dollar rate
- The precision-heavy operating point is a deliberate, stated choice, not the only
  reasonable one — see "Why precision-heavy?" above
- This is a research/portfolio exercise using a fictional "loneliness interceptor"
  framing, not a validated screening tool
- `SDHISOLT` was deployed in an optional module that was only administered in some states, 
  and state level differences are not accounted for
- Survey weights were not used
- Single split with no confidence interval

## Next Steps

- Use SHAP interaction values (`TreeExplainer.shap_interaction_values`) to identify which
  specific features drive the within-category variation observed in features like
  `MARITAL_1.0`, rather than relying on single-feature attribution alone

## Repo Structure

- `Notebooks/` — main analysis notebook (cleaning → modeling → evaluation → SHAP)
- `Data/` — place downloaded BRFSS `.XPT` files here (not included in repo)
- `models/` — pickled pipeline (preprocessing + trained model)

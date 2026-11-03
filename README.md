# Claim Frequency & Severity Pricing (freMTPL2)

Pure premium models for French motor third-party liability insurance: Poisson and gamma GLMs
against LightGBM, served through a FastAPI backend and a Next.js UI where analysts compare
segments. A local LLM drafts plain language segment notes, and a deterministic checker rejects
any note whose numbers or claims do not match the model output.

**Live UI (holdout snapshot):** https://abhayjuloori.github.io/claim-frequency-severity-fremtpl2/

## Data

[freMTPL2freq](https://www.openml.org/d/41214) and [freMTPL2sev](https://www.openml.org/d/41215)
(CASdatasets, via OpenML): 678,013 policies, 358,360 policy-years of exposure, 26,639 claim amounts.

Cleaning rules (counts from the run):

| Rule | Rows |
|---|---|
| Policies reporting claims with no amount in the severity table -> set to 0 claims | 9,116 |
| Claim amounts whose policy is not in the frequency table (dropped by the join) | 6 policies |
| ClaimNb above 4 capped at 4 | 8 |
| Exposure above 1 year capped at 1 | 1,224 |
| Claim amounts above 200,000 capped | 20 |

36,102 reported claims become 26,406 usable claims. The no-amount rule follows the
scikit-learn reference study on this dataset; it is the biggest modeling choice here and is
counted, not hidden.

## Method

- **Split:** one row per policy, 75/25 random holdout (seed 0): 508,509 train, 169,504 test. LightGBM early stopping uses a 10% slice of train only.
- **Frequency:** target ClaimNb / Exposure with Exposure as sample weight (same likelihood as a Poisson model on counts with a log-exposure offset).
  GLM: one-hot brand, power, fuel, region, area; 10 quantile bins of vehicle and driver age; scaled bonus-malus and log density.
  LightGBM: raw features, native categoricals, `objective=poisson`.
- **Severity:** average claim amount on policies with claims, weighted by claim count. Gamma GLM vs LightGBM `objective=gamma`.
- **Pure premium** = frequency x severity.
- **Metrics on the holdout:** mean Poisson deviance (and D2 vs a mean-only model), gamma deviance, Gini from the ordered Lorenz curve (exposure weighted), decile calibration, total predicted vs observed losses.

## Results (holdout, 169,504 policies)

| Frequency | Poisson deviance | D2 | Gini |
|---|---|---|---|
| Mean only | 0.4765 | 0 | 0 |
| Poisson GLM | 0.4562 | 0.043 | 0.277 |
| LightGBM | **0.4461** | **0.064** | **0.323** |

| Severity | Gamma deviance | D2 |
|---|---|---|
| Mean only | 1.4009 | 0 |
| Gamma GLM | 1.3994 | 0.001 |
| LightGBM | 1.4001 | 0.001 (early stopping kept 1 tree) |

| Pure premium | Gini | Predicted / observed losses |
|---|---|---|
| GLM | 0.306 | 1.009 |
| LightGBM | **0.342** | 0.998 |

- LightGBM beats the GLM on frequency by a modest margin, in line with the scikit-learn study on the same data.
- **Severity is close to unpredictable from these rating factors**: both models barely beat the mean. LightGBM's early stopping kept a single tree. Almost all of the pure premium ranking comes from frequency.
- Decile calibration (frequency, observed vs predicted): LightGBM top decile 0.211 vs 0.211, bottom 0.029 vs 0.028. The GLM overpredicts the second decile (0.044 vs 0.040) and underpredicts the ninth (0.100 vs 0.110).
- Segment view: the GLM underpredicts drivers aged 18-20 (0.155 vs observed 0.203 claims per year) because quantile bins merge them with older drivers; LightGBM gives 0.211.

## Segment notes and the checker

`src/pricing/notes.py` sends each segment table to a local model (llama3.1:8b via Ollama by
default; any OpenAI-compatible endpoint via `LLM_PROVIDER=openrouter`) and asks for a three
sentence note. The checker then requires:

1. every number in the note to appear in the table, **in the row of the segment the sentence names** (so numbers cannot be swapped between rows; invented percentages fail);
2. a sentence claiming the highest or lowest relativity to name the true segment;
3. a claim that the model is above or below observed to match the row's direction.

Up to three drafts per table. Result of the run: **4 of 6 notes verified** (all four checked by hand
as correct); **2 rejected** on every try (driver age and region: the model named the wrong lowest
segment). Rejected notes are not shown; the UI shows the table only. An earlier version of the
checker that only matched numbers to the table passed notes with reversed directions and the
wrong segment, which is why the row binding and direction checks exist.

## App

- `api/main.py` (FastAPI): `GET /metrics`, `GET /segments/{dimension}`, `GET /segments-all`, `GET /notes`, `POST /predict` (validated policy fields -> GLM and LightGBM frequency, severity, pure premium).
- `web/` (Next.js, static export): metrics, a segment comparison table by rating factor, the verified note, and a pricing form when `NEXT_PUBLIC_API_URL` points at the API. The GitHub Pages build reads an exported holdout snapshot instead of the API.

## Limitations

- One country, one portfolio, one year of exposure per row; no time split is possible because the data has no dates.
- Severity features are weak; a real severity model would need claim type and injury data.
- Dropping 9,116 claims with no amount understates frequency in absolute terms (the reference study makes the same choice).
- The notes checker verifies numbers, ranking claims and direction. It does not verify every possible statement (for example, sentences that compare two segments at once are not direction checked).

## How to run

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
export PYTHONPATH=src:.
.venv/bin/python -c "from pricing.data import download; download()"   # OpenML, ~30 MB
.venv/bin/python -m pricing.models          # trains 4 models, artifacts/metrics.json (~15 s)
.venv/bin/python -m pricing.segments        # artifacts/segments.json
ollama pull llama3.1:8b && .venv/bin/python -m pricing.notes   # artifacts/notes.json
.venv/bin/python -m pricing.export_web      # snapshot for the static UI
.venv/bin/uvicorn api.main:app --port 8000
cd web && npm install && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
.venv/bin/python -m pytest -q               # 16 tests: cleaning rules, metrics, checker, API
```

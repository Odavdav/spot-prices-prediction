# French day-ahead power price forecasting

Hourly forecasts of the French day-ahead (SDAC) electricity price, built only from information
available before the auction closes at 12:00 on D-1. The project measures **what each block of market
information is worth** (calendar → price history → fundamentals → fuels and neighbours), tests the
differences statistically, and compares a gradient-boosting model with the literature's linear
reference (LEAR) and a zero-shot time-series foundation model (Chronos-2).

**Headline results (test year, Nov. 2025 – Oct. 2026, run once):**

- The main model (LightGBM with fundamentals, gas and CO₂ costs and German data, retrained daily)
  reaches an MAE of **17.3 €/MWh**, i.e. **44 %** of the naive "same hour last week" error.
- **Against the literature's linear benchmark (LEAR), the information matters more than the model:**
  given the same inputs and the same daily retraining, LightGBM beats LEAR by only **0.9 €/MWh**
  (Diebold-Mariano p = 0.02), and almost all of it on negative-price hours. Giving LEAR the gas, CO₂ and
  German data is worth 2.3 €/MWh on its own.
- **Retraining frequency matters as much as model choice:** daily instead of monthly retraining
  cuts the error by 2.5 €/MWh.
- **A zero-shot foundation model given the same fundamentals does better still** (Chronos-2:
  14.4 €/MWh, p < 0.001), except on negative-price hours, where the two are level.
- **Post-test extension:** borrowing Chronos-2's ingredients brings LightGBM down to **15.6 €/MWh**
  (daily retraining). Those ingredients are recent-error correction, recency weights, and day-shape and
  regime features. That closes most of the gap, but Chronos-2 stays significantly ahead (p = 0.006).

![Models at equal information](results/figures/04_models.png)

---

## Contents

1. [Questions](#questions)
2. [Data and information set](#data-and-information-set)
3. [Method](#method)
4. [Results](#results)
5. [Post-test extension: borrowing from Chronos-2](#post-test-extension-borrowing-from-chronos-2)
6. [What drives the forecasts](#what-drives-the-forecasts)
7. [Market takeaways](#market-takeaways)
8. [Limitations](#limitations)
9. [Reproducing the results](#reproducing-the-results)
10. [Acknowledgements](#acknowledgements)

---

## Questions

| Hypothesis (set before modelling) | Verdict |
|---|---|
| Fundamentals, above all residual load, bring most of the gain beyond price lags, especially on negative-price hours and peaks. | **Confirmed.** Adding French fundamentals cuts the error by 18 % (validation) to 20 % (test); on negative hours by 27 % on validation but only 13 % on the test. |
| At a one-day horizon, gas and CO₂ add little in a stable regime but matter as a *scale* when prices change regime, hence normalising the target by the gas-plant cost (SRMC). | **Confirmed.** Normalising changes nothing in calm years (p = 0.73) but cuts the error by 17 % in the 2021–2022 gas crisis (p < 0.001). |
| A "universal" model that only sees past prices (MSTL, Chronos) stays below the model with fundamentals. | **True for price-only models, false once the foundation model gets the fundamentals.** |

## Data and information set

**Golden rule:** every input for delivery day D must be published before the SDAC auction closes at
**12:00 (Paris time) on D-1**. No realised data (actual generation, physical flows) is ever used.

| Data | Source | Known at decision time |
|---|---|---|
| French day-ahead price (target and lags) | ENTSO-E Transparency (`entsoe-py`) | up to D-1 |
| German-Luxembourg day-ahead price | ENTSO-E | up to D-1 |
| Load forecast D-1, France and Germany | ENTSO-E | before 10:00 on D-1 |
| Wind and solar forecast D-1, France and Germany | ENTSO-E | assumed before 12:00 on D-1 (see [limitations](#limitations)) |
| TTF gas, front month | Yahoo Finance (`TTF=F`) | last close dated ≤ D-2 |
| EUA carbon, December contract | Investing.com (manual CSV, not redistributed) | last close dated ≤ D-2 |

- **Period:** 1 Oct. 2018 (creation of the DE-LU bidding zone) to 20 Nov. 2026, 71,353 hours.
- **Resolution:** hourly averages, because the European day-ahead market moved to 15-minute products on 1 Oct. 2025.
- **Data is not versioned.** The scripts rebuild it from the APIs (see [reproducing](#reproducing-the-results)).

**Leakage control:** a register declares when each raw input is published and which inputs each
feature uses. `tests/test_no_leakage.py` checks the declarations and runs a "time-travel" test. It
erases everything published after the auction closes and checks that the features for day D do not
change, around tricky days (clock changes, Easter). A deliberately planted leak makes it fail.

## Method

**Features (`src/features.py`).**
- Calendar: hour, weekday, month, French holidays, bridge days, clock changes.
- Price lags: same local hour on D-1, D-2 and D-7; mean, min and max of D-1.
- French fundamentals: load, wind and solar forecasts, residual load (load − wind − solar), renewables share.
- Fuels and neighbours:
  - CCGT short-run marginal cost: SRMC = TTF / 0.55 + EUA × 0.37 + 2 €/MWh.
  - German price D-1, German residual load.

**Ablation: one main model (LightGBM, fixed hyperparameters), information added step by step.**

| Step | Adds |
|---|---|
| A1 | calendar |
| A2 | + price lags |
| B1 | + French fundamentals |
| B2 | + gas, CO₂ (via the SRMC) and Germany; target = price − SRMC |

**Benchmarks (`src/benchmarks.py`).**
- **Naive forecasts** D-1 and D-7. D-7 is the reference for the relative MAE (rMAE).
- **MSTL:** daily and weekly seasonal decomposition of prices only.
- **LEAR** (Lago et al., 2021, *Applied Energy* 293): one LASSO regression per hour, averaged over 56, 84, 1,092 and 1,456-day windows, re-estimated monthly, weekly or daily. Standard LEAR has the same information as B1 (prices and French fundamentals).
- **LEAR-B2:** the same model with B2's information added (German price D-1, German load and renewables forecasts, SRMC), re-estimated daily. It isolates the model effect from the information effect.
- **Chronos-2** (Amazon, Oct. 2025): zero-shot, with prices only or with the same fundamentals as B2.

**Protocol (`src/backtest.py`).**
- **Walk-forward with an expanding window:** each retraining uses every delivery before the retraining date.
- **Validation** (Nov. 2024 – Oct. 2025) for every modelling choice; **test** (Nov. 2025 – Oct. 2026) run once at the end; plus a **crisis** robustness period (Jul. 2021 – Dec. 2022).
- **Metrics:** MAE and rMAE, overall and by regime: negative-price hours, evening peak (18:00–22:00), other hours. No MAPE, since prices cross zero.
- **Significance:** Diebold-Mariano test on daily-averaged absolute-error differences, with Newey-West (HAC) variance and comparison pairs fixed in advance.

## Results

Each result below answers one question and varies **one thing at a time**: the information, the
retraining pace, the model or the target. Full tables (every model, every period, every
Diebold-Mariano test) are in [`results/`](results/).

**Summary (MAE in €/MWh; rMAE vs naive D-7 in brackets):**

| Model | Information | Updated | Validation | Test | Test, negative hours | Test, evening peak |
|---|---|---|---|---|---|---|
| Naive D-7 | last week's price | – | 32.4 | 39.7 | 30.6 | 42.3 |
| LEAR | B1 (no gas, no Germany) | daily | 15.0 | 20.5 (0.52) | 24.7 | 19.4 |
| LEAR-B2 | B2 | daily | 14.2 | 18.2 (0.46) | 22.1 | 18.1 |
| **LightGBM B2 (main model)** | B2 | daily | **13.5** | 17.3 (0.44) | **18.2** | 18.3 |
| LightGBM final (post-test extension) | B3 + error correction | daily | 12.4 | 15.6 (0.39) | 22.9 | **15.7** |
| **Chronos-2 + fundamentals** | B2 | zero-shot | not run | **14.4 (0.36)** | 19.6 | 15.8 |

Chronos-2 was released on 17 Oct. 2025, after the validation year had started, so it may have seen
those prices during pre-training. It was therefore run on the test year only, which it cannot have seen.

### 1. What each block of information is worth

![Information blocks](results/figures/02_information.png)

Same model (LightGBM), same weekly retraining; one block of information added at each step.

- **Price lags** do most of the work: calendar effects alone are worse than the naive forecast.
- **French fundamentals** cut the error by a further 18 % (validation) to 20 % (test), p < 0.001 on both. On negative-price hours the gain is 27 % on validation but only 13 % on the test: less stable than the overall gain.
- **Gas, CO₂ and Germany** add a smaller but significant 1.35 (validation, p < 0.001) and 0.95 €/MWh (test, p = 0.03), mostly at the evening peak, when gas plants set the price.

### 2. How often to retrain

![Retraining frequency](results/figures/03_retraining.png)

Same information (B2 for LightGBM, B1 for LEAR); only the retraining pace changes.

- Updating often helps both model families, with diminishing returns. For LightGBM, weekly instead of monthly is worth 1.76 €/MWh on the test and daily instead of weekly another 0.78 (both p < 0.001).
- LEAR, whose short calibration windows go stale, collapses when re-estimated only monthly.

### 3. Which model, at equal information

See the figure at the top of the page. Every model is updated daily (MSTL and Chronos-2 re-read the latest prices at each forecast).

| Comparison (test year) | MAE gap (€/MWh) | p-value |
|---|---|---|
| LightGBM vs LEAR, both with B2's information | −0.90 | 0.02 |
| LEAR with vs without gas, CO₂ and Germany | −2.30 | < 0.001 |
| Chronos-2 with vs without fundamentals | −5.41 | < 0.001 |
| LightGBM vs Chronos-2, both with B2's information (chosen after seeing the results) | +2.87 | < 0.001 |

- **With prices only,** the foundation model (Chronos-2) clearly beats the classical decomposition (MSTL), most of all on negative-price hours.
- **With the same information, LightGBM beats LEAR by only 0.9 €/MWh**, and almost all of it on negative-price hours (18.2 vs 22.1). The two are level at the evening peak. The information matters more than the model: gas, CO₂ and German data cut LEAR's error by 2.3 €/MWh.
- **Chronos-2 with fundamentals is the best model overall** (14.4), but not on negative-price hours, where LightGBM is better (18.2 vs 19.6).

### 4. Normalising the target by the gas-plant cost

![Normalisation](results/figures/05_normalisation.png)

Same model, same information (B2); only the target changes: raw price, price − SRMC, or price − w(RL) × SRMC,
where w is the pass-through of gas cost to the French price by residual-load decile (0.7 to 1.5).

- **Crisis (Jul. 2021 – Dec. 2022):** predicting price − SRMC cuts the error from 38.0 to 31.4 €/MWh (p < 0.001). The gain grows with the gas price, reaching 16 €/MWh in Q3 2022.
- **Calm years:** no difference (monthly retraining: validation p = 0.73, test p = 0.09).
- **Negative-price hours are the exception:** on the test year the raw-price target does better there (17.4 vs 18.8 €/MWh, weekly retraining); the two are level on validation. When the price is negative, gas is not setting it, so subtracting the gas cost adds noise.
- **Weighting by residual load** adds nothing on any period (p = 0.61 to 0.74): LightGBM already learns that interaction.

### 5. One test week

![Test week](results/figures/07_test_week.png)

The models track the daily profile closely. Chronos-2 is closest on most days but overshoots the
midday dip on Friday 8 May, forecasting −50 €/MWh against an actual price of about 0. That
illustrates why its edge disappears on negative-price hours.

## Post-test extension: borrowing from Chronos-2

After seeing Chronos-2 win on the test year, we tried to give LightGBM the ingredients that probably
explain its edge. Each idea was developed and **chosen on the validation year only**, then run once on
the test year with comparisons fixed in advance. Because the idea itself came from a test result,
this section is reported separately from the main results.

| Chronos-2 behaviour | LightGBM counterpart |
|---|---|
| Re-learns from the last weeks at every forecast | **Recency weights:** a training row's weight halves every 90 days (half-life chosen among 90, 180 and 365 days on validation). |
| Corrects itself quickly | **Recent-error correction:** a second-stage median regression predicts today's error from yesterday's error and the last 7 days' errors. Yesterday's error is known at 12:00 on D-1, since D-1 prices are published on D-2. Tested for leakage. |
| Sees the whole day and the current regime | **Step B3 = B2 +** the shape of day D (residual-load minimum, maximum, mean and trough hour; each hour's position in the day; solar peak) **+** regime features (price − SRMC gap, volatility and negative-hour share over 14 days; price sensitivity to residual load over 28 days). |

![Extension](results/figures/06_extension.png)

- **Error correction is the robust idea:** −1.16 €/MWh on the main model (p < 0.001), with negative-price hours unchanged.
- **Day-shape and regime features help** a little more (B3 vs B2 at the same pace: −0.69 €/MWh with weekly retraining, p = 0.003).
- **Recency weights backfired on negative-price hours** (22.9 against 17.6 €/MWh without them). A 90-day half-life makes the model forget earlier springs, which is precisely when negative prices happen. They looked harmless on validation.
- **The final model beats the main model by 1.7 €/MWh** (p < 0.001), but **Chronos-2 stays ahead by 1.2 €/MWh** (p = 0.006).

**Choosing the final version.** Daily retraining was too slow to run for every variant (about 2.5 hours
per year of backtest), so only the two best weekly versions were rerun daily. The rule was written down
before the runs: keep the daily version with the lowest validation MAE on Feb.–Oct. 2025 (the error
correction needs a 90-day history first) and report its test figure, whatever it is. Recency weights
won on validation (12.33 against 13.15), so they are in the final model, even though the version
without them did better on the test (15.87, and 17.6 on negative-price hours).

**The lesson:** adapting fast to recent data helps ordinary hours but is risky for rare, seasonal
events. A foundation model that has seen many regimes handles that trade-off better than a
single-market model.

## What drives the forecasts

The SHAP values below are for B2, trained on all data before the test and explained on the test hours
(`src/importance.py`). They explain the spread to the gas-plant cost (the model's target). The SRMC
level itself is added back afterwards.

![Feature importance](results/figures/08_importance.png)

- **Yesterday's price at the same hour** dominates (16 €/MWh of average contribution). Prices are persistent from one day to the next.
- **French and German residual load** come next, ahead of the raw load, wind and solar forecasts. The model reasons in "what is left for dispatchable plants to cover", across the coupled FR-DE zone. Solar alone looks minor because residual load and the renewables share already carry its effect.

![Importance by regime and hour](results/figures/09_importance_regime_hour.png)

- **Negative-price hours** are driven more by residual load (FR and DE: 16 % each, against 11 % at the evening peak). Calendar effects matter much less (3 % against 7 %). Prices collapse when renewables exceed demand, whatever the day.
- **Evening hours (19:00–21:00)** lean most on yesterday's price, because the evening peak is the most repetitive part of the day.
- **Midday hours** (10:00–17:00) give more weight to the renewables share and residual load, which is the solar effect.

The merit-order picture behind these results:

![Merit order](results/figures/01_merit_order.png)

## Market takeaways

1. **Fundamentals beat history, especially when prices go negative.** Residual load forecasts cut the error by about a fifth on top of price lags. On negative-price hours the gain was a quarter on validation but only 13 % on the test year, so it is less stable than the overall gain.
2. **Gas increasingly fails to set the French price.** The average spread between price and CCGT cost went from +1–6 €/MWh (2018–2022) to −35 €/MWh (2024–2026), as nuclear recovered and solar grew. Normalising by gas cost protects against fuel shocks, but tells less and less about everyday prices, and on the test year it even hurt on negative-price hours, where gas plays no role.
3. **Model freshness is a first-order choice.** Retraining daily instead of monthly is worth 2.5 €/MWh on the test year and costs about 8 seconds a day in production.
4. **Linear models are competitive in normal hours; non-linearity earns its keep around renewables thresholds.** With the same information and daily retraining, LightGBM beats LEAR by 0.9 €/MWh on the test (p = 0.02) and 0.8 on validation (p = 0.07). Its edge comes from negative-price hours (18.2 vs 22.1 €/MWh); LEAR is level at the evening peak (18.1 vs 18.3). The information mattered more: gas, CO₂ and German data cut LEAR's error by 2.3 €/MWh.
5. **In-context foundation models are a serious contender.** Fed the same fundamentals, a zero-shot model, never trained on this market, beat a gradient-boosting model trained on seven years of French data by 16 %. It adapts to the latest weeks of data at every forecast, but it can overshoot extreme renewables hours.
6. **Correcting yesterday's bias is the cheapest win; forgetting the past is the riskiest.** A recent-error correction is worth 1.2 €/MWh. Down-weighting old data helps ordinary hours, but it erases the seasonal memory needed for negative prices.

## Limitations

- **Wind and solar forecasts:** EU rules only require D-1 forecasts by 18:00 on D-1. We assume they are available at 12:00, as is standard practice in the literature; ENTSO-E keeps only the latest version.
- **Gas and carbon proxies:** free data only.
  - TTF front month from Yahoo and the EUA December contract from Investing.com.
  - Contract rolls create small jumps: about 1 €/MWh on the SRMC for EUA; a few TTF rolls are visible, e.g. Sept. 2019.
  - Closes are taken at D-2, which is conservative: traders see intraday TTF on the morning of D-1.
- **Hourly averages** since the move to 15-minute products in Oct. 2025 smooth out part of the intra-hour shape.
- **No nuclear availability.** French nuclear outage notices (UMMs) were left out: the as-announced history is hard to rebuild reliably. They are the most promising missing input, especially in winter.
- **Chronos-2:**
  - Its training data is not published; contamination of the test is ruled out by its release date only.
  - It could not be checked on the validation year.
  - The B2 daily vs Chronos-2 comparison was not fixed in advance.
- **One test year and fixed hyperparameters.** No tuning was done. The negative-hour statistics rest on 513 (validation) and 565 (test) hours.
- **LEAR-B2 and the weekly ablation steps A1 and A2 were added after the test year had been run**, in answer to a review: they isolate the model effect in the LEAR comparison and put the whole ablation at one retraining pace. Their design and comparisons were fixed before they were run, and no choice depends on them.
- **The post-test extension was motivated by a test result.** Each idea was developed and chosen on validation, but the test year has now been used more than once, so its test figures are less clean than the main results.
- The gas pass-through weights w(RL) are estimated before the validation year and **drift over time** (0.26 at low residual load in 2024 against 0.5–0.8 before).

## Reproducing the results

```bash
pip install -r requirements.txt          # on macOS with conda, install PyTorch from conda-forge
cp .env.example .env                     # add your ENTSO-E API key
# download the EUA December futures history (Investing.com) to data/raw/eua_investing.csv
python -m src.get_data                   # ENTSO-E, TTF and EUA -> data/processed/dataset_hourly.parquet
python -m pytest                         # leakage tests
python -m src.backtest                   # validation year
python -m src.backtest --period crisis   # 2021-2022 gas crisis
python -m src.backtest --period test     # test year, run once (about 3 to 4 hours on a laptop CPU)
python -m src.backtest --period burn_in --models B3_weekly_hl90   # error history before validation
python -m src.correction --base B3_weekly_hl90 --period validation --history burn_in
python -m src.backtest --period test --models B3_daily_hl90   # final model, retrained daily
python -m src.correction --base B3_daily_hl90 --period test --history validation
python -m src.importance                 # SHAP importance
python -m src.figures                    # figures
```

```
src/
  get_data.py     download and build the hourly dataset
  features.py     features, ablation steps, publication register
  backtest.py     walk-forward, metrics, Diebold-Mariano
  benchmarks.py   MSTL, LEAR, Chronos-2
  correction.py   recent-error correction (second stage)
  importance.py   SHAP feature importance
  figures.py      figures
tests/            leakage tests
results/          metrics, tests and figures (versioned)
```

## Acknowledgements

Code and analysis were developed with the help of Claude (Anthropic), used as an AI coding assistant.
Research design, data choices and interpretation of the results are my own.

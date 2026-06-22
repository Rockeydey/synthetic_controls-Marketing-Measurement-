# Marketing Incrementality with Multivariate Prophet Synthetic Controls

This repository is designed for estimating **incremental sales lift** and **return on ad spend (ROAS)** for retail media campaigns using a **multivariate synthetic control framework** powered by **Prophet-based time-series components**.

The goal is to estimate a counterfactual:

> “What would treated product sales have been if the campaign had not run?”

---

## Why this approach

Traditional attribution often captures only last-touch effects and can overstate campaign impact. A synthetic control framework improves causal measurement by:

- **Estimating true incrementality** rather than raw attributed revenue.
- **Controlling for confounders** such as seasonality, holidays, and promotion cycles.
- **Reducing bias** via a weighted donor pool of unexposed but similar products/regions.
- **Improving robustness** with multivariate covariates and pre-period fit diagnostics.

---

## Method overview

### 1) Define treated and donor groups

- **Treated unit(s):** SKU, category, brand, region, or store exposed to campaign media.
- **Donor pool:** Comparable units not exposed to the campaign during the same period.
- **Covariates:** Exogenous drivers such as competitor pricing, baseline promotions, calendar/holiday effects, and macro factors.

### 2) Model baseline dynamics

Use Prophet-style decomposition to model:

- Trend (including non-linear growth where appropriate)
- Weekly/yearly seasonality (e.g., Fourier terms)
- Holiday/event effects (e.g., Black Friday, Prime Day, Christmas)

### 3) Build synthetic control

Fit a weighted combination of donor units to match treated unit behavior in the pre-treatment window. The weighted donor blend acts as the **counterfactual baseline** for the treatment period.

### 4) Estimate lift and ROAS

For each post-treatment date $t$:

$$
	ext{Lift}_t = y_t^{\text{treated}} - y_t^{\text{synthetic}}
$$

Aggregate incremental sales:

$$
	ext{Incremental Sales} = \sum_{t \in \text{post}} \text{Lift}_t
$$

ROAS calculation:

$$
	ext{ROAS} = \frac{\text{Incremental Sales}}{\text{Media Spend}}
$$

---

## Suggested project workflow

1. Ingest and validate sales + campaign + covariate data.
2. Split data into pre-treatment and post-treatment periods.
3. Fit Prophet components and synthetic control weights.
4. Evaluate pre-period fit (error diagnostics and placebo checks).
5. Estimate lift, confidence intervals, and ROAS.
6. Export decision-ready reporting tables and visuals.

---

## Repository structure

Current workspace snapshot:

- [README.md](README.md)
- [requirements.txt](requirements.txt)
- [data/](data/)
- [test](test)

> You can extend this into a standard layout (e.g., `src/`, `notebooks/`, `tests/`, `reports/`) as implementation code is added.

---

## Environment setup

1. Create and activate a Python environment.
2. Add required libraries to [requirements.txt](requirements.txt).
3. Install dependencies.

Recommended package families for this project:

- Time series: Prophet
- Causal/synthetic control tooling: CausalPy, AugSynth-equivalent methods
- Data stack: pandas, numpy, scipy
- Visualization: matplotlib, seaborn, plotly

---

## Data requirements

Minimum required fields (example):

- `date`
- `unit_id` (SKU/store/region)
- `sales`
- `treated_flag`
- `media_spend`
- Covariates (promo flags, price index, holiday/event indicators, etc.)

Best practices:

- Keep a consistent daily/weekly grain.
- Ensure donor units are never contaminated by treatment.
- Avoid leakage of post-treatment information into training.

---

## Validation and quality checks

Before trusting uplift estimates, validate:

- Pre-period fit quality (MAPE/RMSE and visual overlap)
- Placebo/permutation tests across donor units
- Sensitivity to donor-pool composition
- Stability across alternate pre-period windows

---

## Status

This repository is currently in **initial setup mode**. The documentation is ready and structured for implementation of the full modeling pipeline.

---

## License

Add a project license file (for example, MIT or internal enterprise license) before distribution.
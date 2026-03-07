# Time Series Analysis — Group 5

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/markshanehaines-ZIG/timeseries-group5/blob/main/notebooks/MAICEN_1125_M5_U1_Group_5_Assignment.ipynb)

**Module 5 · Unit 1 — AI in Project Optimisation, Innovation and Ethics**
**Zigurat Institute of Technology — MAICEN 1125**
**Submitted: 15 March 2026**

End-to-end time series analysis of the PJM West hourly electricity consumption dataset using Python. The notebook covers data cleaning, multi-scale visualisation, seasonality analysis, autocorrelation (ACF/PACF), forecasting with Prophet, and an advanced SARIMA model as a bonus exercise.

---

## 1 · Problem Statement

Electricity grids must balance supply and demand in real time. Understanding consumption patterns — daily peaks, weekly rhythms, annual seasonality — is essential for capacity planning, energy procurement, and infrastructure investment in the AECO sector.

**Our goal:** Analyse 16 years of hourly energy consumption data to identify temporal patterns and build forecasting models that support operational decision-making.

### Problem Framing

| Component            | Detail                                                           |
| -------------------- | ---------------------------------------------------------------- |
| Dataset              | PJM West hourly energy consumption (MW), 2002–2018               |
| Records              | ~143,000 hourly observations                                     |
| Patterns of Interest | Daily (24h), weekly (7d), annual (52w) seasonality               |
| Forecasting Models   | Prophet (Meta), SARIMA (statsmodels)                             |
| Evaluation Metrics   | MAE, RMSE, MAPE on a 52-week holdout test set                    |
| Key Principle        | Models as decision-support tools, not replacements for judgement |

---

## 2 · Dataset

| Item            | Detail                                                                                                         |
| --------------- | -------------------------------------------------------------------------------------------------------------- |
| Source          | [PJM Hourly Energy Consumption — Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) |
| File            | `PJMW_hourly.csv`                                                                                              |
| Columns         | `Datetime`, `PJMW_MW`                                                                                          |
| Raw rows        | 143,206                                                                                                        |
| Date range      | 1 April 2002 – 3 August 2018                                                                                   |
| Cleaning issues | 8 duplicate rows (DST transitions), 30 missing hours                                                           |
| Final cleaned   | 143,232 hourly records (gap-free, linearly interpolated)                                                       |

---

## 3 · Assignment Exercises

| Exercise | Topic                                  | Points |
| -------- | -------------------------------------- | ------ |
| 1        | Data Cleaning & Preprocessing          | 2      |
| 2        | Multi-Scale Visualisation              | 2      |
| 3        | Seasonality Analysis (Daily + Weekly)  | 2      |
| 4        | ACF / PACF Statistical Analysis        | 2      |
| 5        | Forecasting with Prophet               | 2      |
| 6        | **BONUS:** Advanced Modelling — SARIMA | 2      |

> Grading: 5 correct exercises = 10/10. The bonus exercise acts as a wildcard.

---

## 4 · Repository Structure

```
├── README.md                                           ← You are here
├── .gitignore                                          ← Python / Jupyter ignores
├── LICENSE                                             ← MIT License
├── notebooks/
│   └── MAICEN_1125_M5_U1_Group_5_Assignment.ipynb      ← Main notebook (Colab-ready)
├── src/
│   └── data_cleaning.py                                ← Data Cleaning standalone extraction
├── data/
│   └── PJMW_hourly.csv                                ← Raw dataset (from Kaggle)
├── results/
│   └── (generated plots saved here when notebook runs)
└── docs/
    └── contributions.md                                ← Detailed contribution log
```

---

## 5 · How to Run

### Option A: Google Colab (Recommended)

1. Open the notebook from GitHub: [`notebooks/MAICEN_1125_M5_U1_Group_5_Assignment.ipynb`](notebooks/MAICEN_1125_M5_U1_Group_5_Assignment.ipynb)
2. Click **"Open in Colab"** or paste the GitHub URL into [colab.research.google.com](https://colab.research.google.com)
3. Upload `PJMW_hourly.csv` to the Colab session files panel (or mount Google Drive)
4. **Runtime → Run all** — total runtime ~5 minutes (Prophet + SARIMA fitting)

### Option B: Local Setup (VS Code + Jupyter)

1. Clone this repository:
   ```bash
   git clone https://github.com/markshanehaines-ZIG/timeseries-group5.git
   cd timeseries-group5
   ```
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn statsmodels prophet scikit-learn
   ```
4. Open `notebooks/MAICEN_1125_M5_U1_Group_5_Assignment.ipynb` in VS Code with the Jupyter extension
5. Run all cells sequentially

### Option C: Contributing via GitHub

Each team member works on their assigned exercise(s):

1. Clone the repo and create a feature branch:
   ```bash
   git clone https://github.com/markshanehaines-ZIG/timeseries-group5.git
   cd timeseries-group5
   git checkout -b feature/ex1-data-cleaning    # use your exercise name
   ```
2. Make your changes in the notebook
3. Commit and push:
   ```bash
   git add .
   git commit -m "Ex.1: Complete data cleaning pipeline"
   git push origin feature/ex1-data-cleaning
   ```
4. Open a **Pull Request** on GitHub for review before merging into `main`

> **Branch naming convention:** `feature/ex<number>-<short-description>`
> e.g., `feature/ex5-prophet-forecasting`, `feature/ex6-sarima-bonus`

### Option D: Standalone Scripts (uv + python)

There is also an extracted codebase inside `src/`. If you want to use the pipeline logic independently of the Jupyter environments, simply run:

```bash
uv run python src/data_cleaning.py
```

This performs the pre-processing checks outined in Exercise 1.

---

## 6 · Team — Group 5

| Member                     | M5 U1 Role                                   | M4 U4 Role (PPE Detection)       |
| -------------------------- | -------------------------------------------- | -------------------------------- |
| Osama Ata                  | Data Cleaning and Preprocessing (Ex. 1)      | Governance and Presentation Lead |
| Marc Azzam                 | Visualisation and Seasonality (Ex. 2 & 3)    | Error Analysis and Evidence Lead |
| Malak Yaseen               | ACF/PACF Statistical Analysis (Ex. 4)        | Model Training Lead              |
| Letícia Cristovam Clemente | Prophet Forecasting and Tuning (Ex. 5)       | Dataset and Annotation Lead      |
| Mark Shane Haines          | SARIMA Bonus (Ex. 6), Integration and Review | Project Lead                     |

> **Rotation rationale:** Task allocation was deliberately rotated from M4 U4 so that each member gains experience across different stages of the data science pipeline.

---

## 7 · Key Results (Summary)

| Model                               | MAE (MW) | RMSE (MW) | MAPE  |
| ----------------------------------- | -------- | --------- | ----- |
| Prophet (Default)                   | ~383     | ~523      | ~6.7% |
| Prophet (Tuned, conservative trend) | ~371     | ~509      | ~6.5% |
| SARIMA(1,1,1)(1,1,1,52)             | ~370     | ~475      | ~6.5% |

All models evaluated on a 52-week holdout test set (Aug 2017 – Aug 2018).

---

## 8 · Dependencies

| Package      | Version | Purpose                    |
| ------------ | ------- | -------------------------- |
| Python       | 3.10+   | Runtime                    |
| pandas       | 2.0+    | Data manipulation          |
| numpy        | 1.24+   | Numerical operations       |
| matplotlib   | 3.7+    | Plotting                   |
| seaborn      | 0.12+   | Plot styling               |
| statsmodels  | 0.14+   | ACF/PACF, SARIMA, ADF test |
| prophet      | 1.1+    | Meta's forecasting library |
| scikit-learn | 1.3+    | MAE / RMSE metrics         |

---

## 9 · License

This project is licensed under the **MIT License** — see `LICENSE` for details.

**Dataset:** [PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) by Rob Mulla, released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain).

---

## 10 · References

- PJM Interconnection LLC — [pjm.com](https://www.pjm.com/)
- Taylor, S.J. & Letham, B. (2018). _Forecasting at Scale._ The American Statistician, 72(1), 37–45.
- Hyndman, R.J. & Athanasopoulos, G. (2021). _Forecasting: Principles and Practice_ (3rd ed.). OTexts.
- Box, G.E.P., Jenkins, G.M. & Reinsel, G.C. (2015). _Time Series Analysis: Forecasting and Control_ (5th ed.). Wiley.
- statsmodels — [statsmodels.org](https://www.statsmodels.org/)
- Prophet — [facebook.github.io/prophet](https://facebook.github.io/prophet/)

---

**Group 5** — Zigurat Institute of Technology, MAICEN 1125

# Contribution Log — Group 5, M5 U1

## Task Rotation Rationale

For M5 U1, we deliberately rotated responsibilities from M4 U4 (PPE Detection) so that each member gains hands-on experience across different stages of the data science pipeline. The table below shows the rotation:

| Member | M4 U4 (PPE Detection) | M5 U1 (Time Series) |
|--------|----------------------|---------------------|
| Osama Ata | Governance & Presentation | Data Cleaning & Preprocessing (Ex. 1) |
| Marc Azzam | Error Analysis & Evidence | Visualisation & Seasonality (Ex. 2 & 3) |
| Malak Yaseen | Model Training | ACF/PACF Statistical Analysis (Ex. 4) |
| Letícia Cristovam Clemente | Dataset & Annotation | Prophet Forecasting & Tuning (Ex. 5) |
| Mark Shane Haines | Project Lead | SARIMA Bonus (Ex. 6), Integration & Review |

## Detailed Contributions

### Osama Ata — Exercise 1: Data Cleaning & Preprocessing
- Datetime conversion and chronological sorting
- Duplicate detection and resolution (DST transitions)
- Hourly frequency enforcement and linear interpolation
- Data quality summary and documentation

### Marc Azzam — Exercises 2 & 3: Visualisation and Seasonality
- Single-day, single-week, and full-dataset plots at multiple temporal scales
- Daily seasonality analysis (hourly averages)
- Weekly seasonality analysis (day-of-week averages)
- Interpretation of patterns and AECO relevance

### Malak Yaseen — Exercise 4: ACF/PACF Statistical Analysis
- Hourly ACF/PACF plots (48 lags) to reveal 24-hour cycle
- Daily ACF/PACF plots (30 lags) to reveal 7-day cycle
- Interpretation of direct vs indirect temporal dependence
- Stationarity discussion and implications for modelling

### Letícia Cristovam Clemente — Exercise 5: Prophet Forecasting
- Weekly resampling and Prophet data formatting
- Train/test split (last 52 weeks held out)
- Default and tuned Prophet models with parameter experimentation
- Evaluation with MAE, RMSE, MAPE and forecast visualisation

### Mark Shane Haines — Exercise 6 (Bonus): SARIMA + Integration
- Augmented Dickey-Fuller stationarity test
- SARIMA(1,1,1)(1,1,1,52) model fitting and forecasting
- Final model comparison table (Prophet vs SARIMA)
- Repository setup, notebook integration, and final review

## Collaboration Workflow

1. Repository created and shared via GitHub
2. Each member worked on their assigned exercise(s) on feature branches
3. Pull requests reviewed before merging into `main`
4. Final integration and testing performed before submission

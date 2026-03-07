# M5U1 | Group Assignment

- Due 16 Mar by 1:59
   - Points 10
   - Submitting a file upload
   - Available 23 Feb at 12:00 - 16 Mar at 1:59

**This is a group assignment. You have to work with your FMP group.**

## **Brief**

The test consists of a total of **6 exercises** (5 Core Python Tasks + 1 Bonus) and must be performed on the dataset ([**PJMW_hourly.csv**](https://canvas.e-zigurat.com/courses/4227/files/886396?wrap=1) [Download PJMW_hourly.csv](https://canvas.e-zigurat.com/courses/4227/files/886396/download?download_frd=1)from Kaggle; instructions on how to load the data into Colab will be provided in class).

- **Evaluation:** Each correctly resolved exercise adds **2 points**.
- **Maximum Grade:** The final grade is out of **10**.

**Use of the Bonus Exercise:** To obtain a **10**, it is sufficient for you to resolve 5 of the 6 proposed exercises correctly. The **Bonus exercise** acts as a "wildcard": if you do not answer one of the previous exercises or make an error, you can use the Bonus to compensate for that score and keep your options for the maximum grade intact.

**Recommendation:** Read all the statements and prioritise the resolution of the 5 exercises you consider most accessible.

**Note:** Regarding the code execution, the solution must be presented in clear, executable Python cells. All libraries allowed: Pandas, Matplotlib/Seaborn, Statsmodels, Prophet, etc…

**Requirement:** Brief comments justifying the logic or interpreting the graph are required for each step. Furthermore, students must briefly explain model assumptions, highlight data limitations and potential risks, and assess whether the forecast is reliable enough to support real-world AECO decisions.

## **EXERCISES**

### 1. **Data Cleaning and Preprocessing**

The raw dataset contains irregularities. Perform the following steps to clean it:

1. Import the data and convert the Date column to a datetime object.
2. Set the timestamp as the index and sort it chronologically.
3. Identify duplicates and handle them by calculating the mean value for that timestamp.
4. *Crucial:* Force the frequency to Hourly ('h') and fill any resulting gaps (missing hours) using *forward fill* or *linear interpolation .*

### 2. **Multi-Scale Visualization**

We need to understand the data behavior at different resolutions. Generate the following plots using the cleaned data:

- One plot showing a single day (24 hours).
- One plot showing a single week.
- One plot showing the full historical dataset.
- *Requirement:* Add a brief comment comparing the visibility of patterns in the "Full Dataset" vs the "Single Week" plot.

### 3. **Seasonality Analysis** 

Analyse the cyclic patterns of the time series data. Generate 2 plots showing the **Average Energy Consumption** for:

- **Daily Seasonality:** Average kW per Hour of the Day (0-23).
- **Weekly Seasonality:** Average kW per Day of the Week (Mon-Sun).
- _Requirement: comment the results obtained. What do you observe? Are there any clear patterns?_

### 4. **Statistical Analysis (ACF/PACF)**

Analyse the dependency of the data on its past values (Autocorrelation).

- **Hourly:** Plot the ACF and PACF for the original hourly data (e.g., Lags = 48).
- **Daily:** Resample the data to a **Daily** frequency (mean) and plot the ACF/PACF again (e.g., Lags = 30).
- *Requirement:* Briefly interpret the pattern seen in the Hourly ACF plot.

### 5. **Forecasting with Prophet**

Calculate a future projection of energy consumption.

1. **Resample** the cleaned data to **Weekly** averages.
2. Format the columns as required by Prophet (ds and y).
3. Train the model with the full dataset (except the last year) and forecast the next **52 weeks** (1 year).
4. Evaluate MAE or RMSE and plot the forecast results.
5. Experiment with different model parameters and comment on the results.

### 6. **BONUS: Advanced Modelling**

Perform an alternative forecast using ARIMA/SARIMA, LSTM or any other forecasting model of your choice that you explored independently.

- - **Recommendation:** You should use Weekly (or Monthly) resampled data. **Do not run this on hourly data**, or it will take hours. 
    - **Output:** Show the code for the model setup, training, and a plot of the prediction against the actual values (or the future forecast).

**Alternatively**, you may use Prophet. But in that case, you must apply it to a different dataset of your choice that is not related to energy consumption.

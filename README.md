# Cash Liquidity Forecasting
<div align="justify">
Suppose there are 100 bank accounts for a group of companies out of 100 business units and a central account. Now let's say those individual accounts that receive variable cash amounts daily (e.g., from stores or other businesses) will deposit the net cash (net cash inflow) into a central account (e.g., after paying out salaries for employees (cash outflow), service charges, and other depreciation costs). Also, since those 100 accounts might be within a country or spread across several countries, let's say the central account will have all the liquid money in US Dollars. How can we model this situation as a cash or liquidity forecasting problem such that we can forecast how much liquid cash will still be in their individual or central account so that the organization can make intelligent decisions about investing the liquid money in profitable businesses? This is crucial because, otherwise that money will just be sitting idle. 
  </div>

## Libraries 
The following are some most widely used Python libraries for time series forecasting tasks:

<div align="justify">

  - [**Darts**](https://github.com/unit8co/darts): Darts is the **most comprehensive and resourceful** Python library for time series forecasting. It provides a variety of models, from classical to deep learning (ARIMA to XGBoost/LightGBM to Time-Series Mixer to exponential smoothing to prophet to transformers). More importantly, it supports both **univariate and multivariate, probabilistic forecasting**, can be used to explain some forecasting models, supports a wide variety of evaluating for the goodness of fit, and supports **back-testing via moving time windows**.
  - [**Skforecast**](https://github.com/JoaquinAmatRodrigo/skforecast/): Skforest is a Python library that eases using scikit-learn regressors as single and multi-step forecasters. It also works with any regressor compatible with the scikit-learn API such as LightGBM, XGBoost, and CatBoost. Besides, it has a **backtesting process** consisting of generating a forecast for each observation in the test set and then comparing the predicted value with the actual value.  
  - **[Prophet](https://github.com/facebook/prophet) and [NeuralProphet](https://github.com/ourownstory/neural_prophet)**: Prophet is designed for forecasting time series data. It's **good for data with strong seasonal effects and several seasons of historical data**. NeuralProphet, on the other hand, is a framework for interpretable time series forecasting. It is built on PyTorch and combines Neural Networks and traditional time-series algorithms, inspired by Facebook Prophet and AR-Net. It is designed for iterative human-in-the-loop model building. This allows building a first model quickly, interpreting the results, improving, and repeating. However, it may not be the most accurate model out of the box. It is **best suited for time series data involving higher frequency** (e.g., sub-daily) and longer duration (e.g., two full periods/years).
  - **Tsfresh**: provides systematic **time-series feature extraction** by combining established algorithms from statistics, time-series analysis, signal processing, and nonlinear dynamics with a robust feature selection algorithm. 
  - [**AutoTS**](https://github.com/winedarksea/AutoTS): all models support forecasting **multivariate and probabilistic forecasts**. **Most models can readily scale to tens to hundreds of thousands of input series**. These models are all designed for integration in an AutoML feature search which automatically finds the best models, preprocessing, and ensembling for a given dataset through genetic algorithms.
  - [**AutoGluon**](https://github.com/autogluon/autogluon): combines various forecasting algorithms, including well-known statical methods like ETS and ARIMA from StatsForecast, efficient tree-based forecasters like LightGBM, deep learning models like DeepAR and Temporal Fusion Transformer from GluonTS, and pretrained zero-shot forecasting model to produce **multi-step ahead probabilistic forecasts for univariate time series data**. It also has **automatic model selection and hyperparameter tuning** functionalities.

</div>

## Standard Python libraries vs. PySpark
<div align="justify">
When comparing standard Python libraries for time series forecasting with PySpark, the key points of comparison are scalability and computational efficiency, especially with large datasets. 

  - Standard Python libraries such as Darts, Prophet, and Skforecast are more specialised, well-maintained, and based on scientific studies, and hence are widely used for time series forecasting. They have support for model development, diagnostics, validation, and even back-testing functions. However, they are primarily designed for single-machine (e.g., they don't parallelize computations without additional frameworks), hence they work well for datasets that fit into the memory of a single machine. Therefore, they can face significant slowdowns or memory issues when dealing with large datasets, leading to slower performance and potential OOM issues. 

  - On the other hand, being designed for distributed computing, PySpark can handle large-scale data processing, allowing parallel processing across multiple nodes in a cluster. This significantly improves scalability and computational speed for large datasets. PySpark's MLlib library provides tools for ML and time series forecasting. The main advantage is its ability to scale horizontally by adding more nodes to the Spark cluster and its ability to handle complex, thus overcoming the computational bottlenecks faced by standard Python libraries. PySpark excels in scenarios where data parallelism can be leveraged, such as when you can partition your data and perform operations on each partition in parallel. Even, standard Python libraries can be used by leveraging its pandas_udf functionality. 

However, it may have fewer specialized time series functions than specialized Python libraries. 
</div>

### Workaround 
<div align="justify">
For example, PySpark's regression models like random forest and gradient-boosted trees are not specialized for time series forecasting, you can engineer features that capture time series characteristics (like lag features, rolling windows, etc.) and use them in these models. 
</div>

#### Python UDF vs. pandas_udf (PySpark's UDFs)
<div align="justify">
  
There are several ways to scale things up as standard Python libraries are not meant with inherent PySpark support. Two options include **pandas_udf** and Python UDFs. Together with Apache Spark and Apache Arrow, **pandas_udfs** use the Pandas library for data manipulation to allow writing more performant UDFs than Python UDFs. They bring the power of Pandas and allow its capabilities to be used in PySpark code. 
  
**pandas_udf** and **applyInPandas** are powerful tools in PySpark for working with grouped data, but their performance and efficiency can vary greatly depending on the use case. But, **pandas_udf** is optimized for grouped operations and can leverage vectorized operations, making it faster than row-at-a-time UDFs: 

- **Scalar pandas_udf**: operates element-wise
- **Grouped map pandas_udf**: it is designed for more complex operations on grouped data.
- **applyInPandas**: allows for arbitrary operations on grouped data and allows for more complex transformations. Hence, it can be more efficient for execution with large datasets. Although similar to the grouped map pandas_udf, its efficiency depends on the specific transformation and the context in which it's used.

The overhead of applying these depends on various factors, including the operations' complexity and the data's size. A sample aggregation operation on 1M rows took 50–55 seconds with Python UDF, whereas pandas_udfs took 40–45 seconds. This is a 25% performance improvement in local mode. Conversely, I noticed the advantage diminishes with smaller data, yet it is a good advantage indicator of using **pandas_udf** compared to Python UDFs in PySpark. 

**Warning**: both applyInPandas and grouped map pandas_udf may lead to **OOM** if the data within a group is too large.  
</div>

## Toy proof-of-concept 
<div align="justify">
We did some quick POC based on synthetic data to assess the technical feasibility. The overall workflow of the methods employed can be described as follows: 

  1. **Data generation:** Generating the data by incorporating real-life scenarios
  2. **Splitting data into a training- and a calibration set**: The calibration set is used to determine the conformity scores, which are essential for conformal prediction.
  3. **Model training**: train the model on the train set by considering factors like seasonality if needed.
  4. **Generating predictions on a calibration set and calculating the conformity scores**: The conformity score can be calculated w.r.t absolute error between the predicted and actual values.
  5. **Setting a significance level**: Set a confidence level, e.g., 0.05 for 95% confidence and use the conformity scores to determine the prediction intervals for new data points.
  6. **Predict future data points:** Predict future data points for a required forecast horizon using the trained model and calculate the upper and lower bounds of the prediction intervals based on the conformity scores and the significance level.
</div>

### Data generation
<div align="justify">
We generate synthetic time series data for 1000 accounts over a specified date that ranges with monthly frequency. The dataset includes simulated cash inflows and outflows with seasonal patterns, seasonality, and external economic indicators such as adjusted inflation factors to comply with a real-world scenario.
</div>

```
  # Set the random seed for reproducibility
  np.random.seed(0)
  
  # Generate month-end dates
  dates = pd.date_range(start=start_date, end=end_date, freq='M')
  
  # Number of accounts
  num_accounts = 1000
  
  # Create a MultiIndex with dates and account IDs
  multi_index = pd.MultiIndex.from_product([dates, range(num_accounts)], names=['date', 'account_id'])
  
  # Initialize an empty DataFrame with the MultiIndex
  df = pd.DataFrame(index=multi_index)
  
  # Define the number of periods (months)
  num_periods = len(dates)
  
  # Simulate seasonal patterns for cash inflows and outflows. Assuming a yearly seasonality with a sinusoidal pattern
  seasonality_inflows = np.sin(np.linspace(0, 2 * np.pi, num_periods))
  seasonality_outflows = np.sin(np.linspace(0, 2 * np.pi, num_periods))
  
  # Generate random cash inflows and outflows for each account on each date and add the seasonality effect
  df['cash_inflows'] = (np.random.normal(loc=100000, scale=20000, size=len(df)) + np.tile(seasonality_inflows, num_accounts))
  df['cash_outflows'] = (np.random.normal(loc=50000, scale=10000, size=len(df)) + np.tile(seasonality_outflows, num_accounts))
  
  # Simulate external economic factors (e.g., inflation rates). Let's assume we have inflation rates for 10 countries
  inflation_rates = [0.03, 0.025, 0.04, 0.02, 0.05, 0.03, 0.045, 0.025, 0.035, 0.03]  # Example rates
  df['inflation_factors'] = np.random.choice(inflation_rates, size=len(df))
  
  # Adjust cash flows based on inflation rates
  df['adjusted_cash_inflows'] = df['cash_inflows'] * (1 + df['inflation_factors'])
  
  # Calculate net cash flow
  df['net_cash_flow'] = df['adjusted_cash_inflows'] - df['cash_outflows']
  
  # Reset the index to turn the MultiIndex into columns
  df.reset_index(inplace=True)
```
<img src="https://github.com/rezacsedu/cash_liquidity_forecasting/blob/main/images/sample_data.png" width="500" height="400">

### Standalone forecasting using Prophet model from Darts library 

```
%%time
from darts import TimeSeries
from darts.models import Prophet
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplcursors
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet

def standalone_model_training_for_liquidity_forecasting(df):
  # Aggregate the net cash flow for each date to get the total liquidity
  total_liquidity = df.groupby('date')['net_cash_flow'].sum().reset_index()

  # Convert the aggregated data into a Darts TimeSeries object
  train_series = TimeSeries.from_dataframe(total_liquidity, time_col='date', value_cols='net_cash_flow', fill_missing_dates=True, freq='M')

  # Create and fit the Prophet model
  model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=4)
  model.fit(train_series)

  return model, train_series

def standalone_liquidity_forecasting(model, forecast_horizon):
  # Forecast the future liquidity for the next 'n' periods
  forecast = model.predict(n=forecast_horizon, num_samples=1000)

  # Extract the quantiles to get the confidence intervals
  lower_quantile = forecast.quantile_timeseries(0.05)
  upper_quantile = forecast.quantile_timeseries(0.95)
  return forecast, lower_quantile, upper_quantile  

def plot_liquidity_forecast(series, forecast, lower_quantile, upper_quantile):
  # Plot the forecast alongside the actual data
  plt.figure(figsize=(13, 6))
  ax = plt.gca()  # Get the current Axes instance

  # Plot actual and forecast series
  actual_plot, = ax.plot(series.time_index, series.values(), label='Actual', color='black')
  forecast_plot, = ax.plot(forecast.time_index, forecast.values(), label='Forecast', color='blue')
  lower_quantile_plot, = ax.plot(lower_quantile.time_index, lower_quantile.values(), label='5th percentile', color='green')
  upper_quantile_plot, = ax.plot(upper_quantile.time_index, upper_quantile.values(), label='95th percentile', color='red')

  # Add vertical dashed lines for important dates
  campaign_start = pd.Timestamp('2022-01-01')
  campaign_end = pd.Timestamp('2022-06-01')
  product_launch = pd.Timestamp('2023-06-01')
  plt.axvline(x=campaign_start, color='red', linestyle='--', linewidth=1, label='Campaign Start')
  plt.axvline(x=campaign_end, color='red', linestyle='--', linewidth=1, label='Campaign End')
  plt.axvline(x=product_launch, color='green', linestyle='--', linewidth=1, label='Product Launch')

  # Annotate with arrows
  plt.annotate('Horizon', xy=(forecast.time_index[-1], forecast.values()[-1]),
  xytext=(forecast.time_index[-30], forecast.values()[-1] + 50000),
  arrowprops=dict(facecolor='orange', shrink=0.05), fontsize=9)
  plt.annotate('Period Focus', xy=(campaign_start, ax.get_ylim()[0]),
  xytext=(campaign_start, ax.get_ylim()[1] - 50000),  # Move the text up by using the upper y-limit
  arrowprops=dict(facecolor='purple', shrink=0.05), fontsize=9)
  plt.annotate('Cut-off', xy=(campaign_end, ax.get_ylim()[0]),
  xytext=(campaign_end, ax.get_ylim()[0] + 50000),
  arrowprops=dict(facecolor='brown', shrink=0.05), fontsize=9)

  # Set title, labels, and legend
  plt.title('Liquidity Forecast with Confidence Intervals')
  plt.xlabel('Date')
  plt.ylabel('Net Cash Flow')
  plt.legend(ncol=len(ax.lines), loc='upper center', bbox_to_anchor=(0.5, -0.1))

  # Improve date formatting on the x-axis
  ax.xaxis.set_major_locator(mdates.YearLocator())
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
  ax.xaxis.set_minor_locator(mdates.AutoDateLocator())
  ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))

  # Rotate date labels for better readability
  plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
  plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)

  # Show gridlines
  plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)

  # Use mplcursors to add interactive hover tooltips
  cursor = mplcursors.cursor(hover=True)
  cursor.connect("add", lambda sel: sel.annotation.set_text(
  'Date: {}\nNet Cash Flow: ${:.2f}'.format(pd.to_datetime(sel.target[0]).strftime('%Y-%m-%d'), sel.target[1])))

  # Show the plot
  plt.tight_layout()
  plt.show()     

model, train_series = standalone_model_training_for_liquidity_forecasting(cash_flow_df)

forecast_horizon = 30
forecast, lower_quantile, upper_quantile = standalone_liquidity_forecasting(model, forecast_horizon)
# Sum all net_cash_flow columns for each date
forecast_pdf = forecast.pd_dataframe()
     
forecast_pdf['forecasted_net_cash_flow'] = forecast_pdf.filter(regex='^net_cash_flow_s').mean(axis=1)
forecast_pdf['date'] = forecast_pdf.index
forecast_pdf = forecast_pdf.reset_index(drop=True)
forecast_pdf = forecast_pdf[['date', 'forecasted_net_cash_flow']]    

plot_liquidity_forecast(train_series, forecast, lower_quantile, upper_quantile)     

```

### Distributed cash liquidity forecasting with PySpark

```
  %%time
  from pyspark.sql import SparkSession
  from pyspark.sql.functions import pandas_udf, PandasUDFType
  from pyspark.sql.types import *
  from pyspark.sql.functions import pandas_udf
  from pyspark.sql.types import StructType, StructField, TimestampType, IntegerType, DoubleType
  
  import pandas as pd
  from darts import TimeSeries
  from darts.models import Prophet
  
  # Initialize a Spark session
  spark = SparkSession.builder.appName("LiquidityForecasting").getOrCreate()
  
  schema = StructType([
  StructField('date', TimestampType(), True),
  StructField('forecast_net_cash_flow', DoubleType(), True)
  ])
  
  # Define the pandas_udf function
  @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
  def forecast_timeseries(df):
    # Convert the data into a Darts TimeSeries object
    train_series = TimeSeries.from_dataframe(df, time_col='date', value_cols='net_cash_flow', fill_missing_dates=True, freq='M')
  
    # Create and fit the Prophet model
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=4)
    model.fit(train_series)
  
    # Forecast the future liquidity
    forecast_horizon = 30  # Adjust the forecast horizon as needed
    forecast = model.predict(n=forecast_horizon)
  
    # Prepare the forecast DataFrame to match the specified schema
    forecast_df = forecast.pd_dataframe()
    forecast_df.reset_index(inplace=True)
    forecast_df.columns = ['date', 'forecast_net_cash_flow']
    #forecast_df['total_mean_forecasted_net_cash_flow'] = forecast_df.filter(regex='^total_mean_forecasted_net_cash_flow_s').mean(axis=1)
  
    # Return the forecast as a Pandas DataFrame
    return forecast_df
  
  # Assuming 'sdf' is a Spark DataFrame with 'date' and 'net_cash_flow' columns. Apply the pandas_udf function to the DataFrame
  total_liquidity = cash_flow_df.groupby('date')['net_cash_flow'].sum().reset_index()
  sdf = spark.createDataFrame(total_liquidity)
  forecast_sdf = sdf.groupby().apply(forecast_timeseries)
```
### Conformal prediciton interval for better uncertanity quantification

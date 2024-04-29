# Cash Liquidity Forecasting
Suppose there are 100 bank accounts for a group of companies for their 100 business units. Now let's say from those accounts variable cash amounts (e.g., from stores to individual accounts) are deposited daily (cash inflow) into a central account and certain amounts will be transferred from those accounts to other accounts like paying salary for the employees (cash outflow) or service charges, etc. Also, since those 100 accounts might be within a single country or spread across countries, the central account will have all the liquid money in US Dollars. How can we model this situation as a cash or liquidity forecasting problem such that we can forecast how much liquid cash will still be in their individual or central account so that the organization can make intelligent decisions about investing the liquid money in profitable businesses? This is crucial because, otherwise that money will just be sitting idle. 

## Libraries 
The following are some most widely used Python libraries for time series forecasting tasks:

  - **Darts**: Darts is the most comprehensive and resourceful Python library for time series forecasting. It provides a variety of models, from classical to deep learning (ARIMA to XGBoost/LightGBM to Time-Series Mixer to exponential smoothing to prophet to transformers). More importantly, it has support for both univariate and multivariate, probabilistic forecasting, can be used to explain some forecasting models, has support for a wide variety of evaluating for the goodness of fit, and its Utilities for simulating historical forecasts, using moving time windows. 
  - **Skforecast**: Skforest is a Python library that eases using scikit-learn regressors as single and multi-step forecasters. It also works with any regressor compatible with the scikit-learn API such as LightGBM, XGBoost, and CatBoost. Besides, it has the backtesting process consists of generating a forecast for each observation in the test set, following the same procedure as it would be done in production, and then comparing the predicted value with the actual value. 
  - **Prophet**: Developed by Facebook, Prophet is designed for forecasting time series data. It's good for data with strong seasonal effects and several seasons of historical data.
  - **NeuralProphet**: NeuralProphet is a framework for interpretable time series forecasting. It is built on PyTorch and combines Neural Networks and traditional time-series algorithms, inspired by Facebook Prophet and AR-Net. It is designed for iterative human-in-the-loop model building. This allows building a first model quickly, interpreting the results, improving, and repeating. However, it may not be the most accurate model out-of-the-box; so, don't hesitate to adjust and iterate until you like your results. However, it is best suited for time series data involving higher frequency (e.g., sub-daily) and longer duration (at least two full periods/years).
  - **Sktime**: Sktime provides a unified framework for time series machine learning.
  - **Tsfresh**: Tsfresh is focused on feature extraction for time series data. It's useful for preparing time series data for machine learning models.
  - **AutoTS**: AutoTS is a library that provides automated time series forecasting. It's designed to be easy to use and to handle a wide range of time series forecasting problems.
  - **AutoGluon**: an auto forecast library that forecasts the future values of multiple time series given the historical data and other related covariates. A single call to AutoGluon TimeSeriesPredictor’s fit() method trains multiple models to generate accurate probabilistic forecasts, and does not require manually dealing with cumbersome issues like model selection and hyperparameter tuning. It combines various state-of-the-art forecasting algorithms that include well-known statical methods like ETS and ARIMA from StatsForecast, efficient tree-based forecasters like LightGBM, deep learning models like DeepAR and Temporal Fusion Transformer from GluonTS, and pretrained zero-shot forecasting model, and Chronos. 

## Standard Python libraries vs. PySpark
When comparing standard Python libraries for time series forecasting with PySpark, the key points of comparison are scalability and computational efficiency, especially with large datasets. 

Standard Python libraries such as Darts, Prophet, and Skforecast are more specialised, well-maintained, and based on scientific studies, and hence are widely used for time series forecasting. They provide a rich set of features for model development, diagnostics, validation, and even back-testing functions. However, they are primarily designed for single-machine (e.g., they don't parallelize computations without additional frameworks), hence they work well for datasets that fit into the memory of a single machine. Therefore, they can face significant slowdowns or memory issues when dealing with large datasets, leading to slower performance and potential OOM issues. 

On the other hand, being designed for distributed computing, PySpark can handle large-scale data processing, allowing parallel processing across multiple nodes in a cluster. This significantly improves scalability and computational speed for large datasets. PySpark's MLlib library provides tools for ML and time series forecasting. The main advantage is its ability to scale horizontally by adding more nodes to the Spark cluster and its ability to handle complex, thus overcoming the computational bottlenecks faced by standard Python libraries. PySpark excels in scenarios where data parallelism can be leveraged, such as when you can partition your data and perform operations on each partition in parallel. Even, standard Python libraries can be used by leveraging its pandas_udf functionality. 

However, it may have fewer specialized time series functions than specialized Python libraries. **Workaround**? For example, PySpark's regression models like random forest and gradient-boosted trees are not specialized for time series forecasting, you can engineer features that capture time series characteristics (like lag features, rolling windows, etc.) and use them in these models.

## Methods 
We did some quick POC based on synthetic data to assess the technical feasibility. The overall workflow of the methods employed can be described as follows: 

  1. **Data generation:** Generating the data by incorporating real-life scenarios
  2. **Splitting data into a training- and a calibration set**: The calibration set is used to determine the conformity scores, which are essential for conformal prediction.
  3. **Model training**: train the model on the train set by considering factors like seasonality if needed.
  4. **Generating predictions on a calibration set and calculating the conformity scores**: The conformity score can be calculated w.r.t absolute error between the predicted and actual values.
  5. **Setting a significance level**: Set a confidence level, e.g., 0.05 for 95% confidence and use the conformity scores to determine the prediction intervals for new data points.
  6. **Predict future data points:** Predict future data points for a required forecast horizon using the trained model and calculate the upper and lower bounds of the prediction intervals based on the conformity scores and the significance level.

## Data generation
Following code to generate synthetic data by including factors such as different types of cash inflows and outflows, seasonality, trends, and perhaps external economic indicators to comply with a real-world scenario. It creates a time series dataset for 1000 accounts over a specified date range with monthly frequency. The dataset includes simulated cash inflows and outflows with seasonal patterns and adjusts for inflation factors.

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
### Standalone forecasting using Prophet model from Darts library 
### Distributed cash liquidity forecasting with PySpark
### Conformal prediciton interval for better uncertanity quantification

## Pandas vs. PySpark and using PySpark's UDFs vs. Python UDFs
There are several ways to scale things up as standard Python libraries are not meant with inherent PySPark support. Therefore, I considered several factors w.r.t Pandas UDFs vs. Python UDFs. 

### Python UDF vs. PySpark's UDFs
Together with Apache Spark and Apache Arrow, pandas_udfs use the Pandas library for data manipulation to allow writing more performant UDFs as compared to Python UDFs. More advantageously, they also bring the power of the Pandas library and allow its capabilities to be used in PySpark code. For a sample aggregation operation on 1M rows, it took 50–55 seconds with Python UDF, whereas pandas_udfs took 40–45 seconds. This is 25% improvement in performance in local mode. Performance advantage diminishes with smaller data, but this is a good indicator of the performance advantage of pandas_udfs compared to Python UDFs in PySpark. 

### Consideration with pandas_udf 
Generally, pandas_udf is optimized for grouped operations and can leverage vectorized operations, which are faster than row-at-a-time UDFs: 

- **Scalar pandas_udf**: operates element-wise
- **Grouped map pandas_udf**: it is designed for more complex operations on grouped data.
- **applyInPandas**: allows for arbitrary operations on grouped data and allows for more complex transformations. Hence, it can be more efficient for execution with large datasets. Although similar to the grouped map pandas_udf, its efficiency depends on the specific transformation and the context in which it's used.

The overhead for **applyInPandas** and **pandas_udf** depends on various factors, including the operations' complexity and the data's size. Both **applyInPandas** and grouped map **pandas_udf** can have significant overhead if not used carefully and may lead to **Out-Of-Memory (OOM)** errors if the data within a group is too large. Therefore, it's essential to ensure that the data can be partitioned into manageable sizes to fit into the memory of the worker nodes. While **pandas_udf** and **applyInPandas** are powerful tools in PySpark for working with grouped data, their performance and efficiency can vary greatly depending on the use case. 

# Cash Liquidity Forecasting
<div align="justify">
Suppose there are 100 bank accounts for a group of companies out of 100 business units and a central account. Now let's say those individual accounts that receive variable cash amounts daily (e.g., from stores or other businesses) will deposit the net cash (net cash inflow) into a central account (e.g., after paying out salaries for employees (cash outflow), service charges, and other depreciation costs). Also, since those 100 accounts might be within a country or spread across several countries, let's say the central account will have all the liquid money in US Dollars. </div>

<div align="justify">
How can we model this situation as a cash or liquidity forecasting problem such that we can forecast how much liquid cash will still be in their individual or central account so that the organization can make intelligent decisions about investing the liquid money in profitable businesses? This is crucial because, otherwise that money will just be sitting idle. 
  </div>

## Libraries 
The following are some most widely used Python libraries for time series forecasting tasks:

  - [**Darts**](https://github.com/unit8co/darts): Darts is the **most comprehensive and resourceful** Python library for time series forecasting. It provides a variety of models, from classical to deep learning (ARIMA to XGBoost/LightGBM to Time-Series Mixer to exponential smoothing to prophet to transformers). More importantly, it supports both **univariate and multivariate, probabilistic forecasting**, can be used to explain some forecasting models, supports a wide variety of evaluating for the goodness of fit, and supports **back-testing via moving time windows**. 
  - [**Skforecast**](https://github.com/JoaquinAmatRodrigo/skforecast/): Skforest is a Python library that eases using scikit-learn regressors as single and multi-step forecasters. It also works with any regressor compatible with the scikit-learn API such as LightGBM, XGBoost, and CatBoost. Besides, it has a **backtesting process** consisting of generating a forecast for each observation in the test set and then comparing the predicted value with the actual value.  
  - **[Prophet](https://github.com/facebook/prophet) and [NeuralProphet](https://github.com/ourownstory/neural_prophet)**: Prophet is designed for forecasting time series data. It's **good for data with strong seasonal effects and several seasons of historical data**. NeuralProphet, on the other hand, is a framework for interpretable time series forecasting. It is built on PyTorch and combines Neural Networks and traditional time-series algorithms, inspired by Facebook Prophet and AR-Net. It is designed for iterative human-in-the-loop model building. This allows building a first model quickly, interpreting the results, improving, and repeating. However, it may not be the most accurate model out of the box. It is **best suited for time series data involving higher frequency** (e.g., sub-daily) and longer duration (e.g., two full periods/years).
  - **Tsfresh**: provides systematic **time-series feature extraction** by combining established algorithms from statistics, time-series analysis, signal processing, and nonlinear dynamics with a robust feature selection algorithm. 
  - [**AutoTS**](https://github.com/winedarksea/AutoTS): all models support forecasting **multivariate and probabilistic forecasts**. **Most models can readily scale to tens to hundreds of thousands of input series**. These models are all designed for integration in an AutoML feature search which automatically finds the best models, preprocessing, and ensembling for a given dataset through genetic algorithms.
  - [**AutoGluon**](https://github.com/autogluon/autogluon): combines various forecasting algorithms, including well-known statical methods like ETS and ARIMA from StatsForecast, efficient tree-based forecasters like LightGBM, deep learning models like DeepAR and Temporal Fusion Transformer from GluonTS, and pretrained zero-shot forecasting model to produce **multi-step ahead probabilistic forecasts for univariate time series data**. It also has **automatic model selection and hyperparameter tuning** functionalities. 

## Standard Python libraries vs. PySpark
When comparing standard Python libraries for time series forecasting with PySpark, the key points of comparison are scalability and computational efficiency, especially with large datasets. 

Standard Python libraries such as Darts, Prophet, and Skforecast are more specialised, well-maintained, and based on scientific studies, and hence are widely used for time series forecasting. They have support for model development, diagnostics, validation, and even back-testing functions. However, they are primarily designed for single-machine (e.g., they don't parallelize computations without additional frameworks), hence they work well for datasets that fit into the memory of a single machine. Therefore, they can face significant slowdowns or memory issues when dealing with large datasets, leading to slower performance and potential OOM issues. 

On the other hand, being designed for distributed computing, PySpark can handle large-scale data processing, allowing parallel processing across multiple nodes in a cluster. This significantly improves scalability and computational speed for large datasets. PySpark's MLlib library provides tools for ML and time series forecasting. The main advantage is its ability to scale horizontally by adding more nodes to the Spark cluster and its ability to handle complex, thus overcoming the computational bottlenecks faced by standard Python libraries. PySpark excels in scenarios where data parallelism can be leveraged, such as when you can partition your data and perform operations on each partition in parallel. Even, standard Python libraries can be used by leveraging its pandas_udf functionality. 

However, it may have fewer specialized time series functions than specialized Python libraries. 

### Workaround 
For example, PySpark's regression models like random forest and gradient-boosted trees are not specialized for time series forecasting, you can engineer features that capture time series characteristics (like lag features, rolling windows, etc.) and use them in these models. 

#### Python UDF vs. pandas_udf (PySpark's UDFs)
There are several ways to scale things up as standard Python libraries are not meant with inherent PySPark support. Therefore, I considered several factors w.r.t **pandas_udf** vs. Python UDFs. Together with Apache Spark and Apache Arrow, **pandas_udfs** use the Pandas library for data manipulation to allow writing more performant UDFs than Python UDFs. They bring the power of Pandas and allow its capabilities to be used in PySpark code. 


**pandas_udf** and **applyInPandas** are powerful tools in PySpark for working with grouped data, but their performance and efficiency can vary greatly depending on the use case. But, **pandas_udf** is optimized for grouped operations and can leverage vectorized operations, making it faster than row-at-a-time UDFs: 

- **Scalar pandas_udf**: operates element-wise
- **Grouped map pandas_udf**: it is designed for more complex operations on grouped data.
- **applyInPandas**: allows for arbitrary operations on grouped data and allows for more complex transformations. Hence, it can be more efficient for execution with large datasets. Although similar to the grouped map pandas_udf, its efficiency depends on the specific transformation and the context in which it's used.

The overhead of applying these depends on various factors, including the operations' complexity and the data's size. A sample aggregation operation on 1M rows took 50–55 seconds with Python UDF, whereas pandas_udfs took 40–45 seconds. This is a 25% performance improvement in local mode. Conversely, I noticed the advantage diminishes with smaller data, yet it is a good advantage indicator of using **pandas_udfs** compared to Python UDFs in PySpark. 

**Warning**: both applyInPandas and grouped map pandas_udf may lead to **OOM** if the data within a group is too large.  

## Toy proof-of-concept 
We did some quick POC based on synthetic data to assess the technical feasibility. The overall workflow of the methods employed can be described as follows: 

  1. **Data generation:** Generating the data by incorporating real-life scenarios
  2. **Splitting data into a training- and a calibration set**: The calibration set is used to determine the conformity scores, which are essential for conformal prediction.
  3. **Model training**: train the model on the train set by considering factors like seasonality if needed.
  4. **Generating predictions on a calibration set and calculating the conformity scores**: The conformity score can be calculated w.r.t absolute error between the predicted and actual values.
  5. **Setting a significance level**: Set a confidence level, e.g., 0.05 for 95% confidence and use the conformity scores to determine the prediction intervals for new data points.
  6. **Predict future data points:** Predict future data points for a required forecast horizon using the trained model and calculate the upper and lower bounds of the prediction intervals based on the conformity scores and the significance level.

### Data generation
We generate synthetic time series data for 1000 accounts over a specified date that ranges with monthly frequency. The dataset includes simulated cash inflows and outflows with seasonal patterns, seasonality, and external economic indicators such as adjusted inflation factors to comply with a real-world scenario.

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

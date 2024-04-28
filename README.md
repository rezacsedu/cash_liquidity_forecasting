# Cash Liquidity Forecasting
Suppose there are 100 bank accounts for a group of companies for their 100 business units. Now let's say from those accounts variable cash amounts (e.g., from stores to individual accounts) are deposited daily (cash inflow) into a central account and certain amounts will be transferred from those accounts to other accounts like paying salary for the employees (cash outflow) or service charges, etc. Also, since those 100 accounts might be within a single country or spread across countries, the central account will have all the liquid money in US Dollars. 

How can we model this situation in the form of a cash or liquidity forecasting problem such that we can forecast how much liquid cash will still be in their individual or central account so that the organization can make intelligent decisions about investing the liquid money in profitable businesses? This is crucial because, otherwise that money will just be sitting idle. 

## Data generation
Following code to generate synthetic data, by including factors such as different types of cash inflows and outflows, seasonality, trends, and perhaps external economic indicators to comply with a real-world scenario. It creates a time series dataset for 1000 accounts over a specified date range with monthly frequency. The dataset includes simulated cash inflows and outflows with seasonal patterns and adjusts for inflation factors.

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

## Methods 
To introduce conformal prediction into your time series forecasting following steps are needed:

  1. **Splitting data into a training- and a calibration set**: The calibration set is used to determine the conformity scores which are essential for conformal prediction.
  
  2. **Model training**: train the model on the train set by taking into consideration factors like seasonality if needed. 
  
  3. **Generating predictions on a calibration set and calculating the conformity scores**: The conformity score can be calculated w.r.t absolute error between the predicted and actual values.
  
  4. **Setting a significance level**: Set a confidence level, e.g., 0.05 for 95% confidence and use the conformity scores to determine the prediction intervals for new data points.
  
  5. **Predict future data points:** Predict future data points for a required forecast horizon using the trained model and calculate the upper and lower bounds of the prediction intervals based on the conformity scores and the significance level. 

### Standalone forecasting using Prophet model from Darts library 
### Distributed cash liquidity forecasting with PySpark
### Conformal prediciton interval for better uncertanity quantification

## Scalability issue: Pandas vs. PySpark: using Pandas UDFs vs. Python UDFs
Together with Apache Spark and Apache Arrow, Pandas User Defined Functions (UDFs) use the Pandas library for data manipulation in order to allow writing more performant UDFs as compared to Python UDFs. More advantageously, they also bring the power of the Pandas library and allow its capabilities to be used in PySpark code. For a sample aggregation operation on 1M rows, it took 50–55 seconds with Python UDF, whereas Pandas UDF took 40–45 seconds. This is 25% improvement in performance in local mode. Performance advantage diminishes with smaller data, but this is a good indicator of the performance advantage of Pandas UDFs compared to Python UDFs in PySpark. 

### Consideration with pandas_udf 
Generally, pandas_udf is optimized for grouped operations, such as applying a function after a groupBy. This allows for vectorized operations which are typically faster than regular User-Defined Functions (UDFs). For simple element-wise operations, a regular Spark UDF might suffice and be faster. On the other hand, applyInPandas is used with groupBy operations on a DataFrame and allows for more complex transformations. It can be more efficient for execution with large datasets, despite requiring some additional syntax to learn. The overhead for applyInPandas is generally lower compared to pandas_udf, especially as the data size increases.

It's important to note that both applyInPandas and pandas_udf can lead to an Out-Of-Memory (OOM) risk if the data for a group is too large to fit in memory. Therefore, it's crucial to be aware of the potential memory constraints when working with large datasets. In summary, if we're performing complex transformations on grouped data, applyInPandas might be more efficient, especially for large datasets. For simpler, element-wise operations, pandas_udf could be more suitable. It's always a good practice to test both methods on your specific dataset to determine which one provides better performance.

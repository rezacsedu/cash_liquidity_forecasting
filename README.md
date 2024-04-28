# Cash Liquidity Forecasting
Suppose there are 100 bank accounts for a group of companies for their 100 different business units. Now let's say from those accounts variable cash amounts (e.g., from stores to individual accounts) are deposited on daily basis (cash inflow) into a central account and certain amounts will be transferred from those accounts to other accounts like paying salary for the employees (cash outflow) or service charges, etc. Also, since those 100 accounts might be within a single country or spread across countries, let's say the central account will have all the liquid money in US Dollar. How can we model this situation in the form of cash or liquidity forecasting problem such that we can forecast how much liquid cash will still be there individual or central account so that the organization can take intelligent decision about investing the liquid money in profitable businesses. This is crucial because, otherwise those money will just be sitting idle. 

## Data generation
Following code to generate synthetic data, by including factors such as different types of cash inflows and outflows, seasonality, trends, and perhaps external economic indicators to comply a real-world scenario. It creates a time series dataset for 1000 accounts over a specified date range with monthly frequency. The dataset includes simulated cash inflows and outflows with seasonal patterns and adjusts for inflation factors.

## Methods 
To introduce conformal prediction into your time series forecasting following steps are needed:

  1. **Spliting data into a training- and a calibration set**: The calibration set is used to determine the conformity scores which are essential for conformal prediction.
  
  2. **Model training**: train the model on the train set by taking into consideration factors like seasonality if needed. 
  
  3. **Generating predictions on calibration set and calculating the conformity scores**: The conformity score can be calculated w.r.t absolute error between the predicted and actual values.
  
  4. **Setting a significance level**: Set confidence level, e.g., 0.05 for 95% confidence and use the conformity scores to determine the prediction intervals for new data points.
  
  5. **Predict future data points:** predict future data points for a required forecast horizon using the trained model and calculate the upper and lower bounds of the prediction intervals based on the conformity scores and the significance level. 


### Standalone forecasting using Prophet model from Darts library 


### Distributed cash liquidity forecasting with PySpark


### Conformal prediciton interval for better uncertanity quantification


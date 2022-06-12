#!/usr/bin/env python
# coding: utf-8

# # Forecasting Simple Time-Series

# This example shows how to estimate time-series that are relatively constant or have a defined trend, but other than that do not have any patterns over time. If there are other patterns in the data, such as seasonality, see Forecasting Complex Time-Series.

# ## Load in the Data

# We'll load in the data with `pandas`, which should be review.

# In[1]:


import pandas as pd
df = pd.read_excel('Sales COGS.xlsx')


# In[2]:


df.head()


# Here we should use Sales, Cost of Goods sold as the index. Load in by setting the index column.

# In[3]:


df = pd.read_excel('Sales COGS.xlsx', index_col=0)
df.head()


# Now that looks better.

# ## Plot Data

# For an effective plot, we will need to transpose the data, so that the dates are the index (x-axis on plot), and the data types are columns (series on plot). Thankfully this is as simple as `df.T`.

# In[37]:


df.T


# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')

df.T.plot.line()


# ## Forecast Using Most Recent Value
# 
# This is the simplest forecast, just keep it the same as it was. 
# 
# Right now we have the dates as the columns. So access the columns and take the max to find the latest date.

# In[7]:


last_date = df.columns.max()
last_date


# Now select the values which have the latest date as the forecast

# In[10]:


fcst_sales = df.loc['Sales'][last_date]
fcst_cogs = df.loc['Cost of Goods Sold'][last_date]
print(f'The forecasted value for sales is ${fcst_sales:,.0f} and for COGS is ${fcst_cogs:,.0f}')


# ## Forecast Using Average
# 
# We have already seen how to take averages of `pandas` `Series`:

# In[11]:


fcst_sales = df.loc['Sales'].mean()
fcst_cogs = df.loc['Cost of Goods Sold'].mean()
print(f'The forecasted value for sales is ${fcst_sales:,.0f} and for COGS is ${fcst_cogs:,.0f}')


# ## Forecast Using Trend
# 
# There are two methods to forecast using the trend.

# ### Trend Method 1: By Regression
# 
# We will estimate the following regression model:
# $$y_t = a + \beta t + \epsilon_t$$

# #### Create DataFrame with $t$

# First we need to create a `DataFrame` which has a column for the $y$ and a column for the $t$:.
# 
# To do this, first we can create a `DataFrame` from the `Series` we want to forecast.

# In[14]:


for_fcst_df = pd.DataFrame(df.loc['Sales'])
for_fcst_df


# We can use `reset_index(drop=True)` to get rid of the date index.

# In[15]:


for_fcst_df = for_fcst_df.reset_index(drop=True)
for_fcst_df


# We can now use `reset_index()` without the `drop=True` to get this new 0, 1, 2 index as a column.

# In[16]:


for_fcst_df = for_fcst_df.reset_index()
for_fcst_df


# We can also rename that column.

# In[17]:


for_fcst_df = for_fcst_df.rename(columns={'index': 't'})
for_fcst_df


# Let's wrap this all up in a function as we'll need to use this for COGS as well, and for the CAGR approaches.

# In[19]:


def for_forecast_df_from_orig_df(orig_df, series_name):
    """
    From a DataFrame where index is name of series to be forecasted and columns are time periods, create a 
    DataFrame with two columns, t in periods and the value to be forecasted, and rows are time periods.
    """
    for_fcst_df = pd.DataFrame(orig_df.loc[series_name])
    for_fcst_df = for_fcst_df.reset_index(drop=True).reset_index()
    for_fcst_df = for_fcst_df.rename(columns={'index': 't'})
    return for_fcst_df

for_forecast_df_from_orig_df(df, 'Cost of Goods Sold')


# #### Now Run Regression

# This should be review from the cost of equity exercise.

# In[24]:


import statsmodels.api as sm

model = sm.OLS(for_fcst_df['Sales'], sm.add_constant(for_fcst_df['t']), hasconst=True)
results = model.fit()
results.summary()


# Now get the intercept and $\beta$ from the regression results.

# In[26]:


intercept = results.params['const']
beta = results.params['t']


# In[27]:


intercept


# In[28]:


beta


# #### Now Predict from Regression Results

# In[29]:


fcst_sales = intercept + beta * 3
fcst_sales


# Now let's wrap up the regression approach into a function to use it with COGS as well.

# In[33]:


def intercept_and_t_beta_from_for_forecast_df(for_fcst_df, series_name):
    """
    Calculates intercept and beta of time periods from DataFrame set up for forecasting
    """
    model = sm.OLS(for_fcst_df[series_name], sm.add_constant(for_fcst_df['t']), hasconst=True)
    results = model.fit()
    intercept = results.params['const']
    beta = results.params['t']
    return intercept, beta

def predict_from_intercept_beta_and_t(intercept, beta, t):
    """
    Predicts value in period t based off regression intercept and beta
    """
    return intercept + beta * t

intercept, beta = intercept_and_t_beta_from_for_forecast_df(for_fcst_df, 'Sales')
fcst_sales = predict_from_intercept_beta_and_t(intercept, beta, 3)
fcst_sales


# #### Entire Approach for COGS Using Functions

# Let's write one more function to do the entire forecast, putting everything together.

# In[39]:


def forecast_trend_reg(df, series_name, t):
    """
    Full workflow of forecasting trend via regression
    """
    for_fcst_df = for_forecast_df_from_orig_df(df, series_name)
    intercept, beta = intercept_and_t_beta_from_for_forecast_df(for_fcst_df, series_name)
    fcst = predict_from_intercept_beta_and_t(intercept, beta, t)
    return fcst

fcst_sales = forecast_trend_reg(df, 'Sales', 3)
fcst_cogs = forecast_trend_reg(df, 'Cost of Goods Sold', 3)
print(f'The forecasted value for sales is ${fcst_sales:,.0f} and for COGS is ${fcst_cogs:,.0f}')


# ### Trend Method 2: By CAGR

# Let's work off the `DataFrame` set up for forecasting from the regression approach.

# In[40]:


for_fcst_df = for_forecast_df_from_orig_df(df, 'Sales')
for_fcst_df


# We want to calculate $$\frac{y_T}{y_0}^{\frac{1}{n}} - 1$$

# We can use `.iloc` (integer location) to get the first and last values of sales.

# In[41]:


y_0 = for_fcst_df['Sales'].iloc[0]
y_0


# In[42]:


y_T = for_fcst_df['Sales'].iloc[-1]
y_T


# We can get the number of time periods elapsed in a similar way.

# In[43]:


n = for_fcst_df['t'].iloc[-1] - for_fcst_df['t'].iloc[0]
n


# Now just calculate

# In[44]:


cagr = (y_T / y_0)**(1 / n) - 1
cagr


# Now to get the predicted value for period 3

# Let's wrap this up into functions.

# In[47]:


def cagr_from_for_forecast_df(for_fcst_df, series_name):
    """
    Calculates CAGR from DataFrame set up for forecasting
    """
    y_0 = for_fcst_df[series_name].iloc[0]
    y_T = for_fcst_df[series_name].iloc[-1]
    n = for_fcst_df['t'].iloc[-1] - for_fcst_df['t'].iloc[0]
    cagr = (y_T / y_0)**(1 / n) - 1
    return cagr


def predict_from_for_forecast_df_and_cagr(for_fcst_df, series_name, cagr, t):
    """
    Forecast value from DataFrame set up for forecasting and calculated CAGR
    """
    y_T = for_fcst_df[series_name].iloc[-1]
    n = for_fcst_df['t'].iloc[-1] - for_fcst_df['t'].iloc[0]
    
    future_nper = t - n
    fcst = y_T * (1 + cagr)**future_nper
    return fcst


cagr = cagr_from_for_forecast_df(for_fcst_df, 'Sales')
fcst_sales = predict_from_for_forecast_df_and_cagr(for_fcst_df, 'Sales', cagr, 3)
fcst_sales


# Now let's put the entire approach in one function.

# In[48]:


def forecast_trend_cagr(df, series_name, t):
    """
    Full workflow of forecasting trend via CAGR
    """
    for_fcst_df = for_forecast_df_from_orig_df(df, series_name)
    cagr = cagr_from_for_forecast_df(for_fcst_df, series_name)
    fcst = predict_from_for_forecast_df_and_cagr(for_fcst_df, series_name, cagr, t)
    return fcst

fcst_sales = forecast_trend_cagr(df, 'Sales', 3)
fcst_cogs = forecast_trend_cagr(df, 'Cost of Goods Sold', 3)
print(f'The forecasted value for sales is ${fcst_sales:,.0f} and for COGS is ${fcst_cogs:,.0f}')


# ## Forecasting as a %

# We can estimate COGS as a percentage of sales. To do this, we must first forecast sales, then forecast the percentage of sales, then combine the two. We already have a sales forecast from the last section, so let's keep that. Next is forecasting the percentage of sales. To do this we must first calculate the historical percentage of sales.

# In[50]:


df.loc['COGS % Sales'] = df.loc['Cost of Goods Sold'] / df.loc['Sales']
df


# Now we can forecast this by any of the available methods.

# In[53]:


fcst_cogs_pct_sales = forecast_trend_reg(df, 'COGS % Sales', 3)
fcst_cogs_pct_sales


# Now combine with the existing sales forecast.

# In[54]:


fcst_cogs = fcst_sales * fcst_cogs_pct_sales
print(f'The forecasted value for sales is ${fcst_sales:,.0f} and for COGS is ${fcst_cogs:,.0f}')


# ## All the Approaches, Together
# 
# There is a dizzying array of forecast options, even only considering simple forecast methods. Here is a quick overview of the approaches.

# In[56]:


def forecast_by_method(df, series_name, method, t):
    if method == 'average':
        return df.loc[series_name].mean()
    elif method == 'recent':
        last_date = df.columns.max()
        return df.loc[series_name][last_date]
    elif method == 'trend reg':
        return forecast_trend_reg(df, series_name, t)
    elif method == 'trend cagr':
        return forecast_trend_cagr(df, series_name, t)

methods = [
    'average',
    'recent',
    'trend reg',
    'trend cagr'
]

t = 3

cogs_forecasts = []
for sales_method in methods:
    fcst_sales = forecast_by_method(df, 'Sales', sales_method, t)
    for cogs_method in methods:
        # Handle levels for COGS
        fcst_cogs = forecast_by_method(df, 'Cost of Goods Sold', cogs_method, t)
        cogs_forecasts.append(fcst_cogs)
        print(f'The forecasted value for sales ({sales_method}) is ${fcst_sales:,.0f} and for COGS ({cogs_method}) is ${fcst_cogs:,.0f}')
        # Handle % of sales for COGS
        fcst_cogs_pct = forecast_by_method(df, 'COGS % Sales', cogs_method, t)
        fcst_cogs = fcst_cogs_pct * fcst_sales
        cogs_forecasts.append(fcst_cogs)
        print(f'The forecasted value for sales ({sales_method}) is ${fcst_sales:,.0f} and for COGS (% of Sales, {cogs_method}) is ${fcst_cogs:,.0f}')


# In[58]:


pd.DataFrame(cogs_forecasts).plot.box()



'''
###Qwiklab 1 - Building a Regression Model in AI Platform Notebooks

##Overview
#In this lab, you will build and evaluate a simple linear regression model to predict AAPL closing stock prices using Scikit-Learn and BigQuery.

##Objectives
#In this lab, you learn to perform the following tasks:
#1.Load data from BigQuery into a Pandas DataFrame
#2.Build a linear regression model in Scikit-Learn
#3.Use AI Platform Notebooks

##Set up your environment
#What you'll need
To complete this lab, you’ll need:

1.Access to a standard internet browser (Chrome browser recommended).

2.Time. Note the lab’s Completion time in Qwiklabs. This is an estimate of the time it should take to complete all steps. Plan your schedule so you have time to complete the lab. Once you start the lab, you will not be able to pause and return later (you begin at step 1 every time you start a lab).

3.The lab's Access time is how long your lab resources will be available. If you finish your lab with access time still available, you will be able to explore the Google Cloud Platform or work on any section of the lab that was marked "if you have time". Once the Access time runs out, your lab will end and all resources will terminate.

4.You DO NOT need a Google Cloud Platform account or project. An account, project and associated resources are provided to you as part of this lab.

5.If you already have your own GCP account, make sure you do not use it for this lab.

6.If your lab prompts you to log into the console, use only the student account provided to you by the lab. This prevents you from incurring charges for lab activities in your personal GCP account.

#Start your lab
When you are ready, click Start Lab. You can track your lab’s progress with the status bar at the top of your screen.

Important:What is happening during this time? Your lab is spinning up GCP resources for you behind the scenes, including an account, a project, resources within the project, and permission for you to control the resources needed to run the lab. This means that instead of spending time manually setting up a project and building resources from scratch as part of your lab, you can begin learning more quickly.

#Find Your Lab’s GCP Username and Password
To access the resources and console for this lab, locate the Connection Details panel in Qwiklabs. Here you will find the account ID and password for the account you will use to log in to the Google Cloud Platform:

If your lab provides other resource identifiers or connection-related information, it will appear on this panel as well.

#Log in to Google Cloud Console
Using the Qwiklabs browser tab/window or the separate browser you are using for the Qwiklabs session, copy the Username from the Connection Details panel and click the Open Google Console button.

You'll be asked to Choose an account. Click Use another account.
Paste in the Username, and then the Password as prompted:
Accept the terms and conditions.

Since this is a temporary account, which you will only have to access for this one lab:

#Do not add recovery options
#Do not sign up for free trials

##Launch AI Platform Notebooks
To launch AI Platform Notebooks:

#Step 1

Click on the Navigation Menu. Navigate to AI Platform, then to Notebooks.

#Step 2

On the Notebook instances page, click NEW INSTANCE. Select a 1.XX version of TensorFlow (not a 2.0) without GPUs. In the following example, you would select Tensorflow Enterprise 1.15 > Without GPUs:
Tensorflow 1.XX versions change semi-frequently, so the version you pick may be different.

In the pop-up, confirm the name of the deep learning VM and click Create.
The new VM will take 2-3 minutes to start.

#Step 3

Click Open JupyterLab. A JupyterLab window will open in a new tab.

##Clone Course Repo within your AI Platform Notebooks Instance
To clone the training-data-analyst notebook in your JupyterLab instance:

#Step 1

In JupyterLab, click the Terminal icon to open a new terminal.

#Step 2

At the command-line prompt, type in the following command and press Enter.

git clone https://github.com/GoogleCloudPlatform/training-data-analyst 

#Step 3

Confirm that you have cloned the repository by double clicking on the training-data-analyst directory and ensuring that you can see its contents. The files for all the Jupyter notebook-based labs throughout this course are available in this directory.

##Regression Model for AAPL Closing Price

#Step 1

In the notebook interface, navigate to training-data-analyst > courses > ai-for-finance > practice and open aapl_regression_scikit_learn.ipynb.

#Step 2

In the notebook interface, click on Edit > Clear All Outputs (click on Edit, then in the drop-down menu, select Clear All Outputs).

#Step 3

Read the narrative and execute each cell in turn. Complete each cell with a # TODO comment. If you get stuck, feel free to consult the solutions file by opening training-data-analyst > courses > ai-for-finance > solution > aapl_regression_scikit_learn.ipynb.


##Next Steps / Learn More

Official documentation for AI Platform Notebooks: https://cloud.google.com/ai-platform/notebooks/docs/


##End your lab
When you have completed your lab, click End Lab. Qwiklabs removes the resources you’ve used and cleans the account for you.

You will be given an opportunity to rate the lab experience. Select the applicable number of stars, type a comment, and then click Submit.

The number of stars indicates the following:

1 star = Very dissatisfied
2 stars = Dissatisfied
3 stars = Neutral
4 stars = Satisfied
5 stars = Very satisfied
You can close the dialog box if you don't want to provide feedback.

For feedback, suggestions, or corrections, please use the Support tab.

Building a Regression Model for a Financial DatasetIn this notebook, you will build a simple linear regression model to predict the closing AAPL stock price. The lab objectives are:
	* 
Pull data from BigQuery into a Pandas dataframe
	* 
Use Matplotlib to visualize data
	* 
Use Scikit-Learn to build a regression model


'''

%%bash

bq mk -d ai4f
bq load --autodetect --source_format=CSV ai4f.AAPL10Y gs://cloud-training/ai4f/AAPL10Y.csv
%matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

plt.rc('figure', figsize=(12, 8.0))

%%bigquery?

'''
Docstring:
::

%_cell_magic [--destination_table DESTINATION_TABLE] [--project PROJECT]
                [--max_results MAX_RESULTS]
                [--maximum_bytes_billed MAXIMUM_BYTES_BILLED] [--dry_run]
                [--use_legacy_sql] [--use_bqstorage_api] [--verbose]
                [--params PARAMS [PARAMS ...]]
                [destination_var]

Underlying function for bigquery cell magic

Note:
    This function contains the underlying logic for the 'bigquery' cell
    magic. This function is not meant to be called directly.

Args:
    line (str): "%%bigquery" followed by arguments as required
    query (str): SQL query to run

Returns:
    pandas.DataFrame: the query results.

positional arguments:
destination_var       If provided, save the output to this variable instead
                        of displaying it.

optional arguments:
--destination_table DESTINATION_TABLE
                        If provided, save the output of the query to a new
                        BigQuery table. Variable should be in a format
                        <dataset_id>.<table_id>. If table does not exists, it
                        will be created. If table already exists, its data
                        will be overwritten.
--project PROJECT     Project to use for executing this query. Defaults to
                        the context project.
--max_results MAX_RESULTS
                        Maximum number of rows in dataframe returned from
                        executing the query.Defaults to returning all rows.
--maximum_bytes_billed MAXIMUM_BYTES_BILLED
                        maximum_bytes_billed to use for executing this query.
                        Defaults to the context
                        default_query_job_config.maximum_bytes_billed.
--dry_run             Sets query to be a dry run to estimate costs. Defaults
                        to executing the query instead of dry run if this
                        argument is not used.
--use_legacy_sql      Sets query to use Legacy SQL instead of Standard SQL.
                        Defaults to Standard SQL if this argument is not used.
--use_bqstorage_api   [Beta] Use the BigQuery Storage API to download large
                        query results. To use this option, install the google-
                        cloud-bigquery-storage and fastavro packages, and
                        enable the BigQuery Storage API.
--verbose             If set, print verbose output, including the query job
                        ID and the amount of time for the query to finish. By
                        default, this information will be displayed as the
                        query runs, but will be cleared after the query is
                        finished.
--params <PARAMS [PARAMS ...]>
                        Parameters to format the query string. If present, the
                        --params flag should be followed by a string
                        representation of a dictionary in the format
                        {'param_name': 'param_value'} (ex. {"num": 17}), or a
                        reference to a dictionary in the same format. The
                        dictionary reference can be made by including a '$'
                        before the variable name (ex. $my_dict_var).
File:      /usr/local/lib/python3.5/dist-packages/google/cloud/bigquery/magics.py
The query below selects everything you'll need to build a regression model to predict the closing price of AAPL stock. The model will be very simple for the purposes of demonstrating BQML functionality. The only features you'll use as input into the model are the previous day's closing price and a three day trend value. The trend value can only take on two values, either -1 or +1. If the AAPL stock price has increased over any two of the previous three days then the trend will be +1. Otherwise, the trend value will be -1.

Note, the features you'll need can be generated from the raw table ai4f.AAPL10Y using Pandas functions. However, it's better to take advantage of the serverless-ness of BigQuery to do the data pre-processing rather than applying the necessary transformations locally.
'''

%%bigquery df
WITH
  raw AS (
  SELECT
    date,
    close,
    LAG(close, 1) OVER(ORDER BY date) AS min_1_close,
    LAG(close, 2) OVER(ORDER BY date) AS min_2_close,
    LAG(close, 3) OVER(ORDER BY date) AS min_3_close,
    LAG(close, 4) OVER(ORDER BY date) AS min_4_close
  FROM
    `ai4f.AAPL10Y`
  ORDER BY
    date DESC ),
  raw_plus_trend AS (
  SELECT
    date,
    close,
    min_1_close,
    IF (min_1_close - min_2_close > 0, 1, -1) AS min_1_trend,
    IF (min_2_close - min_3_close > 0, 1, -1) AS min_2_trend,
    IF (min_3_close - min_4_close > 0, 1, -1) AS min_3_trend
  FROM
    raw ),
  train_data AS (
  SELECT
    date,
    close,
    min_1_close AS day_prev_close,
    IF (min_1_trend + min_2_trend + min_3_trend > 0, 1, -1) AS trend_3_day
  FROM
    raw_plus_trend
  ORDER BY
    date ASC )
SELECT
  *
FROM
  train_data

'''
View the first five rows of the query's output. Note that the object df containing the query output is a Pandas Dataframe.
'''
print(type(df))
df.dropna(inplace=True)
df.head()

'''
Visualize data
The simplest plot you can make is to show the closing stock price as a time series. Pandas DataFrames have built in plotting funtionality based on Matplotlib.
'''

df.plot(x='date', y='close');

'''
You can also embed the trend_3_day variable into the time series above.
'''

start_date = '2018-06-01'
end_date = '2018-07-31'

plt.plot(
    'date', 'close', 'k--',
    data = (
        df.loc[pd.to_datetime(df.date).between(start_date, end_date)]
    )
)

plt.scatter(
    'date', 'close', color='b', label='pos trend', 
    data = (
        df.loc[df.trend_3_day == 1 & pd.to_datetime(df.date).between(start_date, end_date)]
    )
)

plt.scatter(
    'date', 'close', color='r', label='neg trend',
    data = (
        df.loc[(df.trend_3_day == -1) & pd.to_datetime(df.date).between(start_date, end_date)]
    )
)

plt.legend()
plt.xticks(rotation = 90);

'''
Build a Regression Model in Scikit-Learn
In this section you'll train a linear regression model to predict AAPL closing prices when given the previous day's closing price day_prev_close and the three day trend trend_3_day. A training set and test set are created by sequentially splitting the data after 2000 rows.
'''

features = ['day_prev_close', 'trend_3_day']
target = 'close'

X_train, X_test = df.loc[:2000, features], df.loc[2000:, features]
y_train, y_test = df.loc[:2000, target], df.loc[2000:, target]

# Create linear regression object. Don't include an intercept,
# TODO
regr = linear_model.LinearRegression(fit_intercept = False)

# Train the model using the training set
# TODO
regr.fit(X_train,y_train)

# Make predictaions using the testing set
# TODO
y_pred = regr.predict(X_test)

# Print the root mean squared error of your predictions
# TODO
# The mean squared error
print('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

# Print the variance score (1 is perfect prediction)
# TODO
print('Variance Score: {0:.2f}'.format(r2_score(y_test, y_pred)))

# Plot the predicted values against their corresponding true values
# TODO
plt.scatter(y_test, y_pred)
plt.plot([140, 240], [140, 240], 'r--', label='perfect fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend();

# TODO
print('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, X_test.day_prev_close))))


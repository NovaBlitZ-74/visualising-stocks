pip install quandl

# this program predicts stock prices by using machine learning models
# Install the dependencies

import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#get the stock data

df = quandl.get("WIKI/AMZN")

# Take a look at the data

print(df.head())

#get the Adjusted Close Price

df = df[['Adj. Close']] #this is our independent variable

#Take a look at the new data

print(df.head())

# A variable for predicting 'n' days out in the future

forecast_out = 30 #this is our target variable, what we want to predict

#Create another column (the target or dependent variable) shifted 'n' units up

df['prediction'] = df[['Adj. Close']].shift(-forecast_out)

#print the new data set

print(df.tail())

### Create the independent data set (X) ####
# Convert the dataframe to a numpy array 
# We are gonna drop that new column that we called 'prediction' beacuse that we're 
#using the independent variable just adjusted close

X = np.array(df.drop(['prediction'],1))

# Remove the last 'n' rows

X = X[:-forecast_out] #last 30 rows of every column is removed
print(X)

### Create the dependent data set (y) ####
# Convert the dataframe to a numpy array (All of the values including the NaN's)

y = np.array(df['prediction'])

#Get all of the y values except the last 'n' rows

y = y[:-forecast_out]
print(y)

# Spplit the data into 80% training and 20% testing

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the Support Vector Machine (Regressor)
# we'll be using the method svr

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

# Testing Model : Score returns the coefficient of determination of R^2 of the prediction.
# The best possible score is 1.0

svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence : ", svm_confidence)

#Support Vector Regresson is better than Linear Regressor

# Create and train the Linear Regression Model

lr = LinearRegression()

# Train the model

lr.fit(x_train, y_train)

# Testing Model : Score returns the coefficient of determination of R^2 of the prediction.
# The best possible score is 1.0

lr_confidence = lr.score(x_test, y_test)
print("lr confidence : ", lr_confidence)

# Create the values that we are gonna forecast on
# Set x_foercast equal to the last 30 rows of the original data set from Adj. Close column

x_forecast = np.array(df.drop(['prediction'],1))[-forecast_out:] #this last syntasx will give me the last 30 rows from all of the columns in this set

# we will drop the prediciton column from our original data set

print(x_forecast)

# Print linear regression model for the next 'n' days

lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

# Print linear regression model for the next 'n' days
svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)
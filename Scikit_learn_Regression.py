"""linear Regression"""
"""The most significant difference between regression vs classification is that 
- regression helps predict a continuous quantity (example: the price) 
- These values are numerical: price of a house, quantity of oxygen in the air of a city, etc...
The target variable can therefore take an infinity of values."""


"""univariate linear model"""
# we have two variables:
# -** y ** called target variable or target(label) and ** x ** called explanatory variable(feature).
# -**Linear regression** consists in modeling the **link between** these two **variables** by an affine function.
# -Thus, the formula of the univariate linear model is given by:
# y≈β1x+β0
"""
-y is the variable we want to **predict**.
-x is the **explanatory variable**.
-β1 and β0 are the parameters of the affine function. 
-β1 will define its slope and β0 will define its intercept (also called bias)."""

"""
The goal of linear regression is to estimate the best parameters β0 and β1 
to predict the variable y from a given value of x."""


"""Multiple Linear Regression"""
# Multiple linear regression consists in modeling the link between a :
# target variable y and several explanatory variables x1 , x2 , ... , xp ,**(features)**:
# y≈β0+β1x1+β2x2+⋯+βpxp
# ≈β0+∑j=1pβjxj


"""linear regression example"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd

df = pd.read_csv("data/automobiles.csv")

# df.iloc[0:5,:]

a = df.drop(df.iloc[:, 2:9], axis=1)


df = a.drop(a.loc[:, ["engine-type", "num-of-cylinders", "fuel-system"]], axis=1)
df.dtypes
df.head()
# 	    symboling	normalized-losses	wheel-base	length	width	height	curb-weight	    engine-size 	bore	stroke	    compression-ratio	horsepower	peak-rpm	city-mpg	highway-mpg 	price
# 0	        3	        ?	            88.6	    168.8	64.1	48.8	2548	         130	        3.47	2.68	        9.0	            111	            5000	21	            27	        13495
# 1	        3	        ?	            88.6	    168.8	64.1	48.8	2548	         130	        3.47	2.68	        9.0	            111	            5000	21	            27	        16500
# 2	        1	        ?	            94.5	    171.2	65.5	52.4	2823             152	        2.68	3.47	        9.0	            154	            5000	19	            26	        16500
# 3	        2	        164	            99.8	    176.6	66.2	54.3	2337	         109	        3.19	3.4	            10.0	        102	            5500	24	            30	        13950
# 4	        2	        164	            99.4	    176.6	66.4	54.3	2824	         136	        3.19	3.4	            8.0	            115	            5500	18	            22	        17450

# df.isna().any()
# df['price'] = df['price'].replace('?',0)
df.replace(r"?", np.nan, inplace=True)

# df = df.astype(float)

# symboling              int64
# normalized-losses     object
# wheel-base           float64
# length               float64
# width                float64
# height               float64
# curb-weight            int64
# engine-size            int64
# bore                  object
# stroke                object
# compression-ratio    float64
# horsepower            object
# peak-rpm              object
# city-mpg               int64
# highway-mpg            int64
# price                 object

# df['normalized-losses'] = df['normalized-losses'].fillna(0)

df.update(
    df[
        ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"]
    ].fillna(0)
)
df.head()

df.isna().any()
# df.dtypes
df = df.astype(float)
df.mean()

df = df.replace(0, df.mean())
# df.replace(0, df.mean())
df.head()


"""The last price variable corresponds to the selling price of the vehicle. 
This is the variable that we will seek to predict."""

# we must separate the datarame to two dataframes
# first one (features)
X = df.drop(["price"], axis=1)

# second one (target)
y = df["price"]


"""we must import the (train_test_split from sklearn.model_selection)"""

# This function is used as follows:
"""
- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
- X_train and y_train are the explanatory and target variables of the **training** dataset.
- X_test and y_test are the explanatory and target variables of the **test dataset**.
- The **test_size** argument is the proportion of the dataset we want to keep for the test set. 
- In the previous example, this proportion corresponds to 20% of the initial data set."""
"""To eliminate the randomness of the train_test_split function we use **randomstate**"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20
)


"""
To train a linear regression model on this dataset, we will use 
the LinearRegression class contained in the linear_model submodule of scikit-learn."""
# from sklearn.linear_model import LinearRegression


"""All scikit-learn model classes have the following two methods:
- fit: Train the model on a dataset.
- predict: Makes a prediction from explanatory variables(features)."""

"""Example"""
# Here is an example of using a model with scikit-learn:

# # Model instantiation
# linreg = LinearRegression()

# # Train the model on the training set
# linreg.fit(X_train, y_train)

# Prediction of the target(label) variable for the test dataset.
# These predictions are stored in y_pred

# y_pred = linreg.predict(X_test)


lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)

y_pred_test = lr.predict(X_test)


"""
-In order to evaluate the quality of the model predictions obtained thanks to the parameters β0, ..,βj, 
there are several metrics in the scikit-learn library:

-One of the most used metrics for regression is the (Mean Squared Error: MSE) 
which exists under the name mean_squared_error in the metrics sub-module of scikit-learn.

-This function consists of **calculating** the average of the **distances** between the **target variables**
 and the **predictions** obtained using the regression function.
"""

"""The mean_squared_error function of scikit-learn is used as follows:
mean_squared_error(y_train, y_pred)

where :
- y_true(y_train) corresponds to the true values of the target(label) variable.
- y_pred corresponds to the values predicted by our model."""

## we need to import  sklearn.metrics from mean_squared_error ##
mse_train = mean_squared_error(y_train, y_pred_train)

mse_test = mse_test = mean_squared_error(y_test, y_pred_test)

print(f"MSE train = {mse_train}")
print(f"MSE test = {mse_test}")

# MSE train = 10287138.750342472
# MSE test = 23512524.963038027

"""
The root mean square error you find should be in the millions on the test data, 
which can be difficult to interpret.

This is why we are going to use another metric, the (Mean Absolute Error : MAE), 
which is on the same scale as the target(label) variable.
"""

## we need to import mean_absolute_error from sklearn.metrics ##

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"MAE train = {mae_train}")
print(f"MAE test = {mae_test}")

# MAE train = 2403.405021005788
# MAE test = 2625.9322421492175


# From the DataFrame df, calculate the average purchase price across all vehicles.
mean_price = df["price"].mean()

# Do the model's predictions seem reliable to you?
print("\nRelative error", mae_test / mean_price)
# Relative error 0.1989026012174232

# The average error is around 20% of the average price, which is not optimal
# but is still a good baseline for testing more advanced models.
# ------------------------------------------------------------------------------
"""GradientBoostingRegressor"""

"""because we have an error more than 20%:
we will create another regression model which learns very well on the training data 
but which generalizes very poorly on the test data: this is called **overfitting**"""

## we will import GradientBoostingRegressor from sklearn.ensemble ##
gbr = GradientBoostingRegressor(
    n_estimators=1000, max_depth=10000, max_features=15, validation_fraction=0
)
gbr.fit(X_train, y_train)

y_pred_gbr_train = gbr.predict(X_train)

y_pred_gbr_test = gbr.predict(X_test)

# we calculate the Mse with th new regression model (overfitting)
mse_gbr_train = mean_squared_error(y_train, y_pred_gbr_train)

mse_gbr_test = mean_squared_error(y_test, y_pred_gbr_test)

print(f"MSE_GB_TR = {mse_gbr_train}")
print(f"MSE_GB_TS = {mse_gbr_test}")

# MSE_GB_TR = 15254.234756097561
# MSE_GB_TS = 25965848.604480684

# we calculate the Mae with th new regression model (overfitting)
mae_gbr_train = mean_absolute_error(y_train, y_pred_gbr_train)

mae_gbr_test = mean_absolute_error(y_test, y_pred_gbr_test)

print(f"MAE_GB_TR = {mae_gbr_train}")
print(f"MAE_GB_TS = {mae_gbr_test}")

# MAE_GB_TR = 26.628048780497206
# MAE_GB_TS = 2342.5183240091987


# Do the model's predictions seem reliable to you?
mean_price = df["price"].mean()
print(f"Relative error = {mae_gbr_test/mean_price}")
# Relative error = 0.17743526682301644

# ------------------------------------------------------------------------------------------------
"""Polynomial Linear Regression"""
"""
-in many cases, the relationship between the x and y variables is **not linear**.
This does not allow us to use linear regression to predict y. 
We could then propose a **quadratic model** such as:
y=β0+β1x+β2x2

-
Polynomial linear regression amounts to performing a classical linear regression from polynomial functions
of the explanatory variable of arbitrary degree.
"""

##  we need to import PolynomialFeatures from sklearn.preprocessing ##
"""
The degree parameter defines the degree of the polynomial features to be calculated.
"""
# poly_feature_extractor = PolynomialFeatures(degree = 2)

"""
using the fit_transform method.
We can compute the polynomial features on X_train and X_test like this:"""

poly_feature_extractor = PolynomialFeatures(degree=3)

X_train_poly = poly_feature_extractor.fit_transform(X_train)

X_test_poly = poly_feature_extractor.fit_transform(X_test)

poly_reg = LinearRegression()

poly_reg.fit(X_train_poly, y_train)

y_pred_train = poly_reg.predict(X_train_poly)

mae_train = mean_absolute_error(y_train, y_pred_train)

print(f"MAE train = {mae_train}")
# MAE train = 36.703946119265225

y_pred_test = poly_reg.predict(X_test_poly)

mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"MAE test = {mae_test}")
# MAE test = 134144.6944210154

mean_price = df["price"].mean()
print("\nRelative error", mae_test / mean_price)
# Relative error 10.160859534600338

"""
- We are absolutely in an overfitting regime.
- Polynomial regression model performs well on training data but not on test data.
- The 3rd order polynomial regression model performs much worse than simple linear regression."""

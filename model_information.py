# Supervised Learning
"""
Continuous Target (ex. Profit Sales)

- Linear Regression
- Lasso
- Ridge
- ElasticNet

Linear Regression:
	- tidy problems
	- retrospective analysis
	- analysis in general
y ~ b0 + b1X1 + b2X2 ... bnXn -> solve by minimizing (RSS)
*** an increase in X will increase y by the beta coefficient ***

Lasso:
	- too many irrelevant variables
	- variables that are noise
	- goal of feature selection and eliminate variables
	- can also be used with Logistic Regression
y ~ b0 + b1X1 + b2X2 ... bnXn -> solve by minimizing (RSS + alpha∑|b|)

Ridge:
	- correlated variables
	- make variables equally affect the target variable
	- can also be used with Logistic Regression

ElasticNet:
	- Lasso + Ridge
	- Usually stronger than Lasso and Ridge
	- can also be used with Logistic Regression

What are Lasso, Ridge and ElasticNet doing?
Increasing Bias
"""

# Machine Learning
"""
Decision Trees
Ensemble Methods:
	- Random Forests
	- Bagging
	- Boosting
"""

# Centering vs Standardizing
"""
Centering - subtracting the mean
1,2,3,4,5 -> -2,-1,0,1,2

Standardizing - center and divide by standard deviation
1,2,3,4,5 -> -1.3, -0.5, 0.5, 1.3

Any penalty on regularization needs standardization
If you have 2 variables, 1 is 0-1 and 1 is 0-1,000,000 and the 0-1 is the better predictor, Lasso
may be reduced or completely remove the better predictors. Putting all variables on the same scale
allows correct analysis by Lasso.
"""

# Things to consider for picking models
"""

"""
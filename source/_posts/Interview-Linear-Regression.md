---
title: 'DS Interview: Linear Regression'
date: 2017-03-09 16:52:30
tags: 
- Data Science Interviews
- Machine Learning
category: 
- 时习之
- Machine Learning
description: Data Science Interview Questions for Linear Regression
---

#### Definition
* **How would linear regression be described and explained in layman's terms?** <br>
  Linear regression is to *find the best fitting line* given a bunch of points. The distance to the line is the "error" - because ideally we prefer every point is on that line. Mathematically, we want a line that produces as small total error as possible. That's what we do in linear regression. 

* **What is an intuitive explanation of a multivariate regression?** <br>
  Normally regression only has one outcome (Y), and several predictor variables (X). In Multivariate regression, there are more than one outcomes (Y). 

* **What is gradient descent method? Will gradient descent methods always converge to the same point?**<br>
  Gradient descent is an interative algorithm to find optimal solution. Sometimes, for example, in k-means clustering it will converge to different local optimals with different initializations. In regression, yes, it will always converge to the global optimal point.

* **What is the difference between linear regression and least squares?** <br>
  Linear regression is a statistical inference problem / machine learning model; and least squares describes one way to achieve the solution to the problem/ model. Alternatively, we can use MAE as measurement and gradient descent to find the solution to linear regression model.

#### Statistical Perspective

* **What are the assumptions required for linear regression? What if some of these assumptions are violated?** <br>
  First, variables are **normal distributed**. Can check with histogram/ QQ plot. <br> 
  Second, relationship is **linear** between independent and dependent variables (not polynomial etc.). Scatter plot is helpful in two dimensional case. <br>  *Solution: change the model to include polynomial terms.* <br>
  Third, no linear dependency between predictors (no **multi-collinearity**). Use correlation/ covariance matrix to detect. <br>*Solution: dimensional reduction (PCA), select one particular variable from highly correlated variable set, or  ridge regression (without expecting the coefficient of one single variable explain much)* <br>
  Fourth, there is no correlation between error terms (**no autocorrelation**). This means, dependent variables (y) are independent of each other. Use residual plot or Durbin – Watson test. <br> *Solution: time-series modelling* <br>
  Last, **homoscedasticity**. It means the residual remains same as x changes. Scatter plot with x,y,fitting line, residuals is a good measurement. <br> *Solution: log transformation, box-cox transformation*

* **What is collinearity and what to do with it? How to remove multicollinearity?** <br>
  This means there are at least two predictors are highly correlated. It will make interpretation of coefficients, lead to overfitting, or even failure of inverting the matrix. 
  To remove multicollinearity, there are generally four ways:
  1, drop affected variables; 2, dimensional reduction (PCA); 3, ridge regression; 4, partial least square regression.

* **Can you derive the ordinary least square regression formula?** <br>


#### Regularization 
* **What is an intuitive explanation of regularization?** <br>
  Regularization is a technique to prevent overfitting. It enables the model to learn just the right amount of information about the data. Basically it penalizes the model if it goes too detailed that even learn specific patterns that will not necessarily present in the future.

* **What is the difference between L1 and L2 regularization?** <br>
  L1 regularization gives sparse estimation by giving a Lapace prior for coefficients. It will force most of coefficients to be zero.
  L2 regularization will retain all predictors with Gaussian prior for coefficients. Some of the coefficients get larger and some get smaller. It's easier to compute.

* **What are the benefits and drawbacks of specific methods, such as ridge regression and lasso?** <br>
  Ridge regression is faster than lasso; lasso regression is a feature selection method itself. But we can't claim which method to use without observing the distribution of data. <br>

#### Model Tuning/ Selection
* **How do you choose the right number of predictors?** <br>
  One lazy method is lasso regression. We can also go through a feature selection process: First we decide the metric, such as MSE or R square. Then we can use greedy search and cross validation to build models with different combinations of predictors; and evaluate the models with test set. <br>

* **How to check if the regression model fits the data well?** <br>
  Adjusted R square, MSE

* **What is the difference between squared error and absolute error? and Are there instances where root mean squared error might be used rather than mean absolute error? and How would a model change if we minimized absolute error instead of squared error? What about the other way around?** <br>
  Squared error/ root mean squared error  - penalize more on big errors; while MAE is more robust to outliers. This is because taking square of a small item (< 1) is even smaller, square of a large item is even larger.


#### Others
* **What are the drawbacks of linear model? Are you familiar with alternatives (Lasso, ridge regression, boosted trees)?** <br>
  Drawbacks: Rigid assumptions; overfitting problem; linearity; see previous regularization part; boosted trees turn a series of weak prediction models into a strong prediction by voting.

* **Assume you need to generate a predictive model using multiple regression. Explain how you intend to validate this model**<br>
  Cross validation; R^2 or MSE; Residual analysis (see statistical part)

* **Do we always need the intercept term in a regression model?** <br> Yes. Intuitively speaking, without the constant, the regression line has to cross the origin, but it's not always the case. Statistically speaking, it allows the mean of residuals to be 0.

* **Provide three examples of relevant phenomena that have long tails. Why are they important in classification and regression problems?** <br>
  Natural language (Zipf's law); customer purchase; wealth. The problem associated include it violates the normal distribution and linearity assumption for linear model and create accuracy paradox in classification problem. We will need to perform log transformation for long tailed data. 
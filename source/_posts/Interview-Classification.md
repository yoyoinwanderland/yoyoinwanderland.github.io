---
title: 'DS Interview: Classification'
date: 2017-03-09 16:52:45
tags: 
- Data Science Interviews
- Machine Learning
category: 
- 时习之
- Machine Learning
description: Data Science Interview Questions for Classification
---

#### Definition
* **Explain what logistic regression is. How do we train a logistic regression model? How do we interpret its coefficients?** <br>
  Logistic Regression is mapping inputs to probabilities. The sigmoid function turns a non-differentiable cost function to a convex one. So training a logistic regression model is a matter of optimization problem, we can deploy gradient descent ( cost function) or gradient ascent ( likelihood function). The coefficient means unit of increase of the log odd [log P(y=1) - log P(y=0)] for one unit increase of a predictor, given other predictors constant.

* **How do random forests work (in layman's terms)?** <br>
  Random forest works in the following way: <br>
  First, the random forests compromise of decision trees (that's why it is a random forest!). The decision trees learn and VOTE; output with majority vote will be taken. <br>Second, the random forests takes an tree bagging (boostrap aggregation) process: each of the decision trees are built with only a subset of the original train data - it samples with replacement. <br> Third, random forests takes a feature bagging process: each of the decision trees will only train on a subset of all features. 

* **What is the maximal margin classifier? How this margin can be achieved and why is it beneficial? How do we train SVM?** <br>
  Data with different labels are separated by a hyperplane. Maximal margin classifier minimizes the total distance from the points to the hyperplane. Maximal margin classifier tolerate more noises thus more robust to overfitting. We can train it either using linear programming or gradient descent. 

* **What is a kernel? How to choose kernel? Explain the kernel trick.** <br>
  Kernel is legal definition of dot product. Choosing kernel requires domain knowledge, or we can simply apply cross validation. When the data are not seperable in the current feature space, a common approach is to map them into high-dimensional space and then find the hyperplane. The kernel trick is, we don't have to compute the mapping coordinates of those features in high dimensional space - we only perform inner product multiplication in the current space and then raise to the power of n. Thus it saves a lot of computational resources.
* **Why is naive Bayes so bad? How would you improve a spam detection algorithm that uses naive Bayes?** <br> 
  Naive bayes has its name because it assumes all the features to be independent from each other. This is not the case, for example in NLP tasks. Ways to improve Naive Bayes: <br> Complement Naive Bayes: also count words NOT IN the category;<br> Feature engineering, such as retaining words with high kappa, stemming etc; <br>PCA / covariance matrix could be employed. 

* **What is an Artificial Neural Network? What is back propagation?**
  It is a classification method that model brain neuras to solve complex classification problems. Back propagation is a "learn from mistake" algorithm for neural network training. Back propagation, in its nature, is gradient descent. It compares the output with the label, and "propagate" the error back onto the previous layers and adjust the weights accordingly. 

#### Model Comparison
* **What are the advantages of different classification algorithms? (Decision tree, Naive Bayes, logistic regression, SVM, Random Forest)** <br> 
  **Decision Tree:** <br> Pros: 1, Easy to interpret; 2, more robust to outliers; 3, works for non-linear separable data <br> Cons:  1, Tree structures change; 2, Easy to overfit; 3, Performs relatively worse in linear separable tasks <br> **Naive Bayes:**<br>Pros: 1, Easy to train, converge quickly; 2, Gives generative story; <br>Cons: 1, Constraint: Independence assumption; 2, Easily affected by outliers <br> **Logistic Regression:** <br> Pros: 1, Can apply regularization to prevent overfitting; 2, Easy to interpret with probabilities; 3, Well-performed; <br>Cons: 1, Linear separable; 2, Complex for multi-class problem <br> **SVM:** <br>Pros: 1, Good performance; 2, Can apply different kernel (linear/ non-linear separable) <br>Cons: 1, Slow to train, memory intensive (especially with cross validation for kernel selection); 2, complicated for multi-class problem <br> **Random Forests:**<br>Pros: 1, Very good performance; 2, Handles large numbers of features; 3, Tells which features matter more; 4, Deals well with missing data <br>Cons: 1, Unlike decision tree, the process is hard to interpret; 2, Was observed to be overfitting on some datasets; 3, Slow to predict

* **How can I classify supervised data that is probabilistic rather than deterministic?** <br> Either logistic regression, or linear regression with normalization.


#### Metrics
* **How to define/select metrics?** <br>
* **What are benefits and weaknesses of various binary classification metrics? What is the best out-of-sample accuracy test for a logistic regression and why?**  <br> 
* **Provide examples when false positives are more important than false negatives, false negatives are more important than false positives** <br> False positive is more important during early stage health scanning. At this moment the cost of delaying small problems are not big as false alarms bringing unnecessary worries to many patients. <br>False negative is more important during cancer related examinations. Misses will cost lives.

#### Cross validation
* **What is cross-validation?** <br>
  Cross validation is a technique to assess how well the model perform on unseen data. It is used for either, model selection, or model performance estimation. Basically it randomly seperates the data into k-folds and train on k-1 folds and test on the last fold. k is always chosen to be 3, 5, 10.

* **What are the pitfalls on relying on cross-validation to select models?**<br> Generally we use Cross validation for two purpose, model selection& estimate model accuracy for future use. <br>For model selection, cross validation actually measures "accuracy". Then it will easily falls into accuracy paradox - that increasing accuracy doesn't lead to a more desired model (the email spam filter could increase accuracy by setting the rule "no email is spammed", as TP < FP).  So CV is not robust in this sense. <br> For estimate model accuracy - normally it is training and test data falls in the same pattern, but the future data might not have the same pattern as your training/ test data. Potentially data shift exists.


---
title: 'Learn ML from Sklearn: Cross Validation'
date: 2017-08-16 09:56:23
tags: 
- Model Selection
- Machine Learning
- Python Packages
category: 
- 时习之
- Machine Learning
description: study notes for scikit-learn 2
---

## Overfitting in Two Ways

* Learn parameters and test the model in the same dataset

  Solution: train-test

  ```
  X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.4, random_state=0)
  ```

* Tune the hyperparameters and test the model in the same dataset

  > When evaluating different settings (“hyperparameters”) for estimators, such as the `C` setting that must be manually set for an SVM, there is still a risk of overfitting *on the test set* because the parameters can be tweaked until the estimator performs optimally. This way, knowledge about the test set can “leak” into the model and evaluation metrics no longer report on generalization performance. 

  Solution: Train-validation-test, Cross validation 

  ```
  scores = cross_val_score(model, iris.data, iris.target, cv=5, scoring='f1_macro')
  ```

  Note: from my point of view, cross validation is not a very clean solution for  hyperparameter tuning. A small test set is still needed to see the generalization error. But it is a good way to see if the model is stable. If the validation error varies amongst different left out samples, then there might be some problems.

### Visualize Overfitting & Underfitting

#### Effect of a hyper-parameter

![overfitting hyperparameter](http://scikit-learn.org/stable/_images/sphx_glr_plot_validation_curve_0011.png)

```
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge

train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha", np.logspace(-7, 3, 3))
```



#### Effect of the number of training samples

![overfitting training no](http://scikit-learn.org/stable/_images/sphx_glr_plot_learning_curve_0021.png)

```
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

train_sizes, train_scores, valid_scores = learning_curve(
     SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)
```



## K Folds

### K-Fold

> [`KFold`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold) divides all the samples in k groups of samples, called folds ( if `k=n` this is equivalent to the *Leave One Out* strategy), of equal sizes (if possible). The prediction function is learned using `k - 1 ` folds, and the fold left out is used for test.

```
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, random_state = 1)
for train_index, test_index in kf.split(X):
     X_train, y_train = X[train_index], y[train_index]
```

### Stratified K-Fold

Use stratified K-Fold when the class is unbalanced.

> [`StratifiedKFold`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold) is a variation of *k-fold* which returns *stratified* folds: each set contains approximately the same percentage of samples of each target class as the complete set.

```
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, random_state = 1)
for train, test in skf.split(X, y):
	X_train, y_train = X[train_index], y[train_index]
```

### Group K-Fold

> An example would be when there is medical data collected from multiple patients, with multiple samples taken from each patient. And such data is likely to be dependent on the individual group. In our example, the patient id for each sample will be its group identifier.
>
> In this case we would like to know if a model trained on a particular set of groups generalizes well to the unseen groups. To measure this, we need to ensure that all the samples in the validation fold come from groups that are not represented at all in the paired training fold.

```
from sklearn.model_selection import GroupKFold

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

gkf = GroupKFold(n_splits=5)
for train, test in gkf.split(X, y, groups=groups):
     print("%s %s" % (train, test))
```

### Time Series Split

> Time series data is characterised by the correlation between observations that are near in time (*autocorrelation*). However, classical cross-validation techniques assume the samples are independent and identically distributed, and would result in unreasonable correlation between training and testing instances (yielding poor estimates of generalisation error) on time series data. Therefore, it is very important to evaluate our model for time series data on the “future” observations least like those that are used to train the model. 

> Note that unlike standard cross-validation methods, successive training sets are supersets of those that come before them. Also, it adds all surplus data to the first training partition, which is always used to train the model.

```
from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
tscv = TimeSeriesSplit(n_splits=3)
for train, test in tscv.split(X):
     print("%s %s" % (train, test))
#>>> [0 1 2] [3]
#>>> [0 1 2 3] [4]
#>>> [0 1 2 3 4] [5]
```

## Tuning Hyper-parameters

So Scikit-learn provides tools to tune hyper-parameters. That's to say, we don't have start with train-validation-test and then input different hyper-parameter and then print out validation error. We can input the desire model, and a list of hyper-parameters to choose from, and then scikit-learn will iterate and gives the best combination.

> Model selection by evaluating various parameter settings can be seen as a way to use the labeled data to “train” the parameters of the grid. When evaluating the resulting model it is important to do it on held-out samples that were not seen during the grid search process: it is recommended to split the data into a **development set** (to be fed to the `GridSearchCV` instance) and an **evaluation set** to compute performance metrics.



There are two ways to tune hyper-parameters.

### Grid Search

> The grid search provided by [`GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) exhaustively generates candidates from a grid of parameter values specified with the `param_grid` parameter. 

```
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100}]

clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='precision')
clf.fit(X_train, y_train)
print (clf.best_params_)
print (clf.cv_results_['mean_test_score'])

y_true, y_pred = y_test, clf.predict(X_test)
```



### Randomized Search

> [`RandomizedSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV) implements a randomized search over parameters, where each setting is sampled from a distribution over possible parameter values. This has two main benefits over an exhaustive search:
>
> - A budget can be chosen independent of the number of parameters and possible values.
> - Adding parameters that do not influence the performance does not decrease efficiency.

```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
              
clf = RandomForestClassifier(n_estimators=20)

# run randomized search
random_search = RandomizedSearchCV(clf, param_dist, n_iter=20)
random_search.fit(X, y)

print (clf.best_params_)
print (clf.cv_results_['mean_test_score'])

y_true, y_pred = y_test, clf.predict(X_test)
```

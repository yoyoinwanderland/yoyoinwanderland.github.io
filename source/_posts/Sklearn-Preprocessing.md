---
title: Learn ML from Sklearn Preprocessing
date: 2017-06-25 10:59:41
tags: 
- Preprocessing
- Machine Learning
- Python Packages
category: 
- 时习之
- Machine Learning
description: study notes for scikit-learn 1
---
## Preprocessing

### Encoding Categorical Features

> Integer representation can not be used directly with scikit-learn estimators, as these expect continuous input, and would interpret the categories as being ordered, which is often not desired (i.e. the set of browsers was ordered arbitrarily).
>
> One possibility to convert categorical features to features that can be used with scikit-learn estimators is to use a one-of-K or one-hot encoding, which is implemented in `OneHotEncoder` This estimator transforms each categorical feature with `m` possible values into `m` binary features, with only one active.

```
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
X_encoded = enc.fit_transform(X)  


#if X contains missing categorical features, one has to explicitly set n_values
enc = preprocessing.OneHotEncoder(n_values=[1, 2, 3, 4])
X_encoded = enc.fit_transform(X)  
```

### Imputation of Missing Values

> A basic strategy to use incomplete datasets is to discard entire rows and/or columns containing missing values. However, this comes at the price of losing data which may be valuable (even though incomplete). A better strategy is to impute the missing values, i.e., to infer them from the known part of the data.

> Basic strategies for imputing missing values: either using the **mean**, the **median** or the **most frequent value** of the row or column in which the missing values are located.

```
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_imputed = imp.fit(X)
```



### Standardization

> It is sometimes not enough to center and scale the features independently, since a downstream model can further make some assumption on the linear independence of the features. To address this issue you can use PCA with `whiten=True` to further remove the linear correlation across features.

#### Standardization

> **Standardization** of datasets is a **common requirement for many machine learning estimators** implemented in scikit-learn.
>
> If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.

```
from sklearn import preprocessing
X_scaled = preprocessing.scale(X)
```

> Just as it is important to test a predictor on data held-out from training, preprocessing (such as standardization, feature selection, etc.) should be learnt from a training set and applied to held-out data for prediction:

```
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4)
# instead of preprocessing.scale, it's recommended to use scaler
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)
```

#### Scaling sparse data

Summary: do the standardization without centering.

> **Centering** sparse data would destroy the sparseness structure in the data, and thus rarely is a sensible thing to do. However, it can make sense to scale sparse inputs *without centering*, especially if features are on different scales. 

```
from sklearn import preprocessing
max_abs_scaler = preprocessing.MaxAbsScaler()
X_scaled = max_abs_scaler.fit_transform(X)

## alternative
scaler = preprocessing.StandardScaler(with_means = False)
X_scaled = scaler.fit(X)
```



#### Scaling data with outliers

Summary: use median instead of mean, use IQR instead of standard deviation.

> This Scaler removes the **median** and scales the data according to the **quantile range** (defaults to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
>
> Standardization of a dataset is a common requirement for many machine learning estimators. Typically this is done by removing the mean and scaling to unit variance. However, outliers can often influence the sample mean / variance in a negative way. In such cases, the median and the interquartile range often give better results.

```
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
X_scaled = robust_scaler.fit_transform(X)
```



### Normalization

> **Normalization** is the process of **scaling individual samples to have unit norm**. 
>
>  This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples. This assumption is the base of the *Vector Space Model* often used in **text classification** and **clustering** contexts.

```
from sklearn import preprocessing
X_l2_normalized = preprocessing.normalize(X, norm='l2')
X_l1_normalized = preprocessing.normalize(X, norm='l1')
```



### Binarization

> **Feature binarization** is the process of **thresholding numerical features to get boolean values**. This can be useful for downstream probabilistic estimators that make assumption that the input data is distributed according to a multi-variate Bernoulli distribution.

```
from sklean import preprocessing 
binarizer = preprocessing.Binarizer(threshold=1.1)
binarizer.transform(X)
```
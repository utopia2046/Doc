# Feature Pre-processing using Scikit-learn

## Scaling

### Standard Scaler (mean to 0 and std to 1)

``` python
from sklearn import preprocessing
import numpy as np

# create a standard sclaer on training set and then apply it to test set
scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
```

### Scaling into a range [mix, max]

``` python
min_max_scaler = preprocessing.MinMaxScaler
X_train_minmax = mi_max_scaler.fit_transform(X_train)
```

## Other scalers

* `MaxAbsScaler` which devide each value with max
* `StandardScaler` can access `scipy.sparse` matrices as input, with `with_mean=False`
* If there are outliers in your data, you can use `robust_scale` and `RobustScaler` as drop-in replacements instead.
* `QuantileTransformer` performs a rank transforming (leass influenced by outliers)

### Normalization

``` python
normalizer = preprocessing.Normalizer().fit(X)
X_norm = normalizer.transform(X)
```

### Encoding categorial features

``` python
enc = preprocessing.OneHotEncoder()
enc.fit(X_train)
enc.transform(X_train).toarray()
```

### Fill missing values

``` python
import numpy as np
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X_train)

```

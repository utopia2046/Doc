# Hands on Machine Learning with Scrikit Learn and TensorFlow Reading Notes

- [Hands on Machine Learning with Scrikit Learn and TensorFlow Reading Notes](#hands-on-machine-learning-with-scrikit-learn-and-tensorflow-reading-notes)
  - [Fundamental](#fundamental)
    - [Categories](#categories)
    - [Main Challenges](#main-challenges)
    - [No Free Lunch (NFL) Theorem](#no-free-lunch-nfl-theorem)
    - [Online Datasets](#online-datasets)
    - [Example: Housing Price Regression](#example-housing-price-regression)
  - [Classification](#classification)
  - [Regression](#regression)
  - [Support Vector Machines](#support-vector-machines)
  - [Decision Trees](#decision-trees)
  - [Ensemble Learning and Random Forests](#ensemble-learning-and-random-forests)
  - [Dimensionality Reduction](#dimensionality-reduction)
  - [Neural Networks and Deep Learning](#neural-networks-and-deep-learning)
  - [Convolutional Neural Networks](#convolutional-neural-networks)
  - [Recurrent Neural Networks](#recurrent-neural-networks)
  - [Autoencoders](#autoencoders)
  - [Reinforcement Learning](#reinforcement-learning)

## Fundamental

### Categories

- Based on human supervision
  - supervised
    - k-Nearest Neighbors
    - Linear Regression
    - Logistic Regression
    - Support Vector Machines (SVMs)
    - Decision Trees and Random Forests
    - Neural networks
  - unsupervised
    - Clustering
      - k-Means
      - Hierarchical Cluster Analysis (HCA)
      - Expectation Maximization
    - Visualization and dimensionality reduction
      - Principal Component Analysis (PCA)
      - Kernel PCA
      - Locally-Linear Embedding (LLE)
      - t-distributed Stochastic Neighbor Embedding (t-SNE)
    - Association rule learning
      - Apriori
      - Eclat
  - semisupervised
    - Deep Belief Networks (DBN)
  - reinforcement learning
    1. Observe
    2. Select action using policy
    3. Action
    4. Get reward or penalty
    5. Update polity (learning step)
    6. Iterate until an optimal policy is found
- Incrementally?
  - online learning (incremental learning)
  - batch learning
- Instance-based or model-based
  - comparing new data points to known data points
  - detect patterns in the training data and build a predictive model

### Main Challenges

- Insufficient Quantity of Training Data (data matters more than algorithms for complex problems)
- Nonrepresentative Training Data
- Poor-Quality Data
- Irrelevant Features, feature engineering involves:
  - Feature selection: selecting the most useful features to train on among existing features.
  - Feature extraction: combining existing features to produce a more useful one.
  - Creating new features by gathering new data.
- Overfitting the Training Data, solutions:
  - To simplify the model by selecting one with fewer parameters, by reducing the number of attributes in the training data or by constraining the model (adjust by tuning regularization hyper-parameters)
  - To gather more training data
  - To reduce the noise in the training data (e.g., fix data errors
  and remove outliers)
- Underfitting the Training Data, solutions:
  - Selecting a more powerful model, with more parameters
  - Feeding better features to the learning algorithm (feature engineering)
  - Reducing the constraints on the model (e.g., reducing the regularization hyper‐parameter)

### No Free Lunch (NFL) Theorem

If you make absolutely no assumption about the data, then there is no reason to prefer one model over any other. [^DW1996]

[^DW1996] The Lack of A Priori Distinctions Between Learning Algorithms, D. Wolperts (1996)

### Online Datasets

- [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/)
- [Kaggle Dataset](https://www.kaggle.com/datasets)
- [AWS Open Data](https://registry.opendata.aws/)
- [Data Portal](http://dataportals.org/)
- [Open Data Monitor](http://opendatamonitor.eu/)
- [Nasdaq Data Link](https://data.nasdaq.com/)

### Example: Housing Price Regression

``` python
# fetch and load data
%run script/handsonml-houseprice.py
fetch_housing_data()
housing = load_housing_data()

# check housing data
housing.head()
housing.info()
housing.describe()

# check what categories exist and how many districts belong to each category for ocean_proximity column
housing["ocean_proximity"].value_counts()

# plot histogram for each column (feature)
%matplotlib
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))

# split train and test set
train_set, test_set = split_train_test(housing, 0.2)
# or use scikit-learn function to split train test set
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# split median income to 5 categories
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# use stratified shuffle splot to make sure train/test set have same distribution on income_cat
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# drop labels for training set
data = strat_train_set.drop("median_house_value", axis=1)
data_labels = strat_train_set["median_house_value"].copy()

# fill missing value using sklearn imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = data.drop('ocean_proximity', axis=1) # drop non-number properties
imputer.fit(housing_num)
# transform training set
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=data.index)

# transform catogorical property to OneHot encoding
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# number pipeline: fill missing value + add extra feature + normalization
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])
# full pipeline: number pipeline + onehot encoder text attributes
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
housing_prepared = full_pipeline.fit_transform(data)

# fit random forest
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse) # 21933.31414779769
# cross validation
from sklearn.model_selection import cross_val_score
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

# use grid search to tune hyper-parameters (or RandomizedSearch for many hyper-parameters)
from sklearn.model_selection import GridSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
grid_search.best_estimator_
```

Ref:

- <http://contrib.scikit-learn.org/category_encoders/#>
- <https://blog.csdn.net/sinat_26917383/article/details/107851162>

Tips: Show all properties of an object in IPython

1. vars(obj) # show all attributes
2. dir(obj)  # show all attributes and methods, including inherited ones
3. help(obj) # show help document

## Classification

``` python
# load MNIST dataset
mnist = load_mnist_dataset()
X, y = mnist.data, mnist.target
# seperate train & test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target,
    train_size=60000, test_size=10000, random_state=42)

# Binary classifier
y_train_5 = (y_train == '5') # True for all 5s, False for all other digits.
y_test_5 = (y_test == '5')
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict(X_test[:10])
# evaluate
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy") # [0.96095, 0.96775, 0.9201]

```

<!---
TBD below:
-->

## Regression

## Support Vector Machines

## Decision Trees

## Ensemble Learning and Random Forests

## Dimensionality Reduction

## Neural Networks and Deep Learning

## Convolutional Neural Networks

## Recurrent Neural Networks

## Autoencoders

## Reinforcement Learning

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

```

<!---
TBD below:
-->

## Classification

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

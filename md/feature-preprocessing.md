# Feature Preprocessing

## Useful libraries

Powerful model
* Gradient Boosted Decision Tree
* Deep Neural Network

Tree-based libraries
* dmlc/XGBoost
* Microsoft /LightGBM
* Keras
* danielfrg/tsne

Other libraries
* VOWPAL WABBIT
* srendle/libfm
* guestwalk/libffm
* baidu/fast_rgf


## Numeric data pre-processing

### Scaling
tree-based model is not sensitive to scale but non-tree-based models are highly sensitive to scale

`sklearn.preprocessing.MinMaxScaler` transform data to [0, 1]
```
X = (X - X.min())/(X.max() - X.min())
```

`sklearn.preprocessing.StandardScaler` transform data to mean=0, std=1
```
X = (X - X.mean())/X.std()
```

### Exclude outliers

1. clipping
choose 1st and 99st percentile data, usually a histogram will show feature x value distribution

``` python
pd.Series(x).hist(bins=30)
UPPERBOUND, LOWERBOUND = np.percentile(x, [1, 99])
y = np.clip(x, UPPERBOUND, LOWERBOUND)
pd.Series(y).hist(bins=30);
```

2. rank
change value to their indexes

``` python
scipy.stats.rankdata
rank([-100, 0, 1e5]) => [0,1,2]
rank([1000, 1, 10]) => [2,0,1]
```

3. non-linear functions
Drag too big values to the feature's average value. And the values near zero are a bit more distinguishable.

``` python
# Log transform
np.log(1 + x)

# Raising to the power < 1
np.sqrt(x + 2/3)
```

### Feature generation

Based on prior knowledge, or exploratory data analysis.

Examples:
1. multiplication / division
```
ex. x, y => d = sqrt(x*x + y*y)
```

2. factional part
```
ex. 99 => 0.99
```

## Categorical and ordinal data pre-processing

Example of Ordinal feature (order is meaningful):
* Ticket class: 1, 2, 3
* Drivers license: A, B, C, D
* Education: kindergarden, school, undergraduate, bachelor, master, doctoral

Tree-based model can handle ordinal data well.

### Label encoding

1. Alphabetical (sorted)
```
[S,C,Q] -> [2,1,3]
sklearn.preprocessing.LabelEncoder
```
2. Order of appearance
```
[S,C,Q] -> [1,2,3]
Pandas.factorize
```

### Frequency encoding

```
[S,C,Q] -> [0.5, 0.3, 0.2]  # appearance frequency
encoding = titanic.groupby('Embarked').size()
encoding = encoding/len(titanic)
titanic['enc'] = titanic.Embarked.map(encoding)

from scipy.stats import rankdata
```

Non-tree-based models don't usually handle ordinal data very well.

### One-hot encoding

target|1|0|1
------|-|-|-
pclass|1|2|3
pclass==1|1|0|0
pclass==2|0|1|0
pclass==3|0|0|1

```
pandas.get_dummies
sklearn.preprocessing.OneHotEncoder
```
Since the feature could be high-dimensional when the categories are many, save it to sparse matrices
`XGBoost`, `LightGBM`, `sklearn` all has sparse matrices data type for effective storage

### Feature Generation

Combine 2 or more categorial features.

pclass|sex|pclass_sex
------|---|----------
3|male|3male
1|female|1female
3|female|3female


## Date & time, coordinates

1. Periodity, e.g. day in week, minute, second
2. Time since
  * a. Row-independent moment. e.g. since 00:00:00 UTC, 1 Jan 1970;
  * b. Row-dependent important moment. e.g. number of days left until next holiday
3. Time span
days till last purchase day

## Coordinates

1. Distance
2. Interesting places number in area (schools, shops)
3. Centers of clusters
4. Aggregated statistics (population, total road miles)


## Handle missing values

Identify the missing values: null, < 0, empty string, NaN

1. draw histogram to identify the missing number
2. replace with -999, mean, median
3. try to reconstruct value
  * add a new feature ==isNull
  * interpolate in time series
  * ignore this sample
  * use frequency encoding for categorial feature that not exist in training set


## Feature extraction from Text -> vector

### Preprocessing
  * lowercase
  * lemmatization and stemming
  * stopwords: `sklearn.feature_extraction.text.CountVectorizer: max_df`

### Bag of words
  * Word count: `sklearn.feature_extraction.text.CountVectorizer`
  * TF-IDF: `sklearn.feature_extraction.text.TfidfVectorizer`
  * N-grams: bigrams, trigrams, `sklearn.feature_extraction.text.CountVectorizer: Ngram_range, analyzer`

### Word2vec, Glove, FastText, Doc2vec (pretrained models)

Vector expression of word that similar meaning words (replacible in same context) have similar vectors


## Feature extraction from Image -> vector

1. Descriptors
2. Train network from scratch
3. Finetuning

```
original image (224x224x3)
↓ convolution+ReLU (224x224x64)
↓ convolution+ReLU (224x224x64)
↓ max pooling (112x112x128)
↓ convolution+ReLU (112x112x128)
↓ convolution+ReLU (112x112x128)
↓ max pooling (56x56x256)
↓ convolution+ReLU (56x56x256)
↓ convolution+ReLU (56x56x256)
↓ convolution+ReLU (56x56x256)
↓ max pooling (14x14x512)
↓ convolution+ReLU (14x14x512)
↓ convolution+ReLU (14x14x512)
↓ convolution+ReLU (14x14x512)
↓ max pooling (7x7x512)
↓ fully connected+ReLU (1x1x4096)
↓ fully connected+ReLU (1x1x4096)
↓ fully connected+ReLU (1x1x1000)
↓ softmax (1x1x1000)
```

Keras, PyTorch, Caffe


# Exploratary Data Analysis (EDA)

## Visualization

Libraries

* seaborn: https://seaborn.pydata.org/
* Plotly: https://plot.ly/python/
* bokeh: https://github.com/bokeh/bokeh
* ggplot: http://ggplot.yhathq.com/
* NetworkX: https://networkx.github.io/


* Histograms: `plt.hist(x)`
* Plot index versus value: `plt.plot(x,'.')`, `plt.scatter(range(len(x)), x, c=y)`
* Descriptions: `df.describe()`, `x.mean()`, `x.var()`
* other tools: `x.value_counts()`, `x.isnull()`
* scatter plot of 2 features combination: `plt.scatter(x1, x2)`, `pd.scatter_matrix(df)`, `df.corr(), plt.matshow(...)`
* feature groups: `df.mean().sort_values().plot(style='.')`

## Dataset cleaning

* remove constant feature: `traintest.nunique(axis=1) == 1`
* remove duplicate: 
```
for f in categorical_feats:
  traintest[f] = traintest[f].factorize()
traintest.T.drop_duplicates()
```
* check for leaks


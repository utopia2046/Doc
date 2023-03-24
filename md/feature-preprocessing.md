# Feature Preprocessing

- [Feature Preprocessing](#feature-preprocessing)
  - [Useful libraries](#useful-libraries)
  - [Numeric data pre-processing](#numeric-data-pre-processing)
    - [Scaling](#scaling)
    - [Exclude outliers](#exclude-outliers)
    - [Feature generation](#feature-generation)
  - [Categorical and ordinal data pre-processing](#categorical-and-ordinal-data-pre-processing)
    - [Label encoding](#label-encoding)
    - [Frequency encoding](#frequency-encoding)
    - [One-hot encoding](#one-hot-encoding)
    - [Feature Generation](#feature-generation-1)
  - [Date \& time, coordinates](#date--time-coordinates)
  - [Coordinates](#coordinates)
  - [Handle missing values](#handle-missing-values)
  - [Text Data](#text-data)
    - [Preprocessing](#preprocessing)
    - [Convert to Bag of words](#convert-to-bag-of-words)
    - [Convert to Vectors: Word2vec, Glove, FastText, Doc2vec (pretrained models)](#convert-to-vectors-word2vec-glove-fasttext-doc2vec-pretrained-models)
  - [Feature extraction from Image -\> vector](#feature-extraction-from-image---vector)
- [Exploratary Data Analysis (EDA)](#exploratary-data-analysis-eda)
  - [Visualization](#visualization)
  - [Dataset cleaning](#dataset-cleaning)

## Useful libraries

Powerful model

- Gradient Boosted Decision Tree
- Deep Neural Network

Tree-based libraries

- dmlc/XGBoost
- Microsoft /LightGBM
- Keras
- danielfrg/tsne

Other libraries

- VOWPAL WABBIT
- srendle/libfm
- guestwalk/libffm
- baidu/fast_rgf

## Numeric data pre-processing

### Scaling

tree-based model is not sensitive to scale but non-tree-based models are highly sensitive to scale

`sklearn.preprocessing.MinMaxScaler` transform data to [0, 1]

``` python
X = (X - X.min())/(X.max() - X.min())
```

`sklearn.preprocessing.StandardScaler` transform data to mean=0, std=1

``` python
X = (X - X.mean())/X.std()
```

### Exclude outliers

1. clipping: choose 1st and 99st percentile data, usually a histogram will show feature x value distribution

    ``` python
    pd.Series(x).hist(bins=30)
    UPPERBOUND, LOWERBOUND = np.percentile(x, [1, 99])
    y = np.clip(x, UPPERBOUND, LOWERBOUND)
    pd.Series(y).hist(bins=30);
    ```

2. rank: change value to their indexes

    ``` python
    scipy.stats.rankdata
    rank([-100, 0, 1e5]) => [0,1,2]
    rank([1000, 1, 10]) => [2,0,1]
    ```

3. non-linear functions: drag too big values to the feature's average value. And the values near zero are a bit more distinguishable.

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

    ex. `x, y` => `d = sqrt(x*x + y*y)`

2. factional part

    ex. `99` => `0.99`

## Categorical and ordinal data pre-processing

Example of Ordinal feature (order is meaningful):

- Ticket class: 1, 2, 3
- Drivers license: A, B, C, D
- Education: kindergarden, school, undergraduate, bachelor, master, doctoral

Tree-based model can handle ordinal data well.

### Label encoding

1. Alphabetical (sorted)

    [S,C,Q] -> [2,1,3]
    sklearn.preprocessing.LabelEncoder

2. Order of appearance

    [S,C,Q] -> [1,2,3]
    Pandas.factorize

### Frequency encoding

``` python
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

``` python
pandas.get_dummies
sklearn.preprocessing.OneHotEncoder
```

Since the feature could be high-dimensional when the categories are many, save it to sparse matrices
`XGBoost`, `LightGBM`, `sklearn` all has sparse matrices data type for effective storage

``` python
# Example: Turn neighborhood enum into sparse matrix
from sklearn.feature_extraction import DictVectorizer

data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]

# vectorize the category into sparse matrix
vec = DictVectorizer(sparse=True, dtype=int)
out = vec.fit_transform(data)
print(out)

vec.get_feature_names()
```

Many (though not yet all) of the Scikit-Learn estimators accept such sparse inputs when fitting and evaluating models. `sklearn.preprocessing.OneHotEncoder` and `sklearn.feature_extraction.FeatureHasher` are two additional tools that Scikit-Learn includes to support this type of encoding.

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
    - a. Row-independent moment. e.g. since 00:00:00 UTC, 1 Jan 1970;
    - b. Row-dependent important moment. e.g. number of days left until next holiday
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
    - add a new feature ==isNull
    - interpolate in time series
    - ignore this sample
    - use frequency encoding for categorial feature that not exist in training set

## Text Data

### Preprocessing

- lowercase
- lemmatization and stemming
- stopwords: `sklearn.feature_extraction.text.CountVectorizer: max_df`

### Convert to Bag of words

- Word count: `sklearn.feature_extraction.text.CountVectorizer`
- TF-IDF: `sklearn.feature_extraction.text.TfidfVectorizer`
- N-grams: bigrams, trigrams, `sklearn.feature_extraction.text.CountVectorizer: Ngram_range, analyzer`

### Convert to Vectors: Word2vec, Glove, FastText, Doc2vec (pretrained models)

Vector expression of word that similar meaning words (replacible in same context) have similar vectors

``` python
from sklearn.feature_extraction.text import CountVectorizer

sample = ['problem of evil',
          'evil queen',
          'horizon problem']

# vectorize text into bag of words matrix
vec = CountVectorizer()
X = vec.fit_transform(sample)

# Turn the sparse matrix into a DataFrame
import pandas as pd
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

# TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

# Tokenize using Keras
from keras.preprocessing.text import Tokenizer
t = Tokenizer()
t.fit_on_texts(sample)
dir(t)
vars(t)

# Words embedding using Gensim Python library
# install Gensim
pip install -U gensim

sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
['this', 'is', 'the', 'second', 'sentence'],
['yet', 'another', 'sentence'],
['one', 'more', 'sentence'],
['and', 'the', 'final', 'sentence']]

from gensim.models import Word2Vec
# train model
model = Word2Vec(sentences, min_count=1)
print(model)
words = list(model.wv.vocab)
print(word)
print(model['sentence'])

# fit a 2D PCA model to the vector
from sklearn.decomposition import PCA
from matplotlib import pyplot

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
```

## Feature extraction from Image -> vector

1. Descriptors
2. Train network from scratch
3. Finetuning

layer|size
-|-
original image|(224x224x3)
↓ convolution+ReLU|(224x224x64)
↓ convolution+ReLU|(224x224x64)
↓ max pooling|(112x112x128)
↓ convolution+ReLU|(112x112x128)
↓ convolution+ReLU|(112x112x128)
↓ max pooling|(56x56x256)
↓ convolution+ReLU|(56x56x256)
↓ convolution+ReLU|(56x56x256)
↓ convolution+ReLU|(56x56x256)
↓ max pooling|(14x14x512)
↓ convolution+ReLU|(14x14x512)
↓ convolution+ReLU|(14x14x512)
↓ convolution+ReLU|(14x14x512)
↓ max pooling|(7x7x512)
↓ fully connected+ReLU|(1x1x4096)
↓ fully connected+ReLU|(1x1x4096)
↓ fully connected+ReLU|(1x1x1000)
↓ softmax|(1x1x1000)

Keras, PyTorch, Caffe

# Exploratary Data Analysis (EDA)

## Visualization

Libraries

- seaborn: https://seaborn.pydata.org/
- Plotly: https://plot.ly/python/
- bokeh: https://github.com/bokeh/bokeh
- ggplot: http://ggplot.yhathq.com/
- NetworkX: https://networkx.github.io/

- Histograms: `plt.hist(x)`
- Plot index versus value: `plt.plot(x,'.')`, `plt.scatter(range(len(x)), x, c=y)`
- Descriptions: `df.describe()`, `x.mean()`, `x.var()`
- other tools: `x.value_counts()`, `x.isnull()`
- scatter plot of 2 features combination: `plt.scatter(x1, x2)`, `pd.scatter_matrix(df)`, `df.corr(), plt.matshow(...)`
- feature groups: `df.mean().sort_values().plot(style='.')`

## Dataset cleaning

- remove constant feature: `traintest.nunique(axis=1) == 1`
- remove duplicate:

    ``` python
    for f in categorical_feats:
      traintest[f] = traintest[f].factorize()
    traintest.T.drop_duplicates()
    ```

- check for leaks

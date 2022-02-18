# Natural Language Processing using Scikit Package

## Reference
Python Data Science Handbook
by:
Jake VanderPlas
https://github.com/jakevdp/PythonDataScienceHandbook

## Feature Engineeing

### Enum (Category) data

Turn it into sparse matrix

``` python
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


### Text data

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

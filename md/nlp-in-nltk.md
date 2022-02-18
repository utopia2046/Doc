# Natural Language Processing using NLTK Package

## Reading raw text and cleaning up

``` python
# Read a text file
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

# split by space
words = text.split()
words = [word.lower() for word in words]

# filter out numbers
import re
words = re.split(r'\W+', text)

# filter out punctuation
import string
print(string.punctuation)
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in words]

# split into sentences
from nltk import sent_tokenize
sentences = sent_tokenize(text)

# tokenize to words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
words = [word for word in tokens if word.isalpha()]
len(tokens)
tokens[:20]

# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]

# stemming of words
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in words]

```


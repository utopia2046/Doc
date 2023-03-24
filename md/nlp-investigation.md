# NLP Investigation

## Natural Language Processing using NLTK Package

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

## ChatGPT Investigation

Acronyms:

- GPT: Generative Pretrained Transformer
- BERT: Bidirectional Encoder Representations from Transformers
- RLHF: Reinforcement Learning with Human Feedback
- RLAIF: Reinforcement Learning from AI Feedback
- COT: Chain of Thought

Important papers:

- In-context Learning survey:
  - [Pre-trained models for natural language processing: A survey. 2020](https://arxiv.org/abs/2003.08271)
  - [AMMUS: A Survey of Transformer-based Pretrained Models in Natural Language Processing. 2021](https://arxiv.org/abs/2108.05542)
  - Transformer: [Transformer models: an introduction and catalog.2023](https://arxiv.org/abs/2302.07730)
- [Open AI Research Publications](https://openai.com/research)
  - GPT3: [Language Models are Few-Shot Learners. 2020](https://arxiv.org/abs/2005.14165)
  - GPT4: [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
  - Open AI Blog: <https://openai.com/blog>
- RLHF: [Deep TAMER: Interactive Agent Shaping in High-Dimensional State Spaces](https://arxiv.org/abs/1709.10163)
  - [Teaching language models to support answers with verified quotes. 2022](https://arxiv.org/abs/2203.11147)
  - [Training language models to follow instructions with human feedback. 2022](https://arxiv.org/abs/2203.02155)
- RLAIF: [Constitutional AI: Harmlessness from AI Feedback. 2022](https://arxiv.org/abs/2212.08073)
- BERT: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 2018](https://arxiv.org/abs/1810.04805)
  - [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. 2020](https://arxiv.org/abs/1909.11942)
  - [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

In-context learning

- zero-shot, instruction only, no example
- one-shot, instruction with one example
- few-shot, instruction with few examples

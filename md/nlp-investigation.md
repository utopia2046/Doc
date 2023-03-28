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

## Training Frameworks

- [PyTorch](https://pytorch.org)
- [TensorFlow](https://www.tensorflow.org)
- [PaddlePaddle](https://www.paddlepaddle.org.cn)
- [MindSpore](https://mindspore.cn)
- [OneFlow](https://oneflow.ai)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)

Deploy Frameworks

- TensorFlow Serving
- ONNX Runtime
- OpenVINO
- TensorRT
- TorchScript

## Corpus

- BooksCorpus
  - [book1](https://hyper.ai/datasets/13642)
  - [book3](https://the-eye.eu/public/AI/pile_preliminary_components/books3.tar.gz)
  - [training code](https://the-eye.eu/public/AI/pile_preliminary_components/github.tar)
- [Common Crawl](https://commoncrawl.org/)
- [Roots Dataset](https://huggingface.co/bigscience-data)
- [Pile](https://pile.eleuther.ai/)
- [CoQA](https://stanfordnlp.github.io/coqa/)
- [悟道](https://data.baai.ac.cn/details/WUDaoCorporaText)
- [CLUE Corpus](https://github.com/CLUEbenchmark/CLUECorpus2020)
  - [CLUE Pre-trained Model](https://github.com/CLUEbenchmark/CLUEPretrainedModels)
- [MNBVC (Massive Never-ending BT Vast Chinese Corpus) 超大规模中文语料集](https://github.com/esbatmop/MNBVC) <https://huggingface.co/datasets/liwu/MNBVC>
- CCL语料库
  - <https://languageresources.github.io/2018/03/07/%E5%B4%94%E6%AC%A3%E7%AD%89_CCL%E8%AF%AD%E6%96%99%E5%BA%93/>
  - <http://ccl.pku.edu.cn:8080/ccl_corpus/index.jsp>
- BCC语料库
  - <https://languageresources.github.io/2018/03/07/%E8%82%96%E4%B8%B9%E7%AD%89_BCC%E8%AF%AD%E6%96%99%E5%BA%93/>
- [Fun NLP](https://github.com/fighting41love/funNLP)
- [OpenSLR](http://www.openslr.org/18)
- <http://corpus.zhonghuayuwen.org/resources.aspx>
- <https://github.com/jaaack-wang/ChineseNLPCorpus>

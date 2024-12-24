# Keyword Extraction

This repository contains a Jupyter Notebook for performing keyword extraction from a dataset of NIPS papers. The notebook demonstrates data preprocessing, including removing HTML tags and special characters, tokenizing text, removing stopwords, and stemming words. It then applies TF-IDF to extract keywords.

## Installation

To run the notebook, you need to install the required libraries. Use the following commands to set up your environment:

```bash
pip install pandas
pip install kaggle
pip install nltk
pip install scikit-learn
```

## Usage

1. **Download the Dataset**:
   - Ensure you have a `kaggle.json` file with your Kaggle API credentials.
   - Upload the `kaggle.json` file in the notebook.

```python
from google.colab import files
files.upload()  # Select your kaggle.json file

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!pip install kaggle
!kaggle datasets download -d benhamner/nips-papers

import zipfile
with zipfile.ZipFile("nips-papers.zip", 'r') as zip_ref:
    zip_ref.extractall("nips-papers")
```

2. **Load the Data**:
   - Load the dataset into a Pandas DataFrame.

```python
import pandas as pd
df = pd.read_csv("/content/nips-papers/papers.csv")
df.head()
```

3. **Process the Text**:
   - Preprocess the text data by converting to lowercase, removing HTML tags and special characters, tokenizing, removing stopwords, and stemming.

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt_tab')
Stop_Words = set(stopwords.words('english'))

# Define additional stopwords
new_words = ["fig", "figure", "sample", "using", "image", "show", "result", "large", "also", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
Stop_Words = list(Stop_Words.union(new_words))

def processing_text(txt):
    txt = txt.lower()
    txt = re.sub(r'<.*?>', ' ', txt)
    txt = re.sub(r'[^a-zA-Z]', ' ', txt)
    txt = nltk.word_tokenize(txt)
    txt = [word for word in txt if word not in Stop_Words]
    txt = [word for word in txt if len(word) > 3]
    stemming = PorterStemmer()
    txt = [stemming.stem(word) for word in txt]
    return txt

docs = df['paper_text'].apply(lambda x: processing_text(x))
```

4. **Apply TF-IDF**:
   - Use CountVectorizer and TfidfTransformer from scikit-learn to compute TF-IDF scores for the processed text.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

cv = CountVectorizer(max_df=95, max_features=5000, ngram_range=(1, 3))
word_count_vectors = cv.fit_transform(docs)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vectors)
```

## Keywords

- Keyword Extraction
- Text Processing
- Data Preprocessing
- TF-IDF
- Natural Language Processing (NLP)

## Libraries

- `pandas`
- `kaggle`
- `nltk`
- `scikit-learn`

## Dataset

The dataset used in this project is from Kaggle: [NIPS Papers](https://www.kaggle.com/datasets/benhamner/nips-papers).

## License

The dataset is licensed under ODbL-1.0.

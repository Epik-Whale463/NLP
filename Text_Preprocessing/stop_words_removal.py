# Text is about Machine Learning

text = ""
with open("../text.txt", "r") as f:
    text = f.read()
    
## Removing stop words is often helpful for tasks like information retrieval, topic modeling, and text summarization.

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')
nltk.download('punkt')

tokens = word_tokenize(text=text)
print(f"Tokens size {len(tokens)}")
tokens_uniq = set(tokens)
print(f"unique Tokens size {len(tokens_uniq)}")


stop_words = set(stopwords.words("english"))

# Filter out the stopwords from the tokens

filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
print("stop words Filtered tokens")
print(len(set(filtered_tokens)))


## Removing Symbols from the text

filtered_tokens2 = [token for token in tokens if token.isalnum()]
print("Symbol removed tokens")
print(len(set(filtered_tokens2)))

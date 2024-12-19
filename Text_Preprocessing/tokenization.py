# Text is about Machine Learning

text = ""
with open("../text.txt", "r") as f:
    text = f.read()

## Tokenization - breaks the words of the raw text into individial tokens , assigning them unique numbers.

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

tokens = word_tokenize(text=text)
tokens_uniq = set(word_tokenize(text=text)) # Remove duplicates

print(tokens)
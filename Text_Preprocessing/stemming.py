# Text is about Machine Learning

text = ""
with open("../text.txt", "r") as f:
    text = f.read()

## Tokenization

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

tokens = word_tokenize(text=text)
tokens_uniq = set(word_tokenize(text=text)) # Remove duplicates

#print(tokens)

## Stemming - reduces the words to root word ( might or might not be meaningful) studies -> studi

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in tokens]
print("These are the stems \n")
print(stems)
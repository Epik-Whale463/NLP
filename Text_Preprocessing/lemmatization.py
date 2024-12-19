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

#print(tokens)

## Lemmatization - reduces to the dictionary base word considering context (will be meaningful) studies -> study
# Generally , for lemmatization POS tags are needed for better accuracy but for normal rough data preprocessing it considers the deafult POS wich is Noun

from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

lemmas = [lemmatizer.lemmatize(word) for word in tokens]
print("These are lemmatizers")
print(lemmas)

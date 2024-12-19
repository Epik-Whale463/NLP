"""
Tokenization: Breaking down the text into individual words.
Lowercasing: Standardizing the text to ensure case differences don’t affect token matching.
Removing Stopwords: Filtering out common words (like "is", "the") that don’t carry much meaning.
Stemming/Lemmatization (Optional)

Must already be done in order to Implement Bag of Words on a Text corpus

Create a Vocabulary:
The vocabulary is the set of unique words across the entire text corpus.
Each unique word will correspond to a feature in the BoW vector.

Vectorize the Text:
Each document is represented as a vector, where each element corresponds to a word in the vocabulary.
The value of the vector is the frequency of the word in the document.

"""
# Read the text file
text = ""
with open("../text.txt", "r", encoding="utf-8") as f:
    text = f.read()


import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
# Tokenization
tokens = word_tokenize(text=text)
tokens_uniq = set(tokens)

# Lowercasing
lower_case_tokens = [token.lower() for token in tokens]
print("Lower case tokens")
#print(lower_case_tokens)

# Stop words removal

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

stopwords_list = set(stopwords.words('english'))
filtered_tokens = [token for token in lower_case_tokens if token not in stopwords_list]

# Opting not to perform Stemming or Lemmatization

vocabulary = set(filtered_tokens)

# BoW
from nltk.probability import FreqDist

freq_dist = FreqDist(filtered_tokens)

print("Frequency Dictionary")
print(freq_dist)

# BoW representation

bow_vector = [freq_dist[word] for word in vocabulary]
print("BoW Vector")
print(bow_vector)

# Sort the tokens by frequency
sorted_tokens = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)

# Display the most frequent tokens
for token, freq in sorted_tokens[:10]:
    print(f"Token: {token}, Frequency: {freq}")
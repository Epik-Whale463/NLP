import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk
import string

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Read the text from the file
text = ""
with open("../text.txt", "r") as f:
    text = f.read()

# Split the text into paragraphs (documents)
paragraphs = text.split('\n\n')
paragraphs = [para.strip() for para in paragraphs if para.strip()]
total_documents = len(paragraphs)

# Tokenize the entire text
tokens = word_tokenize(text)
tokens = [token.lower() for token in tokens if token not in string.punctuation]

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token not in stop_words]

# Define TF calculation function
def calculate_TF(tokens):
    tf = {}
    word_count = Counter(tokens)
    total_words = len(tokens)
    for word, count in word_count.items():
        tf[word] = count / total_words
    return tf

# Define IDF calculation function
def calculate_IDF(paragraphs, tokens):
    idf = {}
    total_documents = len(paragraphs)
    unique_tokens = set(tokens)  # Get unique tokens from the filtered text
    
    for token in unique_tokens:
        docs_with_token = sum(1 for para in paragraphs if token in word_tokenize(para.lower()))
        idf[token] = math.log(total_documents / (1 + docs_with_token))   + 1 # Add 1 to avoid division by zero
    return idf

# Calculate TF for the filtered tokens
tf = calculate_TF(filtered_tokens)

# Calculate IDF based on paragraphs (documents)
idf = calculate_IDF(paragraphs, filtered_tokens)

# Calculate TF-IDF
def calculate_TFIDF(tf, idf):
    tfidf = {}
    for word, tf_value in tf.items():
        tfidf[word] = tf_value * idf.get(word, 0)  # Use IDF only if the word exists in IDF dictionary
    return tfidf

# Calculate TF-IDF
tfidf = calculate_TFIDF(tf, idf)

# Sort TF-IDF scores in descending order
sorted_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)

print("Top 5 words by TF-IDF:")
for word, score in sorted_tfidf[:5]:
    print(f"Word: {word}, TF-IDF Score: {score}")

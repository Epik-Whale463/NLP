import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

## Rule based POS tagging - already defined rules will dictate the POS tag for each individial token

# Read the text file
text = ""
with open("../text.txt", "r") as f:
    text = f.read()

def rule_based_pos_tagger(word):
    if word.endswith("ing"):
        return "VBG"  # Verb, Gerund/Present Participle
    elif word.endswith("ed"):
        return "VBD"  # Verb, Past Tense
    elif word.endswith("s"):
        return "NNS"  # Noun, Plural
    elif word.lower() in ["is", "are", "was", "were", "am", "be", "been", "being"]:
        return "VBZ"  # Verb, Present Tense (3rd Person Singular)
    elif word.lower() in ["a", "an", "the"]:
        return "DT"  # Determiner
    elif word.istitle():
        return "NNP"  # Proper Noun, Singular
    elif word.isdigit():
        return "CD"  # Cardinal Number
    else:
        return "NN"  # Default: Noun, Singular or Mass

tokens = word_tokenize(text)

tokens_uniq = list(set(tokens))

pos_set = {}

def create_pos_set(tokens):
    for token in tokens:
        pos_set[token] = rule_based_pos_tagger(token)

create_pos_set(tokens)

for word, pos in pos_set.items():
    print(f"{word}: {pos}")


## In advanced concepts like Hiddne Markov models and CRF we can use Statistical tagging based on a large text corpora to predict tags.
#Neural POS Tagging - Leverages neural networks like LSTMs, BiLSTMs, and Transformers for tagging.
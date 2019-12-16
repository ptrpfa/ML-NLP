# Referenced program for pickled vectorizers and models used in data-mining.py (must be in the same directory)

import pandas as pd
import numpy as np
import re # REGEX
import spacy # NLP

# Suppress scikit-learn FutureWarnings
from warnings import simplefilter
simplefilter (action = 'ignore', category = FutureWarning) # Ignore Future Warnings

# Function to tokenize documents
def tokenize (document):

    # Convert document into a spaCy tokens document
    document = nlp (document)

    # Initialise list to contain tokens
    list_tokens = []

    # Loop to tokenize text
    for token in document:

        # Check if token is a stop word
        if (token.is_stop):

            # Skip current for-loop iteration if token is a stop word
            continue
        
        # Get lemmatised form of token
        lemmatised = token.lemma_

        # Check if lemmatised token is -PRON- (all English pronouns are lemmatized to the special token -PRON-)
        if (lemmatised == "-PRON-"):

            # Skip current for-loop iteration
            continue

        # Check if lemmatised token is a single non-word character
        if (re.match (r"[^a-zA-Z0-9]", lemmatised)):

            # Skip current for-loop iteration
            continue

        # Add lemmatised token into list of tokens
        list_tokens.append (lemmatised)
    
    # Return list of tokens to calling program
    return (list_tokens)
    
# Global NLP Objects
# Create spaCy NLP object
nlp = spacy.load ("en_core_web_sm")

# Custom list of stop words to add to spaCy's existing stop word list
list_custom_stopwords = ["I", "i",  "yer", "ya", "yar", "u", "loh", "lor", "lah", "leh", "lei", "lar", "liao", "hmm", "hmmm", "mmm", "mmmmmm", "wah", "eh"] 

# Add custom stop words to spaCy's stop word list
for word in list_custom_stopwords:

    # Add custom word to stopword word list
    nlp.vocab [word].is_stop = True

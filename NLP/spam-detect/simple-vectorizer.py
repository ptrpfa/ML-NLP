import pandas as pd
import numpy as np
import re # REGEX
import string
import html
import unidecode
import pickle 
import scipy
import datetime
import spacy # NLP
from sklearn.model_selection import train_test_split # 4) For splitting dataset into train/test sets
from sklearn import linear_model # 4) Linear Regression classifier
from sklearn.naive_bayes import MultinomialNB # 4) Naive Bayes classifier
from sklearn import svm # 4) SVM classifier
from sklearn.linear_model import LogisticRegression # 4) Logistic Regression classifier
from sklearn.model_selection import GridSearchCV # 4) For model hyperparameters tuning
from sklearn.feature_extraction.text import TfidfVectorizer # NLP Vectorizer
import matplotlib.pyplot as plt # For visualisations
import matplotlib # For visualisations
from sklearn.metrics import accuracy_score # 4.5) Accuracy scorer
from sklearn.model_selection import cross_val_score # 4.5) Cross validation scorer
from sklearn.metrics import confusion_matrix # 4.5) For determination of model accuracy
from sklearn.utils.multiclass import unique_labels # 4.5) For determination of model accuracy
from sklearn.metrics import classification_report # 4.5) For determination of model accuracy
import sklearn.metrics as metrics # 4.5) For determination of model accuracy

# Suppress scikit-learn FutureWarnings
from warnings import simplefilter
simplefilter (action = 'ignore', category = FutureWarning) # Ignore Future Warnings

"""CAN CONSIDER MAKING A SEPARATE FUNCTION FOR CLEANING TEST DATA AS THIS CLEANSING IS SPECIFIC TO TRAINING DATA!"""
# Function to clean corpus (accepts sequence-type corpus and returns a list of all cleaned documents)
def clean_document (corpus):

    # Initialise list containing cleaned documents
    list_cleaned_documents = []

    # Loop to clean documents in the sequence object
    for document in corpus:
        
        # Decode HTML encoded characters (&amp; -> &)
        document = html.unescape (document)

        # Remove character accents (Má -> Ma) 
        document = unidecode.unidecode (document)

        # Change text to lowercase (omitted as may affect the number of tokens created as well as may lose semantic value)
        # document = document.lower ()

        # Remove heading and trailing whitespaces
        document = document.strip ()

        # Get the number of hyperlink protocols contained in the document (http, https, ftp..)
        hyperlink_protocol_count = document.count ("http") + document.count ("ftp")

        # Intialise list containing the document's hyperlinks
        list_document_hyperlinks = []

        # Initialise counter variable
        counter = 0

        # Loop to extract hyperlinks with heading protocols within document
        while (counter < hyperlink_protocol_count):
            
            # Get match object
            match = re.match (r"(.*)((?:http|ftp|https):?\/{1,2})([^\s]*)(.*)", document) # Group 1: Text in front of link, Group 2: Protocol, Group 3: Hyperlink, Group 4: Trailing text

            # Check if a match object is obtained
            if (match == None): # For redundancy
                
                # Increment counter
                counter += 1

                # Skip current iteration
                continue

            # Get hyperlink
            hyperlink = match.group (3)

            # Remove heading and trailing periods from hyperlink (hyperlink is delimited previously by whitespace, so may have trailing period)
            hyperlink = hyperlink.strip (".")

            # Append hyperlink to list
            list_document_hyperlinks.append (hyperlink)

            # Remove hyperlink from document (note in this case the sequence where the hyperlink is in is lost [sequence does not matter for tokenization])
            document = re.sub (r"(.*)((?:http|ftp|https):?\/{1,2})([^\s]*)(.*)", r"\1 \4", document) 

            # Increment counter
            counter += 1
        
        # Get the number of hyperlinks without heading protocols contained in the document (ie www.google.com, without https:// in front)
        hyperlink_count = document.count ("www") 

        # Initialise counter variable
        counter = 0

        # Loop to extract hyperlinks without heading protocols within document
        while (counter < hyperlink_count):
            
            # Get match object
            match = re.match (r"(.*)(www\.[^\s]*)(.*)", document) # Group 1: Text in front of link, Group 2: Hyperlink, Group 3: Trailing text

            # Check if a match object is obtained (may have mismatches ie "awww")
            if (match == None): # For redundancy
                
                # Increment counter
                counter += 1

                # Skip current iteration
                continue

            # Get hyperlink
            hyperlink = match.group (2)

            # Remove heading and trailing periods from hyperlink (hyperlink is delimited previously by whitespace, so may have trailing period)
            hyperlink = hyperlink.strip (".")

            # Append hyperlink to list
            list_document_hyperlinks.append (hyperlink)

            # Remove hyperlink from document (note in this case the sequence where the hyperlink is in is lost [sequence does not matter for tokenization])
            document = re.sub (r"(.*)(www\.[^\s]*)(.*)", r"\1 \3", document) 

            # Increment counter
            counter += 1
        
        #  Remove punctuations from text
        # document = re.sub('[%s]' % re.escape(string.punctuation), '', document)

        # Remove any non-word characters from the document (Used over string.punctuation as provides more granular control)
        document = re.sub (r"[^a-zA-Z0-9 ']", " ", document) # Apostrophe included even though it will result in weird tokenizations (for words like I'll, She's..)

        # Alternative REGEX check for extracting words with embedded special characters (ie weed-deficient)
        # (([a-zA-Z]+)([^a-zA-Z]+)([a-zA-Z]+)){1,}

        # Extract words embedded within digits
        document = re.sub (r"(\d+)([a-zA-Z]+)(\d+)", r"\1 \2 \3", document)

        # Extract digits embedded within words
        document = re.sub (r"([a-zA-Z]+)(\d+)([a-zA-Z]+)", r"\1 \2 \3", document)

        # Insert back previously extracted hyperlinks into the document (maybe don't insert back until text is lemmatised/stemmed/tokenized)
        for hyperlink in list_document_hyperlinks:
            
            # Hyperlink is inserted at a later stage as we don't want some REGEX transformations to be applied to it
            document = document + " " + hyperlink

        # Replace multiple consecutive spaces with a single space
        document = re.sub (r"[ ]{2,}", " ", document)
        
        # Check if document is just a single space
        if (document == " "):

            # Replace document with an empty string if it consists of just a single non-word character
            document = ""

        # Append cleaned document into the list of cleaned documents
        list_cleaned_documents.append (document)

        # For debugging
        # print ("document: ",document)

    # Return list of cleaned documents
    return list_cleaned_documents

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
 
# Global variables
train_file_path = "/home/p/Desktop/csitml/NLP/spam-detect/data/spam-ham-reduced.txt" # Dataset file path
clean_file_path = '/home/p/Desktop/csitml/NLP/spam-detect/data/clean-spam-reduced.csv' # Cleaned dataset file path

# Global NLP Objects
# Create spaCy NLP object
nlp = spacy.load ("en_core_web_sm")

# Custom list of stop words to add to spaCy's existing stop word list
list_custom_stopwords = ["I", "i",  "yer", "ya", "yar", "u", "loh", "lor", "lah", "leh", "lei", "lar", "liao", "hmm", "hmmm", "mmm", "mmmmmm", "wah"] 

# Add custom stop words to spaCy's stop word list
for word in list_custom_stopwords:

    # Add custom word to stopword word list
    nlp.vocab [word].is_stop = True

# Program starts here
program_start_time = datetime.datetime.now ()

# 1) Get data
train_data = pd.read_csv (train_file_path, sep = "\t", encoding = 'utf-8')

# 3) Data pre-processing
train_data.text = clean_document (train_data.text) # Clean text

# Change spam/ham label to numeric values
train_data.label = train_data.label.map ({'spam': 1, 'ham': 0})

# Drop empty rows/columns
train_data.dropna (how = "all", inplace = True) # Drop empty rows
train_data.dropna (how = "all", axis = 1, inplace = True) # Drop empty columns

# Remove with rows containing empty texts
train_data = train_data [train_data.text != ""]

# Save cleaned dataset to CSV
train_data.to_csv (clean_file_path, index = False, encoding="utf-8")

# Assign target and features variables
target = train_data.label
features = train_data.text

print ("Distribution of spam/ham data:")
print (train_data.label.map ({1:'spam', 0:'ham'}).value_counts (normalize = True), "\n")

# Create DTM of dataset (features)
 # Create vectorizer object
vectorizer = TfidfVectorizer (encoding = "utf-8", lowercase = False, strip_accents = 'unicode', stop_words = 'english', tokenizer = tokenize, ngram_range = (1,2), max_df = 0.95)

# Fit data to vectorizer
features = vectorizer.fit_transform (features) # Returns a sparse matrix

# Print information on vectorised words
# print (features, type (features)) # Sparse matrix
print ("Tokens:")
print (vectorizer.get_feature_names()) # Get features (words)
data_dtm = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names()) # Convert DTM to DataFrame
data_dtm.to_csv ("/home/p/Desktop/csitml/NLP/spam-detect/data/dtm_reduced.csv", index = False, encoding="utf-8") # Save DTM

# Program end time
program_end_time = datetime.datetime.now ()
program_run_time = program_end_time - program_start_time

print ("\nProgram start time: ", program_start_time)
print ("Program end time: ", program_end_time)
print ("Program runtime: ", program_run_time.seconds, "seconds")

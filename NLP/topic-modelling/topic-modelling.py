import pandas as pd
import mysql.connector # MySQL
from sqlalchemy import create_engine # MySQL
import numpy as np
import re # REGEX
import json
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

# Function to tokenize each feedback's subject and main text (accepts a Series object of each row in the FeedbackML DataFrame and returns a tokenized Series object)
def tokenize_dataframe (series):

    # Tokenize feedback's subject
    series ['SubjectTokens'] = str (tokenize (series ['Subject']))

    # Tokenize feedback's main text
    series ['MainTextTokens'] = str (tokenize (series ['MainText']))

    # Debugging
    # print (series ['Id'], ": SubjectTokens:", series ['SubjectTokens'], ": MainTextTokens:", series ['MainTextTokens'])

    # Return tokenized series object
    return series

# Global variables
# File paths
train_file_path = "/home/p/Desktop/csitml/NLP/topic-modelling/data/feedback-ml.csv" # Dataset file path
topic_file_path = '/home/p/Desktop/csitml/NLP/topic-modelling/data/feedback-ml-topics.csv' # Topic modelled dataset file path
manual_tagging_file_path = '/home/p/Desktop/csitml/NLP/topic-modelling/data/manual-tagging.txt' # Manually tagged topic-tokens file path
pickles_file_path = "/home/p/Desktop/csitml/NLP/topic-modelling/pickles/" # File path containing pickled objects
accuracy_file_path = "/home/p/Desktop/csitml/NLP/topic-modelling/accuracies/" # Model accuracy results file path

# Boolean triggers global variables
topic_model_data = True # Boolean to trigger application of Topic Modelling model on Feedback data in the database (Default value is TRUE)
preliminary_check = True # Boolean to trigger display of preliminary dataset visualisations and presentations
use_manual_tag = True # Boolean to trigger whether to use manually tagged topics (Reads from manual-tagging.txt)
use_pickle = False # Boolean to trigger whether to use pickled objects or not
display_visuals = True # Boolean to trigger display of visualisations

# Database global variables
mysql_user = "root"                 # MySQL username
mysql_password = "csitroot"         # MySQL password
mysql_host = "localhost"            # MySQL host
mysql_schema = "csitDB"             # MySQL schema (NOTE: MySQL in Windows is case-insensitive)
feedback_table = "Feedback"         # Name of feedback table in database
feedback_ml_table = "FeedbackML"    # Name of feedback table in database used for machine learning

# Create spaCy NLP object
nlp = spacy.load ("en_core_web_sm")

# Custom list of stop words to add to spaCy's existing stop word list
list_custom_stopwords = ["I", "i",  "yer", "ya", "yar", "u", "loh", "lor", "lah", "leh", "lei", "lar", "liao", "hmm", "hmmm", "mmm", "mmmmmm", "wah", "eh"] 

# Add custom stop words to spaCy's stop word list
for word in list_custom_stopwords:

    # Add custom word to stopword word list
    nlp.vocab [word].is_stop = True

# Program starts here
program_start_time = datetime.datetime.now ()
print ("Start time: ", program_start_time)

# 1) Get dataset (GENERAL FEEDBACK FIRST!)
try:

    # Create MySQL connection object to the database
    db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)

    # Create SQL query to get FeedbackML table values (Feature Engineering)
    sql_query = "SELECT CONCAT(WebAppID, \'_\', FeedbackID, \'_\', CategoryID) as `Id`, SubjectCleaned as `Subject`, MainTextCleaned as `MainText` FROM %s WHERE SpamStatus = 0 AND CategoryID = 4;" % (feedback_ml_table)

    # Execute query and convert FeedbackML table into a pandas DataFrame
    feedback_ml_df = pd.read_sql (sql_query, db_connection)

    # Check if dataframe obtained is empty
    if (feedback_ml_df.empty == True):

        # Set boolean to apply topic modelling model on data to False if dataframe obtained is empty
        topic_model_data = False

    else:

        # Set boolean to apply topic modelling model on data to True (default value) if dataframe obtained is not empty
        topic_model_data = True

    """
    Selected Feedback features:
    -Id (WebAppID + FeedbackID + CategoryID) [Not ID as will cause Excel .sylk file intepretation error]
    -SubjectCleaned (processed)
    -MainTextCleaned (processed)

    --> Dataset obtained at this point contains pre-processed Feedback data that are NOT trash records, NOT whitelisted and classified as ham (NOT SPAM)
    """

except mysql.connector.Error as error:

    # Print MySQL connection error
    print ("MySQL error when trying to get unmined records from the FeedbackML table:", error)

except:

    # Print other errors
    print ("Error occurred attempting to establish database connection to get unmined records from the FeedbackML table")

finally:

    # Close connection object once Feedback has been obtained
    db_connection.close () # Close MySQL connection

# Check boolean variable to see whether or not to apply Topic Modelling model on Feedback dataset
if (topic_model_data == True):

    # 2) Further feature engineering
    # Drop empty rows/columns
    feedback_ml_df.dropna (how = "all", inplace = True) # Drop empty rows
    feedback_ml_df.dropna (how = "all", axis = 1, inplace = True) # Drop empty columns

    # Remove rows containing empty main texts (trash records)
    feedback_ml_df = feedback_ml_df [feedback_ml_df.MainText != ""] # For REDUNDANCY

    # Add columns to assign topic to each feedback's subject and main text
    feedback_ml_df ['SubjectTokens'] = ""
    feedback_ml_df ['MainTextTokens'] = ""
    feedback_ml_df ['SubjectTopics'] = ""
    feedback_ml_df ['MainTextTopics'] = ""

    # Some Subject may be BLANK after cleaning (need to accomodate for this!)

    # Save topic modelling dataset to CSV
    feedback_ml_df.to_csv (train_file_path, index = False, encoding = "utf-8")

    # 3) Understand dataset
    if (preliminary_check == True): # Check boolean to display preliminary information

        # Print some information of about the data
        print ("\nPreliminary information about Topic Modelling dataset:")
        print ("Dimensions: ", feedback_ml_df.shape, "\n")
        print ("First few records:")
        print (feedback_ml_df.head (), "\n")
        print ("Columns and data types:")
        print (feedback_ml_df.dtypes, "\n")

    # 4) Apply topic modelling transformations and models

    print ("Tokenizing subject and main text of feedback..")
    # Tokenize subject and main text
    feedback_ml_df.apply (tokenize_dataframe, axis = 1) # Access row by row

    # Save file
    feedback_ml_df.to_csv (topic_file_path, index = False, encoding = "utf-8")
    

    # Convert corpus to DTM
    pass

    # Implement manual tagging (from a specified set of tagged words, tag topics and assign them to feedbacks ie if contain the word Pinterest, put in the same topic)
    # Check boolean to see whether or not to label feedbacks with manually tagged tokens
    if (use_manual_tag == True):

        # Manually tagged topic-tokens file format:
        # { topic_name: ['token1', 'token2'], topic_2: ['token3'] } 

        # Initialise dictionary containing manually-tagged topic-word mappings
        dictionary_manual_tag = json.load (open (manual_tagging_file_path))

        print ("Manual tagging [INCOMPLETE!]")

        # Loop through each topic in the manually-tagged topic-word mapping
        for topic in dictionary_manual_tag:

            # See if tokens in DTM match any token in list of tokens in each topic in dictionary_manual_tag
            # dictionary_manual_tag [topic] # List of tokens
            print (dictionary_manual_tag [topic])

    # Create topics in Topic table
    pass

    # Insert Feedback-Topic mappings in FeedbackTopic table
    pass

# Print debugging message if topic modelling not carried out
else:

    print ("Topic modelling not carried out")

# Program end time
program_end_time = datetime.datetime.now ()
program_run_time = program_end_time - program_start_time

print ("\nProgram start time: ", program_start_time)
print ("Program end time: ", program_end_time)
print ("Program runtime: ", (program_run_time.seconds + program_run_time.microseconds / (10**6)), "seconds")
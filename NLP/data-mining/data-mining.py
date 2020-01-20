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
import os
from sklearn import svm # SVM classifier
from sklearn.linear_model import LogisticRegression # Logistic Regression classifier
from sklearn.feature_extraction.text import TfidfVectorizer # NLP Vectorizer
from textblob import TextBlob # Naive sentiment analysis
import spacy # NLP
from gensim import matutils, models # Gensim topic modelling
import gensim.corpora as corpora # Gensim topic modelling
import scipy.sparse # Gensim topic modelling
import logging # Gensim topic modelling logging
import pyLDAvis.gensim # For topic modelling visualisations
import pyLDAvis # For topic modelling visualisations 
# NOTE: Edit ~/anaconda3/lib/python3.7/site-packages/pyLDAvis/utils.py to comment out warnings.simplefilter("always", DeprecationWarning) to suppress DeprecationWarnings raised by pyLDAvis

# Suppress scikit-learn FutureWarnings
from warnings import simplefilter
simplefilter (action = 'ignore', category = FutureWarning)      # Ignore Future Warnings
simplefilter (action = 'ignore', category = DeprecationWarning) # Ignore Deprecation Warnings

# Pre-requisite:
# MySQL Setting:
# SET SQL Setting SET SQL_SAFE_UPDATES = 0; # To allow for updates using non-key columns in the WHERE clause

"""
NOTE: 
1) Since the database is only accessed by the database administrator/programs, 
it is assumed that the records inserted will be CORRECT and validated
--> Calculation of OverallScore of each Feedback is only done on INSERTs into the Feedback table ONCE. 
UPDATEs to the Feedback table will not re-calculate the OverallScore of each Feedback
--> Calculation of PriorityScore of each Topic is only done on INSERTs into the FeedbackTopic table ONCE. 
UPDATEs to the FeedbackTopic table will not re-calculate the OverallScore of each Topic.

Use case:
This data mining program is designed to only run ONCE on Feedback data that HAVE NOT been data mined before
--> The data mining program does not respond to UPDATEs of features that can only be derived after data mining
(ie change of SpamStatus from 0 to 1 does not trigger the running of Sentiment Analysis/Topic Modelling) OR any
new INSERTIONS of feedback data into the pool of Feedback data collected (in this case topic modelling will not
run properly due to the affected projected number of topics)

2) This program uses PICKLED models for data mining (Spam-Detection and Topic Modelling models)
--> Tuned models are required to be pickled and in the pickle file directory in order to run this program

3) This program conducts Topic Modelling on multiple categories specified in list_models in which their optimal 
number of topics have been obtained after hypertuning. To conduct Topic modelling on a specified CATEGORY and WEB_APP_ID, 
run the topic-modelling-single.py program

"""

# Function to get the factor of each category from the database for computing the overall scores of feedbacks
def get_category_factor (): 

    # Connect to database to get the factor value of each feedback's category
    try:

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # SQL query to get factor of category
        sql = "SELECT CategoryID, Factor FROM Category"

        # Execute query
        db_cursor.execute (sql)

        # Edit global dicitionary variable
        global category_factor
        
        # Assign category-factor values obtained to global dictionary variable
        category_factor = dict (db_cursor.fetchall ())

    # Catch MySQL Exception
    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error occurred when trying to get feedback's category factors:", error)

    # Catch other errors
    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection to get feedback's category factors")

    finally:

        # Close connection objects once factor is obtained
        db_cursor.close ()
        db_connection.close () # Close MySQL connection

# Function to compute each feedback's overall score (accepts a Series object of each row in the feedback Dataframe and returns a Series object with the computed values)
def overall_score_dataframe (series):

	# Compute overall score of feedback
    series ['OverallScore'] = series ['Rating'] * category_factor [series ['CategoryID']]

	# Return series with computed overall score
    return series

# Function to whitelist Feedback (accepts a Series object of each row in the feedback DataFrame and returns a labelled Series object)
def whitelist_dataframe (series): # Used over iterrows as faster and more efficient

    # Initialise check variables
    bugcode_match = False   # By default false
    whitelist_match = False # By default false
    
    # Check for BUGCODE
    match_subject = re.match (bugcode_regex, series ['Subject']) # Check for BUGCODE matches in Subject (uncleaned)
    match_maintext = re.match (bugcode_regex, series ['MainText']) # Check for BUGCODE matches in MainText (uncleaned)

    # Check for BUGCODE matches in Subject and MainText
    if (match_subject != None or match_maintext != None):

        # Set bugcode_match to True if either Subject or MainText contains a BUGCODE
        bugcode_match = True

    # Check for WHITELIST matches
    for whitelisted_string in whitelist:

        # Check if whitelisted string is in either the Subject or MainText of the Feedback
        if ((whitelisted_string in series ['Subject'].lower ()) or (whitelisted_string in series ['MainText'].lower ())): # Check if whitelisted string is in UNPROCESSED text data
            
            # Set whitelist_match to true
            whitelist_match = True

            # Immediately break out of loop if a match is found in either Subject or MainText
            break

    # Check if a BUGCODE or whitelisted word was detected in the Subject/MainText of the Feedback
    if (bugcode_match == True or whitelist_match == True):

        # Set whitelist column as True [1] if a whitelisted string or bugcode was found in the Subject or MainText of the Feedback
        series ['Whitelist'] = 1

        # Set spam statuses of whitelisted column as NOT SPAM automatically
        series ['SubjectSpam'] = 0
        series ['MainTextSpam'] = 0
        series ['SpamStatus'] = 0

        # Debugging
        # print ("Whitelisted:", series ['FeedbackID'])

    else:

        # Set whitelist column as False [0] if no whitelisted string or bugcode was found in the Subject or MainText of the Feedback
        series ['Whitelist'] = 0 # Default value of Whitelisted is 2 (to indicate that feedback has not been processed)

        # Set spam statuses of as default value of 2 (to indicate that they have not been processed)
        series ['SubjectSpam'] = 2
        series ['MainTextSpam'] = 2
        series ['SpamStatus'] = 2

    # Return series object
    return series

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

            # Check if a match object is obtained (as may have mismatches ie "awww")
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
    
        # Remove any non-word characters from the document (replace characters with space)
        document = re.sub (r"[^a-zA-Z0-9 ']", " ", document) # Apostrophe included even though it will result in weird tokenizations (for words like I'll, She's..)

        # Alternative REGEX check for extracting words with embedded special characters (ie weed-deficient)
        # (([a-zA-Z]+)([^a-zA-Z]+)([a-zA-Z]+)){1,}

        # Extract words embedded within digits
        document = re.sub (r"(\d+)([a-zA-Z]+)(\d+)", r"\1 \2 \3", document)

        # Extract digits embedded within words
        document = re.sub (r"([a-zA-Z]+)(\d+)([a-zA-Z]+)", r"\1 \2 \3", document)

        # Insert back previously extracted hyperlinks into the document
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

    # Return list of cleaned documents
    return list_cleaned_documents

# Function to tokenize each row in the series object passed from the FeedbackML table
def tokenize_document (series):

    # Initialise document variable
    document = series ['Text']

    # Convert document into a spaCy tokens document
    document = nlp (document)

    # Initialise list to contain tokens of current document being tokenized
    list_tokens = []

    # Loop to tokenize text in document
    for token in document:
        
        # Check if token is whitelisted (whitelisted terms are special terms that are returned in their normal form [non-lemmatised])
        if (token.text.lower () in token_whitelist):

            # Append current token to list of tokens
            list_tokens.append (token.text)
            
        # Proceed with series of checks if token is not whitelisted
        else:

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
    
    # Update current Feedback's TextTokens
    series ['TextTokens'] = list_tokens
    
    # Return tokenized series object
    return series

# Function to execute SQL UPDATE for each row in the series object passed to update processed Feedback data
def update_feedback_dataframe (series, cursor, connection): 

    # Create SQL statement to update Feedback table values
    sql = "UPDATE %s " % (feedback_table)
    sql = sql + "SET OverallScore = %s, Whitelist = %s, PreprocessStatus = %s, Status = %s WHERE FeedbackID = %s AND CategoryID = %s AND WebAppID = %s;" 

    # Execute SQL statement
    cursor.execute (sql, (series ['OverallScore'], series ['Whitelist'], series ['PreprocessStatus'], series ['Status'], series ['FeedbackID'], series ['CategoryID'], series ['WebAppID']))

    # Commit changes made
    connection.commit ()

# Function to insert each row in the series object passed to the FeedbackML table
def insert_feedback_ml_dataframe (series, cursor, connection): 

    # Create SQL statement to update Feedback table values
    sql = "INSERT INTO %s (FeedbackID, CategoryID, WebAppID, SubjectCleaned, MainTextCleaned, SubjectSpam, MainTextSpam, SpamStatus, Subjectivity, Polarity, TextTokens) " % (feedback_ml_table)
    sql = sql + "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);" 
    
    # Execute SQL statement
    cursor.execute (sql, (series ['FeedbackID'], series ['CategoryID'], series ['WebAppID'], series ['SubjectCleaned'], series ['MainTextCleaned'], series ['SubjectSpam'], series ['MainTextSpam'], series ['SpamStatus'], series ['Subjectivity'], series ['Polarity'], str (series ['TextTokens'])))

    # Commit changes made
    connection.commit ()

# Function to compute overall Spam Status of Feedback
def spam_status_dataframe (series):

    # Get current row's SubjectSpam and MainTextSpam values
    subject_status = series ['SubjectSpam']
    main_text_status = series ['MainTextSpam']

    # Assign SpamStatus of current feedback (if either Subject or MainText is labelled as Spam, label entire Feedback as spam)
    spam_status = bool (subject_status or main_text_status)

    # Assign spam status accordingly
    if (spam_status == True):

        # Assign spam status to True if either Subject or MainText is labelled spam
        series ['SpamStatus'] = 1

        # Debugging
        # print ("Spam record:", series ['Id'])

    else:

        # Otherwise label feedback as not spam
        series ['SpamStatus'] = 0    

    # Return series object
    return series

# Function to execute SQL UPDATE for each row in the series object passed to update the SpamStatus of Feedback data
def update_spam_status_dataframe (series, cursor, connection): 

    # Split Id (WebAppID_FeedbackID_CategoryID) into a list
    list_id = series ['Id'].split ('_') # Each ID component is delimited by underscore
    
    # Create SQL statement to update FeedbackML table values
    sql = "UPDATE %s " % (feedback_ml_table)
    sql = sql + "SET SubjectSpam = %s, MainTextSpam = %s, SpamStatus = %s, Subjectivity = %s, Polarity = %s WHERE WebAppID = %sAND FeedbackID = %s AND CategoryID = %s;" 

    # Execute SQL statement
    cursor.execute (sql, (series ['SubjectSpam'], series ['MainTextSpam'], series ['SpamStatus'], series ['Subjectivity'], series ['Polarity'], list_id [0], list_id [1], list_id [2]))

    # Commit changes made
    connection.commit ()

# Function to execute SQL UPDATE for each row in the series object passed to update the sentiment values (Subjectivity & Polarity) of Feedback data
def update_sentiment_values_dataframe (series, cursor, connection):

    # Split Id (WebAppID_FeedbackID_CategoryID) into a list
    list_id = series ['Id'].split ('_') # Each ID component is delimited by underscore
    
    # Create SQL statement to update FeedbackML table values
    sql = "UPDATE %s " % (feedback_ml_table)
    sql = sql + "SET Subjectivity = %s, Polarity = %s WHERE WebAppID = %sAND FeedbackID = %s AND CategoryID = %s;" 

    # Execute SQL statement
    cursor.execute (sql, (series ['Subjectivity'], series ['Polarity'], list_id [0], list_id [1], list_id [2]))

    # Commit changes made
    connection.commit ()

# Function to get the unique categories of feedback within the selected web application
def get_unique_categories (): 

    # Connect to database to get the unique categories of feedback within the selected web application
    try:

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # SQL query to get the unique categories of feedback within the selected web application
        sql = "SELECT DISTINCT (CategoryID) FROM %s WHERE WebAppID = %s" % (feedback_ml_table, web_app_id)

        # Execute query
        db_cursor.execute (sql)

        # Edit global list variable
        global list_category
        
        # Populate the list with each unique CategoryID obtained
        list_category = [result [0] for result in db_cursor.fetchall ()]

    # Catch MySQL Exception
    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error occurred when trying to get the unique categories of feedback within the selected web application:", error)

    # Catch other errors
    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection to get the unique categories of feedback within the selected web application")

    finally:

        # Close connection objects once factor is obtained
        db_cursor.close ()
        db_connection.close () # Close MySQL connection

# Function to strip heading and trailing whitespaces in the Text of Feedback (accepts a Series object of each row in the FeedbackML DataFrame and returns a cleaned Series object)
def strip_dataframe (series):

    # Remove heading and trailing whitespaces in Text
    series ['Text'] = series ['Text'].strip ()

    # Return cleaned series object
    return series

# Function to tokenize documents (Normal tokenizer function without any POS tagging)
def tm_tokenize_corpus (corpus):

    # Initialise list containing tokenized documents (list of lists)
    list_tokenized_documents = []

    # Loop to tokenize documents in the sequence object
    for document in corpus:

        # Convert document into a spaCy tokens document
        document = nlp (document)

        # Initialise list to contain tokens of current document being tokenized
        list_tokens = []

        # Loop to tokenize text in document
        for token in document:
            
            # Check if token is whitelisted (whitelisted terms are special terms that are returned in their normal form [non-lemmatised])
            if (token.text.lower () in token_whitelist):

                # Append current token to list of tokens
                list_tokens.append (token.text)
                
            # Proceed with series of checks if token is not whitelisted
            else:

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
        
        # Append list of tokens of current document to list containing tokenized documents
        list_tokenized_documents.append (list_tokens)
    
    # Return list of tokenized documents to calling program
    return (list_tokenized_documents)

# Function to tokenize documents (Only accepts POS: Nouns and Adjectives)
def tm_tokenize_corpus_pos_nouns_adj (corpus):

    # Initialise list containing tokenized documents (list of lists)
    list_tokenized_documents = []

    # Loop to tokenize documents in the sequence object
    for document in corpus:

        # Convert document into a spaCy tokens document
        document = nlp (document)

        # Initialise list to contain tokens of current document being tokenized
        list_tokens = []

        # Loop to tokenize text in document
        for token in document:
            
            # Check if token is whitelisted (whitelisted terms are special terms that are returned in their normal form [non-lemmatised])
            if (token.text.lower () in token_whitelist):

                # Append current token to list of tokens
                list_tokens.append (token.text)
                
            # Proceed with series of checks if token is not whitelisted
            else:

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
                
                # Get Part-of-Speech of token
                token_pos = token.pos_

                # Check if token is a Named Entity
                entity_check = str (token) in [str (entity) for entity in document.ents]

                # Check if token's POS is not a NOUN or ADJECTIVE OR NAMED ENTITY
                if (token_pos != "NOUN" and token_pos != "ADJ" and entity_check != True):

                    # Skip current for-loop iteration
                    continue

                # Add lemmatised token into list of tokens
                list_tokens.append (lemmatised)
        
        # Append list of tokens of current document to list containing tokenized documents
        list_tokenized_documents.append (list_tokens)
    
    # Return list of tokenized documents to calling program
    return (list_tokenized_documents)

# Function to tokenize documents (Only accepts POS: Nouns, Adjectives, Verbs and Adverbs)
def tm_tokenize_corpus_pos_nouns_adj_verb_adv (corpus):

    # Initialise list containing tokenized documents (list of lists)
    list_tokenized_documents = []

    # Loop to tokenize documents in the sequence object
    for document in corpus:

        # Convert document into a spaCy tokens document
        document = nlp (document)

        # Initialise list to contain tokens of current document being tokenized
        list_tokens = []

        # Loop to tokenize text in document
        for token in document:
            
            # Check if token is whitelisted (whitelisted terms are special terms that are returned in their normal form [non-lemmatised])
            if (token.text.lower () in token_whitelist):

                # Append current token to list of tokens
                list_tokens.append (token.text)
                
            # Proceed with series of checks if token is not whitelisted
            else:

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
                
                # Get Part-of-Speech of token
                token_pos = token.pos_

                # Check if token is a Named Entity
                entity_check = str (token) in [str (entity) for entity in document.ents]
                
                # Check if token's POS is not a NOUN or ADJECTIVE or VERB or ADVERB or NAMED ENTITY
                if (token_pos != "NOUN" and token_pos != "ADJ" and token_pos != "VERB" and token_pos != "ADV" and entity_check != True):

                    # Skip current for-loop iteration
                    continue

                # Add lemmatised token into list of tokens
                list_tokens.append (lemmatised)
        
        # Append list of tokens of current document to list containing tokenized documents
        list_tokenized_documents.append (list_tokens)
    
    # Return list of tokenized documents to calling program
    return (list_tokenized_documents)

# Function to fill TextTokens columns in DataFrame with tokenized BI-GRAM values (accepts a Series object of each row in the FeedbackML DataFrame and returns a tokenized Series object)
def tokenize_bigram_dataframe (series):

    # Tokenize text and assign list of tokens to row column value
    series ['TextTokens'] = bigram_model [series ['TextTokens']]
    
    # Return tokenized series object
    return series

# Function to fill TextTokens columns in DataFrame with tokenized TRI-GRAM values (accepts a Series object of each row in the FeedbackML DataFrame and returns a tokenized Series object)
def tokenize_trigram_dataframe (series):

    # Tokenize text and assign list of tokens to row column value
    series ['TextTokens'] = trigram_model [series ['TextTokens']]
    
    # Return tokenized series object
    return series

# Function to get the largest TopicID from the database to assign unique IDs to new topics after Topic Modelling
def get_largest_topicid ():

    # Initialise variable to store the largest TopicID value
    largest_id = 0 

    # Connect to database to get the largest TopicID value of each feedback's category
    try:

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # SQL query to get the largest TopicID value in the Topics table
        sql = "SELECT IFNULL(MAX(TopicID), 0) FROM %s;" % (topic_table)
        # sql = "SELECT MAX(TopicID) FROM %s;" % (topic_table)

        # Execute query
        db_cursor.execute (sql)

        # Get the largest TopicID value from the database
        largest_id = db_cursor.fetchone ()[0] 

    # Catch MySQL Exception
    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error occurred when trying to get the largest TopicID value:", error)

    # Catch other errors
    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection to get the largest TopicID value")

    finally:

        # Close connection objects once the largest TopicID value is obtained
        db_cursor.close ()
        db_connection.close () # Close MySQL connection
    
    # Return the largest TopicID value
    return largest_id

# Function to save topics generated by LDA and HDP Models respectively
def save_topics (list_lda_topics, list_hdp_topics, category_id):

    # Store topic information in topics file
    topics_file = open (topics_file_path_dm % category_id, "w") # Create file object (w = write)

    # Write header information in topics file
    topics_file.write ("LDA Model:\n\n")

    # Get current largest TopicID value in the Topics table
    largest_topicid = get_largest_topicid () + 1 # Add extra 1 as Gensim topic IDs start from 0 instead of 1 (added to compensate for this)

    # Loop to store each topic in the topics file (LDA topics)
    for topic in list_lda_topics:

        print ("Topic", (largest_topicid + topic [0]), ":\n", file = topics_file)
        print (topic [1], "\n", file = topics_file)

    # Write header information in topics file
    topics_file.write ("----------------------------------------------------------------------------------\nHDP Model:\n\n")

    # Loop to store each topic in the topics file (HDP topics)
    for topic in list_hdp_topics:

        print ("Topic", (largest_topicid + topic [0]), ":\n", file = topics_file)
        print (topic [1], "\n", file = topics_file)

    # Close file object
    topics_file.close ()

# Function to get Feedback-Topic mappings for LDA/HDP model
def get_feedback_topic_mapping (model, corpus, texts, max_no_topics, minimum_percentage):
    
    # NOTE:
    # max_no_topics = maxium number of topics that can be assigned to a Feedback
    # minimum_percentage = minimum percentage contribution required for a topic to be assigned to a Feedback

    # Apply model on corpus to get a Gensim TransformedCorpus object of mappings (Document-Topic mapping ONLY)
    transformed_gensim_corpus = model [corpus]
    print (transformed_gensim_corpus, type (transformed_gensim_corpus), len (transformed_gensim_corpus))
    # Loop to access mappings in the Gensim transformed corpus (made of lists of document-topic mappings)
    for list_document_topic in transformed_gensim_corpus: # Access document by document
        # print (list_document_topic)
        # Remove topics below specified minimum percentage contribution
        list_document_topic = list (filter (lambda tup: (tup [1] >= minimum_percentage), list_document_topic)) 

        # Sort list of current document's topic mapping according to descending order of its percentages
        list_document_topic.sort (key = lambda tup: tup [1], reverse = True) # Topics with higher percentage contribution to the feedback will be in front

        # Get the specified amount of top few topics
        list_document_topic = list_document_topic [:max_no_topics] # Get up to the specified number of topics in max_no_topics

        # Initialise lists containing topic(s) and percentages of topic(s) of current feedback/document
        list_topics = []
        list_topics_percentages = []

        # Check length of document-topic mapping to see if any topic is assigned to the current feedback/document after filtering
        if (len (list_document_topic) > 0): 

            # Get largest TopicID value from the database 
            largest_topicid = get_largest_topicid () + 1 # Add extra 1 as Gensim topic IDs start from 0 instead of 1 (added to compensate for this)

            # Loop to access list of tuples containing document-topic mappings 
            for feedback_topic in list_document_topic: # List is in the form  of [(topic_no, percentage), ..]
                
                # Add topic to list containing the topics assigned to the current document/feedback
                list_topics.append (largest_topicid + feedback_topic [0]) # List contains topics in descending order of percentage contribution
                list_topics_percentages.append (feedback_topic [1]) # List contains percentages in descending order
    
        else:

            # Add empty lists of topics to the list containing the topics assigned to the current document/feedback if the feedback is not assigned any topic
            list_topics.append ([]) 
            list_topics_percentages.append ([]) 

        # Add lists of topics and percentages assigned to the current feedback/document to the lists containing the document-topic mappings
        feedback_topic_mapping.append (list_topics)  
        feedback_topic_percentage_mapping.append (list_topics_percentages) 

# Function to set no topics to Feedback that do not have any tokens (accepts a Series object of each row in the FeedbackML DataFrame and returns a cleaned Series object)
def unassign_empty_topics_dataframe (series):

    # Check if the current feedback's tokens are empty
    if (series ['TextTokens'] == []):

        # Set topics of current feedback to nothing if its tokens are empty (NOTE: By default, if gensim receives an empty list of tokens, it will assign the document ALL topics!)
        series ['TextTopics'] = []
        series ['TopicPercentages'] = []

    # Check if the current feedback's TextTopics are empty (checking if Gensim did not assign any topics to the feedback)
    if (series ['TextTopics'] == [[]] or series ['TopicPercentages'] == [[]]):

        # Set topics of current feedback to nothing if its an empty 2-Dimensional list
        series ['TextTopics'] = []
        series ['TopicPercentages'] = []

    # Return cleaned series object
    return series

# Function to clean and split the FeedbackTopic dataframe such that one record contains one mapping of Feedback to Topic only (accepts a Series object of each row in the FeedbackTopic DataFrame and returns a cleaned Series object)
def clean_split_feedback_topic_dataframe (series):
    
    # Check length of TextTopics to see if more than one topic is assigned to current Feedback
    if (len (series ['TextTopics']) > 1 ): # Proceed to split feedback into multiple new records if current feedback is assigned to more than one topic

        # Initialise counter variable
        counter = 0

        # Loop to access list of topics assigned to current feedback
        while (counter < len (series ['TextTopics'])):
            
            # Initialise dictionary to store new row information
            dict_feedback_topic = {"Id": series ['Id'], "TextTopics": 0, "TopicPercentages": 0}

            # Assign TopicID and percentage contribution of current topic to dictionary
            dict_feedback_topic ['TextTopics'] = series ['TextTopics'] [counter]
            dict_feedback_topic ['TopicPercentages'] = series ['TopicPercentages'] [counter]
            
            # Append dictionary of new row to list containing new rows to insert into the FeedbackTopic dataframe later on
            list_new_feedback_topic.append (dict_feedback_topic)
            
            # Increment counter
            counter = counter + 1
    
    # Loop to access each topic assigned to the current feedback
    for topic in series ['TextTopics']:

        # Check if current topic is a new topic that has not been added to the list containing all topics that have been assigned to at least one Feedback
        if (topic not in list_topics_assigned):

            # Add topic into the list if it has not been added inside previously
            list_topics_assigned.append (topic)

            # Sort list of unique topics assigned to at least one Feedback in ascending order of TopicIDs
            list_topics_assigned.sort ()

    # Clean and convert datatype of TextTopics
    series ['TextTopics'] = str (series ['TextTopics']).strip ("[]") # Convert TextTopics to a string and remove square brackets
    series ['TopicPercentages'] = str (series ['TopicPercentages']).strip ("[]") # Convert TopicPercentages to a string and remove square brackets

    # Return cleaned series object
    return series

# Function to get Feedback-Topic mappings from manual tagging (accepts a Series object of each row in the FeedbackML DataFrame, a dictionary of the manually tagged topics as well as the minimum percentage contribution for a topic)
def get_manual_feedback_topic_mapping (series, dictionary_manual_tag, minimum_percentage): 

    # NOTE:
    # dictionary_manual_tag is in the format {"topic": (["keyword",..], topic_id)}
    # minimum_percentage = minimum percentage contribution of a topic for it to be assigned to the current Feedback

    # Loop to access the dictionary containing manually tagged topics
    for topic in dictionary_manual_tag.keys ():
        
        # Get the current topic's TopicID
        topic_id = dictionary_manual_tag [topic] [1]

        # Get the current topic's list of keywords and change it to lowercase
        list_keywords = [keyword.lower () for keyword in dictionary_manual_tag [topic] [0]]

        # Initialise counter variable to count the number of times a tagged keyword appears in the current Feedback's list of tokens
        no_occurances = 0

        # Get current Feedback's list of tokens in lowercase
        token_list = series.TextTokens.copy () # Create a copy of the current Feedback's token list 
        token_list = [token.lower () for token in token_list] # Lowercase contents of token list
        # OR token_list = ast.literal_eval (str (token_list).lower ()) # Lowercase token list and convert it back into a list object (ast.literal_eval raises an exception if the input isn't a valid Python datatype)

        # Inner loop to access the list of keywords associated with the current topic
        for keyword in list_keywords:

            # Check if the topic keyword appears at least once in the current Feedback's list of tokens
            if (token_list.count (keyword) > 0):

                # Add the number of times the current topic keyword appears in the feedback token list to the total number of occurances
                no_occurances = no_occurances + token_list.count (keyword) # Max number of occurances will be the length of the token list
        
        # Check if any keywords associated with the current topic appears in the current Feedback
        if (no_occurances > 0):

            # Calculate the percentage contribution of the current topic if the current topic is assigned to the current Feedback
            percentage_contribution = no_occurances / len (token_list) 
            
            # Check if the percentage contribution of the current topic is above the minimum percentage
            if (percentage_contribution >= minimum_percentage): # Can also implement maximum percentage contribution threshold (NOT IMPLEMENTED)

                # Assign the topic and its percentage contribution to the Feedback if the current topic is assigned to the current Feedback (after filtering)
                series ['TextTopics'].append (topic_id)
                series ['TopicPercentages'].append (percentage_contribution)
        
    # Return modified Series object
    return series

# Function to insert each row in the series object passed to the Topics table
def insert_topics_dataframe (series, cursor, connection): 

    # Create SQL statement to insert Topics table values
    sql = "INSERT INTO %s (TopicID, WebAppID, Name, PriorityScore, Remarks, Status) " % (topic_table)
    sql = sql + "VALUES (%s, %s, %s, %s, %s, %s);" 

    # Execute SQL statement
    cursor.execute (sql, (series ['Id'], web_app_id, series ['Name'], series ['PriorityScore'], series ['Remarks'], series ['Status']))

    # Commit changes made
    connection.commit ()

# Function to insert each row in the series object passed to the FeedbackTopic table
def insert_feedback_topic_dataframe (series, cursor, connection): 

    # Split Id (WebAppID_FeedbackID_CategoryID) into a list
    list_id = series ['Id'].split ('_') # Each ID component is delimited by underscore

    # Create SQL statement to insert FeedbackTopic table values
    sql = "INSERT INTO %s (FBWebAppID, FeedbackID, CategoryID, TopicID, TWebAppID, Percentage) " % (feedback_topic_table)
    sql = sql + "VALUES (%s, %s, %s, %s, %s, %s);" 

    # Execute SQL statement
    cursor.execute (sql, (list_id [0], list_id [1], list_id [2], series ['TextTopics'], web_app_id, series ['TopicPercentages']))

    # Commit changes made
    connection.commit ()

# Function to calculate the PriorityScore of each Topic in the Topics DataFrame
def calculate_topic_priority_score (series):

    # NOTE:
    # PriorityScore = Average OverallScore of all Feedbacks assigned to the current Topic
    # --> Sum of all Feedbacks' OverallScore in current Topic / Total number of Feedbacks in current Topic

    # Initialise variables for calculating the PriorityScore of the current Topic
    sum_overall_score = 0 # Sum of all Feedback's OverallScore in the current Topic
    no_feedback = 0       # Total number of Feedbacks in current Topic
    priority_score = 0    # PriorityScore of current Topic

    # Connect to database to get the total number of Feedbacks in the current Topic 
    try:

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # SQL query to get the total number of unique Feedbacks in the current Topic
        sql = "SELECT COUNT(DISTINCT (FeedbackID)) FROM %s WHERE TopicID = %s;" % (feedback_topic_table, series ['Id'])

        # Execute query
        db_cursor.execute (sql)

        # Update the total number of Feedbacks in the current Topic
        no_feedback = db_cursor.fetchone ()[0] 

    # Catch MySQL Exception
    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error occurred when trying to get the total number of Feedbacks in a particular Topic:", error)

    # Catch other errors
    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection to get the total number of Feedbacks in a particular Topic")

    finally:

        # Close connection objects once factor is obtained
        db_cursor.close ()
        db_connection.close () # Close MySQL connection

    # Connect to database to get the sum of all the OverallScore of Feedbacks in the current Topic
    try:

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # SQL query to get the sum of all the OverallScore of Feedbacks in the current Topic
        sql = "SELECT SUM(OverallScore) FROM %s WHERE FeedbackID IN (SELECT DISTINCT (FeedbackID) FROM %s WHERE TopicID = %s);" % (feedback_table, feedback_topic_table, series ['Id'])

        # Execute query
        db_cursor.execute (sql)

        # Update the sum of all Feedback's OverallScore in the current Topic
        sum_overall_score = int (db_cursor.fetchone ()[0])  

    # Catch MySQL Exception
    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error occurred when trying to get the sum of all the OverallScore of Feedbacks in a particular Topic:", error)

    # Catch other errors
    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection to get the sum of all the OverallScore of Feedbacks in a particular Topic")

    finally:

        # Close connection objects once factor is obtained
        db_cursor.close ()
        db_connection.close () # Close MySQL connection

    # Calculate the current Topic's PriorityScore 
    priority_score = round ((sum_overall_score / no_feedback), 2) # PriorityScore is rounded to 2 decimal places

    # Connect to database to update the PriorityScore of the current Topic
    try:

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # Create SQL statement to update Feedback table values
        sql = "UPDATE %s " % (topic_table)
        sql = sql + "SET PriorityScore = %s WHERE TopicID = %s;" 

        # Execute SQL statement
        db_cursor.execute (sql, (priority_score, series ['Id']))

        # Commit changes made
        db_connection.commit ()

    # Catch MySQL Exception
    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error occurred when trying to update the PriorityScore of a particular Topic:", error)

    # Catch other errors
    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection to update the PriorityScore of a particular Topic")

    finally:

        # Close connection objects once factor is obtained
        db_cursor.close ()
        db_connection.close () # Close MySQL connection

    # Update current Topic's PriorityScore in the Topics DataFrame
    series ['PriorityScore'] = priority_score

    # Return modified Series object
    return series

# Function to calculate runtime of specific sections of code
def calculate_runtime (duration, start_time, end_time):

    # Difference in time
    difference = end_time - start_time

    # Calculate runtime
    duration = duration + (difference.seconds + difference.microseconds / (10**6))

    # Return duration to calling program
    return duration

# Custom unpickler to prevent pickle attribute errors (as pickles do not store info on how a class is constructed and needs access to the pickler class when unpickling)
class CustomUnpickler (pickle.Unpickler): # For TF-IDF Vectorizer used during Spam Detection

    def find_class (self, module, name):
        
        # Reference to tokenize function in spam_detect.py (For TF-IDF Vectorizer)
        if name == 'tokenize':

            from pickle_supplement import tokenize # References pickle_supplement.py in the same directory
            return tokenize
            
        return super ().find_class (module, name)

# Function to load pickle object (accepts filename of pickle to load and returns the de-pickled object)
def load_pickle (filename):

     # Get full filepath
    filepath = pickles_file_path + filename

    # Create file object accessing the pickle file
    file_pickle = open (filepath, 'rb') # r = read, b = bytes

    # Custom unpickler to reference pickle object constructed previously
    pickled_object = CustomUnpickler (file_pickle).load() 

    # Close file object
    file_pickle.close ()

    # Return pickle object
    return pickled_object

# Global variables
""" File paths to store data pre-processing and data mining feedback """
folder = "%s-%s:%s" % (str (datetime.date.today ()), str (datetime.datetime.now ().hour), str (datetime.datetime.now ().minute)) # Folder file name (yyyy-mm-dd:hh:mm)
working_directory = "/home/p/Desktop/csitml/NLP/data-mining/" # Working directory of program
pickles_file_path = "%spickles/" % working_directory # File path containing pickled objects

# Data Pre-processing
feedback_file_path_p = '%sdata/%s/pre-processing/feedback.csv' % (working_directory, folder)                   # Dataset file path 
feedback_ml_file_path_p = "%sdata/%s/pre-processing/feedback-ml.csv" % (working_directory, folder)             # Dataset file path 
combined_feedback_file_path_p = "%sdata/%s/pre-processing/combined-feedback.csv" % (working_directory, folder) # Dataset file path 
trash_feedback_file_path_p = "%sdata/%s/pre-processing/trash-feedback.csv" % (working_directory, folder)       # Dataset file path 

# Data Mining
feedback_ml_prior_file_path_spam_dm = '%sdata/%s/data-mining/feedback-ml-spam-before.csv' % (working_directory, folder)       # Raw dataset file path (dataset PRIOR to spam detection) 
feedback_ml_file_path_spam_dm = "%sdata/%s/data-mining/feedback-ml-spam.csv" % (working_directory, folder)                    # Dataset file path (dataset AFTER spam detection)
feedback_ml_prior_file_path_sa_dm = '%sdata/%s/data-mining/feedback-ml-sentiment-before.csv' % (working_directory, folder)    # Raw dataset file path (dataset PRIOR to sentiment analysis)
feedback_ml_file_path_sa_dm = "%sdata/%s/data-mining/feedback-ml-sentiment.csv" % (working_directory, folder)                 # Dataset file path (dataset AFTER sentiment analysis)
topic_file_path_dm = '%sdata/%s/data-mining/feedback-ml-topics' % (working_directory, folder) + '-%s.csv'                     # Topic modelled dataset file path
topics_file_path_dm = '%sdata/%s/data-mining/topics' % (working_directory, folder) + '-%s.txt'                                # File path of topic details
topics_df_file_path_dm = '%sdata/%s/data-mining/topics' % (working_directory, folder) + '-%s.csv'                             # File path of topics table
feedback_topics_df_file_path_dm = '%sdata/%s/data-mining/feedback-topics' % (working_directory, folder) + '-%s.csv'           # File path of feedback-topics table
manual_tagging_file_path_dm = '%sdata/manual-tagging.txt' % (working_directory)                                               # Manually tagged topic-tokens file path
topic_visualise_file_path_dm = '%sdata/%s/data-mining/lda' % (working_directory, folder) + '-%s.html'                         # pyLDAvis topics file path

""" Boolean triggers global variables """
preprocess_data = True          # Boolean to trigger pre-processing of Feedback data in the database (Default value is TRUE)
remove_trash_data = False       # Boolean to trigger deletion of trash Feedback data in the database (Default value is FALSE) [INTRUSIVE]
mine_data = True                # Boolean to trigger data mining of Feedback data in the database (Default value is TRUE)
spam_check_data = True          # Boolean to trigger application of Spam Detection model on Feedback data in the database (Default value is TRUE)
sentiment_check_data = True     # Boolean to trigger application of Naive Sentiment Analysis on Feedback data in the database (Default value is TRUE)
topic_model_data = True         # Boolean to trigger application of Topic Modelling model on Feedback data in the database (Default value is TRUE)
use_manual_tag = True           # Boolean to trigger whether to use manually tagged topics (keyword-based topic modelling) (Default value is TRUE)
use_topic_model_pickle = True  # Boolean to trigger whether or not to use the already pickled Topic Models (For TESTING purposes only as pickled Gensim models will not be applied on new data)
preliminary_check = True        # Boolean to trigger display of preliminary dataset visualisations and presentations

""" Database global variables """
mysql_user = "root"                     # MySQL username
mysql_password = "csitroot"             # MySQL password
mysql_host = "localhost"                # MySQL host
mysql_schema = "csitDB"                 # MySQL schema (NOTE: MySQL in Windows is case-insensitive)
feedback_table = "Feedback"             # Name of feedback table in database
feedback_ml_table = "FeedbackML"        # Name of feedback table in database used for machine learning
topic_table = "Topic"                   # Name of topic table in database 
feedback_topic_table = "FeedbackTopic"  # Name of feedback-topic table in database 
category_factor = {}                    # Initialise empty dictionary to contain mapping of category-factor values for computing overall score of feedback

# Topic modelling specific variables (Need to specify the IDs of the selected Web App and selected Category for topic modelling as the models used will be different for different web apps and categories)
web_app_id = 99  # ID of selected web app whose feedback will be topic modelled
category_id = 4  # ID of selected category whose feedback will be topic modelled (2: Bug Report, 4: General, 5: Feature Request)

""" Data mining specific variables and processings"""
# Token whitelist (to prevent important terms from not being tokenized)
token_whitelist = ["photoshop", "editing", "pinterest", "xperia", "instagram", "facebook", "evernote", "update", "dropbox", "picsart", 
                   "whatsapp", "tripadvisor", "onenote"] 

# Create spaCy NLP object
nlp = spacy.load ("en_core_web_sm")

# Custom list of stop words to add to spaCy's existing stop word list
list_custom_stopwords = ["I", "i",  "yer", "ya", "yar", "u", "loh", "lor", "lah", "leh", "lei", "lar", "liao", "hmm", "hmmm", "mmm", "information", "ok",
                         "man", "giving", "discovery", "seek", "seeking", "rating", "my", "very", "mmmmmm", "wah", "eh", "h", "lol", "guy", "lot", "t", "d",
                         "w", "p", "ve", "y", "s", "m", "aps", "n"]  

# Add custom stop words to spaCy's stop word list
for word in list_custom_stopwords:

    # Add custom word to stopword word list
    nlp.vocab [word].is_stop = True

# Spam Detection specific variables
whitelist = ['csit', 'mindef', 'cve', 'cyber-tech', 'cyber-technology', # Spam Detection whitelist for identifying non-SPAM feedbacks (NOTE: whitelisted words are in lowercase)
            'comms-tech', 'communications-tech', 'comms-technology',
            'communications-technology', 'crypto-tech', 'cryptography-tech',
            'crypto-technology', 'cryptography-technology', 'crash', 'information', 'giving', 'problem', 
            'discovery', 'feature', 'request', 'bug', 'report', 'discover', 'seeking', 'general', 'ui', 
            'ux', 'user', 'password', 'malware', 'malicious', 'vulnerable', 'vulnerability', 'lag', 'hang', 
            'stop', 'usablility', 'usable', 'feedback', 'slow', 'long', 'memory', 'update', 'alert', 
            'install', 'fix', 'future', 'experience']
bugcode_regex = r"(.*)(BUG\d{6}\$)(.*)" # Assume bug code is BUGXXXXXX$ ($ is delimiter)

# Topic Modelling specific variables
list_corpus_tokens = [] # Initialise list containing lists of document tokens in the corpus for Topic Modelling
selected_topic_no = 65 # Set projected number of topics
list_category = [] # Initialise list containing the unique categories of feedback within the selected web application
list_models = [{'CategoryID': 2, 'TopicNo': 90}, {'CategoryID': 4, 'TopicNo': 65}, 
               {'CategoryID': 5, 'TopicNo': 70}] # Set list containing the optimal number of topics for each category of the selected web application (AFTER hypertuning)

# Program starts here
program_start_time = datetime.datetime.now ()
print ("Start time: ", program_start_time)

""" Preliminary preparations before data mining process """
print ("\n***Preliminary preparations before data mining process***\n")

# Check if there are any Feedback in the database which have not been pre-processed yet
try: # Unprocessed Feedback are Feedbacks with PreprocessStatus = 0 and Whitelist = 2 

    # Create MySQL connection object to the database
    db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)

    # Create SQL query to get Feedback table values
    sql_query = "SELECT * FROM %s WHERE PreprocessStatus = 0 AND Whitelist = 2;" % (feedback_table)

    # Execute query and convert Feedback table into a pandas DataFrame
    feedback_df = pd.read_sql (sql_query, db_connection)

    # Check if dataframe obtained is empty
    if (feedback_df.empty == True):

        # Set boolean to pre-process data to False if dataframe obtained is empty
        preprocess_data = False

    else:

        # Set boolean to pre-process data to True (default value) if dataframe obtained is not empty
        preprocess_data = True

# Catch MySQL Exception
except mysql.connector.Error as error:

    # Print MySQL connection error
    print ("MySQL error when trying to get unprocessed feedback:", error)

# Catch other errors
except:

    # Print other errors
    print ("Error occurred attempting to establish database connection when trying to get unprocessed feedback")

finally:

    # Close connection object once Feedback has been obtained
    db_connection.close () # Close MySQL connection

# Check boolean to see whether or not data pre-processing needs to be first carried out on the Feedbacks collected
if (preprocess_data == True): # Pre-process feedback if there are unpre-processed feedback

    # Get start time
    start_time = datetime.datetime.now ()

    # Print debugging message
    print ("Unprocessed feedback data detected!")
    print ("Pre-processing:", len (feedback_df), "record(s)")

    # Create new empty dataframe for feedback database table used for machine learning
    feedback_ml_df = pd.DataFrame (columns = ["FeedbackID", "WebAppID", "CategoryID", "SubjectCleaned", "MainTextCleaned", "SubjectSpam", "MainTextSpam", "SpamStatus", "Subjectivity", "Polarity", "TextTokens"]) 

    # Set index values in FeedbackML (NOTE: Other columns in feedback_ml_df are empty as data mining have not been carried out yet)
    feedback_ml_df.FeedbackID = feedback_df.FeedbackID # Set FeedbackID
    feedback_ml_df.CategoryID = feedback_df.CategoryID # Set CategoryID
    feedback_ml_df.WebAppID = feedback_df.WebAppID     # Set WebAppID

    # Set default values in FeedbackML
    feedback_ml_df ['Subjectivity'] = 2  # Set default Subjectivity value (Value of 2 indicates that the record is UNPROCESSED)
    feedback_ml_df ['Polarity'] = 2      # Set default Polarity value (Value of 2 indicates that the record is UNPROCESSED)
    feedback_ml_df ['TextTokens'] = "[]" # Set default TextTokens value (empty list of tokens)

    # Create temporary dataframe that combines both Feedback and FeedbackML tables (combine dataframes just to get new columns from FeedbackML)
    combined_feedback_df = feedback_df.merge (feedback_ml_df, on = ['FeedbackID', 'CategoryID', 'WebAppID']) # Inner join based on common IDs

    """Preliminary processings """
    # Get category factors to compute feedback overall scores
    get_category_factor ()

    # Compute feedback overall scores 
    combined_feedback_df = combined_feedback_df.apply (overall_score_dataframe, axis = 1) # Access dataframe row by row

    # Check for whitelisted feedbacks (whitelist first before cleaning documents as sometimes some whitelisted documents may be indicated as trash records ie subject contain bugcode but maintext is filled with invalid characters)
    combined_feedback_df = combined_feedback_df.apply (whitelist_dataframe, axis = 1) # Access dataframe row by row (row-iteration)

    """ Clean feedback data """
    print ("Cleaning unprocessed feedbacks..")

    # Clean new Feedback data that have not been pre-processed
    combined_feedback_df.MainTextCleaned = clean_document (combined_feedback_df.MainText) # Clean main text
    combined_feedback_df.SubjectCleaned = clean_document (combined_feedback_df.Subject) # Clean subject text
    
    # Create new dataframe to contain cleaned rows containing empty main texts (invalid trash records to be removed later on)
    combined_feedback_df_trash = combined_feedback_df [combined_feedback_df.MainTextCleaned == ""].copy () # Get feedback with MainTextCleaned set to blank first to work on trash records

    # Update columns accordingly to mark invalid rows as blacklisted invalid trash records
    combined_feedback_df_trash.loc [combined_feedback_df_trash ['Whitelist'] != 1, 'SubjectSpam'] = 3   # Set SubjectSpam status to unable to process (3) [for UNWHITELISTED records]
    combined_feedback_df_trash.loc [combined_feedback_df_trash ['Whitelist'] != 1, 'MainTextSpam'] = 3  # Set MainTextSpam status to unable to process (3) [for UNWHITELISTED records]
    combined_feedback_df_trash.loc [combined_feedback_df_trash ['Whitelist'] != 1, 'SpamStatus'] = 3    # Set SpamStatus to unable to process (3) [for UNWHITELISTED records]
    combined_feedback_df_trash.loc [combined_feedback_df_trash ['Whitelist'] != 1, 'Subjectivity'] = 3  # Set Subjectivity to unable to process (3) [for UNWHITELISTED records] {Note for this, whitelisted trash records will have sentiment value of UNPROCESSED (2)}
    combined_feedback_df_trash.loc [combined_feedback_df_trash ['Whitelist'] != 1, 'Polarity'] = 3      # Set Polarity to unable to process (3) [for UNWHITELISTED records] {Note for this, whitelisted trash records will have sentiment value of UNPROCESSED (2)}
    combined_feedback_df_trash.loc [combined_feedback_df_trash ['Whitelist'] != 1, 'Whitelist'] = 3     # Set whitelisted status to blacklisted (3) [for UNWHITELISTED records]

    # Print debugging message
    print ("Number of trash record(s) found:", len (combined_feedback_df_trash.loc [combined_feedback_df_trash ['Whitelist'] != 1]), "record(s)")

    # Remove rows containing empty texts (remove trash records from current dataframe)
    combined_feedback_df = combined_feedback_df [combined_feedback_df.MainTextCleaned != ""]

    # Combined newly labelled empty rows into the previous dataframe
    combined_feedback_df = combined_feedback_df.append (combined_feedback_df_trash) # Take note of index here! (no changes required)

    # Update preprocessed status of preprocessed feedback
    combined_feedback_df ['PreprocessStatus'] = 1 # Change to 1 to indicate that the feedbacks have been pre-processed

    # Assign column values in combined dataframe to Feedback and FeedbackML dataframes
    # Feedback table
    feedback_df.OverallScore = combined_feedback_df.OverallScore 
    feedback_df.Whitelist = combined_feedback_df.Whitelist
    feedback_df.PreprocessStatus = combined_feedback_df.PreprocessStatus

    # FeedbackML table
    feedback_ml_df.SubjectCleaned = combined_feedback_df.SubjectCleaned
    feedback_ml_df.MainTextCleaned = combined_feedback_df.MainTextCleaned
    feedback_ml_df.SubjectSpam = combined_feedback_df.SubjectSpam
    feedback_ml_df.MainTextSpam = combined_feedback_df.MainTextSpam
    feedback_ml_df.SpamStatus = combined_feedback_df.SpamStatus
    feedback_ml_df.Subjectivity = combined_feedback_df.Subjectivity
    feedback_ml_df.Polarity = combined_feedback_df.Polarity

    # Create new column to combine Subject and MainText fields of FeedbackML
    feedback_ml_df ['Text'] = feedback_ml_df ['SubjectCleaned'] + " " + feedback_ml_df ['MainTextCleaned']

    # Remove heading and trailing whitespaces in Text (to accomodate cases of blank Subjects in header)
    feedback_ml_df = feedback_ml_df.apply (strip_dataframe, axis = 1) # Access row by row 

    # Tokenize feedback
    feedback_ml_df = feedback_ml_df.apply (tokenize_document, axis = 1)        

    # Update default developer status of Feedbacks in categories (4: General, 5: Feature Request) to the value 3 (closed)
    feedback_df.loc [feedback_df ['CategoryID'] == 4, 'Status'] = 3
    feedback_df.loc [feedback_df ['CategoryID'] == 5, 'Status'] = 3

    # Connect to database to UPDATE Feedback table
    try:

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # Update Feedback database table with the newly pre-processed data
        feedback_df.apply (update_feedback_dataframe, axis = 1, args = (db_cursor, db_connection))

        # Print debugging message
        print (len (feedback_df), "record(s) successfully pre-processed")
        
    # Catch MySQL Exception
    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error occurred when trying to update Feedback table:", error)

    # Catch other errors
    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection to update Feedback table")

    finally:

        # Close connection objects once Feedback has been obtained
        db_cursor.close ()
        db_connection.close () # Close MySQL connection

    # Connect to database to INSERT values into FeedbackML table
    try:

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # Update Feedback database table with the newly pre-processed data
        feedback_ml_df.apply (insert_feedback_ml_dataframe, axis = 1, args = (db_cursor, db_connection))

        # Print debugging message
        print (len (feedback_ml_df), "record(s) successfully inserted into FeedbackML table")
        
    # Catch MySQL Exception
    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error occurred when trying to insert values into FeedbackML table:", error)

    # Catch other errors
    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection to insert values into the FeedbackML table")

    finally:

        # Close connection objects once Feedback has been obtained
        db_cursor.close ()
        db_connection.close () # Close MySQL connection

    # Check if folder to store pre-processed feedback exists
    if (not os.path.exists ("%sdata/%s" % (working_directory, folder))):

        # Create folder if it doesn't exist
        os.mkdir ("%sdata/%s" % (working_directory, folder)) 
    
    # Check if sub-folder for pre-processed feedback exists
    if (not os.path.exists ("%sdata/%s/pre-processing/" % (working_directory, folder))):

        # Create sub-folder if it doesn't exist
        os.mkdir ("%sdata/%s/pre-processing/" % (working_directory, folder)) 
    
    # Export dataframes to CSV
    combined_feedback_df.to_csv (combined_feedback_file_path_p, index = False, encoding = "utf-8")
    combined_feedback_df_trash.to_csv (trash_feedback_file_path_p, index = False, encoding = "utf-8")
    feedback_df.to_csv (feedback_file_path_p, index = False, encoding = "utf-8")
    feedback_ml_df.to_csv (feedback_ml_file_path_p, index = False, encoding = "utf-8")

    # Get data pre-processing end time
    end_time = datetime.datetime.now ()

    # Print data pre-processing duration
    print ("\nData pre-processing completed in", calculate_runtime (0, start_time, end_time), "seconds")

# Print debugging message if no data pre-processing is carried out
else:

    print ("Data pre-processing not carried out")

# Check boolean to see whether or not to delete trash records (intrusive)
if (remove_trash_data == True): # By default don't delete TRASH RECORDs as even though they lack quality information, they serve semantic value (metadata)

    # Remove trash Feedback data marked with custom removal Remarks ("TRASH RECORD")
    try: # Trash Feedback are feedback which has its MainTextCleaned empty after data cleaning (meaning that they contain and are made up of all invalid characters)

        print ("Removing TRASH records from the database..")

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # Create SQL statement to delete records (blacklisted records)
        sql = "DELETE FROM %s WHERE Whitelist = 3;" % (feedback_table)

        # Execute SQL statement
        db_cursor.execute (sql)

        # Commit changes made
        db_connection.commit ()

        # Debugging
        print (db_cursor.rowcount, "trash record(s) deleted")

    # Catch MySQL Exception
    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error when trying to remove trash records from the database:", error)

    # Catch other errors
    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection to remove trash records from the database")

    finally:

        # Close connection objects once Feedback has been obtained
        db_cursor.close ()
        db_connection.close () # Close MySQL connection

""" Start DATA MINING process """
print ("\n\n***Data Mining***\n")

# Check if there are any Feedback in the database which have not been data-mined yet
try: 

    # Create MySQL connection object to the database
    db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)

    # Create SQL query to get Feedback table values
    sql_query = "SELECT * FROM %s WHERE MineStatus = 0;" % (feedback_table)

    # Execute query and convert Feedback table into a pandas DataFrame
    feedback_df = pd.read_sql (sql_query, db_connection)

    # Check if dataframe obtained is empty
    if (feedback_df.empty == True):

        # Set boolean to mine data to False if dataframe obtained is empty
        mine_data = False

    else:

        # Set boolean to mine data to True (default value) if dataframe obtained is not empty
        mine_data = True

# Catch MySQL Exception
except mysql.connector.Error as error:

    # Print MySQL connection error
    print ("MySQL error when trying to get unmined feedback:", error)

# Catch other errors
except:

    # Print other errors
    print ("Error occurred attempting to establish database connection when trying to get unmined feedback")

finally:

    # Close connection object once Feedback has been obtained
    db_connection.close () # Close MySQL connection

# Check boolean to see whether or not to data mine feedback
if (mine_data == True):

    # Print debugging message
    print ("Unmined feedback data detected!")
    print ("Mining:", len (feedback_df), "record(s)")

    """ Spam detection (Detect if Feedback data is spam or ham (not spam)) """
    # 1) Get dataset for Spam Detection
    try:

        # Create MySQL connection object to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)

        # Create SQL query to get FeedbackML table values (Feature Engineering)
        sql_query = "SELECT CONCAT(WebAppID, \'_\', FeedbackID, \'_\', CategoryID) as `Id`, SubjectCleaned as `Subject`, MainTextCleaned as `MainText`, SubjectSpam, MainTextSpam, SpamStatus, Subjectivity, Polarity FROM %s WHERE SpamStatus = 2;" % (feedback_ml_table)

        # Execute query and convert FeedbackML table into a pandas DataFrame
        feedback_ml_df = pd.read_sql (sql_query, db_connection)

        # Check if dataframe obtained is empty
        if (feedback_ml_df.empty == True):

            # Set boolean to apply spam detection model on data to False if dataframe obtained is empty
            spam_check_data = False

        else:

            # Set boolean to apply spam detection model on data to True (default value) if dataframe obtained is not empty
            spam_check_data = True

        """
        Selected Feedback features:
        -Id (WebAppID + FeedbackID + CategoryID) [Not ID as will cause Excel .sylk file intepretation error]
        -Subject (pre-processed)
        -MainText (pre-processed)
        -SubjectSpam [target1]
        -MainTextSpam [target2]
        -SpamStatus [target]
        -Subjectivity
        -Polarity

        --> Dataset obtained at this point contains pre-processed Feedback data that are NOT trash records, NOT whitelisted and NOT classified as spam/ham
        """

    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error when trying to get unmined records from the FeedbackML table for spam detection:", error)

    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection to get unmined records from the FeedbackML table for spam detection")

    finally:

        # Close connection object once Feedback has been obtained
        db_connection.close () # Close MySQL connection

    # Check boolean variable to see whether or not to apply Spam Detection model on Feedback data
    if (spam_check_data == True):

        print ("\n(1) Conducting SPAM DETECTION..")

        # Get start time of spam detection
        spam_start_time = datetime.datetime.now ()

        # 2) Further feature engineering (Data pre-processing) [FOR REDUNDANCY]
        # Drop empty rows/columns
        feedback_ml_df.dropna (how = "all", inplace = True) # Drop empty rows
        feedback_ml_df.dropna (how = "all", axis = 1, inplace = True) # Drop empty columns

        # Remove rows containing empty main texts (trash records)
        feedback_ml_df = feedback_ml_df [feedback_ml_df.MainText != ""]

        # Check if folder to store data-mined feedback exists
        if (not os.path.exists ("%sdata/%s" % (working_directory, folder))):

            # Create folder if it doesn't exist
            os.mkdir ("%sdata/%s" % (working_directory, folder)) 

        # Check if sub-folder for data-mined feedback exists
        if (not os.path.exists ("%sdata/%s/data-mining/" % (working_directory, folder))):

            # Create sub-folder if it doesn't exist
            os.mkdir ("%sdata/%s/data-mining/" % (working_directory, folder)) 

        # Save cleaned raw (prior to spam detection data mining) dataset to CSV
        feedback_ml_df.to_csv (feedback_ml_prior_file_path_spam_dm, index = False, encoding = "utf-8")

        # 3) Understand dataset
        if (preliminary_check == True): # Check boolean to display preliminary information

            # Print some information of about the data
            print ("\nPreliminary information about SPAM/HAM dataset:")
            print ("Dimensions: ", feedback_ml_df.shape, "\n")
            print ("First few records:")
            print (feedback_ml_df.head (), "\n")
            print ("Columns and data types:")
            print (feedback_ml_df.dtypes, "\n")

        # 4) Apply spam-detection model
        # Assign target and features variables
        target_subject = feedback_ml_df.SubjectSpam # Target and feature variables for Subject
        feature_subject = feedback_ml_df.Subject

        target_main_text = feedback_ml_df.MainTextSpam # Target and feature variables for MainText
        feature_main_text = feedback_ml_df.MainText 

        # Load pickled/serialised vectorizer from spam-detection program
        start_time = datetime.datetime.now ()
        vectorizer = load_pickle ("tfidf-vectorizer.pkl")
        end_time = datetime.datetime.now ()
        print ("Loaded vectorizer in", calculate_runtime (0, start_time, end_time), "seconds")

        # Fit data to vectorizer [Create DTM of dataset (features)]
        start_time = datetime.datetime.now ()
        feature_subject = vectorizer.transform (feature_subject) 
        end_time = datetime.datetime.now ()
        print ("Transformed subject to DTM in", calculate_runtime (0, start_time, end_time), "seconds")

        start_time = datetime.datetime.now ()
        feature_main_text = vectorizer.transform (feature_main_text) 
        end_time = datetime.datetime.now ()
        print ("Transformed main text to DTM in", calculate_runtime (0, start_time, end_time), "seconds")

        # Initialise model duration
        spam_model_duration = 0 

        # Load pickled model
        start_time = datetime.datetime.now ()
        spam_model = load_pickle ("svm-model.pkl") # Used SVM Model in this case
        # spam_model = load_pickle ("logistic-regression-model.pkl") # Used LR Model in this case
        end_time = datetime.datetime.now ()
        spam_model_duration = calculate_runtime (spam_model_duration, start_time, end_time)

        # Predict whether Subject is spam or not
        print ("\nPredicting whether subjects of feedback is spam..")
        start_time = datetime.datetime.now ()
        model_prediction_subject = spam_model.predict (feature_subject) # Store predicted results of model
        end_time = datetime.datetime.now ()
        spam_model_duration = calculate_runtime (spam_model_duration, start_time, end_time)
        print ("Predicted subject values:", model_prediction_subject)

        # Predict whether MainText is spam or not
        print ("\nPredicting whether main texts of feedback is spam..")
        start_time = datetime.datetime.now ()
        model_prediction_main_text = spam_model.predict (feature_main_text) # Store predicted results of model
        end_time = datetime.datetime.now ()
        spam_model_duration = calculate_runtime (spam_model_duration, start_time, end_time)
        print ("Predicted main text values:", model_prediction_main_text)

        # Print spam model runtime
        print ("\nSpam model runtime: ", spam_model_duration, "seconds\n")

        # Collate results of spam-detection predictions
        feedback_ml_df.SubjectSpam = model_prediction_subject
        feedback_ml_df.MainTextSpam = model_prediction_main_text

        # Get overall SpamStatus of each feedback record
        feedback_ml_df = feedback_ml_df.apply (spam_status_dataframe, axis = 1) 

        # Update Sentiment Analysis Subjectivity and Polarity values of records labelled as SPAM
        feedback_ml_df.loc [feedback_ml_df ['SpamStatus'] == 1, 'Subjectivity'] = 3  # Set Subjectivity value of SPAM records as UNABLE TO PROCESS (3)
        feedback_ml_df.loc [feedback_ml_df ['SpamStatus'] == 1, 'Polarity'] = 3      # Set Polarity value of SPAM records as UNABLE TO PROCESS (3)

        # Save spam-mined dataset to CSV
        feedback_ml_df.to_csv (feedback_ml_file_path_spam_dm, index = False, encoding = "utf-8")

        # Connect to database to UPDATE SpamStatus values of Feedback
        try:

            # Create MySQL connection and cursor objects to the database
            db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
            db_cursor = db_connection.cursor ()

            # Update database table with the newly pre-processed data
            feedback_ml_df.apply (update_spam_status_dataframe, axis = 1, args = (db_cursor, db_connection))

            # Print debugging message
            print (len (feedback_ml_df), "record(s) successfully classified as SPAM/HAM")

        # Catch MySQL Exception
        except mysql.connector.Error as error:

            # Print MySQL connection error
            print ("MySQL error when trying to update spam statuses of FeedbackML records:", error)

        # Catch other errors
        except:

            # Print other errors
            print ("Error occurred attempting to establish database connection to update the spam statuses of FeedbackML records")

        finally:

            # Close connection objects once Feedback has been obtained
            db_cursor.close ()
            db_connection.close () # Close MySQL connection

        # Get end time of spam detection
        spam_end_time = datetime.datetime.now ()
        print ("\nCompleted Spam Detection in", calculate_runtime (0, spam_start_time, spam_end_time), "seconds")

    """ NOTE: Sentiment Analysis and Topic Modelling will only be carried out AFTER Spam Detection is carried out and they will only be applied to NON-SPAM records! """

    """ Naive Sentiment Analysis on Feedback data using TextBlob """
    # 1) Get dataset for naive Sentiment Analysis
    try:

        # Create MySQL connection object to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)

        # Create SQL query to get FeedbackML table values (Feature Engineering) [SpamStatus = 0, Subjectivity = 2 & Polarity = 2] 
        sql_query = "SELECT CONCAT(WebAppID, \'_\', FeedbackID, \'_\', CategoryID) as `Id`, MainTextCleaned as `MainText`, Subjectivity, Polarity FROM %s WHERE SpamStatus = 0 AND Subjectivity = 2 AND Polarity = 2;" % (feedback_ml_table)

        # Execute query and convert FeedbackML table into a pandas DataFrame
        feedback_ml_df = pd.read_sql (sql_query, db_connection)

        # Check if dataframe obtained is empty
        if (feedback_ml_df.empty == True):

            # Set boolean to apply naive sentiment analysis on data to False if dataframe obtained is empty
            sentiment_check_data = False

        else:

            # Set boolean to apply naive sentiment analysis on data to True (default value) if dataframe obtained is not empty
            sentiment_check_data = True

        """
        Selected Feedback features:
        -Id (WebAppID + FeedbackID + CategoryID) [Not ID as will cause Excel .sylk file intepretation error]
        -MainText (processed)
        -Subjectivity [target1]
        -Polarity [target2]

        --> Dataset obtained at this point contains pre-processed Feedback data that are NOT trash records, NOT whitelisted, NOT SPAM and NOT sentiment analysed

        NOTE: Subject of Feedback is omitted from sentiment analysis instead of combining it together with MainText to get a longer text because 1) Sentiment analysis
        takes into account the order of words and 2) Practically, Subject should be a summary or subset of MainText, with MainText giving the real value of the Feedback
        --> Alternatives to include Subject with MainText would be to compute the Subject of each individual Feedback's subjectivity and polarity scores separately and
        combining it with the values of the Feedback's MainText to some degree/factor (ie Subjectivity = (0.3 * SubjectivityScore of Subject) + (0.7 * SubjectivityScore 
        of MainText))
        """

    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error when trying to get unmined records from the FeedbackML table for naive sentiment analysis:", error)

    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection to get unmined records from the FeedbackML table for naive sentiment analysis")

    finally:

        # Close connection object once Feedback has been obtained
        db_connection.close () # Close MySQL connection

    # Check boolean variable to see whether or not to apply Naive Sentiment Analysis on Feedback data
    if (sentiment_check_data == True):

        print ("\n\n(2) Conducting SENTIMENT ANALYSIS..\n")
        
        # Get start time of sentiment analysis
        sentiment_start_time = datetime.datetime.now ()
        
        # 2) Further feature engineering (Data pre-processing) [FOR REDUNDANCY]
        # Drop empty rows/columns
        feedback_ml_df.dropna (how = "all", inplace = True) # Drop empty rows
        feedback_ml_df.dropna (how = "all", axis = 1, inplace = True) # Drop empty columns

        # Check if folder to store data-mined feedback exists
        if (not os.path.exists ("%sdata/%s" % (working_directory, folder))):

            # Create folder if it doesn't exist
            os.mkdir ("%sdata/%s" % (working_directory, folder)) 

        # Check if sub-folder for data-mined feedback exists
        if (not os.path.exists ("%sdata/%s/data-mining/" % (working_directory, folder))):

            # Create sub-folder if it doesn't exist
            os.mkdir ("%sdata/%s/data-mining/" % (working_directory, folder)) 

        # Save cleaned raw (prior to sentiment analysis data mining) dataset to CSV
        feedback_ml_df.to_csv (feedback_ml_prior_file_path_sa_dm, index = False, encoding = "utf-8")

        # 3) Apply TextBlob Naive Sentiment Analysis on Feedback data
        # Create lambda functions for getting the polarity and subjectivity values of each Feedback
        get_polarity = lambda x: TextBlob (x).sentiment.polarity         # Value from -1 (negative) to 1 (positive)
        get_subjectivity = lambda x: TextBlob (x).sentiment.subjectivity # Value from 0 (objective) to 1 (subjective)

        # Apply functions to obtain the naive subjectivity and polarity sentiment values of each Feedback
        feedback_ml_df ['Polarity'] = feedback_ml_df ['MainText'].apply (get_polarity) # Apply lambda function on MainText portion of Feedback [NOTE: Blank records will be given a Subjectivity and Polarity score of 0]
        feedback_ml_df ['Subjectivity'] = feedback_ml_df['MainText'].apply (get_subjectivity) # Apply lambda function on MainText portion of Feedback

        # Save sentiment-analysed-mined dataset to CSV
        feedback_ml_df.to_csv (feedback_ml_file_path_sa_dm, index = False, encoding = "utf-8")

        # Connect to database to UPDATE sentiment analysis values of Feedback
        try:

            # Create MySQL connection and cursor objects to the database
            db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
            db_cursor = db_connection.cursor ()

            # Update database table with the newly pre-processed data
            feedback_ml_df.apply (update_sentiment_values_dataframe, axis = 1, args = (db_cursor, db_connection))

            # Print debugging message
            print (len (feedback_ml_df), "record(s)' sentiment subjectivity and polarity values analysed")

        # Catch MySQL Exception
        except mysql.connector.Error as error:

            # Print MySQL connection error
            print ("MySQL error when trying to update the sentiment values of FeedbackML records:", error)

        # Catch other errors
        except:

            # Print other errors
            print ("Error occurred attempting to establish database connection to update the sentiment values of FeedbackML records")

        finally:

            # Close connection objects once Feedback has been obtained
            db_cursor.close ()
            db_connection.close () # Close MySQL connection

        # Get end time of sentiment analysis
        sentiment_end_time = datetime.datetime.now ()
        print ("\nCompleted Sentiment Analysis in", calculate_runtime (0, sentiment_start_time, sentiment_end_time), "seconds")

    """ Topic Modelling on Feedback data to group similar feedback together for ease of prioritisation of feedbacks for developers in the developer's platform """
    # Get the unique categories of feedback within the selected web application
    get_unique_categories ()
    
    print ("\n\n(3) Conducting TOPIC MODELLING..")

    # Get start time of topic modelling
    topic_start_time = datetime.datetime.now ()

    # Loop to topic model feedbacks within each category in the selected web application (provided that the optimal number of topics of each category is provided)
    for optimal_model in list_models:

        # Check if the CategoryID of the current item is not in the list of unique categories
        if (optimal_model ['CategoryID'] not in list_category):

            # Skip current iteration
            continue

        # Conduct Topic modelling if the CategoryID of the current item is in the list of unique categories

        # Update current CategoryID
        category_id = optimal_model ['CategoryID']

        # Update optimal number of topics for current CategoryID
        selected_topic_no = optimal_model ['TopicNo']

        # Print debugging message
        print ("\nPerforming Topic Modelling on Category %s with %s projected topics..\n" % (category_id, selected_topic_no))

        # 1) Get dataset
        try:

            # Create MySQL connection object to the database
            db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)

            # Create SQL query to get FeedbackML table values (Feature Engineering)
            sql_query = "SELECT CONCAT(WebAppID, \'_\', FeedbackID, \'_\', CategoryID) as `Id`, SubjectCleaned as `Subject`, MainTextCleaned as `MainText` FROM %s WHERE WebAppID = %s AND SpamStatus = 0 AND CategoryID = %s;" % (feedback_ml_table, web_app_id, category_id)

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
            -Text [SubjectCleaned (processed) + MainTextCleaned (processed)]

            --> Dataset obtained at this point contains pre-processed Feedback data that are NOT trash records, NOT whitelisted and classified as ham (NOT SPAM)
            """

        except mysql.connector.Error as error:

            # Print MySQL connection error
            print ("MySQL error when trying to get Category %s HAM records from the FeedbackML table:" % category_id, error)

        except:

            # Print other errors
            print ("Error occurred attempting to establish database connection to get Category %s HAM records from the FeedbackML table" % category_id)

        finally:

            # Close connection object once Feedback has been obtained
            db_connection.close () # Close MySQL connection

        # Check boolean variable to see whether or not to apply Topic Modelling model on current Category Feedback dataset
        if (topic_model_data == True):

            # 2) Further feature engineering and data pre-processings
            # Drop empty rows/columns (for redundancy)
            feedback_ml_df.dropna (how = "all", inplace = True) # Drop empty rows
            feedback_ml_df.dropna (how = "all", axis = 1, inplace = True) # Drop empty columns
            
            # Combine subject and main text into one column [Will apply topic modelling on combined texts of subject together with main text instead of both separately as topic modelling uses the DTM/Bag of Words format, in which the order of words does not matter]
            feedback_ml_df ['Text'] = feedback_ml_df ['Subject'] + " " + feedback_ml_df ['MainText']
            
            # Remove heading and trailing whitespaces in Text (to accomodate cases of blank Subjects in header)
            feedback_ml_df = feedback_ml_df.apply (strip_dataframe, axis = 1) # Access row by row 

            # Create new columns for dataframe
            feedback_ml_df ['TextTokens'] = "[]"       # Default empty list for tokens of feedback text after tokenization
            feedback_ml_df ['TextTopics'] = "[]"       # Default empty list of topics assigned to feedback
            feedback_ml_df ['TopicPercentages'] = "[]" # Default empty list of the percentage contributions of topics assigned to feedback

            # Tokenize texts and assign text tokens to column in DataFrame
            # feedback_ml_df.TextTokens = tm_tokenize_corpus (feedback_ml_df.Text)                        # Default tokenize function without any POS tagging specifications
            feedback_ml_df.TextTokens = tm_tokenize_corpus_pos_nouns_adj (feedback_ml_df.Text)            # Only tokenize NOUNS and ADJECTIVES (will result in many empty token lists)
            # feedback_ml_df.TextTokens = tm_tokenize_corpus_pos_nouns_adj_verb_adv (feedback_ml_df.Text) # Only tokenize NOUNS, ADJECTIVES, VERBS and ADVERBS (will result in many empty token lists)

            # Assign document tokens in DataFrame to global list containing all corpus tokens
            list_corpus_tokens = list (feedback_ml_df.TextTokens)

            # Create bigram and trigram models
            bigram = models.Phrases (list_corpus_tokens, min_count = 5, threshold = 100) # Bigrams must be above threshold in order to be formed
            bigram_model = models.phrases.Phraser (bigram) # Create bigram model
            
            trigram = models.Phrases (bigram [list_corpus_tokens], threshold = 110) # Threshold should be higher (for trigrams to be formed, need to have higher frequency of occurance)
            trigram_model = models.phrases.Phraser (trigram) # Create trigram model

            # Create bigram and trigram tokens in DataFrame
            feedback_ml_df = feedback_ml_df.apply (tokenize_bigram_dataframe, axis = 1) 
            feedback_ml_df = feedback_ml_df.apply (tokenize_trigram_dataframe, axis = 1) 

            # 3) Understand dataset
            if (preliminary_check == True): # Check boolean to display preliminary information

                # Print some information of about the data
                print ("\nPreliminary information about Topic Modelling Category %s dataset:" % category_id)
                print ("Dimensions: ", feedback_ml_df.shape, "\n")
                print ("First few records:")
                print (feedback_ml_df.head (), "\n")
                print ("Columns and data types:")
                print (feedback_ml_df.dtypes, "\n")
            
            # 4) Apply topic modelling transformations and models
            # Convert list of corpus document-tokens into a dictionary of the locations of each token in the format {location: 'term'}
            id2word = corpora.Dictionary (list_corpus_tokens)

            # # Human readable format of corpus (term-frequency)
            # dtm = [[(id2word [id], freq) for id, freq in cp] for cp in corpus[:1]]

            # Get Term-Document Frequency
            gensim_corpus = [id2word.doc2bow (document_tokens) for document_tokens in list_corpus_tokens]

            # Create new models if not using serialised models
            if (not use_topic_model_pickle): # REMOVE THIS!

                # Create Topic Modelling models
                lda_model = models.LdaModel (corpus = gensim_corpus, id2word = id2word, num_topics = selected_topic_no, passes = 100, 
                                            chunksize = 3500, alpha = 'auto', eta = 'auto', random_state = 123, minimum_probability = 0.05) # LDA Model

                hdp_model = models.HdpModel (corpus = gensim_corpus, id2word = id2word, random_state = 123) # HDP Model (infers the number of topics [always generates 150 topics])
            
            # Using pickled objects
            else:

                # Load serialised models
                lda_model = load_pickle ("lda-model-%s.pkl" % category_id)
                hdp_model = load_pickle ("hdp-model-%s.pkl" % category_id)

            """ Get Topics generated """
            # Get topics
            list_lda_topics = lda_model.show_topics (formatted = True, num_topics = selected_topic_no, num_words = 20)
            list_lda_topics.sort (key = lambda tup: tup [0]) # Sort topics according to ascending order

            list_hdp_topics = hdp_model.show_topics (formatted = True, num_topics = 150, num_words = 20)
            list_hdp_topics.sort (key = lambda tup: tup [0]) # Sort topics according to ascending order

            # Check if folder to store data-mined feedback exists
            if (not os.path.exists ("%sdata/%s" % (working_directory, folder))):

                # Create folder if it doesn't exist
                os.mkdir ("%sdata/%s" % (working_directory, folder)) 

            # Check if sub-folder for data-mined feedback exists
            if (not os.path.exists ("%sdata/%s/data-mining/" % (working_directory, folder))):

                # Create sub-folder if it doesn't exist
                os.mkdir ("%sdata/%s/data-mining/" % (working_directory, folder)) 

            # Save topics in topic file
            save_topics (list_lda_topics, list_hdp_topics, category_id)

            """ Get Feedback-Topic mappings """
            # Initialise lists containing feedback-topic and percentage contribution mappings
            feedback_topic_mapping = []
            feedback_topic_percentage_mapping = []

            # Get Feedback-Topic mappings and assign mappings to lists created previously
            get_feedback_topic_mapping (lda_model, gensim_corpus, list_corpus_tokens, 3, 0.3) # Get mappings for LDA model
            # get_feedback_topic_mapping (hdp_model, gensim_corpus, list_corpus_tokens, 3, 0.45) # Get mappings for HDP model

            # Assign topics and topic percentages to feedbacks in the DataFrame
            feedback_ml_df ['TextTopics'] = feedback_topic_mapping
            feedback_ml_df ['TopicPercentages'] = feedback_topic_percentage_mapping

            # Set topics of Feedback with empty TextTokens and TextTopics to nothing (NOTE: By default, if gensim receives an empty list of tokens, it will assign the document ALL topics!)
            feedback_ml_df = feedback_ml_df.apply (unassign_empty_topics_dataframe, axis = 1) # Access row by row 

            """ Create and populate Feedback-Topic DataFrame """
            # Create new dataframe to store all feedback that are assigned with at least one topic after Topic Modelling
            feedback_topic_df = feedback_ml_df [feedback_ml_df.astype (str) ['TextTopics'] != '[]'].copy () # Get feedback that are assigned at least one topic

            # Remove unused columns 
            feedback_topic_df.drop (columns = ['Subject', 'MainText', 'Text', 'TextTokens'], inplace = True)

            # Initialise lists used to store unique topics and new rows to add into the topic-feedback dataframe
            list_topics_assigned = []    # List containing unique topics assigned to at least one Feedback
            list_new_feedback_topic = [] # List containing dictionaries of new rows to add to the topic-feedback dataframe later on

            # Clean and split feedback that are assigned more than one topic into multiple new entries to be added later on in the Feedback-Topic dataframe
            feedback_topic_df = feedback_topic_df.apply (clean_split_feedback_topic_dataframe, axis = 1) 
            
            # Remove feedbacks that are assigned with more than one topic
            feedback_topic_df = feedback_topic_df [feedback_topic_df.TextTopics.str.match (r"^\d*$")] # Only obtain feedbacks whose topics are made of digits (only one topic, since no commas which would be indicative of multiple topics)
            # feedback_topic_df = feedback_topic_df [feedback_topic_df.TextTopics.str.match (r"\d+,\D?\d+")] # Inverse

            # Insert new rows of feedback splitted previously into the FeedbackTopic DataFrame
            feedback_topic_df = feedback_topic_df.append (list_new_feedback_topic, ignore_index = True)

            # Remove duplicate records (for redundancy)
            feedback_topic_df.drop_duplicates (inplace = True)

            """ Create and populate Topics DataFrame """
            # Create new dataframe for Topics
            topic_df = pd.DataFrame (columns = ["Id", "Name", "PriorityScore", "Remarks"]) # Initialise DataFrame to contain topic data 

            # Get current largest TopicID value in the Topics table
            largest_topicid = get_largest_topicid () + 1 # Add extra 1 as Gensim topic IDs start from 0 instead of 1 (added to compensate for this)

            # Initialise dictionary to store information of new row to insert into the Topics DataFrame
            dict_topic = {'Id': 0, 'Name': "", 'PriorityScore': 0, 'Remarks': ""}

            # Populate Topic DataFrame with new topics
            for topic in list_lda_topics: # OR list_hdp_topics for HDP model

                # Assign new ID to current topic
                dict_topic ['Id'] = largest_topicid + topic [0]

                # Get a list of the top 10 most significant words associated with the current topic, along with their weightages [(word, weightage),..]
                list_top_words = lda_model.show_topic (topic [0], topn = 10) # List is by default sorted according to descending order of weightages
                # list_top_words = hdp_model.show_topic (topic [0], topn = 10) # List is by default sorted according to descending order of weightages
                
                # Assign top 10 words associated with the current topic to Remarks
                dict_topic ['Remarks'] = ", ".join ([word for word, weightage in list_top_words])

                # Assign custom name for current topic (model_top_five_words)
                dict_topic ['Name'] = "lda_" + "_".join ([word for word, weightage in list_top_words [:5]])
                # dict_topic ['Name'] = "hdp_" + "_".join ([word for word, weightage in list_top_words [:5]])
                
                # Add new row in Topic DataFrame
                topic_df = topic_df.append (dict_topic, ignore_index = True)

            # Remove topics that have not been assigned to at least one feedback in the Feedback-Topic mapping DataFrame
            topic_df = topic_df [topic_df.Id.isin (list_topics_assigned)]

            """ Manual Tagging """
            # Manually tag topics and assign them to feedbacks from a specified set of tagged words (manual keyword-based topic modelling)
            # ie if feedback contain words related to Pinterest, assign them to the Pinterest topic

            # Check boolean to see whether or not to assign manually labelled topics to feedbacks
            if (use_manual_tag == True): # Implement manual tagging 

                # Manually tagged topic-tokens/words file format:
                # { topic_name: ['token1', 'token2'], topic_2: ['token3'] } 

                # NOTE: Manually-tagged file should contain different topic-word mappings for different datasets (ie a different set of word-tag list should each be used 
                # for datasets about Google VS Facebook). The terms should also be as specific to the specified topic as possible to avoid it being assigned to many topics
                # The terms also should be in lowercase
            
                # Initialise dictionary containing manually-tagged topic-word mappings
                dictionary_manual_tag = json.load (open (manual_tagging_file_path_dm))

                # Get updated value of largest TopicID (after Topic Modelling)
                largest_topicid = max (list_topics_assigned) + 1 # Get largest TopicID in list containing topics that have been assigned to at least one feedback and increment by one for new TopicID

                """ Update Topics DataFrame """
                # Loop through each topic in the manually-tagged topic-word mapping to add topics into the Topic DataFrame
                for topic in dictionary_manual_tag.keys (): 
                    
                    # Add the new topic into the Topics DataFrame
                    topic_df = topic_df.append ({'Id': largest_topicid, 'Name': "manual_" + topic, 'PriorityScore': 0, 
                                                'Remarks': ", ".join ([word for word in dictionary_manual_tag [topic][:10]])}, ignore_index = True)
                    
                    # Update dictionary so that the key-value pair is a tuple in the format {"topic": (["keyword",..], topic_id)}
                    dictionary_manual_tag [topic] = (dictionary_manual_tag [topic], largest_topicid)

                    # Increment the largest TopicID (to prepare for the next topic)
                    largest_topicid = largest_topicid + 1

                """ Update FeedbackTopic DataFrame """
                # Get Feedback-Topic mappings and update the FeedbackML DataFrame accordingly
                feedback_ml_df = feedback_ml_df.apply (get_manual_feedback_topic_mapping, args = (dictionary_manual_tag, 0.3), axis = 1)
            
                # Create Feedback-Topic DataFrame again to store all feedback that are assigned with at least one topic after Topic Modelling
                feedback_topic_df = feedback_ml_df [feedback_ml_df.astype (str) ['TextTopics'] != '[]'].copy () # Get feedback that are assigned at least one topic

                # Remove unused columns from re-created dataframe
                feedback_topic_df.drop (columns = ['Subject', 'MainText', 'Text', 'TextTokens'], inplace = True)

                # Re-initialise list used to store new rows to add into the topic-feedback dataframe
                list_new_feedback_topic = [] # List containing dictionaries of new rows to add to the topic-feedback dataframe later on

                # Clean and split feedback that are assigned more than one topic into multiple entries to add later on in the Feedback-Topic dataframe
                feedback_topic_df = feedback_topic_df.apply (clean_split_feedback_topic_dataframe, axis = 1) 
                
                # Remove feedbacks that are assigned with more than one topic
                feedback_topic_df = feedback_topic_df [feedback_topic_df.TextTopics.str.match (r"^\d*$")] # Only obtain feedbacks whose topics are made of digits (only one topic, since no commas which would be indicative of multiple topics)

                # Insert new rows of feedback splitted previously into the FeedbackTopic DataFrame
                feedback_topic_df = feedback_topic_df.append (list_new_feedback_topic, ignore_index = True)

                # Remove duplicate records (for redundancy)
                feedback_topic_df.drop_duplicates (inplace = True)

                # Remove topics that have not been assigned to at least one feedback in the Feedback-Topic mapping DataFrame
                topic_df = topic_df [topic_df.Id.isin (list_topics_assigned)]

            """ Update Topic status """
            # Check current category
            if (category_id == 4 or category_id == 5):

                # Add new column to Topic DataFrame
                topic_df ['Status'] = 3 # Set default value of Closed for developer's status on Topic if category is 4:General or 5:Feature Request

            else:

                # Add new column to Topic DataFrame
                topic_df ['Status'] = 0 # Default value of Pending for developer's status on Topic for any other category

            """ Database updates """
            # Connect to database to INSERT new topics into the Topics table (need to first insert into the Topics table as FeedbackTopic insertions later on have foreign key references to the Topics table)
            try:

                # Create MySQL connection and cursor objects to the database
                db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
                db_cursor = db_connection.cursor ()

                # Insert the new topics into the Topics database table
                topic_df.apply (insert_topics_dataframe, axis = 1, args = (db_cursor, db_connection))

                # Print debugging message
                print (len (topic_df), "record(s) successfully inserted into Topics table for Category %s" % category_id)
                
            # Catch MySQL Exception
            except mysql.connector.Error as error:

                # Print MySQL connection error
                print ("MySQL error occurred when trying to insert values into Topics table for Category %s:" % category_id, error)

            # Catch other errors
            except:

                # Print other errors
                print ("Error occurred attempting to establish database connection to insert values into the Topics table for Category %s" % category_id)

            finally:

                # Close connection objects once Feedback has been obtained
                db_cursor.close ()
                db_connection.close () # Close MySQL connection

            # Connect to database to INSERT new Feedback-Topic mappings into the FeedbackTopic table
            try:

                # Create MySQL connection and cursor objects to the database
                db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
                db_cursor = db_connection.cursor ()

                # Insert the new Feedback-Topic mappings into the FeedbackTopic database table
                feedback_topic_df.apply (insert_feedback_topic_dataframe, axis = 1, args = (db_cursor, db_connection))

                # Print debugging message
                print (len (feedback_topic_df), "record(s) successfully inserted into FeedbackTopic table for Category %s" % category_id)
                
            # Catch MySQL Exception
            except mysql.connector.Error as error:

                # Print MySQL connection error
                print ("MySQL error occurred when trying to insert values into FeedbackTopic table for Category %s:" % category_id, error)

            # Catch other errors
            except:

                # Print other errors
                print ("Error occurred attempting to establish database connection to insert values into the FeedbackTopic table for Category %s" % category_id)

            finally:

                # Close connection objects once Feedback has been obtained
                db_cursor.close ()
                db_connection.close () # Close MySQL connection

            # Calculate the PriorityScore of each Topic and update the Topics table
            topic_df.apply (calculate_topic_priority_score, axis = 1) # Access each topic row by row

            # Print debugging message
            print (len (topic_df), "Category %s Topic(s)' PriorityScore updated" % category_id)

            """ Miscellaneous """
            # Create interactive visualisation for LDA model
            lda_visualise = pyLDAvis.gensim.prepare (lda_model, gensim_corpus, id2word, mds = 'mmds') # Create visualisation
            pyLDAvis.save_html (lda_visualise, topic_visualise_file_path_dm % category_id) # Export visualisation to HTML file

            # Export and save DataFrames
            feedback_ml_df.to_csv (topic_file_path_dm % category_id, index = False, encoding = "utf-8") # Save FeedbackML DataFrame
            topic_df.to_csv (topics_df_file_path_dm % category_id, index = False, encoding = "utf-8") # Save Topics DataFrame
            feedback_topic_df.to_csv (feedback_topics_df_file_path_dm % category_id, index = False, encoding = "utf-8") # Save FeedbackTopic DataFrame
            
        # Print debugging message if topic modelling not carried out
        else:

            print ("Topic modelling not carried out for Category:", category_id)

    # Get end time of topic modelling
    topic_end_time = datetime.datetime.now ()
    print ("\nCompleted Topic Modelling in", calculate_runtime (0, topic_start_time, topic_end_time), "seconds")

    """ Post-data-mining preparations """
    # Connect to database to UPDATE MineStatus of Feedback 
    try: 

        print ("\nUpdating data-mined status of feedback..")

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # Create SQL statement to update MineStatus of feedback
        sql = "UPDATE %s SET MineStatus = 1 WHERE MineStatus = 0;" % (feedback_table)

        # Execute SQL statement
        db_cursor.execute (sql)

        # Commit changes made
        db_connection.commit ()

        # Debugging
        print (db_cursor.rowcount, "record(s) successfully mined")

    # Catch MySQL Exception
    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error when trying to update mine status of Feedback:", error)

    # Catch other errors
    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection to update mine status of feedback")

    finally:

        # Close connection objects once Feedback has been obtained
        db_cursor.close ()
        db_connection.close () # Close MySQL connection

# Print debugging message if data mining is not carried out
else:

    print ("Data mining not carried out")

# Program end time
program_end_time = datetime.datetime.now ()
program_run_time = program_end_time - program_start_time

print ("\nProgram start time: ", program_start_time)
print ("Program end time: ", program_end_time)
print ("Program runtime: ", (program_run_time.seconds + program_run_time.microseconds / (10**6)), "seconds")
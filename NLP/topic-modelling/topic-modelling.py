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
from gensim import matutils, models # 4) Gensim topic modelling
import gensim.corpora as corpora # 4) Gensim topic modelling
import scipy.sparse # 4) Gensim topic modelling
import logging # 4) Gensim topic modelling logging
from sklearn.model_selection import GridSearchCV # 4) For model hyperparameters tuning
import matplotlib.pyplot as plt # For visualisations
import matplotlib # For visualisations
import pyLDAvis.gensim # For topic modelling visualisations
import pyLDAvis # For topic modelling visualisations
import sklearn.metrics as metrics # 4.5) For determination of model accuracy

# Suppress scikit-learn FutureWarnings
from warnings import simplefilter
simplefilter (action = 'ignore', category = FutureWarning)      # Ignore Future Warnings
simplefilter (action = 'ignore', category = DeprecationWarning) # Ignore Deprecation Warnings

# Logging configurations
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

""" 
NOTE: The Topic Modelling LDA Model needs to be hyper-tuned for every set of feedback

Future enhancements:
-Further hypertuning the LDA model's hyperparameters such as the alpha, beta and eta values (alpha and beta values from the HDP model could be used via the hdp_to_lda () method provided by Gensim)
-The HDP model, which has a high coherence value could be reverse-engineered into a similarly equivalent LDA model for possibly better performance (Gensim provides the suggested_lda_model () method)

"""

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
def save_topics (list_lda_topics, list_hdp_topics):

    # Store topic information in topics file
    topics_file = open (topics_file_path, "w") # Create file object (w = write)

    # Write header information in topics file
    topics_file.write ("LDA Model:\n\n")

    # Get current largest TopicID value in the Topics table
    largest_topicid = get_largest_topicid () + 1 # Add extra 1 as Gensim topic IDs start from 0 instead of 1    (added to compensate for this)

    # Loop to store each topic in the topics file (LDA topics)
    for topic in list_lda_topics:

        print ("Topic", (largest_topicid + topic [0]), ":\n", file = topics_file)
        print (topic [1], "\n", file = topics_file)

    # Write header information in topics file
    topics_file.write ("\n\nHDP Model:\n\n")

    # Loop to store each topic in the topics file (HDP topics)
    for topic in list_hdp_topics:

        print ("Topic", (largest_topicid + topic [0]), ":\n", file = topics_file)
        print (topic [1], "\n", file = topics_file)

    # Close file object
    topics_file.close ()

# Function to get Feedback-Topic mappings for LDA model (provides additional information like word-topic mapping and word phi values)
def get_lda_feedback_topic_mapping (model, corpus, texts, max_no_topics, minimum_percentage, minimum_prob = 0.02):
    
    # NOTE:
    # max_no_topics = maxium number of topics that can be assigned to a Feedback
    # minimum_percentage = minimum percentage contribution required for a topic to be assigned to a Feedback

    # Apply LDA model on corpus to get a Gensim TransformedCorpus object of mappings (Document-Topic mapping, Word-Topic mapping and Phi values)
    transformed_gensim_corpus = model.get_document_topics (corpus, per_word_topics = True, minimum_probability = minimum_prob)

    # Loop to access mappings in the Gensim transformed corpus (made of tuples of lists)
    for tuple_feedback_mapping in transformed_gensim_corpus: # Access document by document
        
        # Tuple contains three lists
        list_document_topic = tuple_feedback_mapping [0] # List containing tuples of current document/feedback topic-percentage mapping [(topic_no, percentage), ..]
        # list_word_topic = tuple_feedback_mapping [1]     # List containing tuples of word - topic mapping [unused]
        # list_phi_value = tuple_feedback_mapping [2]      # List containing tuples of word phi values (probability of a word in the document belonging to a particular topic) [unused]

        # Remove topics below specified minimum percentage contribution
        list_document_topic = list (filter (lambda tup: (tup [1] >= minimum_percentage), list_document_topic)) 

        # Sort list of current document's topic mapping according to descending order of its percentages
        list_document_topic.sort (key = lambda tup: tup [1], reverse = True) # Topics with higher percentage contribution to the feedback will be in front

        # Get the specified amount of top few topics
        list_document_topic = list_document_topic [:max_no_topics] # Get up to the specified number of topics in max_no_topics

        # Initialise lists containing topic(s) and percentages of topic(s) of current feedback/document
        list_topics = []
        list_topics_percentages = []

        # Check length of document-topic mapping to see if any topic is assigned to the current feedback/document (after filtering)
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

# Function to get Feedback-Topic mappings for LDA/HDP model
def get_feedback_topic_mapping (model, corpus, texts, max_no_topics, minimum_percentage):
    
    # NOTE:
    # max_no_topics = maxium number of topics that can be assigned to a Feedback
    # minimum_percentage = minimum percentage contribution required for a topic to be assigned to a Feedback

    # Apply model on corpus to get a Gensim TransformedCorpus object of mappings (Document-Topic mapping ONLY)
    transformed_gensim_corpus = model [corpus]
    
    # Loop to access mappings in the Gensim transformed corpus (made of lists of document-topic mappings)
    for list_document_topic in transformed_gensim_corpus: # Access document by document

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

# Function to hypertune LDA Model to tune the number of topics
def hypertune_no_topics (dictionary, corpus, texts, limit, start, step):
    
    # Initialise list containing model coherence values
    coherence_values = []
    
    # Loop to create respective models and get respective coherence values
    for no_topics in range (start, limit, step):
        
        # Create LDA model
        model = models.LdaModel (corpus = corpus, id2word = id2word, num_topics = no_topics, passes = 100, 
                                 chunksize = 3500, minimum_probability = 0.05, random_state = 123) 
 
        # Create coherence model
        coherence_model = models.CoherenceModel (model = model, texts = texts, dictionary = dictionary, coherence = 'c_v')
        coherence_values.append (coherence_model.get_coherence ()) # Append coherence value of current LDA model
    
    # Print results
    for m, cv in zip (range (start, limit, step), coherence_values):

        print ("Number of Topics =", m, " has Coherence Value of", cv)

    # Plot graph to visualise results
    x = range (start, limit, step)
    plt.plot (x, coherence_values)
    plt.xlabel ("Number of Topics")
    plt.ylabel ("Coherence value")
    plt.legend (("coherence_values"), loc='best')

    # Save graph
    plt.savefig (accuracy_file_path + "topics-coherence.png")

    # Display visualisations if global variable is True
    if (display_visuals == True):

        # Show visualisations
        plt.show ()

# Function to insert each row in the series object passed to the Topics table
def insert_topics_dataframe (series, cursor, connection): 

    # Create SQL statement to insert Topics table values
    sql = "INSERT INTO %s (TopicID, WebAppID, Name, PriorityScore, Remarks) " % (topic_table)
    sql = sql + "VALUES (%s, %s, %s, %s, %s);" 

    # Execute SQL statement
    cursor.execute (sql, (series ['Id'], web_app_id, series ['Name'], series ['PriorityScore'], series ['Remarks']))

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

# Function to pickle object (accepts object to pickle and its filename to save as)
def pickle_object (pickle_object, filename):

    # Get full filepath
    filepath = pickles_file_path + filename

    # Create file object to store object to pickle
    file_pickle = open (filepath, 'wb') # w = write, b = bytes (overwrite pre-existing files if any)

    # Pickle (serialise) object [store object as a file]
    pickle.dump (pickle_object, file_pickle)

    # Close file object
    file_pickle.close ()

# Function to load pickle object (accepts filename of pickle to load and returns the de-pickled object)
def load_pickle (filename):

     # Get full filepath
    filepath = pickles_file_path + filename

    # Create file object accessing the pickle file
    file_pickle = open (filepath, 'rb') # r = read, b = bytes

    # Get pickled object
    pickled_object = pickle.load (file_pickle)

    # Close file object
    file_pickle.close ()

    # Return pickle object
    return pickled_object

# Global variables
# File paths
train_file_path = "/home/p/Desktop/csitml/NLP/topic-modelling/data/feedback-ml.csv" # Dataset file path
topic_file_path = '/home/p/Desktop/csitml/NLP/topic-modelling/data/feedback-ml-topics.csv' # Topic modelled dataset file path
topics_file_path = '/home/p/Desktop/csitml/NLP/topic-modelling/data/topics.txt' # File path of topic details
topics_df_file_path = '/home/p/Desktop/csitml/NLP/topic-modelling/data/topics.csv' # File path of topics table
feedback_topics_df_file_path = '/home/p/Desktop/csitml/NLP/topic-modelling/data/feedback-topics.csv' # File path of feedback-topics table
manual_tagging_file_path = '/home/p/Desktop/csitml/NLP/topic-modelling/data/manual-tagging.txt' # Manually tagged topic-tokens file path
pickles_file_path = "/home/p/Desktop/csitml/NLP/topic-modelling/pickles/" # File path containing pickled objects
accuracy_file_path = "/home/p/Desktop/csitml/NLP/topic-modelling/accuracies/" # Model accuracy results file path

# Boolean triggers global variables
topic_model_data = True # Boolean to trigger application of Topic Modelling model on Feedback data in the database (Default value is TRUE)
preliminary_check = True # Boolean to trigger display of preliminary dataset visualisations and presentations
use_manual_tag = True # Boolean to trigger whether to use manually tagged topics (Reads from manual-tagging.txt)
use_pickle = True # Boolean to trigger whether to use pickled objects or not
display_visuals = True # Boolean to trigger display of visualisations
modify_database = True # Boolean to trigger modifications of the database

# Database global variables
mysql_user = "root"                     # MySQL username
mysql_password = "csitroot"             # MySQL password
mysql_host = "localhost"                # MySQL host
mysql_schema = "csitDB"                 # MySQL schema (NOTE: MySQL in Windows is case-insensitive)
feedback_table = "Feedback"             # Name of feedback table in database
feedback_ml_table = "FeedbackML"        # Name of feedback table in database used for machine learning
topic_table = "Topic"                   # Name of topic table in database 
feedback_topic_table = "FeedbackTopic"  # Name of feedback-topic table in database 
web_app_id = 99                         # ID of selected web app whose feedback will be topic modelled

# Tokens
list_corpus_tokens = [] # Initialise list containing lists of document tokens in the corpus for training Gensim Bigram and Trigram models and for Topic Modelling
token_whitelist = ["photoshop", "editing", "pinterest", "xperia", "instagram", "facebook", "evernote", "update", "dropbox", "picsart", # Token whitelist (to prevent important terms from not being tokenized)
                   "whatsapp", "tripadvisor", "onenote"] 

# Set projected number of topics
selected_topic_no = 65

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

# Program starts here
program_start_time = datetime.datetime.now ()
print ("Start time: ", program_start_time)

# 1) Get dataset (General Feedback)
try:

    # Create MySQL connection object to the database
    db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)

    # Create SQL query to get FeedbackML table values (Feature Engineering)
    sql_query = "SELECT CONCAT(WebAppID, \'_\', FeedbackID, \'_\', CategoryID) as `Id`, SubjectCleaned as `Subject`, MainTextCleaned as `MainText` FROM %s WHERE WebAppID = %s AND SpamStatus = 0 AND CategoryID = 4;" % (feedback_ml_table, web_app_id)   # General 
    # sql_query = "SELECT CONCAT(WebAppID, \'_\', FeedbackID, \'_\', CategoryID) as `Id`, SubjectCleaned as `Subject`, MainTextCleaned as `MainText` FROM %s WHERE SpamStatus = 0 AND CategoryID = 2;" % (feedback_ml_table) # Bug Report 
    # sql_query = "SELECT CONCAT(WebAppID, \'_\', FeedbackID, \'_\', CategoryID) as `Id`, SubjectCleaned as `Subject`, MainTextCleaned as `MainText` FROM %s WHERE SpamStatus = 0 AND CategoryID = 5;" % (feedback_ml_table) # Feature Request

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
    print ("MySQL error when trying to get General HAM records from the FeedbackML table:", error)
    # print ("MySQL error when trying to get Bug Report HAM records from the FeedbackML table:", error)
    # print ("MySQL error when trying to get Feature Request HAM records from the FeedbackML table:", error)

except:

    # Print other errors
    print ("Error occurred attempting to establish database connection to get General HAM records from the FeedbackML table")
    # print ("Error occurred attempting to establish database connection to get Bug Report HAM records from the FeedbackML table")
    # print ("Error occurred attempting to establish database connection to get Feature Request HAM records from the FeedbackML table")

finally:

    # Close connection object once Feedback has been obtained
    db_connection.close () # Close MySQL connection

# Check boolean variable to see whether or not to apply Topic Modelling model on Feedback dataset
if (topic_model_data == True):

    # 2) Further feature engineering and data pre-processings
    # Drop empty rows/columns (for redundancy)
    feedback_ml_df.dropna (how = "all", inplace = True) # Drop empty rows
    feedback_ml_df.dropna (how = "all", axis = 1, inplace = True) # Drop empty columns
    
    # Combine subject and main text into one column [Will apply topic modelling on combined texts of subject together with main text instead of both separately as topic modelling uses the DTM/Bag of Words format, in which the order of words does not matter]
    feedback_ml_df ['Text'] = feedback_ml_df ['Subject'] + " " + feedback_ml_df ['MainText']
    
    # Remove heading and trailing whitespaces in Text (to accomodate cases of blank Subjects in header)
    feedback_ml_df.apply (strip_dataframe, axis = 1) # Access row by row 

    # Create new columns for dataframe
    feedback_ml_df ['TextTokens'] = "[]"       # Default empty list for tokens of feedback text after tokenization
    feedback_ml_df ['TextTopics'] = "[]"       # Default empty list of topics assigned to feedback
    feedback_ml_df ['TopicPercentages'] = "[]" # Default empty list of the percentage contributions of topics assigned to feedback

    # Save dataframe prior to topic modelling 
    feedback_ml_df.to_csv (train_file_path, index = False, encoding = "utf-8")

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
    feedback_ml_df.apply (tokenize_bigram_dataframe, axis = 1) 
    feedback_ml_df.apply (tokenize_trigram_dataframe, axis = 1) 

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
    # Convert list of corpus document-tokens into a dictionary of the locations of each token in the format {location: 'term'}
    id2word = corpora.Dictionary (list_corpus_tokens)

    # # Human readable format of corpus (term-frequency)
    # dtm = [[(id2word [id], freq) for id, freq in cp] for cp in corpus[:1]]

    # Get Term-Document Frequency
    gensim_corpus = [id2word.doc2bow (document_tokens) for document_tokens in list_corpus_tokens]

    # Create new models if not using serialised models
    if (not use_pickle):

        # Create Topic Modelling models
        lda_model = models.LdaModel (corpus = gensim_corpus, id2word = id2word, num_topics = selected_topic_no, passes = 100, 
                                     chunksize = 3500, alpha = 'auto', eta = 'auto', random_state = 123, minimum_probability = 0.05) # LDA Model

        hdp_model = models.HdpModel (corpus = gensim_corpus, id2word = id2word, random_state = 123) # HDP Model (infers the number of topics [always generates 150 topics])
    
    # Using pickled objects
    else:

        # Load serialised models
        lda_model = load_pickle ("lda-model.pkl")
        hdp_model = load_pickle ("hdp-model.pkl")

    """ Get model performances """
    # Get model performance metrics
    print ("*****Model Performances***\n")

    # Hypertune LDA model
    # print ("Finding optimal number of topics for LDA model..")    
    # hypertune_no_topics (dictionary = id2word, corpus = gensim_corpus, texts = list_corpus_tokens, start = 5, limit = 100, step = 5) # Find optimal number of topics with highest coherence value for LDA model

    # Get equivalent LDA parameters of HDP model 
    # print ("Hypertuned alpha and beta values of a LDA almost equivalent of current HDP:", hdp_model.hdp_to_lda ())
    # print ("Closest LDA model to HDP model:", hdp_model.suggested_lda_model ())

    # LDA Model
    print ("LDA:")

    # Compute Coherence Score
    coherence_model_lda = models.CoherenceModel (model = lda_model, texts = list_corpus_tokens, corpus = gensim_corpus, dictionary = id2word, coherence = 'c_v')
    coherence_lda = coherence_model_lda.get_coherence ()
    print('\nCoherence Score: ', coherence_lda)

    # HDP Model
    print ("\nHDP:")

    # Compute Coherence Score
    coherence_model_hdp = models.CoherenceModel (model = hdp_model, texts = list_corpus_tokens, corpus = gensim_corpus, dictionary = id2word, coherence = 'c_v')
    coherence_hdp = coherence_model_hdp.get_coherence ()
    print('\nCoherence Score: ', coherence_hdp, "\n")

    """ Get Topics generated """
    # Get topics
    list_lda_topics = lda_model.show_topics (formatted = True, num_topics = selected_topic_no, num_words = 20)
    list_lda_topics.sort (key = lambda tup: tup [0]) # Sort topics according to ascending order

    list_hdp_topics = hdp_model.show_topics (formatted = True, num_topics = 150, num_words = 20)
    list_hdp_topics.sort (key = lambda tup: tup [0]) # Sort topics according to ascending order

    # print ("Topics:")
    # print ("LDA:")
    # print (list_lda_topics)
    # print ("HDP:")
    # print (list_hdp_topics)

    # Save topics in topic file
    save_topics (list_lda_topics, list_hdp_topics)

    """ Get Feedback-Topic mappings """
    # Initialise lists containing feedback-topic and percentage contribution mappings
    feedback_topic_mapping = []
    feedback_topic_percentage_mapping = []

    # Get Feedback-Topic mappings and assign mappings to lists created previously
    get_feedback_topic_mapping (lda_model, gensim_corpus, list_corpus_tokens, 3, 0.3) # Get mappings for LDA model
    # get_lda_feedback_topic_mapping (lda_model, gensim_corpus, list_corpus_tokens, 3, 0.3) # Get mappings for LDA model (more information)
    # get_feedback_topic_mapping (hdp_model, gensim_corpus, list_corpus_tokens, 3, 0.45) # Get mappings for HDP model

    # Assign topics and topics percentages to feedbacks in the DataFrame
    feedback_ml_df ['TextTopics'] = feedback_topic_mapping
    feedback_ml_df ['TopicPercentages'] = feedback_topic_percentage_mapping

    # Set topics of Feedback with empty TextTokens and TextTopics to nothing (NOTE: By default, if gensim receives an empty list of tokens, it will assign the document ALL topics!)
    feedback_ml_df.apply (unassign_empty_topics_dataframe, axis = 1) # Access row by row 

    """ Create and populate Feedback-Topic DataFrame """
    # Create new dataframe to store all feedback that are assigned with at least one topic after Topic Modelling
    feedback_topic_df = feedback_ml_df [feedback_ml_df.astype (str) ['TextTopics'] != '[]'].copy () # Get feedback that are assigned at least one topic

    # Remove unused columns 
    feedback_topic_df.drop (columns = ['Subject', 'MainText', 'Text', 'TextTokens'], inplace = True)

    # Initialise lists used to store unique topics and new rows to add into the topic-feedback dataframe
    list_topics_assigned = []    # List containing unique topics assigned to at least one Feedback
    list_new_feedback_topic = [] # List containing dictionaries of new rows to add to the topic-feedback dataframe later on

    # Clean and split feedback that are assigned more than one topic into multiple new entries to be added later on in the Feedback-Topic dataframe
    feedback_topic_df.apply (clean_split_feedback_topic_dataframe, axis = 1) 
    
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
    # Manually tag topics and assign them to feedbacks from a specified set of tagged words
    # ie if feedback contain words related to Pinterest, assign them to the Pinterest topic

    # Check boolean to see whether or not to assign manually labelled topics to feedbacks
    if (use_manual_tag == True): # Implement manual tagging 

        # Manually tagged topic-tokens/words file format:
        # { topic_name: ['token1', 'token2'], topic_2: ['token3'] } 

        # NOTE: Manually-tagged file should contain different topic-word mappings for different datasets (ie a different set of word-tag list should each be used 
        # for datasets about Google VS Facebook). The terms should also be as specific to the specified topic as possible to avoid it being assigned to many topics
        # The terms also should be in lowercase
    
        # Initialise dictionary containing manually-tagged topic-word mappings
        dictionary_manual_tag = json.load (open (manual_tagging_file_path))

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
        feedback_ml_df.apply (get_manual_feedback_topic_mapping, args = (dictionary_manual_tag, 0.3), axis = 1)
    
        # Create Feedback-Topic DataFrame again to store all feedback that are assigned with at least one topic after Topic Modelling
        feedback_topic_df = feedback_ml_df [feedback_ml_df.astype (str) ['TextTopics'] != '[]'].copy () # Get feedback that are assigned at least one topic

        # Remove unused columns from re-created dataframe
        feedback_topic_df.drop (columns = ['Subject', 'MainText', 'Text', 'TextTokens'], inplace = True)

        # Re-initialise list used to store new rows to add into the topic-feedback dataframe
        list_new_feedback_topic = [] # List containing dictionaries of new rows to add to the topic-feedback dataframe later on

        # Clean and split feedback that are assigned more than one topic into multiple entries to add later on in the Feedback-Topic dataframe
        feedback_topic_df.apply (clean_split_feedback_topic_dataframe, axis = 1) 
        
        # Remove feedbacks that are assigned with more than one topic
        feedback_topic_df = feedback_topic_df [feedback_topic_df.TextTopics.str.match (r"^\d*$")] # Only obtain feedbacks whose topics are made of digits (only one topic, since no commas which would be indicative of multiple topics)

        # Insert new rows of feedback splitted previously into the FeedbackTopic DataFrame
        feedback_topic_df = feedback_topic_df.append (list_new_feedback_topic, ignore_index = True)

        # Remove duplicate records (for redundancy)
        feedback_topic_df.drop_duplicates (inplace = True)

        # Remove topics that have not been assigned to at least one feedback in the Feedback-Topic mapping DataFrame
        topic_df = topic_df [topic_df.Id.isin (list_topics_assigned)]

    """ Database updates """
    # Check  global boolean variable to see whether or not to modify the database
    if (modify_database == True): # Execute database modifications

        # Connect to database to INSERT new topics into the Topics table (need to first insert into the Topics table as FeedbackTopic insertions later on have foreign key references to the Topics table)
        try:

            # Create MySQL connection and cursor objects to the database
            db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
            db_cursor = db_connection.cursor ()

            # Insert the new topics into the Topics database table
            topic_df.apply (insert_topics_dataframe, axis = 1, args = (db_cursor, db_connection))

            # Print debugging message
            print (len (topic_df), "record(s) successfully inserted into Topics table")
            
        # Catch MySQL Exception
        except mysql.connector.Error as error:

            # Print MySQL connection error
            print ("MySQL error occurred when trying to insert values into Topics table:", error)

        # Catch other errors
        except:

            # Print other errors
            print ("Error occurred attempting to establish database connection to insert values into the Topics table")

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
            print (len (feedback_topic_df), "record(s) successfully inserted into FeedbackTopic table")
            
        # Catch MySQL Exception
        except mysql.connector.Error as error:

            # Print MySQL connection error
            print ("MySQL error occurred when trying to insert values into FeedbackTopic table:", error)

        # Catch other errors
        except:

            # Print other errors
            print ("Error occurred attempting to establish database connection to insert values into the FeedbackTopic table")

        finally:

            # Close connection objects once Feedback has been obtained
            db_cursor.close ()
            db_connection.close () # Close MySQL connection



        # Calculate priorityscore of topic and update Topics table
        pass


    """ Miscellaneous """
    # Create interactive visualisation for LDA model
    lda_visualise = pyLDAvis.gensim.prepare (lda_model, gensim_corpus, id2word) # Create visualisation
    pyLDAvis.save_html (lda_visualise, accuracy_file_path + 'lda.html') # Export visualisation to HTML file

    # Save models (pickling/serialization)
    pickle_object (lda_model, "lda-model.pkl") # LDA Model
    pickle_object (hdp_model, "hdp-model.pkl") # HDP Model

    # Export and save DataFrames
    feedback_ml_df.to_csv (topic_file_path, index = False, encoding = "utf-8") # Save FeedbackML DataFrame
    topic_df.to_csv (topics_df_file_path, index = False, encoding = "utf-8") # Save Topics DataFrame
    feedback_topic_df.to_csv (feedback_topics_df_file_path, index = False, encoding = "utf-8") # Save FeedbackTopic DataFrame
    
# Print debugging message if topic modelling not carried out
else:

    print ("Topic modelling not carried out")

# Program end time
program_end_time = datetime.datetime.now ()
program_run_time = program_end_time - program_start_time

print ("\nProgram start time: ", program_start_time)
print ("Program end time: ", program_end_time)
print ("Program runtime: ", (program_run_time.seconds + program_run_time.microseconds / (10**6)), "seconds")

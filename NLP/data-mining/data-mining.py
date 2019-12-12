import pandas as pd
import mysql.connector # MySQL
from sqlalchemy import create_engine # MySQL
import numpy as np
import re # REGEX
import string
import html
import unidecode
import pickle 
import datetime
from sklearn import svm # 4) SVM classifier
from sklearn.linear_model import LogisticRegression # 4) Logistic Regression classifier
from sklearn.feature_extraction.text import TfidfVectorizer # NLP Vectorizer

# Suppress scikit-learn FutureWarnings
from warnings import simplefilter
simplefilter (action = 'ignore', category = FutureWarning) # Ignore Future Warnings

# Pre-requisite:
# MySQL Setting:
# SET SQL Setting SET SQL_SAFE_UPDATES = 0; # To allow for updates using non-key columns in the WHERE clause

# Function to whitelist Feedback (accepts a Series object of each row in DataFrame and returns a labelled Series object)
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

        # Debugging
        print ("Whitelisted:", series ['FeedbackID'])

    else:

        # Set whitelist column as False [0] if no whitelisted string or bugcode was found in the Subject or MainText of the Feedback
        series ['Whitelist'] = 0 # Default value of Whitelisted is 2 (to indicate that feedback has not been processed)

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

# Function to execute SQL UPDATE for each row in the series object passed to update processed Feedback data
def update_row_dataframe (series, cursor, connection): # Used over iterrows as faster and more efficient

    # Create SQL statement to get Feedback table values
    sql = "UPDATE %s " % (feedback_table)
    sql = sql + "SET Remarks = %s, MainTextCleaned = %s, SubjectCleaned = %s, Whitelisted = %s, SpamStatus = %s WHERE FeedbackID = %s AND CategoryID = %s AND WebAppID = %s;" 

    # Execute SQL statement
    cursor.execute (sql, (series ['Remarks'], series ['MainTextCleaned'], series ['SubjectCleaned'], series ['Whitelist'], series ['SpamStatus'], series ['FeedbackID'], series ['CategoryID'], series ['WebAppID']))

    # Commit changes made
    connection.commit ()

# Function to compute Spam Status of Feedback
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
        print ("Spam record:", series ['Id'])

    else:

        # Otherwise label feedback as not spam
        series ['SpamStatus'] = 0    

    # Return series object
    return series

# Function to execute SQL UPDATE for each row in the series object passed to update the Spam Status of Feedback data
def update_spam_status_dataframe (series, cursor, connection): # Used over iterrows as faster and more efficient

    # Split Id (WebAppID_FeedbackID_CategoryID) into a list
    list_id = series ['Id'].split ('_') # Each ID component is delimited by underscore
    
    # Create SQL statement to get Feedback table values
    sql = "UPDATE %s " % (feedback_table)
    sql = sql + "SET SpamStatus = %s WHERE WebAppID = %sAND FeedbackID = %s AND CategoryID = %s;" 

    # Execute SQL statement
    cursor.execute (sql, (series ['SpamStatus'], list_id [0], list_id [1], list_id [2]))

    # Commit changes made
    connection.commit ()

# Function to calculate runtime of models
def model_runtime (duration, start_time, end_time):

    # Difference in time
    difference = end_time - start_time

    # Calculate runtime
    duration = duration + (difference.seconds + difference.microseconds / (10**6))

    # Return duration to calling program
    return duration

# Custom unpickler to prevent pickle attribute errors (as pickles do not store info on how a class is constructed and needs access to the pickler class when unpickling)
class CustomUnpickler (pickle.Unpickler):

    def find_class (self, module, name):
        
        # Reference to tokenize function in spam_detect.py
        if name == 'tokenize':

            from spam_detect_supplement import tokenize # References spam_detect_supplement.py in the same directory
            return tokenize
            
        return super().find_class(module, name)

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
raw_feedback_file_path = '/home/p/Desktop/csitml/NLP/data-mining/data/raw-feedback.csv' # Raw dataset file path (dataset PRIOR to data mining) [Features for ML]
feedback_file_path = "/home/p/Desktop/csitml/NLP/data-mining/data/feedback.csv" # Dataset file path (dataset AFTER data mining) [Full dataset in MySQL table]
pickles_file_path = "/home/p/Desktop/csitml/NLP/data-mining/pickles/" # File path containing pickled objects
accuracy_file_path = "/home/p/Desktop/csitml/NLP/data-mining/accuracies/" # Model accuracy results file path
preprocess_data = True # Boolean to trigger pre-processing of Feedback data in the database (Default value is TRUE)
remove_trash_data = False # Boolean to trigger deletion of trash Feedback data in the database (Default value is FALSE)
spam_check_data = True # Boolean to trigger application of Spam Detection model on Feedback data in the database (Default value is TRUE)
preliminary_check = True # Boolean to trigger display of preliminary dataset visualisations and presentations

# Database global variables
mysql_user = "root"             # MySQL username
mysql_password = "csitroot"     # MySQL password
mysql_host = "localhost"        # MySQL host
mysql_schema = "csitDB"         # MySQL schema (NOTE: MySQL in Windows is case-insensitive)
feedback_table = "Feedback"     # Name of feedback table in database
feedback_ml_table = "FeedbackML" # Name of feedback table in database used for machine learning
trash_record = "TRASH RECORD"   # Custom identifier for identifying records to be deleted (NOTE: MySQL is case-insensitive!)

# Whitelisting
whitelist = ['csit', 'mindef', 'cve', 'cyber-tech', 'cyber-technology', # Whitelist for identifying non-SPAM feedbacks (whitelist is in lowercase)
            'comms-tech', 'communications-tech', 'comms-technology',
            'communications-technology', 'crypto-tech', 'cryptography-tech',
            'crypto-technology', 'cryptography-technology', 'crash', 'information', 'giving', 'problem', 
            'discovery', 'feature', 'request', 'bug', 'report', 'discover', 'seeking', 'general', 'ui', 
            'ux', 'user', 'password', 'malware', 'malicious', 'vulnerable', 'vulnerability', 'lag', 'hang', 
            'stop', 'usablility', 'usable', 'feedback', 'slow', 'long', 'memory', 'update', 'alert', 
            'install', 'fix', 'future']
bugcode_regex = r"(.*)(BUG\d{6}\$)(.*)" # Assume bug code is BUGXXXXXX$ ($ is delimiter)

# Program starts here
program_start_time = datetime.datetime.now ()
print ("Start time: ", program_start_time)

""" Preliminary preparations before data mining process """
print ("\n***Preliminary preparations before data mining process***\n")

# Check if there are any Feedback in the database which have not been pre-processed yet
try: # Unprocessed Feedback are Feedbacks with Subject/MainText not cleaned yet and not labelled to be whitelisted or not

    # Create MySQL connection object to the database
    db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)

    # Create SQL query to get Feedback table values
    sql_query = "SELECT * FROM %s WHERE PreprocessStatus = 0;" % (feedback_table)

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
if (preprocess_data == True): # Pre-process feedback if there are texts that have not been cleaned or labelled in the whitelist column

    # Print debugging message
    print ("Unprocessed feedback data detected..")
    print ("Pre-processing:", len (feedback_df), "record(s)")

    """" Feedback WHITELISTING """ 
    # Whitelist first before cleaning documents as sometimes some whitelisted documents may be indicated as 
    # TRASH RECORDS (ie Subject contain BUGCODE but MainText is filled with invalid characters)

    # Default value of Whitelisted = 2 (to indicate absence of whitelist check) [0: Not whitelisted, 1: Whitelisted]
    print ("Checking unprocessed feedbacks for whitelisted strings and custom BUGCODE..")

    # Check for whitelisted strings
    feedback_df = feedback_df.apply (whitelist_dataframe, axis = 1) # Access dataframe row by row (row-iteration)

    """CALCULATE OVERALL SCORE OF FEEDBACK AND UPDATE!
    
    CONVERT TRIGGER FUNCTIONS TO PYTHON CODE HERE!
    """

    """ Feedback CLEANING """
    print ("Cleaning unprocessed feedbacks..")

    # Create new dataframe for feedback database table used for machine learning
    feedback_ml_df = pd.DataFrame (columns = ["FeedbackID", "WebAppID", "CategoryID", "SubjectCleaned", "MainTextCleaned", "SubjectSpam", "MainTextSpam", "SpamStatus"]) 

    # Pre-process new Feedback data that have not been pre-processed
    feedback_ml_df.MainTextCleaned = clean_document (feedback_df.MainText) # Clean main text
    feedback_ml_df.SubjectCleaned = clean_document (feedback_df.Subject) # Clean subject text
    
     # Extract rows containing empty main texts (invalid records to be removed later on)
    feedback_df_trash = feedback_ml_df [feedback_ml_df.MainTextCleaned == ""].copy () # Get feedback with MainTextCleaned set to blank
 
    # Set remarks of empty rows to custom trash record identifier remark for removal ("TRASH RECORD") if feedback is NOT WHITELISTED
    feedback_df_trash.loc [feedback_df_trash ['Whitelist'] != 1, 'Remarks'] = trash_record # NEED TO EDIT THIS FILTER! (FILTER WILL HAVE TO REFERENCE 2 TABLES!)

    # Set SpamStatus of trash records that are not whitelisted as 3 to mark that the feedbacks are unable to be processed as they are trash records
    feedback_df_trash.loc [feedback_df_trash ['Whitelist'] != 1, 'SpamStatus'] = 3 # see how to run this, got warning

    # Print debugging message
    print ("Number of trash record(s) found:", len (feedback_df_trash.loc [feedback_df_trash ['Whitelist'] != 1]), "record(s)")

    # Remove rows containing empty texts (remove trash records from current dataframe)
    feedback_df = feedback_df [feedback_df.MainTextCleaned != ""]

    # Combined newly labelled empty rows into the previous dataframe (Rows with empty MainText now have the Remark 'TRASH RECORD' for later removal [except for those with WHITELISTED STRINGS])
    feedback_df = feedback_df.append (feedback_df_trash, ignore_index = True)
    feedback_df_trash.to_csv ("/home/p/Desktop/csitml/NLP/data-mining/data/trash.csv")
    # Connect to database to update values
    try:

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # Update database table with the newly pre-processed data
        feedback_df.apply (update_row_dataframe, axis = 1, args = (db_cursor, db_connection))

        # Print debugging message
        print (len (feedback_df), "record(s) successfully pre-processed")
        
    # Catch MySQL Exception
    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error:", error)

    # Catch other errors
    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection")

    finally:

        # Close connection objects once Feedback has been obtained
        db_cursor.close ()
        db_connection.close () # Close MySQL connection

# Check boolean to see whether or not to delete records (intrusive) that contain the custom TRASH RECORD identifier in Remarks
if (remove_trash_data == True): # By default don't delete TRASH RECORDs as even though they lack quality information, they serve semantic value (metadata)

    # Remove trash Feedback data marked with custom removal Remarks ("TRASH RECORD")
    try: # Trash Feedback are feedback which has either its SubjectCleaned or MainTextCleaned empty after data cleaning (meaning that they contain and are made up of invalid characters)

        print ("Removing TRASH records from the database..")

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # Create SQL statement to delete records with the Remarks set as 'TRASH RECORD'
        sql = "DELETE FROM %s WHERE Remarks = \'%s\';" % (feedback_table, trash_record)

        # Execute SQL statement
        db_cursor.execute (sql)

        # Commit changes made
        db_connection.commit ()

        # Debugging
        print (db_cursor.rowcount, "trash record(s) deleted")

    # Catch MySQL Exception
    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error:", error)

    # Catch other errors
    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection")

    finally:

        # Close connection objects once Feedback has been obtained
        db_cursor.close ()
        db_connection.close () # Close MySQL connection

""" Start DATA MINING process """
print ("\n\n***Data Mining***\n")

# 1) Get dataset for Spam Detection
try:

    # Create MySQL connection object to the database
    db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)

    # Create SQL query to get Feedback table values (Feature Engineering)
    sql_query = "SELECT CONCAT(WebAppID, \'_\', FeedbackID, \'_\', CategoryID) as `Id`, OverallScore, SubjectCleaned as `Subject`, MainTextCleaned as `MainText`, SpamStatus FROM %s WHERE (Remarks != \'%s\' OR Remarks IS NULL) AND Whitelisted != 1 AND SpamStatus NOT IN (0, 1, 3);" % (feedback_table, trash_record)

    # Execute query and convert Feedback table into a pandas DataFrame
    feedback_df = pd.read_sql (sql_query, db_connection)

    # Check if dataframe obtained is empty
    if (feedback_df.empty == True):

        # Set boolean to apply spam detection model on data to False if dataframe obtained is empty
        spam_check_data = False

    else:

        # Set boolean to apply spam detection model on data to True (default value) if dataframe obtained is not empty
        spam_check_data = True

    """
    Selected Feedback features:
    -Id (WebAppID + FeedbackID + CategoryID) [Not ID as will cause Excel .sylk file intepretation error]
    -Overall score
    -Subject (processed)
    -Main text (processed)
    -Spam status [target1]
    -Topic [added] [target2]

    --> Dataset obtained at this point contains pre-processed Feedback data that are NOT trash records, NOT whitelisted and NOT classified as spam/ham
    """

except mysql.connector.Error as error:

    # Print MySQL connection error
    print ("MySQL error:", error)

except:

    # Print other errors
    print ("Error occurred attempting to establish database connection")

finally:

    # Close connection object once Feedback has been obtained
    db_connection.close () # Close MySQL connection

# Check boolean to see whether or not to apply Spam Detectino model on Feedback data
if (spam_check_data == True):

    # 2) Further feature engineering (Data pre-processing) [FOR REDUNDANCY]
    # Drop empty rows/columns
    feedback_df.dropna (how = "all", inplace = True) # Drop empty rows
    feedback_df.dropna (how = "all", axis = 1, inplace = True) # Drop empty columns

    # Remove rows containing empty main texts (trash records)
    feedback_df = feedback_df [feedback_df.MainText != ""]

    # Add two extra custom columns to identify whether Subject or MainText portion of Feedback is Spam
    feedback_df ['SubjectSpam'] = 0 # Default value of 0 [HAM]
    feedback_df ['MainTextSpam'] = 0 # Default value of 0 [HAM]

    # Save cleaned raw (prior to data mining) dataset to CSV
    feedback_df.to_csv (raw_feedback_file_path, index = False, encoding = "utf-8")

    # 3) Understand dataset
    if (preliminary_check == True): # Check boolean to display preliminary information

        # Print some information of about the data
        print ("Preliminary information about dataset:")
        print ("Dimensions: ", feedback_df.shape, "\n")
        print ("First few records:")
        print (feedback_df.head (), "\n")
        print ("Columns and data types:")
        print (feedback_df.dtypes, "\n")

    # 4) Apply spam-detection model
    # Assign target and features variables
    target_subject = feedback_df.SubjectSpam # Target and feature variables for Subject
    feature_subject = feedback_df.Subject

    target_main_text = feedback_df.MainTextSpam # Target and feature variables for MainText
    feature_main_text = feedback_df.MainText 

    # Load pickled/serialised vectorizer from spam-detection program
    start_time = datetime.datetime.now ()
    vectorizer = load_pickle ("tfidf-vectorizer.pkl")
    end_time = datetime.datetime.now ()
    print ("Loaded vectorizer in", model_runtime (0, start_time, end_time), "seconds")

    # Fit data to vectorizer [Create DTM of dataset (features)]
    start_time = datetime.datetime.now ()
    feature_subject = vectorizer.transform (feature_subject) 
    end_time = datetime.datetime.now ()
    print ("Transformed subject to DTM in", model_runtime (0, start_time, end_time), "seconds")

    start_time = datetime.datetime.now ()
    feature_main_text = vectorizer.transform (feature_main_text) 
    end_time = datetime.datetime.now ()
    print ("Transformed main text to DTM in", model_runtime (0, start_time, end_time), "seconds")

    # Initialise model duration
    spam_model_duration = 0 

    # Load pickled model
    start_time = datetime.datetime.now ()
    spam_model = load_pickle ("svm-model.pkl") # Used SVM Model in this case
    # spam_model = load_pickle ("logistic-regression-model.pkl") # Used LR Model in this case
    end_time = datetime.datetime.now ()
    spam_model_duration = model_runtime (spam_model_duration, start_time, end_time)

    # Predict whether Subject is spam or not
    print ("\nPredicting whether subjects of feedback is spam..")
    start_time = datetime.datetime.now ()
    model_prediction_subject = spam_model.predict (feature_subject) # Store predicted results of model
    end_time = datetime.datetime.now ()
    spam_model_duration = model_runtime (spam_model_duration, start_time, end_time)
    print ("Predicted subject values:", model_prediction_subject)

    # Predict whether MainText is spam or not
    print ("\nPredicting whether main texts of feedback is spam..")
    start_time = datetime.datetime.now ()
    model_prediction_main_text = spam_model.predict (feature_main_text) # Store predicted results of model
    end_time = datetime.datetime.now ()
    spam_model_duration = model_runtime (spam_model_duration, start_time, end_time)
    print ("Predicted main text values:", model_prediction_main_text)

    # Print spam model runtime
    print ("\nSpam model runtime: ", spam_model_duration, "seconds")

    # Collate results of spam-detection predictions
    feedback_df.SubjectSpam = model_prediction_subject
    feedback_df.MainTextSpam = model_prediction_main_text
    feedback_df = feedback_df.apply (spam_status_dataframe, axis = 1) # Access dataframe row by row (row-iteration)

    # Save cleaned (after data mining) dataset to CSV
    feedback_df.to_csv (feedback_file_path, index = False, encoding = "utf-8")

    # Connect to database to update SpamStatus values of Feedback
    try:

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # Update database table with the newly pre-processed data
        feedback_df.apply (update_spam_status_dataframe, axis = 1, args = (db_cursor, db_connection))

        # Print debugging message
        print (len (feedback_df), "record(s) successfully classified as SPAM/HAM")

    # Catch MySQL Exception
    except mysql.connector.Error as error:

        # Print MySQL connection error
        print ("MySQL error:", error)

    # Catch other errors
    except:

        # Print other errors
        print ("Error occurred attempting to establish database connection")

    finally:

        # Close connection objects once Feedback has been obtained
        db_cursor.close ()
        db_connection.close () # Close MySQL connection

# Program end time
program_end_time = datetime.datetime.now ()
program_run_time = program_end_time - program_start_time

print ("\nProgram start time: ", program_start_time)
print ("Program end time: ", program_end_time)
print ("Program runtime: ", (program_run_time.seconds + program_run_time.microseconds / (10**6)), "seconds")
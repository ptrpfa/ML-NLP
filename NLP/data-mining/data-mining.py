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
import os
from sklearn import svm # 4) SVM classifier
from sklearn.linear_model import LogisticRegression # 4) Logistic Regression classifier
from sklearn.feature_extraction.text import TfidfVectorizer # NLP Vectorizer

# Suppress scikit-learn FutureWarnings
from warnings import simplefilter
simplefilter (action = 'ignore', category = FutureWarning) # Ignore Future Warnings

# Pre-requisite:
# MySQL Setting:
# SET SQL Setting SET SQL_SAFE_UPDATES = 0; # To allow for updates using non-key columns in the WHERE clause

"""
NOTE: Since the database is only accessed by the database administrator/programs, 
it is assumed that the records inserted will be CORRECT and validated
--> Calculation of OverallScore of each Feedback is only done on INSERTs into the Feedback table ONCE. 
UPDATEs to the Feedback table will not re-calculate the OverallScore of each Feedback

--> Calculation of PriorityScore of each Topic is only done on INSERTs into the FeedbackTopic table ONCE. 
UPDATEs to the FeedbackTopic table will not re-calculate the OverallScore of each Topic.
--> For this need to add in code that will automatically run upon changes within the FeedbackTopic table!
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

# Function to execute SQL UPDATE for each row in the series object passed to update processed Feedback data
def update_feedback_dataframe (series, cursor, connection): 

    # Create SQL statement to update Feedback table values
    sql = "UPDATE %s " % (feedback_table)
    sql = sql + "SET OverallScore = %s, Whitelist = %s, PreprocessStatus = %s WHERE FeedbackID = %s AND CategoryID = %s AND WebAppID = %s;" 

    # Execute SQL statement
    cursor.execute (sql, (series ['OverallScore'], series ['Whitelist'], series ['PreprocessStatus'], series ['FeedbackID'], series ['CategoryID'], series ['WebAppID']))

    # Commit changes made
    connection.commit ()

# Function to insert each row in the series object passed to the FeedbackML table
def insert_feedback_ml_dataframe (series, cursor, connection): 

    # Create SQL statement to update Feedback table values
    sql = "INSERT INTO %s (FeedbackID, CategoryID, WebAppID, SubjectCleaned, MainTextCleaned, SubjectSpam, MainTextSpam, SpamStatus) " % (feedback_ml_table)
    sql = sql + "VALUES (%s, %s, %s, %s, %s, %s, %s, %s);" 

    # Execute SQL statement
    cursor.execute (sql, (series ['FeedbackID'], series ['CategoryID'], series ['WebAppID'], series ['SubjectCleaned'], series ['MainTextCleaned'], series ['SubjectSpam'], series ['MainTextSpam'], series ['SpamStatus']))

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
    
    # Create SQL statement to get Feedback table values
    sql = "UPDATE %s " % (feedback_ml_table)
    sql = sql + "SET SubjectSpam = %s, MainTextSpam = %s, SpamStatus = %s WHERE WebAppID = %sAND FeedbackID = %s AND CategoryID = %s;" 

    # Execute SQL statement
    cursor.execute (sql, (series ['SubjectSpam'], series ['MainTextSpam'], series ['SpamStatus'], list_id [0], list_id [1], list_id [2]))

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
# File paths to store data pre-processing and data mining feedback
folder = "%s-%s:%s" % (str (datetime.date.today ()), str (datetime.datetime.now ().hour), str (datetime.datetime.now ().minute)) # Folder file name (yyyy-mm-dd:hh:mm)
feedback_file_path_p = '/home/p/Desktop/csitml/NLP/data-mining/data/%s/pre-processing/feedback.csv' % folder # Dataset file path 
feedback_ml_file_path_p = "/home/p/Desktop/csitml/NLP/data-mining/data/%s/pre-processing/feedback-ml.csv" % folder # Dataset file path 
combined_feedback_file_path_p = "/home/p/Desktop/csitml/NLP/data-mining/data/%s/pre-processing/combined-feedback.csv" % folder # Dataset file path 
trash_feedback_file_path_p = "/home/p/Desktop/csitml/NLP/data-mining/data/%s/pre-processing/trash-feedback.csv" % folder # Dataset file path 
feedback_ml_prior_file_path_dm = '/home/p/Desktop/csitml/NLP/data-mining/data/%s/data-mining/feedback-ml-before.csv' % folder # Raw dataset file path (dataset PRIOR to data mining) [Features for ML]
feedback_ml_file_path_dm = "/home/p/Desktop/csitml/NLP/data-mining/data/%s/data-mining/feedback-ml.csv" % folder # Dataset file path (dataset AFTER data mining)
pickles_file_path = "/home/p/Desktop/csitml/NLP/data-mining/pickles/" # File path containing pickled objects

# Boolean triggers global variables
preprocess_data = True # Boolean to trigger pre-processing of Feedback data in the database (Default value is TRUE)
remove_trash_data = False # Boolean to trigger deletion of trash Feedback data in the database (Default value is FALSE) [INTRUSIVE]
mine_data = True # Boolean to trigger data mining of Feedback data in the database (Default value is TRUE)
spam_check_data = True # Boolean to trigger application of Spam Detection model on Feedback data in the database (Default value is TRUE)
preliminary_check = True # Boolean to trigger display of preliminary dataset visualisations and presentations

# Database global variables
mysql_user = "root"                 # MySQL username
mysql_password = "csitroot"         # MySQL password
mysql_host = "localhost"            # MySQL host
mysql_schema = "csitDB"             # MySQL schema (NOTE: MySQL in Windows is case-insensitive)
feedback_table = "Feedback"         # Name of feedback table in database
feedback_ml_table = "FeedbackML"    # Name of feedback table in database used for machine learning
category_factor = {}                # Initialise empty dictionary to contain mapping of category-factor values for computing overall score of feedback

# Whitelisting
whitelist = ['csit', 'mindef', 'cve', 'cyber-tech', 'cyber-technology', # Whitelist for identifying non-SPAM feedbacks (whitelist words are in lowercase)
            'comms-tech', 'communications-tech', 'comms-technology',
            'communications-technology', 'crypto-tech', 'cryptography-tech',
            'crypto-technology', 'cryptography-technology', 'crash', 'information', 'giving', 'problem', 
            'discovery', 'feature', 'request', 'bug', 'report', 'discover', 'seeking', 'general', 'ui', 
            'ux', 'user', 'password', 'malware', 'malicious', 'vulnerable', 'vulnerability', 'lag', 'hang', 
            'stop', 'usablility', 'usable', 'feedback', 'slow', 'long', 'memory', 'update', 'alert', 
            'install', 'fix', 'future', 'experience']
bugcode_regex = r"(.*)(BUG\d{6}\$)(.*)" # Assume bug code is BUGXXXXXX$ ($ is delimiter)

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
if (preprocess_data == True): # Pre-process feedback if there are unpre-processed feedback

    # Get start time
    start_time = datetime.datetime.now ()

    # Print debugging message
    print ("Unprocessed feedback data detected!")
    print ("Pre-processing:", len (feedback_df), "record(s)")

    # Create new empty dataframe for feedback database table used for machine learning
    feedback_ml_df = pd.DataFrame (columns = ["FeedbackID", "WebAppID", "CategoryID", "SubjectCleaned", "MainTextCleaned", "SubjectSpam", "MainTextSpam", "SpamStatus"]) 

    # Set index values in FeedbackML (NOTE: Other columns in feedback_ml_df are empty as data mining have not been carried out here)
    feedback_ml_df.FeedbackID = feedback_df.FeedbackID # Set FeedbackID
    feedback_ml_df.CategoryID = feedback_df.CategoryID # Set CategoryID
    feedback_ml_df.WebAppID = feedback_df.WebAppID     # Set WebAppID

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
    combined_feedback_df_trash = combined_feedback_df [combined_feedback_df.MainTextCleaned == ""].copy () # Get feedback with MainTextCleaned set to blan

    # Update columns accordingly to mark invalid rows as blacklisted invalid trash records
    combined_feedback_df_trash.loc [combined_feedback_df_trash ['Whitelist'] != 1, 'SubjectSpam'] = 3  # Set SubjectSpam status to unable to process (3) [for UNWHITELISTED records]
    combined_feedback_df_trash.loc [combined_feedback_df_trash ['Whitelist'] != 1, 'MainTextSpam'] = 3  # Set MainTextSpam status to unable to process (3) [for UNWHITELISTED records]
    combined_feedback_df_trash.loc [combined_feedback_df_trash ['Whitelist'] != 1, 'SpamStatus'] = 3  # Set SpamStatus to unable to process (3) [for UNWHITELISTED records]
    combined_feedback_df_trash.loc [combined_feedback_df_trash ['Whitelist'] != 1, 'Whitelist'] = 3  # Set whitelisted status to blacklisted (3) [for UNWHITELISTED records]

    # Print debugging message
    print ("Number of trash record(s) found:", len (combined_feedback_df_trash.loc [combined_feedback_df_trash ['Whitelist'] != 1]), "record(s)")

    # Remove rows containing empty texts (remove trash records from current dataframe)
    combined_feedback_df = combined_feedback_df [combined_feedback_df.MainTextCleaned != ""]

    # Combined newly labelled empty rows into the previous dataframe
    combined_feedback_df = combined_feedback_df.append (combined_feedback_df_trash) # Take note of index here! (no change)

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
    if (not os.path.exists ("/home/p/Desktop/csitml/NLP/data-mining/data/%s" % folder)):

        # Create folder if it doesn't exist
        os.mkdir ("/home/p/Desktop/csitml/NLP/data-mining/data/%s" % folder) 
    
    # Check if sub-folder for pre-processed feedback exists
    if (not os.path.exists ("/home/p/Desktop/csitml/NLP/data-mining/data/%s/pre-processing/" % folder)):

        # Create sub-folder if it doesn't exist
        os.mkdir ("/home/p/Desktop/csitml/NLP/data-mining/data/%s/pre-processing/" % folder) 
    
    # Export dataframes to CSV
    combined_feedback_df.to_csv (combined_feedback_file_path_p, index = False, encoding = "utf-8")
    combined_feedback_df_trash.to_csv (trash_feedback_file_path_p, index = False, encoding = "utf-8")
    feedback_df.to_csv (feedback_file_path_p, index = False, encoding = "utf-8")
    feedback_ml_df.to_csv (feedback_ml_file_path_p, index = False, encoding = "utf-8")

    # Get data pre-processing end time
    end_time = datetime.datetime.now ()

    # Print data pre-processing duration
    print ("\nData pre-processing completed in", model_runtime (0, start_time, end_time), "seconds")

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
        sql_query = "SELECT CONCAT(WebAppID, \'_\', FeedbackID, \'_\', CategoryID) as `Id`, SubjectCleaned as `Subject`, MainTextCleaned as `MainText`, SubjectSpam, MainTextSpam, SpamStatus FROM %s WHERE SpamStatus = 2;" % (feedback_ml_table)

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
        -SubjectCleaned (processed)
        -MainTextCleaned (processed)
        -Spam status [target1]
        -Topic [added] [target2]

        --> Dataset obtained at this point contains pre-processed Feedback data that are NOT trash records, NOT whitelisted and NOT classified as spam/ham
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

    # Check boolean variable to see whether or not to apply Spam Detection model on Feedback data
    if (spam_check_data == True):

        # 2) Further feature engineering (Data pre-processing) [FOR REDUNDANCY]
        # Drop empty rows/columns
        feedback_ml_df.dropna (how = "all", inplace = True) # Drop empty rows
        feedback_ml_df.dropna (how = "all", axis = 1, inplace = True) # Drop empty columns

        # Remove rows containing empty main texts (trash records)
        feedback_ml_df = feedback_ml_df [feedback_ml_df.MainText != ""]

        # Check if folder to store data-mined feedback exists
        if (not os.path.exists ("/home/p/Desktop/csitml/NLP/data-mining/data/%s" % folder)):

            # Create folder if it doesn't exist
            os.mkdir ("/home/p/Desktop/csitml/NLP/data-mining/data/%s" % folder) 

        # Check if sub-folder for data-mined feedback exists
        if (not os.path.exists ("/home/p/Desktop/csitml/NLP/data-mining/data/%s/data-mining/" % folder)):

            # Create sub-folder if it doesn't exist
            os.mkdir ("/home/p/Desktop/csitml/NLP/data-mining/data/%s/data-mining/" % folder) 

        # Save cleaned raw (prior to data mining) dataset to CSV
        feedback_ml_df.to_csv (feedback_ml_prior_file_path_dm, index = False, encoding = "utf-8")

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
        print ("\nSpam model runtime: ", spam_model_duration, "seconds\n")

        # Collate results of spam-detection predictions
        feedback_ml_df.SubjectSpam = model_prediction_subject
        feedback_ml_df.MainTextSpam = model_prediction_main_text

        # Get overall SpamStatus of each feedback record
        feedback_ml_df = feedback_ml_df.apply (spam_status_dataframe, axis = 1) 

        # Save spam-mined dataset to CSV
        feedback_ml_df.to_csv (feedback_ml_file_path_dm, index = False, encoding = "utf-8")

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

    """ Topic Modelling on Feedback data to group similar feedback together for ease of prioritisation to developers in the developer's platform """

    # Apply TOPIC MODELLING model
    pass

    # Insert Feedback-Topic mappings to the FeedbackTopic table in the database
    pass

    # Update each topic's PriorityScore in the Topic table in the database (compute average OverallScore of all Feedback in the same topic)
    pass

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
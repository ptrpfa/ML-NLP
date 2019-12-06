import pandas as pd
import mysql.connector # MySQL
from sqlalchemy import create_engine # MySQL
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

# Pre-requisite:
# SET SQL Setting SET SQL_SAFE_UPDATES = 0; # To allow for updates using non-key columns in the WHERE clause

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

    # Return list of cleaned documents
    return list_cleaned_documents

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
raw_feedback_file_path = '/home/p/Desktop/csitml/NLP/data-mining/data/raw-feedback.csv' # Raw dataset file path (dataset PRIOR to data mining)
feedback_file_path = "/home/p/Desktop/csitml/NLP/data-mining/data/feedback.csv" # Dataset file path (dataset AFTER data mining)
pickles_file_path = "/home/p/Desktop/csitml/NLP/data-mining/pickles/" # File path containing pickled objects
accuracy_file_path = "/home/p/Desktop/csitml/NLP/data-mining/accuracies/" # Model accuracy results file path
preprocess_data = True # Boolean to trigger pre-processing of Feedback data in the database (Default value is TRUE)
remove_trash_data = False # Boolean to trigger deletion of trash Feedback data in the database (Default value is FALSE)
preliminary_check = True # Boolean to trigger display of preliminary dataset visualisations and presentations

# Database global variables
mysql_user = "root"             # MySQL username
mysql_password = "csitroot"     # MySQL password
mysql_host = "localhost"        # MySQL host
mysql_schema = "csitDB"         # MySQL schema (NOTE: MySQL in Windows is case-insensitive)
feedback_table = "Feedback"     # Name of feedback table in database
trash_record = "TRASH RECORD"   # Custom identifier for identifying records to be deleted (NOTE: MySQL is case-insensitive!)

# Program starts here
program_start_time = datetime.datetime.now ()
print ("Start time: ", program_start_time)

""" Preliminary preparations before data mining process """
print ("\n***Preliminary preparations before data mining process***\n")

# Check if there are any Feedback in the database which have not been pre-processed yet
try:

    # Create MySQL connection object to the database
    db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)

    # Create SQL query to get Feedback table values
    sql_query = "SELECT * FROM %s WHERE MainTextCleaned IS NULL OR SubjectCleaned IS NULL;" % (feedback_table)

    # Execute query and convert Feedback table into a pandas DataFrame
    feedback_to_clean_df = pd.read_sql (sql_query, db_connection)

    # Check if dataframe obtained is empty
    if (feedback_to_clean_df.empty == True):

        # Set boolean to pre-process data to False if dataframe obtained is empty
        preprocess_data = False

    else:

        # Set boolean to pre-process data to True (default value) if dataframe obtained is not empty
        preprocess_data = True

# Catch MySQL Exception
except mysql.connector.Error as error:

    # Print MySQL connection error
    print ("MySQL error:", error)

# Catch other errors
except:

    # Print other errors
    print ("Error occurred attempting to establish database connection")

finally:

    # Close connection object once Feedback has been obtained
    db_connection.close () # Close MySQL connection

# Check boolean to see whether or not data pre-processing needs to be first carried out on the Feedbacks collected
if (preprocess_data == True): # Pre-process feedback if there are texts that have not been cleaned

    # Print debugging message
    print ("Unprocessed feedback data detected..")
    print ("Pre-processing:", len (feedback_to_clean_df), "record(s)")

    # print ("Records:")
    # print (feedback_to_clean_df)

    # Pre-process new Feedback data that have not been pre-processed
    feedback_to_clean_df.MainTextCleaned = clean_document (feedback_to_clean_df.MainText) # Clean main text
    feedback_to_clean_df.SubjectCleaned = clean_document (feedback_to_clean_df.Subject) # Clean subject text
    
     # Extract rows containing empty texts and combine them into a new dataframe containing trash records (records to be removed later on)
    feedback_to_clean_df_trash = feedback_to_clean_df [feedback_to_clean_df.MainTextCleaned == ""] # Get feedback with MainTextCleaned set to blank
    feedback_to_clean_df_trash = feedback_to_clean_df_trash.append (feedback_to_clean_df [feedback_to_clean_df.SubjectCleaned == ""], ignore_index = True) # Get feedback with SubjectCleaned set to blank
    feedback_to_clean_df_trash ['Remarks'] = trash_record # Set remarks of empty rows to custom trash record identifier remark for removal ("TRASH RECORD")

    # Remove duplicate trash feedbacks (for trash feedback with both blanks for SubjectCleaned and MainTextCleaned)[keeps first occurance of duplicated feedback]
    feedback_to_clean_df_trash.drop_duplicates (subset = "FeedbackID", keep = "first", inplace = True)

    # Print debugging message
    print ("Number of trash record(s) found:", len (feedback_to_clean_df_trash), "record(s)")

    # Remove rows containing empty texts (remove trash records from current dataframe)
    feedback_to_clean_df = feedback_to_clean_df [feedback_to_clean_df.MainTextCleaned != ""]
    feedback_to_clean_df = feedback_to_clean_df [feedback_to_clean_df.SubjectCleaned != ""]

    # Combined newly labelled empty rows into the previous dataframe (Rows with empty Subject or MainText now have the Remark 'TRASH RECORD' for later removal)
    feedback_to_clean_df = feedback_to_clean_df.append (feedback_to_clean_df_trash, ignore_index = True)

    # Connect to database to update values
    try:

        # Create MySQL connection and cursor objects to the database
        db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)
        db_cursor = db_connection.cursor ()

        # Update database table with the newly pre-processed data
        for index, row in feedback_to_clean_df.iterrows ():

            # Create SQL statement to get Feedback table values
            sql = "UPDATE %s " % (feedback_table)
            sql = sql + "SET Remarks = %s, MainTextCleaned = %s, SubjectCleaned = %s WHERE FeedbackID = %s AND CategoryID = %s AND WebAppID = %s;" 

            # Execute SQL statement
            db_cursor.execute (sql, (row ['Remarks'], row ['MainTextCleaned'], row ['SubjectCleaned'], row ['FeedbackID'], row ['CategoryID'], row ['WebAppID']))

            # Commit changes made
            db_connection.commit ()

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

        # Print debugging message
        print (len (feedback_to_clean_df), "record(s) successfully pre-processed")

# Check boolean to see whether or not to delete records (intrusive) that contain the custom TRASH RECORD identifier in Remarks
if (remove_trash_data == True):

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

# 1) Get dataset
try:

    # Create MySQL connection object to the database
    db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)

    # Create SQL query to get Feedback table values
    sql_query = "SELECT CONCAT(WebAppID, \'_\', FeedbackID, \'_\', CategoryID) as `ID`, OverallScore, SubjectCleaned as `Subject`, MainTextCleaned as `MainText` FROM %s;" % (feedback_table)
    # sql_query = "SELECT * FROM %s" % (feedback_table)

    # Execute query and convert Feedback table into a pandas DataFrame
    feedback_df = pd.read_sql (sql_query, db_connection)

    # Save dataframe as a CSV file
    feedback_df.to_csv (raw_feedback_file_path, index = False, encoding = "utf-8")

except mysql.connector.Error as error:

    # Print MySQL connection error
    print ("MySQL error:", error)

except:

    # Print other errors
    print ("Error occurred attempting to establish database connection")

finally:

    # Close connection object once Feedback has been obtained
    db_connection.close () # Close MySQL connection

# 2) Understand dataset
if (preliminary_check == True): # Check boolean to display preliminary information

    # Print some information of about the data
    print ("Preliminary information about dataset:")
    print ("Dimensions: ", feedback_df.shape, "\n")
    print ("First few records:")
    print (feedback_df.head (), "\n")
    print ("Columns and data types:")
    print (feedback_df.dtypes, "\n")

# 3) Feature Engineering (Data pre-processing)
pass

# Create custom identifier (WebAppID_FeedbackID_CategoryID)

"""
Feedback features:
-ID (WebAppID + FeedbackID + CategoryID)
-Subject
-Overall score
-Main text
"""

# 4) Apply spam-detection model

# Program end time
program_end_time = datetime.datetime.now ()
program_run_time = program_end_time - program_start_time

print ("\nProgram start time: ", program_start_time)
print ("Program end time: ", program_end_time)
print ("Program runtime: ", (program_run_time.seconds + program_run_time.microseconds / (10**6)), "seconds")
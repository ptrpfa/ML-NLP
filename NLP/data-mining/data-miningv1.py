import pandas as pd
import mysql.connector #MySQL
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

# Global variables
feedback_table = "Feedback" # Name of feedback table in database
feedback_file_path = "/home/p/Desktop/csitml/NLP/data-mining/data/feedback.csv" # Dataset file path
clean_file_path = '/home/p/Desktop/csitml/NLP/data-mining/data/clean-feedback.csv' # Cleaned dataset file path
pickles_file_path = "/home/p/Desktop/csitml/NLP/data-mining/pickles/" # File path containing pickled objects
accuracy_file_path = "/home/p/Desktop/csitml/NLP/data-mining/accuracies/" # Model accuracy results file path
preliminary_check = True # Boolean to trigger display of preliminary dataset visualisations and presentations

# Database global variables
mysql_user = "root"         # MySQL username
mysql_password = "csitroot" # MySQL password
mysql_host = "localhost"    # MySQL host
mysql_schema = "csitDB"     # MySQL schema (NOTE: MySQL in Windows is case-insensitive)

# Program starts here
program_start_time = datetime.datetime.now ()
print ("Start time: ", program_start_time)

# 1) Get dataset
try:

    # Create MySQL connection object to the database
    db_connection = mysql.connector.connect (host = mysql_host, user = mysql_user, password = mysql_password, database = mysql_schema)

    # Create SQL query to get Feedback table values
    sql_query = "SELECT * FROM %s" % (feedback_table)

    # Execute query and convert Feedback table into a pandas DataFrame
    feedback_df = pd.read_sql (sql_query, db_connection)
    # feedback_df = pd.read_sql (sql_query, db_connection, index_col = "FeedbackID")

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
    print ("\n***Preliminary information about dataset***\n")
    print ("Dimensions: ", feedback_df.shape, "\n")
    print ("First few records:")
    print (feedback_df.head (), "\n")
    print ("Columns and data types:")
    print (feedback_df.dtypes, "\n")

# 3) Data pre-processing
# Create custom identifier (WebAppID_FeedbackID_CategoryID)

"""
Feedback features:
-ID (WebAppID + FeedbackID + CategoryID)
-Subject
-Overall score
-Main text
"""

# Clean data..



# Program end time
program_end_time = datetime.datetime.now ()
program_run_time = program_end_time - program_start_time

print ("\nProgram start time: ", program_start_time)
print ("Program end time: ", program_end_time)
print ("Program runtime: ", (program_run_time.seconds + program_run_time.microseconds / (10**6)), "seconds")
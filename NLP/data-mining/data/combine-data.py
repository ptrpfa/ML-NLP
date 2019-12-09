# Program to combine the data from each of the four datasets used from Hamburg University and Mendeley Data respectively
# Datasets chosen due to credibility of data source as well as QUALITY of datasets
import pandas as pd
import mysql.connector # MySQL
from sqlalchemy import create_engine # MySQL
import datetime

# Pre-requisite:
# MySQL Setting:
# Requires SQL Setting: SET @@global.sql_mode = ''; # To allow insertion of combined dataset into the database

# Global variables
combined_df = pd.DataFrame (columns = ["WebAppID", "CategoryID", "Subject", "MainText", "Rating", "Remarks"]) # Initialise DataFrame to contain combined feedback data
combined_file_path = '/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/clean/combined.csv' # Cleaned dataset file path
clean_file_path = '/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/clean/' # File path for folder containing formatted and cleaned datasets
dataset_webapp_id = 99 # Default WebAppID for datasets obtained from Hamburg University and Mendeley Data
execute_insert = False # Boolean to trigger insertion of combined dataset to the database 

# Datasets
pan_file_path = "/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/Pan_Dataset.xlsx" # Dataset file path
maalej_file_path = "/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/Maalej_Dataset.xlsx" # Dataset file path
rej_file_path = "/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/Unused/REJ/categorised/" # Additional Dataset file path [NOT USED]
bfj_file_path = "/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/Unused/BFJ/Re2015_Training_Set.sql" # Additional Dataset file path [NOT USED]

# Database global variables
mysql_user = "root"         # MySQL username
mysql_password = "csitroot" # MySQL password
mysql_host = "localhost"    # MySQL host
mysql_schema = "csitDB"     # MySQL schema (NOTE: MySQL in Windows is case-insensitive)
feedback_table = "Feedback" # Name of feedback table in database

# Program starts here
program_start_time = datetime.datetime.now ()
print ("Start time: ", program_start_time)

# 1) Processing for Pan Dataset
"""
Pan Dataset:
1390 reviews
Bug Report/Problem Discover (494) [2]
Feature Request (192) [5]
Information Giving (603) (General) [4]
Information Seeking (101) (General) [4]

Will classify the four categories into the three general Feedback categories of:
-Bug (ID: 2)
-Feature Request (ID: 5)
-General (ID: 4)
"""

# Get DataFrame object of the dataset
pan_df = pd.read_excel (pan_file_path)

# Feature engineer Pan dataset
pan_df.id = ["P_" + str (id_no) for id_no in list (pan_df.id)] # Append dataset identifier to ID
pan_df ['Remarks'] = pan_df.id # Fill remarks with the custom dataset identifier of each feedback 
pan_df = pan_df.drop (['id'], axis = 1) # Drop unused ID column

pan_df ['Subject'] = [cname for cname in pan_df.class_name] # Since Pan dataset does not have any Subject values, use its previous class name as the subject
pan_df ['Rating'] = 3 # New column for rating (Default value is set to 3)
pan_df ['WebAppID'] = dataset_webapp_id # New column for WebAppID (Default value is set to 99)

# Re-label categories to respective IDs and change categories to the generalised categories
pan_df.class_name = pan_df.class_name.map ({'feature request': 5, 'information giving': 4, 'information seeking': 4, 'problem discovery': 2}) 

# Rename columns
pan_df = pan_df.rename (columns = {"class_name": "CategoryID", "review": "MainText"})

# Rearrange DataFrame
pan_df = pan_df [["WebAppID", "CategoryID", "Subject", "MainText", "Rating", "Remarks"]]

# Print information about dataset
print ("\n***Pan Dataset***")
print ("Dimensions: ", pan_df.shape)
print (pan_df.head ())
print ("\nColumns and data types:")
print (pan_df.dtypes, "\n")

# Save formatted Pan dataset to CSV
pan_df.to_csv (clean_file_path + "pan.csv", index = False, encoding = "utf-8")

# Append dataset to combined DataFrame
combined_df = combined_df.append (pan_df, ignore_index = True)

# 2) Processing for Maalej Dataset
"""
Maalej Dataset:
3689 reviews
Bug Report/Problem Discovery (370)
Feature Request (252)
Rating (2461) (General)
User Experience (607) (General)

Will classify the four categories into the three general Feedback categories of:
-Bug (ID: 2)
-Feature Request (ID: 5)
-General (ID: 4)
"""

# Get DataFrame object of the dataset
maalej_df = pd.read_excel (maalej_file_path)

# Feature engineer Maalej dataset
# Drop unused columns
maalej_df = maalej_df.drop (['past', 'stopwords_removal', 'reviewer', 'id', 'stemmed', 
                             'fee', 'future', 'lemmatized_comment', 'sentiScore', 'sentiScore_neg', 
                             'reviewId', 'stopwords_removal_nltk', 'present_simple', 'date', 
                             'sentiScore_pos', 'present_con', 'length_words', 'stopwords_removal_lemmatization', 
                             'Exclude'], axis = 1)

maalej_df.id_num = ["M_" + str (id_no) for id_no in list (maalej_df.id_num)] # Append dataset identifier to ID
maalej_df ['Remarks'] = maalej_df.id_num # Fill remarks with the custom dataset identifier of each feedback 
maalej_df = maalej_df.drop (['id_num'], axis = 1) # Drop unused ID column

maalej_df.appId.fillna ("Not stated", inplace = True) # Fill null values
maalej_df.Remarks = maalej_df ['Remarks'] + " AppID: " + maalej_df ['appId'] + " Src: " + maalej_df ['dataSource'] # Append appID and dataSource to Remarks
maalej_df = maalej_df.drop (['appId', 'dataSource'], axis = 1) # Drop unused appId and dataSource columns

maalej_df.task = maalej_df.task.map ({'FR': 'feature request', 'RT': "rating", 'UE': 'user experience', 'PD': 'problem discovery'}) # Re-label categories
maalej_df.title.fillna (maalej_df ['task'], inplace = True) # Fill null values in Title with the Category (task)
maalej_df ['WebAppID'] = dataset_webapp_id # New column for WebAppID (Default value is set to 99)

# Re-label categories to respective IDs and change categories to the generalised categories
maalej_df.task = maalej_df.task.map ({'feature request': 5, 'rating': 4, 'user experience': 4, 'problem discovery': 2})

# Rename columns
maalej_df = maalej_df.rename (columns = {"task": "CategoryID", "review": "MainText", "title": "Subject", "rating": "Rating"})

# Rearrange DataFrame
maalej_df = maalej_df [["WebAppID", "CategoryID", "Subject", "MainText", "Rating", "Remarks"]]

# Print information about dataset
print ("\n***Maalej Dataset***")
print ("Dimensions: ", maalej_df.shape)
print (maalej_df.head ())
print ("\nColumns and data types:")
print (maalej_df.dtypes, "\n")

# Save formatted Maalej dataset to CSV
maalej_df.to_csv (clean_file_path + "maalej.csv", index = False, encoding = "utf-8")

# Append dataset to combined DataFrame
combined_df = combined_df.append (maalej_df, ignore_index = True)


# Additional: 3) Processing for REJ Dataset & 4) Processing for BFJ Dataset
"""
Bug Report
Feature Request
Rating (General)
User Experience (General)

Will classify the four categories into the three general Feedback categories of:
-Bug (ID: 2)
-Feature Request (ID: 5)
-General (ID: 4)
"""
"""
# Get DataFrame object of the dataset
rej_df = pd.DataFrame (columns = ["WebAppID", "CategoryID", "Subject", "MainText", "Rating", "Remarks"]) # Initialise REJ Dataset dataframe

# Note: Not using all.json of REJ dataset as some records are missing category classifications
rej_df_bug = pd.read_json (rej_file_path + "Bug.json", orient = "records")
rej_df_feature = pd.read_json (rej_file_path + "Feature.json", orient = "records")
rej_df_rating = pd.read_json (rej_file_path + "Rating.json", orient = "records")
rej_df_userexp = pd.read_json (rej_file_path + "UserExperience.json", orient = "records")

# # Print information about dataset
print ("\n***REJ Bug Dataset***")
print ("Dimensions: ", rej_df_bug.shape)
print (rej_df_bug.head ())
print ("\nColumns and data types:")
print (rej_df_bug.dtypes, "\n")

print ("\n***REJ Feature Dataset***")
print ("Dimensions: ", rej_df_feature.shape)
print (rej_df_feature.head ())
print ("\nColumns and data types:")
print (rej_df_feature.dtypes, "\n")

print ("\n***REJ Rating Dataset***")
print ("Dimensions: ", rej_df_rating.shape)
print (rej_df_rating.head ())
print ("\nColumns and data types:")
print (rej_df_rating.dtypes, "\n")

print ("\n***REJ User Experience Dataset***")
print ("Dimensions: ", rej_df_userexp.shape)
print (rej_df_userexp.head ())
print ("\nColumns and data types:")
print (rej_df_userexp.dtypes, "\n")

# Save formatted REJ dataset to CSV
rej_df_bug.to_csv (clean_file_path + "rej_bug.csv", index = False, encoding = "utf-8")
rej_df_feature.to_csv (clean_file_path + "rej_feature.csv", index = False, encoding = "utf-8")
rej_df_rating.to_csv (clean_file_path + "rej_rating.csv", index = False, encoding = "utf-8")
rej_df_userexp.to_csv (clean_file_path + "rej_userexperience.csv", index = False, encoding = "utf-8")
"""

# Remove duplicate feedback reviews (keeps first occurance of duplicated feedback)
combined_df.drop_duplicates (subset = "MainText", keep = "first", inplace = True)

# Print combined dataset information
print ("\n***COMBINED Dataset***")
print ("Dimensions: ", combined_df.shape)
print (combined_df.head ())
print ("\nColumns and data types:")
print (combined_df.dtypes, "\n")

# Export combined dataframe
combined_df.to_csv (combined_file_path, index = False, encoding = "utf-8") 

# Check boolean to see whether to execute insertion of combined dataset into the database
if (execute_insert ==  True): # Create database connection and insert combined dataset to database if boolean set to True

    # Create SQLAlchemy engine object [mysql://user:password@host/database]
    db_engine = create_engine ("mysql://{user}:{password}@{host}/{schema}".format (user = mysql_user, password = mysql_password, host = mysql_host, schema = mysql_schema)) 
    db_connection = db_engine.connect () # Establish a connection to the database

    # Insert combined dataframe to the database (Requires SQL Setting: SET @@global.sql_mode = '';)
    combined_df.to_sql (name = feedback_table, con = db_connection, if_exists = "append", index = False, chunksize = 1000) # Insert 1000 rows into database at a time

    # Close connection object once Feedback has been inserted
    db_connection.close () # Close MySQL connection

    # Print debugging statement
    print (len (combined_df), "records inserted into %s.%s" % (mysql_schema, feedback_table))

# Program end time
program_end_time = datetime.datetime.now ()
program_run_time = program_end_time - program_start_time

print ("\nProgram start time: ", program_start_time)
print ("Program end time: ", program_end_time)
print ("Program runtime: ", (program_run_time.seconds + program_run_time.microseconds / (10**6)), "seconds")
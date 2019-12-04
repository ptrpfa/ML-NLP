# Program to combine the data from each of the four datasets used from Hamburg University and Mendeley Data respectively

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

# Global variables
combined_df = pd.DataFrame (columns = ["WebAppID", "CategoryID", "Subject", "MainText", "Rating", "Remarks"]) # Initialise DataFrame to contain combined feedback data
combined_file_path = '/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/clean/combined.csv' # Cleaned dataset file path
clean_file_path = '/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/clean/' # File path for folder containing formatted and cleaned datasets

# Datasets
pan_file_path = "/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/Pan_Dataset.xlsx" # Dataset file path
maalej_file_path = "/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/Maalej_Dataset.xlsx" # Dataset file path
rej_file_path = "/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/REJ/all.json" # Dataset file path
bfj_file_path = "/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/BFJ/Re2015_Training_Set.sql" # Dataset file path

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
pan_df ['WebAppID'] = 99 # New column for WebAppID (Default value is set to 99)

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
3691 reviews
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

# Re-label categories
maalej_df.task = maalej_df.task.map ({'FR': 'feature request', 'RT': "rating", 'UE': 'user experience', 'PD': 'problem discovery'}) 
maalej_df.title.fillna (maalej_df ['task'], inplace = True) # Fill null values

# Print information about dataset
print ("\n***Maalej Dataset***")
print ("Dimensions: ", maalej_df.shape)
print (maalej_df.head ())
print ("\nColumns and data types:")
print (maalej_df.dtypes, "\n")
print (maalej_df.Remarks)

# Save formatted Pan dataset to CSV
maalej_df.to_csv (clean_file_path + "maalej.csv", index = False, encoding = "utf-8")



# Print combined dataset information
# print ("\n***COMBINED Dataset***")
# print ("Dimensions: ", combined_df.shape)
# print (combined_df.head ())
# print ("\nColumns and data types:")
# print (combined_df.dtypes, "\n")

# Export combined dataframe
combined_df.to_csv (combined_file_path, index = False, encoding = "utf-8")

# Program end time
program_end_time = datetime.datetime.now ()
program_run_time = program_end_time - program_start_time

print ("\nProgram start time: ", program_start_time)
print ("Program end time: ", program_end_time)
print ("Program runtime: ", (program_run_time.seconds + program_run_time.microseconds / (10**6)), "seconds")
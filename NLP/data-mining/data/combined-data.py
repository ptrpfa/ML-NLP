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
combined_df = pd.DataFrame () # Initialise DataFrame to contain combined feedback data
file_path = '/home/p/Desktop/csitml/NLP/data-mining/data/combined.csv' # Combined dataset file path
combined_clean_file_path = '/home/p/Desktop/csitml/NLP/data-mining/data/clean-combined.csv' # Cleaned dataset file path
clean_file_path = '/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/clean/' # File path for folder containing formatted and cleaned datasets

# Datasets
pan_file_path = "/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/Pan_Dataset.xlsx" # Dataset file path
maleej_file_path = "/home/p/Desktop/csitml/NLP/data-mining/data/Datasets/Maalej_Dataset.xlsx" # Dataset file path
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
pan_df = pd.read_excel (pan_file_path, index_col = "id")

# Re-label categories to respective IDs and change categories to the generalised categories
pan_df.class_name = pan_df.class_name.map ({'feature request': 5, 'information giving': 4, 'information seeking': 4, 'problem discovery': 2})
pan_df ['rating'] = 3 # New column for rating (Default value is set to 3)

# Save formatted dataset to CSV
pan_df.to_csv (clean_file_path + "pan.csv", index = True, encoding="utf-8")

print ("\n***Preliminary information about dataset***")
print ("Dimensions: ", pan_df.shape)
print (pan_df.head ())
print ("\nColumns and data types:")
print (pan_df.dtypes, "\n")

# Program end time
program_end_time = datetime.datetime.now ()
program_run_time = program_end_time - program_start_time

print ("\nProgram start time: ", program_start_time)
print ("Program end time: ", program_end_time)
print ("Program runtime: ", (program_run_time.seconds + program_run_time.microseconds / (10**6)), "seconds")
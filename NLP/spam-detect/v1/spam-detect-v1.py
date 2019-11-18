import pandas as pd
import numpy as np
import re # REGEX
import string
import html
import pickle 
from sklearn.model_selection import train_test_split # 4) For splitting dataset into train/test sets
from sklearn import linear_model # 4) Linear Regression classifier
from sklearn.naive_bayes import GaussianNB # 4) Naive Bayes classifier
from sklearn.ensemble import RandomForestClassifier # 4) Random Forest classifier
from sklearn import svm # 4) SVM classifier
from sklearn.linear_model import LogisticRegression # 4) Logistic Regression classifier
from sklearn.metrics import accuracy_score # 4) Accuracy scorer
from sklearn.model_selection import cross_val_score # 4) Cross validation scorer
from sklearn.model_selection import GridSearchCV # 4) For model hyperparameters tuning
from nltk import stem # NLP
from nltk.corpus import stopwords # NLP
from sklearn.feature_extraction.text import TfidfVectorizer # NLP Vectorizer
import seaborn as sn # For visualisation
import matplotlib.pyplot as plt # For visualisation

# Suppress scikit-learn FutureWarnings
from warnings import simplefilter
simplefilter (action = 'ignore', category = FutureWarning) # Ignore Future Warnings

"""CAN CONSIDER MAKING A SEPARATE FUNCTION FOR CLEANING TEST DATA AS THIS CLEANSING IS SPECIFIC TO TRAINING DATA!"""
# Function to clean strings (accepts a sequence-typed variable containing dirty strings and returns a list of the cleaned strings)
def clean_string (sequence):

    # Initialise list containing cleaned strings
    list_cleaned_strings = []

    # REGEX for CSIT's custom error code
    bugcoderegex = "" # Still WIP currently [Assume EC is the first string split by space ie '00001 Error occurred' [for subject]]

    # Initialise list containing white-listed strings [NOTE: SHOULD BE IN LOWERCASE]
    whitelist = ['csit', 'mindef', 'cve', 'cyber-tech', 
                'software engineering & analytics', 'comms-tech', 
                'systems & network infrastructure', 'crypto-tech']

    # Loop to clean strings in the sequence object
    for item in sequence:
        
        # Decode HTML encoded characters (&amp; -> &)
        item = html.unescape (item)

        # Change text to lowercase
        item = item.lower ()

        # Apply further cleansing if string is not whitelisted
        if (item not in whitelist):          

            #  Remove punctuations from text
            #item = re.sub('[%s]' % re.escape(string.punctuation), '', item)

            # Remove any non-word characters from the item
            item = re.sub (r"[^a-zA-Z0-9 ]", "", item)

            # Replace multiple consecutive spaces with a single space
            item = re.sub (r"[ ]{2,}", " ", item)

            # Remove heading and trailing whitespaces
            item = item.strip ()

        # Append cleaned string into the list of cleaned strings
        list_cleaned_strings.append (item)

    # Return list of cleaned strings
    return list_cleaned_strings

# Function to pickle object (accepts object to pickle and its filename to save as)
def pickle_object (pickle_object, filename):

    # Get full filepath
    filepath = pickles_file_path + filename

    # Create file object to store object to pickle
    file_pickle = open (filepath, 'ab') # a = append, b = bytes

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

# Function to create spam detection model
def create_spam_detection_model (features, target):

    # 4) Create spam-detection classfiers to filter out spam
    svm_classifier = svm. SVC (C=10, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=1, gamma=0.1, kernel='rbf',
        max_iter=-1, probability=False, random_state=1, shrinking=True, tol=0.001,
        verbose=False)

    logistic_regression_classifier = LogisticRegression(C=1000, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=1, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)

    # GridSearchCV to to tune hyperparameters of models
    """
    # Dictionary to store parameters and parameter values to test
    parameter_grid = {
                    'C': [0.001, 0.01, 0.1, 1, 10,1000],
                    'degree': [1, 3, 5, 10],
                    'gamma': [0.001, 0.01, 0.1, 1],
                    'random_state': [1, 5, 10, 50, 55, 70, 100, 123]
                    }

    # Create Grid Search object
    grid_search = GridSearchCV (estimator = svm_classifier, param_grid = parameter_grid, 
                    scoring = "accuracy", n_jobs = 4, iid = False, cv = 10, verbose = 1)

    # Fit features and target variable to grid search object
    grid_search.fit (features, target)

    # Get fine-tuned details
    print ("Best score: ", grid_search.best_score_)
    print ("Best parameters: ", grid_search.best_params_)
    print ("Best estimator: ", grid_search.best_estimator_)

    # SVM:
    # Best score:  0.981328511060382
    # Best parameters:  {'C': 10, 'degree': 1, 'gamma': 0.1, 'random_state': 1}
    # Best estimator:  SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape='ovr', degree=1, gamma=0.1, kernel='rbf',
    #     max_iter=-1, probability=False, random_state=1, shrinking=True, tol=0.001,
    #     verbose=False)
    """
    """
    # Dictionary to store parameters and parameter values to test
    parameter_grid = {
                    'C': [0.001, 0.01, 0.1, 1, 10,1000],
                    'penalty': ['l1', 'l2'],
                    'random_state': [1, 5, 10, 50, 55, 70, 100, 123]
                    }

    # Create Grid Search object
    grid_search = GridSearchCV (estimator = logistic_regression_classifier, param_grid = parameter_grid, 
                    scoring = "accuracy", n_jobs = 4, iid = False, cv = 10, verbose = 1)

    # Fit features and target variable to grid search object
    grid_search.fit (features, target)

    # Get fine-tuned details
    print ("LR:")
    print ("Best score: ", grid_search.best_score_)
    print ("Best parameters: ", grid_search.best_params_)
    print ("Best estimator: ", grid_search.best_estimator_)
    """

    # Get list of accuracies
    print ("***Model accuracies***")

    """ SVM """
    print ("SVM:")

    # Using cross validation
    list_cross_val_score = cross_val_score (svm_classifier, features, target, cv = 5, scoring = 'accuracy')

    print ("Cross-validation:")
    print ("List of scores: ", list_cross_val_score)
    print ("Mean score: ", np.mean (list_cross_val_score), "\n")

    # Using train-test-split
    print ("Train-test-split:")
    x_train, x_test, y_train, y_test_result = train_test_split (features, target, test_size = 0.3, random_state = 123, stratify = target)
    svm_model = svm_classifier.fit (x_train, y_train) # Fit model with training data
    y_test_svm = svm_model.predict (x_test) # Store predicted results of model

    # Accuracy against training data
    print ("Training data accuracy: ", svm_model.score (x_train, y_train)) # Result is over train data, not test data
    print ("Testing data accuracy: ", accuracy_score (y_test_result, y_test_svm)) # Testing data accuracy
    
    """ Logistic Regression """
    print ("\nLogistic Regression:")

    # Using cross validation
    list_cross_val_score = cross_val_score (logistic_regression_classifier, features, target, cv = 5, scoring = 'accuracy')

    print ("Cross-validation:")
    print ("List of scores: ", list_cross_val_score)
    print ("Mean score: ", np.mean (list_cross_val_score), "\n")

    # Using train-test-split
    print ("Train-test-split:")
    x_train, x_test, y_train, y_test_result = train_test_split (features, target, test_size = 0.3, random_state = 123, stratify = target)
    logistic_regression_model = logistic_regression_classifier.fit (x_train, y_train) # Fit model with training data
    y_test_logistic_regression = logistic_regression_model.predict (x_test) # Store predicted results of model

    # Accuracy against training data
    print ("Training data accuracy: ", logistic_regression_model.score (x_train, y_train)) # Result is over train data, not test data
    print ("Testing data accuracy: ", accuracy_score (y_test_result, y_test_logistic_regression)) # Testing data accuracy

    # Save models (pickling/serialization)
    pickle_object (svm_model, "svm-model.pkl")
    pickle_object (logistic_regression_model, "logistic-regression-model.pkl")


    # Plot confusion matrix
    


    plt.show ()

# Global variables
train_file_path = "/home/p/Desktop/csitml/NLP/spam-detect/v1/data/spam-ham.txt" # Dataset file path
clean_file_path = '/home/p/Desktop/csitml/NLP/spam-detect/v1/data/clean-spam-ham.csv' # Cleaned dataset file path
pickles_file_path = "/home/p/Desktop/csitml/NLP/spam-detect/v1/pickles/" # File path containing pickled objects
preliminary_check = False # Boolean to trigger display of preliminary dataset visualisations and presentations
message_check = False # Boolean to trigger prompt for user message to check whether it is spam or not

# 1) Get data
train_data = pd.read_csv (train_file_path, sep = "\t", encoding = 'utf-8')
# print (train_data)

# 2) Understand dataset
if (preliminary_check == True): # Check boolean to display preliminary information

    # Print some information of about the data
    print ("***Preliminary information about dataset***")
    print ("Dimensions: ", train_data.shape, "\n")
    print (train_data.head ())
    print ("Columns and data types:")
    print (train_data.dtypes, "\n")

# 3) Data pre-processing
train_data.text = clean_string (train_data.text) # Clean text

# Change spam/ham label to numeric values
train_data.label = train_data.label.map ({'spam': 1, 'ham': 0})

# Drop empty rows/columns
train_data.dropna (how = "all", inplace = True) # Drop empty rows
train_data.dropna (how = "all", axis = 1, inplace = True) # Drop empty columns

# Remove with rows containing empty texts
train_data = train_data [train_data.text != ""]

# Save cleaned dataset to CSV
train_data.to_csv (clean_file_path, index = False, encoding="utf-8")

# Assign target and features variables
target = train_data.label
features = train_data.text

# Create DTM of dataset (features)
vectorizer = TfidfVectorizer () # Create vectorizer object

# Fit data to vectorizer
features = vectorizer.fit_transform(features) # Returns a matrix
# print (features ,type (features))
# print (vectorizer.get_feature_names()) # Get features (words)
# data_dtm = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names()) # Convert DTM to DataFrame
# data_dtm.to_csv ("/home/p/Desktop/csitml/NLP/spam-detect/v1/data/dtm.csv", index = False, encoding="utf-8") # Save DTM
pickle_object (vectorizer, "tfid-vectorizer.pkl")

# 4) Load models
create_spam_detection_model (features, target) # Create spam-detection classfier to filter out spam

#svm_model = load_pickle ("svm-model.pkl")

# # JUST TO CHECK
# print ("JUST TO CHECK!")
# x_train, x_test, y_train, y_test_result = train_test_split (features, target, test_size = 0.3, random_state = 123, stratify = target)
# y_test_svm = svm_model.predict (x_test) # Store predicted results of model

# # Accuracy against training data
# print ("Training data accuracy: ", svm_model.score (x_train, y_train)) # Result is over train data, not test data
# print ("Testing data accuracy: ", accuracy_score (y_test_result, y_test_svm), "\n") # Testing data accuracy

# Get input from user and check if it is spam
if (message_check == True):

    # Get input from user and check if it is spam or not
    input_string = input ("Enter message to check: ")

    # Clean string
    # Decode HTML encoded characters (&amp; -> &)
    input_string = html.unescape (input_string)

    # Change text to lowercase
    input_string = input_string.lower ()

    # Remove any non-word characters from the input_string
    input_string = re.sub (r"[^a-zA-Z0-9 ]", "", input_string)

    # Replace multiple consecutive spaces with a single space
    input_string = re.sub (r"[ ]{2,}", " ", input_string)

    # Remove heading and trailing whitespaces
    input_string = input_string.strip ()

    # Create a DataFrame for the input string
    default_values = {'label': 1, 'text': input_string} # Default value of label is SPAM
    df_input_string = pd.DataFrame (default_values, index=[0])

    # Drop empty rows/columns
    df_input_string.dropna (how = "all", inplace = True) # Drop empty rows
    df_input_string.dropna (how = "all", axis = 1, inplace = True) # Drop empty columns

    # Remove with rows containing empty texts
    df_input_string = df_input_string [df_input_string.text != ""]

    # Assign features
    features = df_input_string.text
    target = df_input_string.label

    # Create DTM of dataset (features)
    # Fit data to vectorizer
    features = vectorizer.transform (features) # Not fit_transform!

    # Predict message is a spam or not
    y_test_predict = svm_model.predict (features) # Store predicted results of model
    # OR 
    # y_test_predict = logistic_regression_model.predict (features) # Store predicted results of model

    print (y_test_predict)

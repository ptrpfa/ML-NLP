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

"""
NOTE: The Spam-detection models are CONSERVATIVE
Models tend to classify spam messages as not spam
-> However, it is better to be more conservative as we would rather mislabel spam as not spam rather than have
   genuine messages be labeled as spam, which would result in the lost of valuable feedback
-> Model is better at identifying true negative (ham) over true postive (spam) [which is better for a conservative approach]

Since the accuracy of the model is quite high, we will not go into further enhancements on the spam-detection model 
such as applying Boosting (due to time-constraint and lack of practicality as the model is the first pass/layer 
in the data mining process)

Improvements that could be done:
-Better and more dataset added
-Further model refinements like boosting..
-Can implement another 'layer' for a feedback deception model (detection of false feedback)

Based on results, concluded that SVM model is the best performing model"
-In terms of F1 score (useful for binary classfication of SPAM or HAM (not spam) and unbalanced data distribution)
-In terms of accuracy, ROC, Recall, Precision..

Miscellaneous:
spaCy is used over NLTK for things such as tokenization and lemmatisation due to it being a faster/more efficient library

"""

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

        # Change text to lowercase
        document = document.lower ()

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

            # Check if a match object is obtained (may have mismatches ie "awww")
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
        
        #  Remove punctuations from text
        # document = re.sub('[%s]' % re.escape(string.punctuation), '', document)

        # Remove any non-word characters from the document (Used over string.punctuation as provides more granular control)
        document = re.sub (r"[^a-zA-Z0-9 ]", "", document) # Apostrophe not included as will result in weird tokenizations (for words like I'll, She's..)

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

        # Append cleaned document into the list of cleaned documents
        list_cleaned_documents.append (document)

        # For debugging
        # print ("document: ",document)

    # Return list of cleaned documents
    return list_cleaned_documents

# Function to tokenize documents
def tokenize (document):

    # Convert document into a spaCy tokens document
    document = nlp (document)

    # Initialise list to contain tokens
    list_tokens = []

    # Loop to tokenize text
    for token in document:

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

        # Check if lemmatised token is already in the list of tokens (OMITTED as may affect the TF-IDF score of terms)
        # if (lemmatised not in list_tokens):

        #     # Add new lemmatised token into the list of tokens if it is not inside
        #     list_tokens.append (lemmatised)

        # Add lemmatised token into list of tokens
        list_tokens.append (lemmatised)
    
    # Return list of tokens to calling program
    return (list_tokens)
    
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

# Function to plot confusion matrix of model results
def plot_confusion_matrix (y_test, y_pred, classes, title, filename):

    # Create confusion matrix
    cm = confusion_matrix (y_test, y_pred)

    # Colour map
    cmap = plt.cm.Blues

    # OR (Random colour mapping)
    # cmap = matplotlib.colors.ListedColormap (np.random.rand (256,3))

    # Confusion matrix variable
    classes = classes [unique_labels (y_test, y_pred)]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title= title,
        ylabel='True label',
        xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # Display confusion matrix
    # plt.show ()

    # Save confusion matrix
    plt.savefig (accuracy_file_path + filename)

# Function to plot ROC curves of models
def plot_roc_curve (dictionary): 
    
    # Receives dictionary in the format: 
    # dictionary = {"model-name": [false_positive_rate, true_positive_rate, roc_auc, color]}

    plt.title ("ROC Curves of Models")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # For loop to access each model in dictionary
    for model in dictionary:

        false_positive_rate = dictionary [model] [0]
        true_positive_rate = dictionary [model] [1]
        roc_auc = dictionary [model] [2]
        random_color = dictionary [model] [3] # OR random_color = "#" + str (random.randint (0,999999)) 

        plt.plot (false_positive_rate, true_positive_rate, random_color, label = "%s, AUC=%f" % (model, roc_auc))
        plt.legend (loc = 'lower right')
    
    # Save ROC Curve figure
    plt.savefig (accuracy_file_path + "roc-curves.png")

# Function to get Classification Accuracies of model
def classification_accuracy (classifier, model, features, target, x_train, y_train, y_test, y_test_result, scoring, title, target_name):

    # Print title
    print ("*" * 30) # Separator
    print (title)

    # Cross validation score
    list_cross_val_score = cross_val_score (classifier, features, target, cv = 5, scoring = scoring)

    print ("Cross-validation:")
    print ("List of scores: ", list_cross_val_score)
    print ("Mean score: ", np.mean (list_cross_val_score), "\n")

    # Using train-test-split
    print ("Train-test-split:")

    # Accuracy against training data
    print ("Training data accuracy: ", model.score (x_train, y_train)) # Result is over train data, not test data
    print ("Testing data accuracy: ", accuracy_score (y_test_result, y_test), "\n") # Testing data accuracy

    # Classification report
    print ("Classification Report:")
    print (classification_report (y_test_result, y_test, target_names = target_name), "\n")

    # Precision scores
    print ("Precision Scores: (Ability of classifier not to label as positive a sample that is negative)")
    print ("Macro average: ", metrics.precision_score (y_test_result, y_test, average = 'macro'))  
    print ("Micro average: ", metrics.precision_score (y_test_result, y_test, average = 'micro'))  
    print ("Weighted average: ", metrics.precision_score (y_test_result, y_test, average = 'weighted'), "\n") 

    # Recall scores
    print ("Recall Scores: (Ability of the classifier to find all the positive samples)")
    print ("Macro average: ", metrics.recall_score (y_test_result, y_test, average = 'macro'))  
    print ("Micro average: ", metrics.recall_score (y_test_result, y_test, average = 'micro'))  
    print ("Weighted average: ", metrics.recall_score (y_test_result, y_test, average = 'weighted'), "\n")  

    # F1 Scoring
    print ("F1 Scores:")
    print ("Macro average: ", metrics.f1_score (y_test_result, y_test, average = 'macro'))  
    print ("Micro average: ", metrics.f1_score (y_test_result, y_test, average = 'micro'))  
    print ("Weighted average: ", metrics.f1_score (y_test_result, y_test, average = 'weighted'), "\n")  

    # Logarithmic Loss score (closer to 0 better)
    print ("Log Loss: ", metrics.log_loss (y_test_result, y_test), "\n")  

# Global variables
train_file_path = "/home/p/Desktop/csitml/NLP/spam-detect/data/spam-ham.txt" # Dataset file path
clean_file_path = '/home/p/Desktop/csitml/NLP/spam-detect/data/clean-spam-ham.csv' # Cleaned dataset file path
pickles_file_path = "/home/p/Desktop/csitml/NLP/spam-detect/pickles/" # File path containing pickled objects
accuracy_file_path = "/home/p/Desktop/csitml/NLP/spam-detect/accuracies/" # Model accuracy results file path
preliminary_check = False # Boolean to trigger display of preliminary dataset visualisations and presentations
use_pickle = True # Boolean to trigger whether to use pickled objects or not
message_check = False # Boolean to trigger prompt for user message to check whether it is spam or not

# Whitelisting
whitelist = ['csit', 'mindef', 'cve', 'cyber-tech', 'cyber-technology', # Whitelist for identifying non-SPAM feedbacks
            'comms-tech', 'communications-tech', 'comms-technology',
            'communications-technology', 'crypto-tech', 'cryptography-tech',
            'crypto-technology', 'cryptography-technology']
bugcode_regex = r"(.*)(BUG\d{6}\$)(.*)" # Assume bug code is BUGXXXXXX$ ($ is delimiter)


# Global NLP Objects
# Create spaCy NLP object
nlp = spacy.load ("en_core_web_sm")

# Custom list of stop words to add to spaCy's existing stop word list
list_custom_stopwords = ["I", "i",  "yer", "ya", "yar", "u", "loh", "lor", "lah", "leh", "lei", "lar", "liao", "hmm", "hmmm", "mmm", "mmmmmm", "wah", "eh"] 

# Add custom stop words to spaCy's stop word list
for word in list_custom_stopwords:

    # Add custom word to stopword word list
    nlp.vocab [word].is_stop = True

# Program starts here
program_start_time = datetime.datetime.now ()
print ("Start time: ", program_start_time)

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
train_data.text = clean_document (train_data.text) # Clean text

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

# Create new vectorizers if not using pickled objects
if (not use_pickle):
    
    # Create vectorizer object
    vectorizer = TfidfVectorizer (encoding = "utf-8", lowercase = True, strip_accents = 'unicode', stop_words = 'english', tokenizer = tokenize, ngram_range = (1,2), max_df = 0.95) 
    
    # Should try HashingVectorizer combined with TFIDF Transformer
    pass

    # Fit data to vectorizer (Create DTM of dataset (features))
    features = vectorizer.fit_transform (features) # Returns a sparse matrix
    
    # Print information on vectorised words
    # print (features, type (features)) # Sparse matrix
    print ("Tokens:")
    print (vectorizer.get_feature_names()) # Get features (words)
    data_dtm = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names()) # Convert DTM to DataFrame
    data_dtm.to_csv ("/home/p/Desktop/csitml/NLP/spam-detect/data/large/dtm.csv", index = False, encoding="utf-8") # Save DTM

# Using pickled objects
else:

    # Load serialised vectorizer
    vectorizer = load_pickle ("tfidf-vectorizer.pkl")

    # Load serialised features (sparse matrix)
    # Load DTM and convert it into a sparse matrix
    # features_dtm = pd.read_csv ("/home/p/Desktop/csitml/NLP/spam-detect/data/dtm.csv") # Not used as DTM file is very large
    # features = scipy.sparse.csr_matrix (features_dtm.values)

    # OR 

    # Load serialised object
    features = load_pickle ("features.pkl")

    # Print information on vectorised words
    # print (features, type (features)) # Sparse matrix
    print ("Tokens:")
    print (vectorizer.get_feature_names()) # Get features (words)

# 4) # Create spam-detection models to filter out spam
# Create spam-detection classfiers to filter out spam
svm_classifier = svm.SVC (C=1000, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=1, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=True, random_state=1, shrinking=True, tol=0.001,
    verbose=False)

logistic_regression_classifier = LogisticRegression (C=1000, class_weight=None, dual=False, fit_intercept=True,
    intercept_scaling=1, l1_ratio=None, max_iter=100,
    multi_class='warn', n_jobs=None, penalty='l1',
    random_state=1, solver='liblinear', tol=0.0001, verbose=0,
    warm_start=False)

multinomialnb_classifier = MultinomialNB ()

# GridSearchCV to tune hyperparameters of models
"""
# SVM:
# Dictionary to store parameters and parameter values to test
parameter_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10,1000],
                'degree': [1, 3, 5, 10],
                'gamma': [0.001, 0.01, 0.1, 1],
                'random_state': [1, 5, 10, 50, 55, 70, 100, 123]
                }

# Create Grid Search object
grid_search = GridSearchCV (estimator = svm_classifier, param_grid = parameter_grid, 
                scoring = "f1", n_jobs = 4, iid = False, cv = 10, verbose = 1)

# Fit features and target variable to grid search object
grid_search.fit (features, target)

# Get fine-tuned details
print ("Best score: ", grid_search.best_score_)
print ("Best parameters: ", grid_search.best_params_)
print ("Best estimator: ", grid_search.best_estimator_)

# SVM:
# 
# F1 scoring:
# Best score:  0.8973526061818614
# Best parameters:  {'C': 1000, 'degree': 1, 'gamma': 0.001, 'random_state': 1}
# Best estimator:  SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=1, gamma=0.001, kernel='rbf',
#     max_iter=-1, probability=True, random_state=1, shrinking=True, tol=0.001,
#     verbose=False)
"""

"""
# Logistic Regression:
# Dictionary to store parameters and parameter values to test
parameter_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10,1000],
                'penalty': ['l1', 'l2'],
                'random_state': [1, 5, 10, 50, 55, 70, 100, 123]
                }

# Create Grid Search object
grid_search = GridSearchCV (estimator = logistic_regression_classifier, param_grid = parameter_grid, 
                scoring = "f1", n_jobs = 4, iid = False, cv = 10, verbose = 1)

# Fit features and target variable to grid search object
grid_search.fit (features, target)

# Get fine-tuned details
print ("LR:")
print ("Best score: ", grid_search.best_score_)
print ("Best parameters: ", grid_search.best_params_)
print ("Best estimator: ", grid_search.best_estimator_)

# Logistic Regression
# 
# F1 Scoring:
# Best score:  0.9048442354890881
# Best parameters:  {'C': 1000, 'penalty': 'l1', 'random_state': 1}
# Best estimator:  LogisticRegression(C=1000, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=100,
#                    multi_class='warn', n_jobs=None, penalty='l1',
#                    random_state=1, solver='liblinear', tol=0.0001, verbose=0,
#                    warm_start=False)
"""

# Get model performance metrics
print ("*** Model Performance metrics ***", "\n")

""" Classification Accuracy """
print ("--- Classification Accuracies: ---")
print ("(Only effective if have BALANCED DATA aka distribution of categories/classes is EQUAL!)")
print ("Distribution of spam/ham data:")
print (train_data.label.map ({1:'spam', 0:'ham'}).value_counts (normalize = True), "\n")

""" SVM """
# Train-test-split
x_train, x_test, y_train, y_test_result = train_test_split (features, target, test_size = 0.3, random_state = 123, stratify = target)
target_names = ['ham', 'spam'] # For labelling target variable in classification report

# Create new models if not using serialised models
if (not use_pickle):

    # Fit classifiers (models) with training data
    svm_model = svm_classifier.fit (x_train, y_train) 
    logistic_regression_model = logistic_regression_classifier.fit (x_train, y_train) 
    multinomialnb_model = multinomialnb_classifier.fit (x_train.todense (), y_train) # Need to convert sparse matrix (due to vectorizer) to a dense numpy array

# Using pickled objects
else:

    # Load pickled models
    svm_model = load_pickle ("svm-model.pkl") 
    logistic_regression_model = load_pickle ("logistic-regression-model.pkl")
    multinomialnb_model = load_pickle ("naive-bayes-model.pkl")

# Store predicted results of models
y_test_svm = svm_model.predict (x_test) 
y_test_lr = logistic_regression_model.predict (x_test) 
y_test_mnb = multinomialnb_model.predict (x_test.todense ()) # Need to convert sparse matrix (due to vectorizer) to a dense numpy array

# # Store predicted results (in probabilty) of models
y_test_svm_prob = svm_model.predict_proba (x_test) 
y_test_lr_prob = logistic_regression_model.predict_proba (x_test) 
y_test_mnb_prob = multinomialnb_model.predict_proba (x_test.todense ()) # Need to convert sparse matrix (due to vectorizer) to a dense numpy array

# Remove first column of probability of results
y_test_svm_prob = y_test_svm_prob [:, 1]
y_test_lr_prob = y_test_lr_prob [:, 1]
y_test_mnb_prob = y_test_mnb_prob [:, 1]

# Get False Positive Rate and True Positive Rate of models
false_positive_rate_svm, true_positive_rate_svm, threshold_svm = metrics.roc_curve (y_test_result, y_test_svm_prob)
false_positive_rate_lr, true_positive_rate_lr, threshold_lr = metrics.roc_curve (y_test_result, y_test_lr_prob)
false_positive_rate_mnb, true_positive_rate_mnb, threshold_mnb = metrics.roc_curve (y_test_result, y_test_mnb_prob)

# Calculate Area Under Curve of models' ROC curves
svm_roc_auc = metrics.auc (false_positive_rate_svm, true_positive_rate_svm)
lr_roc_auc = metrics.auc (false_positive_rate_lr, true_positive_rate_lr)
mnb_roc_auc = metrics.auc (false_positive_rate_mnb, true_positive_rate_mnb)

# Get classification accuracies of models
# SVM
classification_accuracy (svm_classifier, svm_model, features, target, x_train, y_train,
                         y_test_svm, y_test_result, "f1", "SVM classification accuracy:", target_names)

# Logistic Regression
classification_accuracy (logistic_regression_classifier, logistic_regression_model, features, target, x_train, y_train,
                         y_test_lr, y_test_result, "f1", "Logistic Regression classification accuracy:", target_names)

# Naive Bayes
classification_accuracy (multinomialnb_classifier, multinomialnb_model, features.todense (), target, x_train.todense (), y_train,
                         y_test_mnb, y_test_result, "f1", "Naive Bayes classification accuracy:", target_names)

""" Visualisations of model performance """
# Plot ROC (Receiver Operating Characteristics) Curves
dict_roc = { # Dictionary for passing model values to function
    "SVM": [false_positive_rate_svm, true_positive_rate_svm, svm_roc_auc, 'red'],
    "LR": [false_positive_rate_lr, true_positive_rate_lr, lr_roc_auc, 'blue'],
    "MNB": [false_positive_rate_mnb, true_positive_rate_mnb, mnb_roc_auc, 'green']
    }

# Plot ROC curves
plot_roc_curve (dict_roc)

# Plot confusion matrices
classes = train_data.label.map ({1:'spam', 0:'ham'}).unique () # Classes refer to possible unique values of the target variable

# SVM Confusion Matrix
plot_confusion_matrix (y_test_result, y_test_svm, classes, "SVM Confusion Matrix", "svm-confusion-matrix.png")

# Logistic Regression Confusion Matrix
plot_confusion_matrix (y_test_result, y_test_lr, classes, "Logistic Regression Confusion Matrix", "logistic-regression-confusion-matrix.png")

# Naive Bayes Confusion Matrix
plot_confusion_matrix (y_test_result, y_test_mnb, classes, "Naive Bayes Confusion Matrix", "naive-bayes-confusion-matrix.png")

# Display visualisations
plt.show ()

# Save models (pickling/serialization)
pickle_object (features, "features.pkl") # Sparse Matrix of features
pickle_object (vectorizer, "tfidf-vectorizer.pkl") # TF-IDF Vectorizer
pickle_object (svm_model, "svm-model.pkl") # SVM Model
pickle_object (logistic_regression_model, "logistic-regression-model.pkl") # Logistic Regression Model
pickle_object (multinomialnb_model, "naive-bayes-model.pkl") # Naive Bayes Model


# Get input from user and check if it is spam
if (message_check == True):

    # Get input from user and check if it is spam or not
    input_string = input ("Enter message to check: ")

    # Set initial value of y_test_predict (results of check)
    y_test_predict = 1 # Default value is SPAM [0 = HAM, 1 = SPAM]

    # Initialise check variables
    bugcode_match = False   # By default false
    whitelist_match = False # By default false
    
    # Check for BUGCODE
    match = re.match (bugcode_regex, input_string)

    # Check for BUGCODE matches
    if (match != None):

        # Set bugcode_match to true
        bugcode_match = True

    # Check for WHITELIST matches
    for whitelisted_string in whitelist:

         # Check if whitelisted string is in the input string
        if (whitelisted_string in input_string.lower ()):
            
            # Set whitelist_match to true
            whitelist_match = True

            # Break out of loop
            break

    # Check if a BUGCODE or whitelisted word was detected in the input
    if (bugcode_match == True or whitelist_match == True):

        # Set input_string as HAM immediately if bugcode or whitelisted string is detected
        y_test_predict = 0 

    # Use models to predict SPAM or HAM if no whitelisted item or bugcode was detected
    else:

         # Create a DataFrame for the input string
        default_values = {'label': 1, 'text': input_string} # Default value of label is SPAM
        df_input_string = pd.DataFrame (default_values, index = [0])

        # Data pre-processing
        df_input_string.text = clean_document (df_input_string.text) # Clean text

        # Drop empty rows/columns
        df_input_string.dropna (how = "all", inplace = True) # Drop empty rows
        df_input_string.dropna (how = "all", axis = 1, inplace = True) # Drop empty columns

        # Remove with rows containing empty texts
        df_input_string =  df_input_string [df_input_string.text != ""]

        # Assign target and features variables
        target =  df_input_string.label
        features =  df_input_string.text

        # Fit data to vectorizer [Create DTM of dataset (features)]
        features = vectorizer.transform (features) # Not fit_transform!

        # Predict message is a spam or not
        y_test_predict = svm_model.predict (features) # Store predicted results of model
        # OR 
        # y_test_predict = logistic_regression_model.predict (features) # Store predicted results of model
        # OR 
        # y_test_predict = multinomialnb_model.predict (features) # Store predicted results of model
    
        print ("SVM: ", svm_model.predict (features))
        print ("LR: ", logistic_regression_model.predict (features))
        print ("MNB: ", multinomialnb_model.predict (features))
    
    # Print results
    if (y_test_predict == 1):

        # Spam
        print ("Message was a SPAM! ->", y_test_predict)

    else:

        # Not Spam
        print ("Message was a HAM! ->", y_test_predict)


# Program end time
program_end_time = datetime.datetime.now ()
program_run_time = program_end_time - program_start_time

print ("\nProgram start time: ", program_start_time)
print ("Program end time: ", program_end_time)
print ("Program runtime: ", program_run_time.seconds, "seconds")

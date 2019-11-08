import pandas as pd
import numpy as np
import re # REGEX
import pickle 
from sklearn.model_selection import train_test_split # 4) For splitting dataset into train/test sets
from sklearn import linear_model # 4) Linear Regression classifier
from sklearn.naive_bayes import GaussianNB # 4) Naive Bayes classifier
from sklearn.ensemble import RandomForestClassifier # 4) Random Forest classifier
from sklearn.metrics import accuracy_score # 4) Accuracy scorer
from sklearn.model_selection import cross_val_score # 4) Cross validation scorer
from sklearn.model_selection import GridSearchCV # 4) For model hyperparameters tuning

# Global variables
bugcoderegex = "" # Still WIP currently [Assume EC is the first string split by space ie '00001 Error occurred' [for subject]]

# Create spam-detection model to filter out spam


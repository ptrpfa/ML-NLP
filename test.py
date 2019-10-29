import pandas as pd
import numpy as np
import re # REGEX
import matplotlib.pyplot as plt # 2) For understanding dataset
from pandas.plotting import scatter_matrix # 2) For understanding dataset
from sklearn.preprocessing import StandardScaler # 3) For scaling numerical data (Data pre-processing)
from sklearn.preprocessing import PolynomialFeatures # 4) For converting features to polynomial
from sklearn import linear_model # 4) Linear Regression classifier

''' Functions '''

# Function to clean strings (accepts a sequence-typed variable containing dirty strings and returns a list of the cleaned strings)
def clean_string (sequence):

    # Initialise list containing cleaned strings
    list_cleaned_strings = []

    # Loop to clean strings in the sequence object
    for item in sequence:

        # Convert item into a string
        #item = str (item)

        # Remove heading and trailing whitespaces
        item = item.strip ()

        # Remove any non-word characters from the item
        item = re.sub (r"\W", "", item)

        # Append cleaned string into the list of cleaned strings
        list_cleaned_strings.append (item)

    # Return list of cleaned strings
    return list_cleaned_strings

# Global variables
train_file_path = "/home/p/Desktop/csitml/train.csv"
test_file_path = "/home/p/Desktop/csitml/test.csv"
clean_file_path = '/home/p/Desktop/csitml/clean.csv' # Cleaned dataset file path
list_columns_empty = [] # Empty list to store the names of columns containing null values
preliminary_check = False # Boolean to trigger display of preliminary dataset visualisations and presentations

# 1) Get data
train_data = pd.read_csv (train_file_path, index_col="PassengerId")

# 2) Understand dataset
# Print some information of about the data
print ("***Preliminary information about dataset***")
print ("Dimensions: ", train_data.shape, "\n")
print ("Columns and data types:")
print (train_data.dtypes, "\n")
print ("Columns with empty values: ")

# Loop to print out columns in dataset that contain empty/null values
for column in dict (train_data.count()): 

    if int (dict (train_data.count())[column]) < train_data.shape[0]: # or if (True in pd.isnull (train_data.column)): 

        # Add column name to list containing columns that contain null values
        list_columns_empty.append (column)

        # Print column details
        print ("  ", column, dict (train_data.count())[column])

# Plot/visualise dataset of boolean set to True
if (preliminary_check == True):

    # train_data.hist() # Univariate histogram
    # train_data.plot(kind='density', subplots=True, layout=(3,3), sharex=False) # Univariate density plot
    # train_data.plot(kind = 'box', subplots = True, layout = (3,3), sharex = False,sharey = False) # Univariate box plot
    # scatter_matrix (train_data) # Multivariate scatter matrix

    print ("\nBasic statistics (for numeric columns!): \n", train_data.describe (), "\n")

    # Visualise data

    # Create empty figure object to set size of figures
    fig = plt.figure (figsize = (15,8))

    # Bar chart of survival rates
    plt.subplot2grid ((8,2), (0,0)) # Place chart in a position in a grid (as will be displaying multiple graphs)
    plt.title ("Survival rate") # Title of chart
    plt.xlabel ("Survived")
    plt.ylabel ("Percentage")
    train_data.Survived.value_counts (normalize=True).plot (kind = "bar", alpha = 0.5)

    # Scatter plot of survival rate with regards to Age
    plt.subplot2grid ((8,2), (0,1)) # Place chart in a position in a grid (as will be displaying multiple graphs)
    plt.title ("Survival rate wrt Age") # Title of chart
    plt.xlabel ("Survived")
    plt.ylabel ("Age")
    plt.scatter (train_data.Survived, train_data.Age, alpha = 0.1) # x-axis is Survived, y-axis is Age


    # Bar chart of Passenger class distribution
    plt.subplot2grid ((8,2), (1,0)) # Place chart in a position in a grid (as will be displaying multiple graphs)
    plt.title ("Passenger class distribution") # Title of chart
    plt.xlabel ("Passenger class")
    plt.ylabel ("Distribution")
    train_data.Pclass.value_counts (normalize=True).plot (kind = "bar", alpha = 0.5) # x-axis is Survived, y-axis is Age

    # Bar chart of Passenger Embarked distribution
    plt.subplot2grid ((8,2), (1,1)) # Place chart in a position in a grid (as will be displaying multiple graphs)
    plt.title ("Passenger Embarked distribution") # Title of chart
    plt.xlabel ("Embarked from")
    plt.ylabel ("Distribution")
    train_data.Embarked.value_counts (normalize=True).plot (kind = "bar", alpha = 0.5) # x-axis is Survived, y-axis is Age

    # Kernel density estimation of Passenger class with regards to Age
    plt.subplot2grid ((8,2), (2,0), colspan=2) # colspan to specify that graph will occupy two columns
    plt.title ("Passenger class wrt Age") # Title of chart
    plt.xlabel ("Age")
    plt.ylabel ("Distribution")
    for pclass in train_data.Pclass.unique(): # Loop to draw one line graph per passenger class

        # Get ages for a given class and plot
        train_data.Age [train_data.Pclass == pclass].plot (kind = "kde")
    plt.legend (tuple (train_data.Pclass.unique())) # Legend of each graph (need to be added only after graphs are plotted!)    

    # Kernel density estimation of Survival rate with regards to Age
    plt.subplot2grid ((8,2), (3,0), colspan=2) # colspan to specify that graph will occupy two columns
    plt.title ("Survival rate wrt Age") # Title of chart
    plt.xlabel ("Age")
    plt.ylabel ("Distribution")
    for survival in train_data.Survived.unique(): # Loop to draw one line graph per survival status

        # Get age for a given survival status and plot graph
        train_data.Age [train_data.Survived == survival].plot (kind = "kde")
    plt.legend (tuple (train_data.Survived.unique())) # Legend of each graph (need to be added after graphs are plotted!)    

    # Bar chart of survival rates amongst males
    plt.subplot2grid ((8,2), (4,0)) # Place chart in a position in a grid (as will be displaying multiple graphs)
    plt.title ("Survival distribution of males") # Title of chart
    plt.xlabel ("Survived")
    plt.ylabel ("Percentage")
    train_data.Survived [train_data.Sex == 'male'].value_counts (normalize=True).plot (kind = "bar", alpha = 0.5)

    # Bar chart of survival rates amongst females
    plt.subplot2grid ((8,2), (4,1)) # Place chart in a position in a grid (as will be displaying multiple graphs)
    plt.title ("Survival distribution of females") # Title of chart
    plt.xlabel ("Survived")
    plt.ylabel ("Percentage")
    train_data.Survived [train_data.Sex == 'female'].value_counts (normalize=True).plot (kind = "bar", alpha = 0.5)

    # Bar chart of survival rates amongst males wrt class
    plt.subplot2grid ((8,2), (5,0), colspan=2) # colspan to specify that graph will occupy two columns
    plt.title ("Survival rate wrt Passenger Class (male)") # Title of chart
    plt.xlabel ("Survival rate")
    plt.ylabel ("Distribution")
    for pclass in train_data.Pclass.unique(): # Loop to draw one line graph per survival status

        # Get age for a given survival status and plot graph
        train_data.Survived [(train_data.Pclass == pclass) & (train_data.Sex == 'male')].plot (kind = "kde")
    plt.legend (tuple (train_data.Pclass.unique())) # Legend of each graph (need to be added after graphs are plotted!)    

    # Bar chart of survival rates amongst females wrt class
    plt.subplot2grid ((8,2), (6,0), colspan=2) # colspan to specify that graph will occupy two columns
    plt.title ("Survival rate wrt Passenger Class (female)") # Title of chart
    plt.xlabel ("Survival rate")
    plt.ylabel ("Distribution")
    for pclass in train_data.Pclass.unique(): # Loop to draw one line graph per survival status

        # Get age for a given survival status and plot graph
        train_data.Survived [(train_data.Pclass == pclass) & (train_data.Sex == 'female')].plot (kind = "kde")
    plt.legend (tuple (train_data.Pclass.unique())) # Legend of each graph (need to be added after graphs are plotted!)   

    # Display visualisations (will open up a new window and pause program flow) 
    plt.show ()

# 3) Data pre-processing

# Selected features:
# Survived      int64
# Pclass        int64
# Sex          object
# Age         float64
# SibSp         int64
# Parch         int64
# Fare        float64
# Embarked     object

# During data pre-processing, should code such that any missing value/abnormal valued data will be normalised (cater to any situation of poor data quality)
# -> DP should be flexible enough to cater to any data field but also apply column-specific processings (ie Male and Female to 1/0..)

# Note that ORDER in data pre-processing is important!

''' Empty values processing '''
# Drop empty rows and columns
train_data.dropna (how = "all", inplace = True) # Drop empty rows
train_data.dropna (how = "all", axis = 1, inplace = True) # Drop empty columns

# Deal with columns containing null/empty values
if (len(list_columns_empty) > 0): # Check if list containing empty columns has anything inside

    # For the given Titanic dataset, the columns with empty values are: Age, Cabin & Embarked

    # Loop to access each column
    for column in list_columns_empty:

        # Get datatype of column
        datatype = train_data [column].dtype
        
        # Check datatype of column
        if (datatype ==  np.float64 or datatype ==  np.int64):

            # Get average/median value of numeric column (after removing null values)
            mean = train_data [column].dropna ().mean ()
            median = train_data [column].dropna ().median ()

            # Convert mean and median to 1dp
            mean = float("{0:.1f}".format(mean))
            median = float("{0:.1f}".format(median))

            # Replace empty values in column with the mean/median
            train_data [column].fillna (mean, inplace = True) # OR train_data.fillna (median)

        # Deal with object-type/string datatypes    
        else:    

            # For the filling of string/object-typed data, depends a lot on common sense
            # 1) We can either drop the data (if there are a lot of missing values ie cabin in this case)
            # 2) We can try to reassign the data to something else ie use the mode (this method has the limitations of biasness, in cases where the column has a lot of
            # missing values like cabin in this case)
            # Advanced processing can include NLP, viewing the correlations btwn existing fields.. before assigning the value
            # Will not really expand too much on this as quite intuitive and depends on a case by case basis
            
            # Get mode of string value
            mode_string = train_data [column].mode () [0]

            # Fill empty values in column with the most recurring value
            train_data [column].fillna (mode_string, inplace = True)

# Drop unused/unselected features
train_data = train_data.drop (['Ticket', 'Cabin', 'Name'], axis = 1)

''' Numeric processing '''
# Scale Age and Fare
scaled_columns = ['Age', 'Fare']
scaler = StandardScaler () 

# Loop to scale columns in the scaled_columns list
for column in scaled_columns:

    # Scale column
    scaled = scaler.fit_transform (train_data [[column]].values)

    # Update dataset with scaled values
    train_data [column] = scaled

''' String processing '''
# Label encoding to change labels to numeric values
# Change gender to numeric values
train_data.loc [train_data ["Sex"] == "male", "Sex"] = 0
train_data.loc [train_data ["Sex"] == "female", "Sex"] = 1
# Change embarked port to numeric values
train_data.loc [train_data ["Embarked"] == "S", "Embarked"] = 0
train_data.loc [train_data ["Embarked"] == "C", "Embarked"] = 1
train_data.loc [train_data ["Embarked"] == "Q", "Embarked"] = 2

# Clean stringed columns and column values (use REGEX and exercise common sense for EACH DIFFERENT column!)
# Clean column names/labels
train_data.columns = clean_string (train_data.columns) # Call clean_string function and update dataset with cleaned column names

# Clean string column data values
for column in train_data.columns: # Loop to access list of column names

    # Get data type of column
    column_datatype = train_data [column].dtype

    # Check if datatype of column is a string
    if (column_datatype == np.object):

        # Call clean_string function and update dataset column with the cleaned column values
        train_data [column] = clean_string (train_data [column].values)

# Export cleaned dataset into a CSV file
train_data.to_csv (clean_file_path)

# Get cleaned data
clean_data = pd.read_csv (clean_file_path, index_col = "PassengerId")

# 4) Train model

# ***Experiment with which algorithms to use, what features to add and drop to obtain a higher accuracy! 
# Can also try more advanced things like cross validation, transforming the dataset into a different from..
# -> Just trial and error until get the best accuracy, however need to make sure of the bias-variance tradeoff! (difference of >= 3.5% in accuracy may be indicative of this)

# Specify target feature/result
target = train_data.Survived

# Specify selected features
features = train_data [["Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked", "Sex" ]].values

# Sometimes features are better represented as polynomials
# Create polynomial converter
polynomial = PolynomialFeatures (degree = 2)

# Convert features to polynomial
poly_features = polynomial.fit_transform (features)

# Create algorithm classfiers
linear_regression_classifier = linear_model.LogisticRegression ()

# Fit features and target to algorithm classifiers and get results
linear_regression_result = linear_regression_classifier.fit (features, target)
#poly_linear_regression_result = linear_regression_classifier.fit (poly_features, target)

# Get score of classifier (check against training dataset)
print ("Linear regression: ", linear_regression_result.score (features, target)) # Result is over train data, not test data
# print ("Linear regression: ", poly_linear_regression_result.score (poly_features, target)) # Result is over train data, not test data

import pandas as pd
import matplotlib.pyplot as plt # 2) For understanding dataset
from matplotlib import pyplot # 2) For understanding dataset
from pandas.plotting import scatter_matrix # 2) For understanding dataset

# Global variables
train_file_path = "/home/p/Desktop/csitml/train.csv"

# 1) Get data
train_data = pd.read_csv (train_file_path, index_col="PassengerId")

# 2) Understand dataset
# train_data.hist() # Univariate histogram
# train_data.plot(kind='density', subplots=True, layout=(3,3), sharex=False) # Univariate density plot
# train_data.plot(kind = 'box', subplots = True, layout = (3,3), sharex = False,sharey = False) # Univariate box plot
# scatter_matrix (train_data) # Multivariate scatter matrix

# Print some information of about the data
print ("***Preliminary information about dataset***")
print ("Dimensions: ", train_data.shape, "\n")
print ("Columns and data types:")
print (train_data.dtypes, "\n")
print ("Columns with empty values: ")

# Loop to print out columns in dataset that contain empty/null values
for column in dict (train_data.count()): 

    if int (dict (train_data.count())[column]) < train_data.shape[0]: 

        print ("  ", column, dict (train_data.count())[column])

print ("\nBasic statistics: \n", train_data.describe (), "\n")

# Visualise data

# Create empty figure object to set size of figures
fig = plt.figure (figsize = (15,8))

# Bar chart of survival rates
plt.subplot2grid ((10,2), (0,0)) # Place chart in a position in a grid (as will be displaying multiple graphs)
plt.title ("Survival rate") # Title of chart
plt.xlabel ("Survived")
plt.ylabel ("Percentage")
train_data.Survived.value_counts (normalize=True).plot (kind = "bar", alpha = 0.5)

# Scatter plot of survival rate with regards to Age
plt.subplot2grid ((10,2), (0,1)) # Place chart in a position in a grid (as will be displaying multiple graphs)
plt.title ("Survival rate wrt Age") # Title of chart
plt.xlabel ("Survived")
plt.ylabel ("Age")
plt.scatter (train_data.Survived, train_data.Age, alpha = 0.1) # x-axis is Survived, y-axis is Age


# Bar chart of Passenger class distribution
plt.subplot2grid ((10,2), (1,0)) # Place chart in a position in a grid (as will be displaying multiple graphs)
plt.title ("Passenger class distribution") # Title of chart
plt.xlabel ("Passenger class")
plt.ylabel ("Distribution")
train_data.Pclass.value_counts (normalize=True).plot (kind = "bar", alpha = 0.5) # x-axis is Survived, y-axis is Age

# Bar chart of Passenger Embarked distribution
plt.subplot2grid ((10,2), (1,1)) # Place chart in a position in a grid (as will be displaying multiple graphs)
plt.title ("Passenger Embarked distribution") # Title of chart
plt.xlabel ("Embarked from")
plt.ylabel ("Distribution")
train_data.Embarked.value_counts (normalize=True).plot (kind = "bar", alpha = 0.5) # x-axis is Survived, y-axis is Age

# Kernel density estimation of Passenger class with regards to Age
plt.subplot2grid ((10,2), (2,0), colspan=2) # colspan to specify that graph will occupy two columns
plt.title ("Passenger class wrt Age") # Title of chart
plt.xlabel ("Age")
plt.ylabel ("Distribution")
for pclass in train_data.Pclass.unique(): # Loop to draw one line graph per passenger class

    # Get ages for a given class and plot
    train_data.Age [train_data.Pclass == pclass].plot (kind = "kde")
plt.legend (tuple (train_data.Pclass.unique())) # Legend of each graph (need to be added after graphs are plotted!)    

# Kernel density estimation of Survival rate with regards to Age
plt.subplot2grid ((10,2), (3,0), colspan=2) # colspan to specify that graph will occupy two columns
plt.title ("Survival rate wrt Age") # Title of chart
plt.xlabel ("Age")
plt.ylabel ("Distribution")
for survival in train_data.Survived.unique(): # Loop to draw one line graph per survival status

    # Get age for a given survival status and plot graph
    train_data.Age [train_data.Survived == survival].plot (kind = "kde")
plt.legend (tuple (train_data.Survived.unique())) # Legend of each graph (need to be added after graphs are plotted!)    

# Bar chart of survival rates amongst males
plt.subplot2grid ((10,2), (4,0)) # Place chart in a position in a grid (as will be displaying multiple graphs)
plt.title ("Survival distribution of males") # Title of chart
plt.xlabel ("Survived")
plt.ylabel ("Percentage")
train_data.Survived [train_data.Sex == 'male'].value_counts (normalize=True).plot (kind = "bar", alpha = 0.5)

# Bar chart of survival rates amongst females
plt.subplot2grid ((10,2), (4,1)) # Place chart in a position in a grid (as will be displaying multiple graphs)
plt.title ("Survival distribution of females") # Title of chart
plt.xlabel ("Survived")
plt.ylabel ("Percentage")
train_data.Survived [train_data.Sex == 'female'].value_counts (normalize=True).plot (kind = "bar", alpha = 0.5)




# Display visualisations
plt.show ()

# 3) Data pre-processing

# First clean column names/labels

# Remove whitespaces and weird characters from column names
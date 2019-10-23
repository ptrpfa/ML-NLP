import pandas as pd
from matplotlib import pyplot # 2) For understanding dataset
from pandas.plotting import scatter_matrix # 2) For understanding dataset

# 1) Get data
train_file_path = "/home/p/Desktop/csitml/train.csv"

train_data = pd.read_csv (train_file_path, index_col="PassengerId")

# 2) Understand dataset
# train_data.hist() # Univariate histogram
# train_data.plot(kind='density', subplots=True, layout=(3,3), sharex=False) # Univariate density plot
# train_data.plot(kind = 'box', subplots = True, layout = (3,3), sharex = False,sharey = False) # Univariate box plot
# scatter_matrix (train_data) # Multivariate scatter matrix

print (train_data.head ()) # View dataset
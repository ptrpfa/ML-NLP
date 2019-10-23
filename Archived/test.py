import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB # algorithm
from sklearn.svm import LinearSVC # algorithm
from sklearn.neighbors import KNeighborsClassifier #algorithm
from sklearn.metrics import accuracy_score

dataset = pd.read_csv ("./WA_Fn-UseC_-Sales-Win-Loss.csv")

# View top few data 
print(dataset.head(n=2))

print ('Separator*' * 15)

# View bottom data
print(dataset.tail())

# Get datatypes of each column in dataset
print(dataset.dtypes)

# Box plot
# set the background colour of the plot to white
#sns.set(style="whitegrid", color_codes=True)
# setting the plot size for all plots
#sns.set(rc={'figure.figsize':(11.7,8.27)})
# create a countplot
#sns.countplot('Route To Market',data=dataset,hue = 'Opportunity Result')
# Remove the top and down margin
#sns.despine(offset=10, trim=True)
# display the plotplt.show()

# violine plot
# sns.set(rc={'figure.figsize':(16.7,13.27)})
# plotting the violinplot
# sns.violinplot(x="Opportunity Result",y="Client Size By Revenue", hue="Opportunity Result", data=dataset);
# plt.show()

# Data Preprocessing
# Get the unique values for each column
print("Supplies Subgroup' : ",dataset['Supplies Subgroup'].unique())
print("Region : ",dataset['Region'].unique())
print("Route To Market : ",dataset['Route To Market'].unique())
print("Opportunity Result : ",dataset['Opportunity Result'].unique())
print("Competitor Type : ",dataset['Competitor Type'].unique())
print("'Supplies Group : ",dataset['Supplies Group'].unique())

# create the Labelencoder object
le = preprocessing.LabelEncoder()
#convert the categorical columns into numeric
dataset['Supplies Subgroup'] = le.fit_transform(dataset['Supplies Subgroup'])
dataset['Region'] = le.fit_transform(dataset['Region'])
dataset['Route To Market'] = le.fit_transform(dataset['Route To Market'])
dataset['Opportunity Result'] = le.fit_transform(dataset['Opportunity Result'])
dataset['Competitor Type'] = le.fit_transform(dataset['Competitor Type'])
dataset['Supplies Group'] = le.fit_transform(dataset['Supplies Group'])
#display the initial records
print(dataset.head())

# Remove unused columns for machine learning
# select columns other than 'Opportunity Number','Opportunity Result'
cols = [col for col in dataset.columns if col not in ['Opportunity Number','Opportunity Result']]
# dropping the 'Opportunity Number'and 'Opportunity Result' columns
data = dataset[cols]
#assigning the Oppurtunity Result column as target
target = dataset['Opportunity Result']
print(data.head(n=2))

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)

#create an object of the type GaussianNB
gnb = GaussianNB()
#train the algorithm on training data and predict using the testing data
pred = gnb.fit(data_train, target_train).predict(data_test)
#print(pred.tolist())
#print the accuracy score of the model
print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))

#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0)
#train the algorithm on training data and predict using the testing data
pred = svc_model.fit(data_train, target_train).predict(data_test)
#print the accuracy score of the model
print("LinearSVC accuracy : ",accuracy_score(target_test, pred, normalize = True))

#create object of the lassifier
neigh = KNeighborsClassifier(n_neighbors=3)
#Train the algorithm
neigh.fit(data_train, target_train)
# predict the response
pred = neigh.predict(data_test)
# evaluate accuracy
print ("KNeighbors accuracy score : ",accuracy_score(target_test, pred))
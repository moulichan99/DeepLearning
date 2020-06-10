# Logistic Regression
 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Importing the dataset
dataset = pd.read_csv('matches.csv', index_col='team1')
dataset = dataset.loc['Chennai Super Kings']
data = dataset[['id','city','team2','toss_winner','win_by_runs','win_by_wickets','winner']]

winner_list = data.winner.tolist()
for ind in range(0,len(winner_list)):
    if winner_list[ind] == 'Chennai Super Kings':
        winner_list[ind] = 1
    else:
        winner_list[ind] = 0

Toss_list = data.toss_winner.tolist()
for ind in range(0,len(Toss_list)):
    if Toss_list[ind] == 'Chennai Super Kings':
        Toss_list[ind] = 1
    else:
        Toss_list[ind] = 0

data=data.drop('winner',axis=1) 
data=data.drop('toss_winner',axis=1) 
data['toss_winner'] = Toss_list
data['winner'] = winner_list

X = data.iloc[:,0:6].values
y = data.iloc[:, 6].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
X[:, 1] = label_encoder1.fit_transform(X[:, 1].astype(str))
X[:, 2] = label_encoder2.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [1,2]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = transformer.fit_transform(X).toarray()
X = X.astype('float64')
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train.astype(float))
X_test = sc_X.transform(X_test.astype(float))

#Fitting Logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#predict classifications
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#visualize predictions

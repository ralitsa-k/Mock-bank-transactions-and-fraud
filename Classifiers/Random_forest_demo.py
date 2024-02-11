
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# some sample data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import csv



df = pd.read_table('SourceData/transactions.csv',names=['Descriptions'], delimiter=',')
df[['Descriptions','Amount','Category']] = df['Descriptions'].str.split(',',expand=True)

df1 = df.loc[:,['Descriptions']]


df_train = df.loc[:200, ['Descriptions', 'Category']]
df_test = df.loc[200:293, ['Descriptions', 'Category']]
df.sort_values('Category', ascending=False)


#%% Predict text type
df=pd.DataFrame([
    ['MasterCard online Transaction at GOOGLE *TinderPlus', 'dating'],
    ['CHECKCARD AMAZON AMAZN.COM/BILLWA 125679318', 'shopping'],
    ['CHECKCARD AMAZON AMAZN.COM/BILLWA 467924720', 'shopping'],
    ['Visa Fandango.com CA Fandango.com 787879089', 'entertainment'],
    ['VISA Amazon web services aws.amazon.coWA 12321321', 'work'],
    ['Mastercard Dr Jimmy Smits DDS', 'medical'],
    # ... lots more rows
], columns=['Description','category'])

# convert the input text to something that sklearn can compute on using a TfidfVectorizer
tfidf=TfidfVectorizer()
x_train=tfidf.fit_transform(df.Description)

# we need to also encode the "target" as something the algorithm can handle (numbers)
le=LabelEncoder()
y_train=le.fit_transform(df.category)

# here's the actual ML algorithm
classifier=RandomForestClassifier(n_jobs=-1)

x_train = np.asarray(x_train.todense())
# train the model on your historical data
classifier.fit(x_train, y_train)

# here's our "new" data that we want to get categories for, we need 
#  to treat it the same way
txt_predict=['Amazon web services', 'Harris Teeter', 'Amazon', 'Dr', 'Mastercard Tinder']
x_predict=tfidf.transform(txt_predict)

x_predict = np.asarray(x_predict.todense())
# do the magic prediction!
predicted=classifier.predict(x_predict)

# predict() output is just a bunch of numbers, we need to turn it back into words
actual_answers=le.inverse_transform(predicted)
print(actual_answers)
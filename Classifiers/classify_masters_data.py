#%%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# some sample data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB  # You can choose the appropriate Naive Bayes variant here
import joblib

df = pd.read_csv('OutputData/data_with_classified_scam.csv')
df.columns

# Balance data
scams = df.loc[df.is_scam_transaction == 1,:]
scams.shape[0]
noscams = df.loc[df.is_scam_transaction == 0,:]
noscams.shape[0]

noscams_b = noscams.sample(scams.shape[0])

df = pd.concat([scams, noscams_b])


#%% Check if dimensions will be different from number of variables 
# maybe some are correlated 
df_sel = df.loc[:,['date','Descriptions','Amount','is_scam_transaction']]
le= LabelEncoder()
df_sel['Descriptions'] = le.fit_transform(df_sel['Descriptions'])
df_sel['date'] = le.fit_transform(df_sel['date'])
pca_n = PCA(n_components=0.9).fit(StandardScaler().fit_transform(np.array(df_sel)))
print(pca_n.n_components_)

#%% Basic Naive Bayes 

#Vae
#features = ['transaction_value', 'case_value', 'number_of_transactions_by_case']

df.columns

# Features
X_features = df.loc[:, ['Descriptions', 'Category', 'Amount', 'date']]

# PREDICTOR
y = df.loc[:, ['is_scam_transaction']]

X_train, X_test, y_train, y_test = train_test_split(X_features, y, random_state=0, train_size=0.75)


# FEATURES 
# Encode 'sentences' and 'words' columns using CountVectorizer
max_features = 200  # Adjust this value as needed
# Encode 'sentences' and 'words' columns using CountVectorizer with limited features

# Fit the CountVectorizer on the training data for 'Descriptions' and 'Category' columns
vectorizer = CountVectorizer(max_features=max_features)
X_descr_train = vectorizer.fit_transform(X_train['Descriptions'])
X_categ_train = vectorizer.transform(X_train['Category'])

# Convert the sparse matrices to DataFrames
X_descr_X_train = pd.DataFrame(X_descr_train.toarray(), columns=vectorizer.get_feature_names_out())
X_categ_X_train = pd.DataFrame(X_categ_train.toarray(), columns=vectorizer.get_feature_names_out())

# Standardize 'Amount in pounds' column
scaler = StandardScaler()
X_amount_train = pd.DataFrame(scaler.fit_transform(X_train[['Amount']]), columns=['Amount'])
X_amount_test = pd.DataFrame(scaler.transform(X_test[['Amount']]), columns=['Amount'])

# Encode 'dates' column if it's categorical (e.g., month names)
le = LabelEncoder()
X_dates_train = pd.DataFrame(le.fit_transform(X_train['date']), columns=['date'])
X_dates_test = pd.DataFrame(le.fit_transform(X_test['date']), columns=['date'])

# Transform the test data using the same CountVectorizer and other preprocessing steps
X_descr_test = vectorizer.transform(X_test['Descriptions'])
X_descr_X_test = pd.DataFrame(X_descr_test.toarray(), columns=vectorizer.get_feature_names_out())
X_categ_test = vectorizer.transform(X_test['Category'])
X_categ_X_test = pd.DataFrame(X_categ_test.toarray(), columns=vectorizer.get_feature_names_out())

# Concatenate all the DataFrames
X_features_train = pd.concat([X_descr_X_train, X_categ_X_train, X_amount_train, X_dates_train], axis=1)
X_features_test = pd.concat([X_descr_X_test, X_categ_X_test, X_amount_test, X_dates_test], axis=1)


# Create a Gaussian Naive Bayes classifier (you can choose the appropriate variant)
classifier = GaussianNB()  # Use MultinomialNB if your features are discrete or count-like

# Fit the Naive Bayes classifier on the training data
trained_model = classifier.fit(X_features_train, y_train)

# Now you can use the trained classifier for prediction
predictions = classifier.predict(X_features_test)

y_test = y_test.reset_index(drop=True)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Specify the filename to save the model
model_filename = 'find_fraud_cases.pkl'
# Save the model to a file
joblib.dump(trained_model, model_filename)
joblib.dump(vectorizer, 'vectorizer_filename.pkl')
joblib.dump(scaler, 'scaler_filename.pkl')
joblib.dump(le, 'label_encoder_filename.pkl')



# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
#     Accuracy is a measure of overall correctness in a classification model.
#     Accuracy is a single number that tells you the percentage of all predictions that are correct.
#     It's suitable for balanced datasets where the classes are roughly equal in size.
#     However, accuracy can be misleading when dealing with 
#       imbalanced datasets because a high accuracy score can be achieved by simply 
#       predicting the majority class all the time.

#     Precision is a measure of the model's ability to make correct positive predictions.
#     Precision focuses on the relevant subset of predictions and evaluates how well the 
#           model avoids false positives.
#     It's particularly important when the cost of false positives is high 
#           or when you want to ensure that positive predictions are highly reliable.


#%% Test on new data

loaded_model = joblib.load('find_fraud_cases.pkl')
# Load the saved preprocessing transformers
vectorizer = joblib.load('vectorizer_filename.pkl')
scaler = joblib.load('scaler_filename.pkl')
le = joblib.load('label_encoder_filename.pkl')
df2 = pd.read_csv('final_clean_masters_data_clasified_test_model.csv')

# PREDICTOR
y_test_new = df2.loc[:, ['is_scam_transaction']]
X_test = df2

# FEATURES 
# Encode 'sentences' and 'words' columns using CountVectorizer
# Encode 'sentences' and 'words' columns using CountVectorizer with limited features
X_descr_test = vectorizer.transform(X_test['Descriptions'])
X_categ_test = vectorizer.transform(X_test['Category'])

# Standardize 'amount in pounds' column
X_amount_test = pd.DataFrame(scaler.transform(X_test[['Amount']]), columns = ['Amount'])

# Encode 'dates' column if it's categorical (e.g., month names)
# You can use LabelEncoder for this purpose
X_dates_test = pd.DataFrame(le.fit_transform(X_test['date']), columns = ['date'])

# Convert the sparse matrices to DataFrames
X_descr_X_test = pd.DataFrame(X_descr_test.toarray(), columns=vectorizer.get_feature_names_out())
X_categ_X_test = pd.DataFrame(X_categ_test.toarray(), columns=vectorizer.get_feature_names_out())
# Concatenate all the DataFrames
X_test = pd.concat([X_descr_X_test, X_categ_X_test, X_amount_test, X_dates_test], axis=1)


#%% Test 


predictions = loaded_model.predict(X_test)

accuracy = accuracy_score(y_test_new, predictions)
precision = precision_score(y_test_new, predictions)
recall = recall_score(y_test_new, predictions)
f1 = f1_score(y_test_new, predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# %%

#%% 
""" 
Created by Julian Wang and Kathy Blackburn. 
Original sourceL: https://innersource.soprasteria.com/fraud-detection-proposition/app-scam-prevention.git
# The following code is to build the synthetic data for scam prevention based on a range of distribution we have been able to identify.  
# Where we do not know the distribution of the data we will initially build with a normal curve and as part of the project will consider 
# the impacts if those data where not normally distributed as is likely to be the case.

"""

# %%
import numpy as np
from scipy.stats import skewnorm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import os

# Get the path for this current file 
curr_path = os.path.abspath(__file__)
# Get the root path by deleting everything after the specified folder 
curr_abs_path = curr_path.split('BanksDataGen')[0]

# Define paths for saving files and loading files 
save_path = curr_abs_path + 'BanksDataGen/OutputData/'
source_d_path = curr_abs_path + 'BanksDataGen/SourceData/'

# %% [markdown]
# The following set of code is a defined function that takes the input from a dictionary to calculate the number of rows for a give key pair value set.

def generate_column_values(distribution, total_cases):
    values = []
    all_but_last_dict = {k: v for i, (k, v) in enumerate(distribution.items()) if i < len(distribution) - 1}
    last_key = list(distribution.keys())[-1]

    for key, ratio in all_but_last_dict.items():
        count = int(round(total_cases * ratio))
        values.extend([key] * count)


    remaining_rows = total_cases - len(values)
    values.extend([last_key] * remaining_rows)


    #np.random.shuffle(values)
    return values

# %% [markdown]
# The following code is a defined function that takes the input from a DataFrame to calculate the number of rows for a given set of values

# %%
def generate_column_values2(df_fraud, total_cases):
    values2 = []
    all_but_last_fraud_type = df_fraud['fraud_type'][:-1]
    last_fraud_type = df_fraud['fraud_type'].iloc[-1]

    for fraud in all_but_last_fraud_type:
        fraud_distribution = df_fraud[df_fraud['fraud_type'] == fraud]['fraud_type_distribution']

        fraud_count = int(fraud_distribution.iloc[0] *  total_cases)
        values2.extend([fraud ]* fraud_count)

    remaining_rows = total_cases - len(values2)
    values2.extend([last_fraud_type] * remaining_rows)

    return values2

    

# %% [markdown]
# the following function is used to calculate the case value across the different fraud types

# %%
def generate_column_values3(df_fraud, total_cases, fraud_type):

    fraud=fraud_type
    mean = df_fraud[df_fraud['fraud_type'] == fraud]['avg_value_per_case']
    case_cnt = total_cases
    dev = df_fraud[df_fraud['fraud_type'] == fraud]['val_dev']
    skew = df_fraud[df_fraud['fraud_type'] == fraud]['val_skew']

    value3 = []

    value3 = skewnorm.rvs(skew,mean,dev, case_cnt)
    #value3 = np.random.normal(mean,dev,case_cnt)

    return value3

def generate_column_values4(df_fraud, total_cases, fraud_type):

    fraud=fraud_type
    mean = df_fraud[df_fraud['fraud_type'] == fraud]['avg_trans_per_case']
    case_cnt = total_cases
    dev = df_fraud[df_fraud['fraud_type'] == fraud]['tran_dev']
    skew = df_fraud[df_fraud['fraud_type'] == fraud]['tran_skew']

    value4 = []

    value4 = skewnorm.rvs(skew,mean,dev, case_cnt)

    return value4

total_cases = 200000

case_id = np.random.randint(1000000, 9000000, total_cases)

bank_distribution = {'bankA': 0.2
                     , 'bankB': 0.3
                     , 'bankC': 0.2
                     , 'bankD': 0.1
                     , 'bankE': 0.1
                     , 'bankF': 0.1}
data = {'fraud_type':['Romance','Impersonation','Investment','CEO','Purchase','InvoiceMandate','AdvanceFee'],
        'fraud_type_distribution':[0.017,0.223,0.054,0.002,0.565,0.017,0.121],
        'avg_value_per_case':[9988,4257,11847,39303,578,16822,1265],
        'avg_trans_per_case':[8.4,2,2.9,1.5,1.3,1.5,1.8],
        'total_value':[33200000, 181000000, 122400000, 15800000, 62200000, 54200000, 29200000],
        'val_dev':[5000, 1000, 5000, 10000, 300, 10000, 1000],
        'val_skew':[6,4,3,5,3,10,3],
        'tran_dev':[2, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
        'tran_skew':[1, 0, 0.5, 0, 0, 0, 0]}

df_case_distribution = pd.DataFrame(data)


data = {'fraud_type':['Romance','Impersonation','Investment','CEO','Purchase','InvoiceMandate','AdvanceFee'],
        'fraud_type_distribution':[0.017,0.223,0.054,0.002,0.565,0.017,0.121],
        'avg_value_per_case':[9988,4257,11847,39303,578,16822,1265],
        'avg_trans_per_case':[8.4,2,2.9,1.5,1.3,1.5,1.8],
        'total_value':[33200000, 181000000, 122400000, 15800000, 62200000, 54200000, 29200000],
        'val_dev':[5000, 1000, 5000, 10000, 300, 10000, 1000],
        'val_skew':[6,4,3,5,3,10,3],
        'tran_dev':[2, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
        'tran_skew':[1, 0, 0.5, 0, 0, 0, 0]}


# %%
df_case_distribution.head(10)

# %%
bank = generate_column_values(bank_distribution, total_cases)

# %%
data2 = {'case_id': case_id, 'bank' : bank}
df_cases = pd.DataFrame(data2)

# %%
df_cases.head(5)

# %%
frauds = []
for index, row in df_cases.groupby('bank').count().reset_index().rename(columns={'case_id' : 'count'}).iterrows():
    #print(row['bank'], row['count'])

    subtotal_rows = row['count']
    frauds.extend(generate_column_values2(df_case_distribution, subtotal_rows))
    
df_cases['fraud_type'] = frauds

# %%
case_value = []
for index, row in df_cases.groupby('fraud_type').count().reset_index().rename(columns={'case_id':'count'}).iterrows():
    print(row['fraud_type'], row['count'])

    subtotal_rows = row['count']
    fraud_type = row['fraud_type']
    case_value.extend(generate_column_values3(df_case_distribution, subtotal_rows, fraud_type))

df_cases = df_cases.sort_values(by=['fraud_type'], ignore_index='True')
df_cases = df_cases.reset_index()
df_cases['case_value'] = case_value

# %%
df_cases.groupby('fraud_type').agg({'case_value' : [min, max]})

# %%
tran_cnt = []
for index, row in df_cases.groupby('fraud_type').count().reset_index().rename(columns={'case_id':'count'}).iterrows():
    print(row['fraud_type'], row['count'])

    subtotal_rows = row['count']
    fraud_type = row['fraud_type']
    tran_cnt.extend(generate_column_values4(df_case_distribution, subtotal_rows, fraud_type))

df_cases['number_of_transactions'] = tran_cnt

# %%
df_cases.groupby('fraud_type').agg({'number_of_transactions' : [min, max]})

# %%
df_cases['number_of_transactions'] = round(df_cases['number_of_transactions'],0)
df_cases['number_of_transactions'] = df_cases['number_of_transactions'].astype(int)
df_cases.head(5)

# %%
df_cases.groupby('fraud_type').mean(numeric_only=True)


# %%
def duplicate_rows(id, number_of_transactions):
    temp_id = []
    temp_id.extend([id]*number_of_transactions)
    
    return temp_id

def generate_transaction_value(number_of_transactions, value):
    tran_amt = []
    tran_amt.extend(np.int_(np.random.dirichlet(np.ones(number_of_transactions)) * value))

    return tran_amt

trans = []
val = []

for index, row in df_cases.iterrows():
    case_id = row['case_id']
    transactions = row['number_of_transactions']
    value = row['case_value']
    trans.extend(duplicate_rows(case_id, transactions))
    val.extend(generate_transaction_value(transactions, value))

df_transactions = pd.DataFrame({'case_id':trans, 'transaction_value':val})
df_transactions.head(5)

# %%
total_trans = len(df_transactions)
df_transactions['transaction_id'] = np.random.randint(1000000, 9000000, total_trans)
df_transactions.head(5)

# %%
df_case_transaction = df_transactions.merge(df_cases, how='left', left_on='case_id', right_on='case_id')

df_case_transaction.head(5)
df_case_transaction.to_csv(source_d_path + 'case_transaction.csv')

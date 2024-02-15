

""" 
this file adds fraud transactions to some of the customers and adds some additional columns for the students to work with 
As the generating_bank_transactions.py, the folder and files should be:

    -- BanksDataGen/
        - OutputData 
            |- final_logical_non_fraud.csv    (created with Generating_bank_transactions.py)
        - SourceData 
            |- no_fraud_transactions.csv      (created manually)
            |- fraud_with_descriptions.csv    (created with Descriptions_additions.py)

"""
import time
st = time.time()
import numpy as np
import pandas as pd
import random
from datetime import date
import os
import polars as pl

# Get the path for this current file 
curr_path = os.path.abspath(__file__)
# Get the root path by deleting everything after the specified folder 
curr_abs_path = curr_path.split('banks-data-gen')[0]

# Define paths for saving files and loading files 
save_path = curr_abs_path + 'banks-data-gen/OutputData/'
source_d_path = curr_abs_path + 'banks-data-gen/SourceData/'

# The way it all works now is that non fraud data let's say of
#  N customers is taken and a certain amount of them have fraud interjected in their normal transactions (substituted)
# (so that number of transactions is not indicative if someone is scammed or not)

#%% Prep fraud data and define some parameters 

large_non_fraud_data = pl.read_csv(save_path + 'final_logical_non_fraud2.csv')
large_non_fraud_data = large_non_fraud_data.rename({'category':'Category'})
# define how many to be scammed 
perc_scammes = 18/100
# get N scammed 
n_cust = len(large_non_fraud_data.unique('customer_id'))
n_scammed = int(n_cust*perc_scammes)
scammed_ids = random.choices(range(1,n_cust),k=n_scammed)
# Get the data for scammed customers only 
scammed_sample_data = large_non_fraud_data.filter(pl.col('customer_id').is_in(scammed_ids))

# Load the data with fraud cases with descriptions 
fraud_cases = pl.read_csv(source_d_path + 'fraud_with_descriptions.csv')

# get quartiles of amounts
transaction_quartiles = np.percentile(fraud_cases.select(pl.col('transaction_value')), [25, 50, 75])
# transaction in the upper quartiles 
high_value_transaction = fraud_cases.filter(pl.col('transaction_value') > transaction_quartiles[2])
# prepare high fraud transaction values to be assigned to high-paid individuals later
# first get salaries and obtain the quartiles of salaries of scammed customers 
salaries = (scammed_sample_data.filter(pl.col('Category')=='Income'))
quartiles = np.percentile(salaries.select(pl.col('Amount')), [25, 50, 90])

# match_names of columns of fraud data and non-fraud data 
fraud_cases = fraud_cases.rename({'transaction_value':'Amount',
                             'transaction_description': 'Descriptions',
                             'category':'Category',
                             })

#%% Clean the fraud data and add fraud transactions to selected customers 

# data without CEO scam 
no_ceo = fraud_cases.filter(pl.col('fraud_type')!='CEO')
# divide the rest of the data to Romance cases vs no Romance cases 
# Romance is a scam that we will assign to higher-paid individuals only 
romance_only = no_ceo.filter(pl.col('fraud_type')=='Romance')
no_romance = no_ceo.filter(pl.col('fraud_type')!='Romance')
    
# get number of months in non-fraud data
scammed_sample_data = scammed_sample_data.with_columns(pl.col('date').str.to_datetime().dt.month().alias('month'))
n_of_months = len(scammed_sample_data.select(pl.col('month')).unique())
# define how many fraud transactions we want per month 
# sampling up to 5 per romance and up to 3 per other types of scam 
romance_per_month = n_of_months*5
noromance_per_month = n_of_months*3
    
# From here on we get the data for each scammed customer and apply scam transactions     
fraud_and_no = pl.DataFrame()
for cust in scammed_sample_data.select(pl.col('customer_id')).unique():
    # get the data per customer 
    curr_data = scammed_sample_data.filter(pl.col('customer_id') == cust)
    # add column fraud type 
    curr_data = curr_data.with_columns(pl.lit('none').alias('fraud_type'))
    
    # If the income of this person is in a high quartile, assign Romance to them 
    # If not, assign other types of scam 
    if curr_data.filter(pl.col('Category') == 'Income').select(pl.col('Amount')) > quartiles[2]:
       nn = random.randint(romance_per_month-2, romance_per_month)
       fraud_d = romance_only.sample(nn)
    else:
       nn = random.randint(noromance_per_month-2, noromance_per_month)
       fraud_d = no_romance.sample(nn)
       
    sample_dates = curr_data.select(pl.col('date', 'month', 'customer_id')).sample(nn)  
    # fraud is always spending (for now), so add this column   
    fraud_d['type'] = 'spending'
    # add the dates to fraud transactions 
    fraud_d[['date', 'month', 'customer_id']] = sample_dates
    
    # the dates used for the fraud transactions have to be used to remove some regular transactions 
    # (so that the number of transactions are still the same per customer per month)
    # but making sure income housing and utilities don't get removed 
    sdates = sample_dates.date
    shorten_data = curr_data.loc[(~curr_data['Category'].isin(['Income', 'Utilities', 'Housing'])) & (curr_data['date'].isin(sdates)),:]
    # get the index of the non-fraud data to be removed 
    remove_d = shorten_data.sample(fraud_d.shape[0]).index
    # from the full data remove the random non-fraud transactions given their defined index
    curr_data.drop(index=remove_d)
    curr_data = pd.concat([curr_data, fraud_d])
   
    # add the fraud transactions to the non-fraud data of this customer 
    fraud_and_no = pd.concat([fraud_and_no, curr_data])
    
# add a flag indicated that all of the customers in this data set have been scammed at least once     
fraud_and_no['customer_scammed'] = 1



#%% Add the non-fraud data to the customers who were scammed 

# remove the scammed customers from the original data
non_fraud = large_non_fraud_data.loc[~large_non_fraud_data['customer_id'].isin(scammed_ids)].copy(deep = False)
# make sure ordinary transaction are of fraud_type none 
non_fraud['fraud_type'] = 'none'
# all of these customers are not scammed, so the flag is 0
non_fraud['customer_scammed'] = 0

# combine the two data sets 
data_enriched = pd.concat([fraud_and_no,non_fraud])

# Define if a transaction is a deposit or withdrawal 
deposits_descr = ['Deposit', 'Interest', 'from Savings' ]

# check average Number of fraud transactions per customer per month 
n_fraud_transactions = len(data_enriched.transaction_id.drop_duplicates())
n_fraud_transactions/n_scammed/n_of_months

#%%  Assign banks to tranasactions 

list_banks_uk = ["bank_A", "bank_B", "bank_C", "bank_D", "bank_E"]

data_enriched.loc[:,'bank_to'] = random.choices(list_banks_uk, k = data_enriched.shape[0])
data_enriched.loc[:,'bank_from'] = random.choices(list_banks_uk, k = data_enriched.shape[0])
   
# assigning banks to customers. 
# income is always going into the customers bank
# all spending always going out from customers bank    
def assign_banks(customer_data, own_bank):
    curr_data_copy = customer_data.copy()
    curr_data_copy.reset_index(drop=True, inplace=True)

    # maybe the following code can be optismised 
    for i in range(curr_data_copy.shape[0]):
        if curr_data_copy.loc[i, 'type'] == 'income':
            curr_data_copy.at[i, 'bank_to'] = own_bank
        if curr_data_copy.loc[i, 'type'] == 'spending':
            curr_data_copy.at[i, 'bank_from'] = own_bank

        if curr_data_copy.loc[i, 'Category'] == 'Income':
            # income bank to and bank from, keep constant across months 
            inc = curr_data_copy.loc[curr_data_copy['Category'] == 'Income', ['bank_to', 'bank_from']].iloc[0]
            curr_data_copy.at[i, 'bank_to'] = inc['bank_to']
            curr_data_copy.at[i, 'bank_from'] = inc['bank_from']
            
        # If a description contains 'from Savings or to Savings, it means the transaction
        #   should be going between the bank of the customer 
        for i in range(len(curr_data_copy)):
            # Convert cell value to string
            description = str(curr_data_copy.loc[i, 'Descriptions'])
            
            # Check if any word exists in the description
            if any(word in description for word in deposits_descr):
                curr_data_copy.at[i, 'bank_to'] = own_bank
                curr_data_copy.at[i, 'bank_from'] = own_bank
    
    # Making sure that utilities always end up in the same bank 
    utils = curr_data_copy.loc[(curr_data_copy['Category'] == 'Utilities') & (curr_data_copy['fraud_type'] == 'none'), ['Descriptions', 'bank_to','bank_from']]
    # get the unique utilities only 
    first_occurrences = utils[~utils['Descriptions'].duplicated(keep='first')]
    utils_des = utils.loc[:,['Descriptions']]
    merged_data = utils_des.merge(first_occurrences, on='Descriptions', how='left')  
    utils.reset_index(drop=True, inplace=True)
    merged_data.reset_index(drop=True, inplace=True)
    # Update 'bank_to' and 'bank_from' columns in 'utils' with values from 'merged_data'
    utils['bank_to'] = merged_data['bank_to'].fillna(utils['bank_to'])
    utils['bank_from'] = merged_data['bank_from'].fillna(utils['bank_from'])
    utils.reset_index(drop=True, inplace=True)
    curr_data_copy.reset_index(drop=True, inplace=True)
    curr_data_copy.loc[(curr_data_copy['Category'] == 'Utilities') & (curr_data_copy['fraud_type'] == 'none'), 'bank_to'] = utils['bank_to'].values
    curr_data_copy.loc[(curr_data_copy['Category'] == 'Utilities') & (curr_data_copy['fraud_type'] == 'none'), 'bank_from'] = utils['bank_from'].values

    return curr_data_copy

# Use the above function to assign banks to add bank to and bank from to each customer
data_with_banks = pd.DataFrame()
for cust in data_enriched['customer_id'].unique():
    own_bank = random.choice(list_banks_uk)
    curr_data = data_enriched[data_enriched['customer_id'] == cust].copy()
    curr_data_with_banks = assign_banks(customer_data = curr_data, own_bank = own_bank)
    data_with_banks = pd.concat([data_with_banks, curr_data_with_banks])

# no bank to for withdrawal 
update_bank_to = lambda row: np.nan if row['Category'] == 'Cash Withdrawal' else row['bank_to']
data_with_banks['bank_to'] = data_with_banks.apply(update_bank_to, axis=1)

# deposits (income) have the same 'bank from' every time
# cash withdrawals do not have 'bank to' 
# transfers from and to savings accounts go between the owner (so same bank)
# utilities go to betweeb the same banks every month


#%% Make variable incomes and spendings per customer 

# Increasing spendings and income for some customers 
cust_r = random.choices(data_with_banks['customer_id'].unique(), k=len(data_with_banks['customer_id'].unique()))
# factor by which the amounts will be increased (customer-wise)
factor_increase = np.random.uniform(0.5, 2, size = len(cust_r))

# Create a dictionary to map customer IDs to their corresponding factors
factor_dict = dict(zip(cust_r, factor_increase))
data_with_temp = data_with_banks.copy(deep = False)

# Use the transform function to apply the multiplication based on customer ID
data_with_banks['Amount'] = data_with_banks.groupby('customer_id')['Amount'].transform(
    lambda x: x * factor_dict.get(x.iloc[0], 1.0)  # Use 1.0 as the default factor if customer ID not found
)

cc = data_with_banks.groupby('customer_id').count()/10
data_with_banks.groupby('transaction_id').count()

#%% Add probabilities 

# Probability rules for transactionс 
def assign_transac_probs(row, transaction_quartiles):
    if row['Amount'] <= transaction_quartiles[0]:
        return round(np.random.uniform(0.1, 0.4),2)
    elif row['Amount'] > transaction_quartiles[0] and row['Amount'] <= transaction_quartiles[1]:
        return round(np.random.uniform(0.3, 0.6),2)
    elif row['Amount'] > transaction_quartiles[1] and row['Amount'] <= transaction_quartiles[2]:
        return round(np.random.uniform(0.4, 0.7),2)
    elif row['Amount'] >= transaction_quartiles[2]:
        return round(np.random.uniform(0.5, 0.8),2)
    else:
        return np.nan

# Define risky transaction based on customer spendings 
for cust in data_with_banks['customer_id'].unique():
    curr_data = data_with_banks[data_with_banks['customer_id'] == cust].copy()
    this_average = curr_data.loc[curr_data['type'] == 'spending', :]['Amount'].mean()
    transaction_values = curr_data.loc[:,'Amount']
    transaction_quartiles = np.percentile(transaction_values, [25, 50, 75])
    
    # add transaction probability according to quartiles rule for each customer 
    data_with_banks.loc[data_with_banks.customer_id == cust,'transac_prob'] = data_with_banks.loc[data_with_banks.customer_id == cust, :].apply(assign_transac_probs,args=(transaction_quartiles,), axis=1)
 
    # add customer probability - higher if customer is scammed (for some reason unknown to the students)
    if np.array(curr_data.customer_scammed[[1]])==1:
       data_with_banks.loc[data_with_banks.customer_id == cust, ['customer_prob']] = round(np.random.uniform(0.4, 0.9),2)
    elif np.array(curr_data.customer_scammed[[1]])==0:
       data_with_banks.loc[data_with_banks.customer_id == cust, ['customer_prob']] = round(np.random.uniform(0.2, 0.7),2)
       
    # transaction probability depending on the description (if pooled from fraud descriptions)   

# Low probability of cash withdrawal 
data_with_banks['transac_prob'] = data_with_banks.apply(lambda row: round(np.random.uniform(0.1, 0.3), 2) 
                                                        if row['Category'] == 'Cash Withdrawal' else row['transac_prob'], axis=1)
# na probability if paid-in 
data_with_banks['transac_prob'] = data_with_banks.apply(lambda row: np.nan if row['type'] == 'income' else row['transac_prob'], axis=1)



#%% Add international banks 

temp = data_with_banks.copy()
temp.reset_index(drop=True, inplace=True)
# Transaction probability = between 0.5 and 1 to be fraud if larger than average spendings per customer 
# between 0 and 0.3 if cash withdrawal
# Add international banks 
int_banks = ['Intrnl']

temp.Category.unique()

categ = ['Credit Card Payment', 'Charity',
       'Online Shopping', 'Electronics',
       'Investment', 'Holiday']

# get fraud with transaction probability more than 0.6 
high_prob_fraud = temp.loc[(temp['fraud_type'] != 'none') & (temp['transac_prob'] > 0.6), :]
# get non-fraud from the above categories 
high_prob_nonfraud = temp.loc[(temp['fraud_type'] == 'none') & temp['Category'].isin(categ), :]

# sample some transactions that are fraud or not to go to international banks 
for_intrl = high_prob_fraud.sample(n=high_prob_fraud.shape[0] // 3)
for_intrl_nonfr = high_prob_nonfraud.sample(n = high_prob_nonfraud.shape[0] // 10)
for_intrl = pd.concat([for_intrl, for_intrl_nonfr])
# there were initially a few international banks, which demanded random choice of bank: 
for_intrl['bank_to'] = random.choices(int_banks, k=for_intrl.shape[0])

sum(temp.index.isin(for_intrl.index))
temp.drop(temp[temp.index.isin(for_intrl.index)].index, inplace=True)
# Merge the modified for_intrl DataFrame back into data_with_banks
data_with_intrl_banks = pd.concat([temp, for_intrl])


#%% Save data for Master's projects 

# Clean the data for the project, remove columns and hide some 
clean_masters_data = data_with_intrl_banks.loc[:,['customer_id', 'Descriptions', 'Amount', 'Category', 'date', 'bank_from', 'bank_to',
       'transac_prob', 'customer_prob']]
                                            
clean_masters_data['Amount'] = round(clean_masters_data['Amount'],2)

clean_masters_data.to_csv(save_path + 'data_without_classified_scam.csv')


# Prepare full data with clasified transactions
data_with_intrl_banks['is_scam_transaction'] = data_with_intrl_banks['fraud_type'].apply(lambda x: 0 if x == 'none' else 1)

data_with_intrl_banks.groupby('is_scam_transaction').count()

full_data = data_with_intrl_banks.loc[:,['Descriptions', 'Amount', 'Category',
                                   'date', 'customer_id', 'type','is_scam_transaction', 'fraud_type',
                                    'case_id', 'transaction_id','month',
                                    'customer_scammed', 'bank_to', 'bank_from', 'transac_prob',
                                    'customer_prob']]

full_data['date_DF_created'] = date.today()

full_data = full_data.sort_values('date')
full_data.reset_index(drop = True, inplace = True)
full_data.index = full_data.index + 1
full_data.to_csv(save_path + 'data_with_classified_scam.csv')


# %% Stats and Checks 

# will depend on current parameters 
# months data 
len(data_with_intrl_banks.month.unique())

# n of customers
len(data_with_intrl_banks.customer_id.unique())

# n scammed customers
len(data_with_intrl_banks.loc[data_with_intrl_banks.customer_scammed == 1, :].customer_id.unique())

# average transaactions per person for the whole period (should be around 63)
clean_masters_data.groupby('customer_id').count()['Descriptions'].mean()/len(data_with_intrl_banks.month.unique())

scammed_data = data_with_intrl_banks.loc[data_with_intrl_banks.customer_scammed == 1, :]
# number of scams per person scammed : on average  for the whole period per person (e.g. 26 out of 625)
scammed_data.groupby(['customer_id', 'is_scam_transaction']).count()

# If running the full script at once- this is relevant 
# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

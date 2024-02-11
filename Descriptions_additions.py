
import sys 
import random
import pandas as pd
import numpy as np
import os
import scipy.stats
import seaborn as sns

""" 
This file adds descriptions to case_transaction dataset (created by Julian and Kathy).
The folder structure to begin with should be:

    -- BanksDataGen/
       - OutputData 
         |- 
       - SourceData 
         |- no_fraud_transactions.csv               (created manually)
         |- fraud_transaction_descriptions.csv      (created manually)
         |- case_transaction.csv                    (created by Julian and Kathy, but also: Generating_fraud_data.py)
"""

# Get the path for this current file 
curr_path = os.path.abspath(__file__)
# Get the root path by deleting everything after the specified folder 
curr_abs_path = curr_path.split('BanksDataGen')[0]

# Define paths for saving files and loading files 
save_path = curr_abs_path + 'BanksDataGen/OutputData/'
source_d_path = curr_abs_path + 'BanksDataGen/SourceData/'


#%% LOAD descriptions of no-fraud data (n_rows = 589)
# These were created with chat gpt and manually
non_fraud_data  = pd.read_csv(source_d_path + 'no_fraud_transactions.csv')

#%% Load FRAUD

# Load only descriptions for fraud types and their proposed categories 
fraud_des = pd.read_csv(source_d_path + 'fraud_transaction_descriptions.csv', names = ['fraud_type','Description','Category'])

# Scam Use case data values only
case_df = pd.read_csv(source_d_path + 'case_transaction.csv')
case_df['transaction_value'] = case_df['transaction_value'].apply(lambda x: x/100)


#%% Adding random descriptions for ALL fraud types 
# Create a dictionary to store mappings of fraud types to descriptions
fraud_type_to_descriptions = {}
for fraud_type, group in fraud_des.groupby('fraud_type'):
    descriptions = group['Description'].tolist()
    categories = group['Category'].tolist()
    fraud_type_to_descriptions[fraud_type] = list(zip(descriptions, categories))
    
# Define a function to get a random description for a given fraud type
def get_random_description_and_category(fraud_type):
    descriptions_and_categories = fraud_type_to_descriptions[fraud_type]
    random_pair = random.choice(descriptions_and_categories)
    return random_pair

# Apply the function to create the 'transaction_description' column in 'case_df' (fraud data)
case_df[['transaction_description', 'category']] = case_df['fraud_type'].apply(get_random_description_and_category).apply(pd.Series)

# Match categories that come from different csv files (if the issue applies at all)
case_df['category'] = np.where(case_df['category'] == "Dining out", "Dining Out", case_df['category'])
case_df['category'] = np.where(case_df['category'] == "Travel", "Holiday", case_df['category'])
case_df['category'] = np.where(case_df['category'] == "Online shopping", "Online Shopping", case_df['category'])

# Now this produces a pretty boring fraud data with repeating descriptions, next we will
#   add non-fraud descriptions to this data set 

#%% Embed non-fraud descriptions in fraud data

# --------------------------------------------------------------------------------------------------------
# Bind non-fraud categories to potential fraud (maybe to IMPROVE later)
# eg Romance fraud can include Personal care, dining out etc. 
match_fraud_cat = {'Investment':['Investment'], 
                 'Romance':['Personal care', 'Dining Out', 'Entertainment', 'Home Improvement','Holiday'],
                  'Purchase': ['Online Shopping', 'Credit Card Payment','Electronics','Healthcare'], 
                 'Impersonation': ['Loan Payment', 'Utilities', 'Charity'],
                 'InvoiceMandate':['Transfers'],
                 'AdvanceFee': ['Bank Fees']
                }

# Get non-fraud data and asign fraud according to above mapping
all_cats_with_fraud = pd.DataFrame()
for cat in match_fraud_cat:
    assign_fraud = non_fraud_data.loc[non_fraud_data.Category.isin(match_fraud_cat[cat]),:].copy()
    assign_fraud.loc[:,'fraud_type'] = cat
    all_cats_with_fraud = pd.concat([all_cats_with_fraud, assign_fraud])
all_cats_with_fraud = all_cats_with_fraud.drop(['min','max','mean'], axis = 1)
all_cats_with_fraud.columns = ['transaction_description', 'category', 'subcategory', 'fraud_type']

# --------------------------------------------------------------------------------------------------------
# get all original categories from non-fraud data and count 
counts = all_cats_with_fraud.groupby('category').count()
temp_case = case_df.copy(deep=False)

for cc in counts.index:
        # if category not in fraud, sample random categories and still add some descriptions
    if len(temp_case.loc[temp_case.category == cc, :]) == 0:
        # sample fraud data cases, the same number of rows as the current non-fraud category count
        sampled_t = temp_case.sample(counts.loc[cc].transaction_description, axis =0).reset_index(drop = True)
    else:
        # get all fraud data of current Category and select only half of the cases 
        sampled_t = temp_case.loc[temp_case.category == cc, :].sample(len(temp_case.loc[temp_case.category == cc, :])//2).reset_index(drop = True)
    # sample non-fraud descriptions and assign to samples fraud cases
    cat_non_fraud = all_cats_with_fraud.loc[all_cats_with_fraud.category == cc,:].sample(len(sampled_t), replace = True)
    sampled_t[['transaction_description','fraud_type','category','subcategory']] = cat_non_fraud.loc[:,['transaction_description','fraud_type','category','subcategory']].reset_index(drop = True)
    sampled_t['data_ver'] = 'non_fraud'
    temp_case = pd.merge(temp_case, sampled_t, how='outer')
               
# some categories will not be fraud: withdrawals, groceries, housing, income, interest, transport 
fraud_set = set(temp_case.category.unique())
non_Fraud_set = set(non_fraud_data['Category'].unique())
non_Fraud_set.difference(fraud_set)


#%% Fix some fraud categories because their original descriptions are too small in numbers 

cats = ['Entertainment', 'Healthcare' 'Personal care', 'Dining Out']
temp_case_better_cats = pd.DataFrame()
for c in cats:
    # take only current category and all the null cases which are old transactions descriptions
    add_to = temp_case.loc[(temp_case.category == c) & (temp_case.subcategory.isnull()), :]
    # decide how many of these old transactions to keep and how many to rewrite
       # keeping 1/3rd of original bad descriptions 
    simply_concat = add_to.sample(add_to.shape[0]//3)
    temp_case2 = temp_case.loc[~((temp_case.category == c) & (temp_case.subcategory.isnull())), :]
    # All categories are stored with reduced current category
    add_at_end = pd.concat([temp_case2, simply_concat])
    
    # from the kept transactions take the current category and null subcategory
    change_origin = add_at_end.loc[(add_at_end.category == c) & (add_at_end.subcategory.isnull()), :]
    # Take some of these bad descriptions and change them 
    add_to_keep = change_origin.sample(int(change_origin.shape[0]//1.2))
    # extract the leftover data 
    change_origin = change_origin.drop(add_to_keep.index)    
    
    add_to_keep = add_to_keep.reset_index(drop = True)
    add_to_keep['transaction_description'] = all_cats_with_fraud.loc[all_cats_with_fraud.category == c,'transaction_description'].sample(add_to_keep.shape[0], replace = True).reset_index(drop = True)
    temp_case_better_cats = pd.concat([add_to_keep, add_at_end.loc[add_at_end.category!=c,:], change_origin])

temp_case_better_cats.groupby('category').count()
# At this point it doesn't matter how many transactions each category has 
# Because fraud transactions for the main data set are taken with predefined numbers per category 

#%% Get the correct sum for Case value and save fraud data 

# filter transactions with at least 1 pounds
temp_case_better_cats = temp_case_better_cats.loc[temp_case_better_cats.transaction_value > 1, :]

# Get case value (sum of all transactions within a case)
d2 = pd.DataFrame(columns = ['case_id', 'case_value'])
d1 = temp_case_better_cats.groupby('case_id')
d2[['case_id', 'case_value']] = d1['transaction_value'].sum().reset_index()

temp_case_better_cats = temp_case_better_cats.drop(columns = ['case_value'])
temp_case_better_cats = pd.merge(temp_case_better_cats, d2, on = 'case_id',how='left')

temp_case_better_cats.to_csv(source_d_path + 'fraud_with_descriptions.csv')


#%% Create more of non-fraud data from 588 to 300000 (WITH REPLACEMENT of categories, but random amounts)

# function to obtain distribution for each category 
def gen_dist(min_val, max_val, mean_v, size):
    std_dev = (max_val - min_val) / 4
    random_numbers = np.random.normal(loc=mean_v, scale=std_dev, size=size)
    return random_numbers

sample_descriptions = non_fraud_data.sample(400000, replace= True).reset_index(drop = True)
updated_amounts = pd.DataFrame(columns = sample_descriptions.columns)
for subc in sample_descriptions['Subcategory'].unique():
    curr_cat = sample_descriptions.loc[sample_descriptions['Subcategory']==subc, :].reset_index(drop = True)
    size = len(curr_cat)
    amounts = gen_dist(curr_cat['min'].iloc[0], curr_cat['max'].iloc[0], curr_cat['mean'].iloc[0], size*10)
    amounts2 = amounts[((amounts > curr_cat['min'].iloc[0]) & (amounts < curr_cat['max'].iloc[0]))]    
    curr_cat.loc[:,'Amount'] = np.random.choice(amounts2, size)
    updated_amounts = pd.concat([updated_amounts, curr_cat])
rm_values = updated_amounts.loc[updated_amounts['Amount']<1, :].index
updated_amounts = updated_amounts.drop(index = rm_values)
updated_amounts['Amount'] = updated_amounts['Amount'].round(2)
non_fraud_data = updated_amounts.drop(columns=['min', 'max', 'mean'], axis = 1)

print('amount current range: ', non_fraud_data.Amount.min() ,'to' ,non_fraud_data.Amount.max())

non_fraud_data['origin'] = 'non-fraud'

non_fraud_data= non_fraud_data.rename(columns={'Category':'category',
                                               'Descriptions':'transaction_description'})

non_fraud_data.to_csv(save_path + 'longer_non_fraud_data.csv')
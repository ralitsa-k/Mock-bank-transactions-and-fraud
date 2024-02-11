#%% Set up
import numpy as np
from scipy.stats import skewnorm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime, timedelta
import os
import time
import polars as pl
start_time = time.time()

""" 
Generating realistic bank transactions (non-fraud)

This script is within the folder named 'BanksDataGen'.
Within BanksDataGen create folder 'OutputData' and 'SourceData'.
SourceData contains no_fraud_transactions.csv. 
And this script creates a csv file within SourceData folder as shown below: 

    -- BanksDataGen/
       - OutputData 
         |- 
       - SourceData 
         |- no_fraud_transactions.csv    (created manually)
         |- final_logical_non_fraud (created by the current script)
"""

# Prepare paths
curr_path = os.path.abspath(__file__)
curr_abs_path = curr_path.split('BanksDataGen')[0]
save_path = curr_abs_path + 'BanksDataGen/OutputData/'
source_d_path = curr_abs_path + 'BanksDataGen/SourceData/'

# N. customers simulated
n_customers_no_Fraud = 10
# Data starting date
initial_date = '2023-10-01' #oct

# Load realistic transaction Descriptions with Categories and Amounts
des_df = pl.read_csv(save_path + 'longer_non_fraud_data.csv')
des_df = des_df.filter(pl.col('Subcategory')!='Luxury shopping') 

# %% Define transaction numbers per month, given categories

categories = list(des_df.unique('category').select('category').to_series())
# Use a dictionary to store number of transactions per month for each category 
sample_dic = {'Bank Fees': 4,
            'Cash Withdrawal': 3,
            'Charity': 4,
            'Credit Card Payment': 4,
            'Dining Out': 2,
            'Electronics': 2,
            'Entertainment': 6,
            'Groceries': 6,
            'Healthcare': 3,
            'Holiday': 2,
            'Home Improvement': 2,
            'Housing': 1,
            'Income': 1,
            'Interest': 3,
            'Investment': 3,
            'Loan Payment': 2,
            'Online Shopping': 5,
            'Personal care': 5,
            'Shopping': 5,
            'Transfers': 8,
            'Transportation': 10,
            'Utilities': 5
            }

#%% Prep timeline of transactions
def get_month_frame(start):
    month_dic = dict()
    month_dic['start'] = datetime.strptime(start, '%Y-%m-%d').date()
    first = month_dic['start'].replace(day=1)
    first_next = (first + timedelta(days = 35)).replace(day=1)
    month_dic['end'] = first_next - timedelta(days=1)
    return month_dic

# example 
get_month_frame('2024-02-02')

def get_all_months(initial_date, n_months):
    start_date = datetime.strptime(initial_date, '%Y-%m-%d').date()
    month_frames = [
        get_month_frame(str(start_date + timedelta(days=i)))
        for i in range(n_months)
    ]
    return month_frames

get_all_months(initial_date, 10)


#%% Now get timelines of the data set 
# To get one month data, 
# - we go through categories from the dictionary that stores the number of occurences of each category 
# - we load the descriptions for this category only
# - we sample number of transactions as defined in the dictionary sample_dic

def get_one_month_data():
    month_data = pl.DataFrame()
    for this_Cat in categories:
        n_cat = sample_dic[this_Cat]
        reps = random.randint(1, n_cat)
        sampled_df = des_df.filter(pl.col('category')==this_Cat).sample(reps)
        month_data = pl.concat([month_data,sampled_df], how = 'vertical')
    return month_data

# %%

month_frames = get_all_months('2024-02-02', n_months = 4)
# change general spendings of customers 
# initially they will be very similar
customer_spending_variance = random.uniform(0.1, 0.5) 
full_data = pl.DataFrame()
n_customers_no_Fraud = 10
one_month_data = get_one_month_data()
one_month_data.shape[0]

sets = ['Income', 'Utilities', 'Housing']
one_month_data.filter(pl.col('category').is_in(['Income', 'Utilities', 'Housing']))

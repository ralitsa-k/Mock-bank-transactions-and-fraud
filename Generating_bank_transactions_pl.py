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
import calendar
""" 
Generating realistic bank transactions (non-fraud)

This script is within the folder named 'banks-data-gen'.
Within banks-data-gen create folder 'OutputData' and 'SourceData'.
SourceData contains no_fraud_transactions.csv. 
And this script creates a csv file within SourceData folder as shown below: 

    -- banks-data-gen/
       - OutputData 
         |- 
       - SourceData 
         |- no_fraud_transactions.csv    (created manually)
         |- final_logical_non_fraud (created by the current script)
"""

# Prepare paths
curr_path = os.path.abspath(__file__)
curr_abs_path = curr_path.split('banks-data-gen')[0]
save_path = curr_abs_path + 'banks-data-gen/OutputData/'
source_d_path = curr_abs_path + 'banks-data-gen/SourceData/'

# N. customers simulated
n_customers_no_Fraud = 200
n_months = 10
# Data starting date
initial_date = '2024-02-02' 

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
            'Interest': 3,
            'Investment': 3,
            'Loan Payment': 2,
            'Online Shopping': 5,
            'Personal care': 5,
            'Shopping': 5,
            'Transfers': 8,
            'Transportation': 10,
            'Housing': 1,
            'Income': 1,
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

# get months start and end dates 
def get_all_months(initial_date, n_months):
    start_date = initial_date 
    timeranges1 = get_month_frame(str(initial_date))
    month_frames = []
    for i in range(n_months):
        start = initial_date if i == 0 else timeranges1['end'] + timedelta(days=1)
        timeranges1 = get_month_frame(str(start))
        month_frames.append(timeranges1)
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

# Get last friday of month
def last_friday(year, month):
    num_days = calendar.monthrange(year, month)[1]
    for day in range(num_days, 0, -1):
        if calendar.weekday(year, month, day) == calendar.FRIDAY:
            return datetime(year, month, day).date()

# %% Create the full data set 

# change general spendings of customers 
# initially they will be very similar

# get sim data for as many month as desired
def gen_sim_data(n_months):
    month_frames = get_all_months('2024-02-02' , n_months)
    full_data = pl.DataFrame()
    for kk in range(n_customers_no_Fraud):
        out_data = pl.DataFrame()
        customer_spending_variance = random.uniform(0.1, 0.5) 
        for _ in range(n_months):
            one_month_data = get_one_month_data()
            if _ == 0:  
                # Extract only first month utilities and repeat every month 
                #   on same date 
                one_month_data_bills = one_month_data.filter(pl.col('category').is_in(['Income', 'Utilities', 'Housing']))
            else:
                # if second loop and above, remove utilities and add from first month
                one_month_data = pl.concat([one_month_data.filter(~pl.col('category').is_in(['Income', 'Utilities', 'Housing'])), 
                                                  one_month_data_bills], how = 'vertical')
            # get random dates for current month and assign transactions
            date_range = pd.date_range(start=month_frames[_]['start'], end=month_frames[_]['end'])
            dates = np.random.choice(date_range, size=len(one_month_data), replace=True)
            one_month_data = one_month_data.with_columns(date = dates)   
            # make bills to be at beginning of month
            one_month_data = one_month_data.with_columns(date = pl.when(pl.col("category").is_in(['Utilities', 'Housing']))
                                                         .then(pl.lit(month_frames[_]['start']))
                                                         .otherwise(pl.col('date')))
            # make payment to be at last day of the month
            friday = last_friday(month_frames[_]['start'].year, month_frames[_]['start'].month)
            one_month_data = one_month_data.with_columns(date = pl.when(pl.col("category")=='Income')
                                                .then(pl.lit(friday))
                                                .otherwise(pl.col('date')))
            out_data = pl.concat([out_data, one_month_data], how = 'vertical')
            
        out_data = out_data.with_columns(customer_id = kk + 1)
        # add variability per customer 
        out_data = out_data.with_columns(pl.col('Amount')*customer_spending_variance)
        full_data = pl.concat([full_data, out_data])

    return full_data

dd = gen_sim_data(1)

# Add Category income vs spendings    
dd = dd.with_columns(type = pl.when(pl.col('category').is_in(['Income','Interest']))
                    .then(pl.lit('income'))
                    .otherwise(pl.lit('spending')))
# Add transactions to current account to be counted as income 
dd = dd.with_columns(type = pl.when(pl.col('transaction_description').str.contains('to Current'))
                    .then(pl.lit('income'))
                    .otherwise(pl.col('type')))

dd.write_csv(save_path + '/final_logical_non_fraud2.csv')
print("--- %s seconds ---" % (time.time() - start_time))

# to fix, one transaction states 'Savings Account interest - October'
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
start_time = time.time()

""" 
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

# Get the path for this current file 
curr_path = os.path.abspath(__file__)
# Get the root path by deleting everything after the specified folder 
curr_abs_path = curr_path.split('BanksDataGen')[0]

# Define paths for saving files and loading files 
save_path = curr_abs_path + 'BanksDataGen/OutputData/'
source_d_path = curr_abs_path + 'BanksDataGen/SourceData/'

# Define how many customers you want simulated 
n_customers_no_Fraud = 10
# Define the date you want the data to start from 
initial_date = '2023-10-01'

# Load realistic transaction descriptions with amounts and category
des_df = pd.read_csv(save_path + 'longer_non_fraud_data.csv')

# remove Luxury category for now
des_df = des_df.loc[des_df.Subcategory!='Luxury',:]
des_df.rename(columns = {'category':'Category'}, inplace = True) 

#%% From here on we define how many transactions per category occur per month 

# Get unique Categories of transactions
categories = list(des_df.Category.unique())
categories.sort()

# Use a dictionary to store number of transactions per month for each category 
sample_dic = {'Bank Fees': 1,
            'Cash Withdrawal': 12,
            'Charity': 1,
            'Credit Card Payment': 5,
            'Dining Out': 5,
            'Electronics': 1,
            'Entertainment': 4,
            'Groceries': 10,
            'Healthcare': 3,
            'Holiday': 1,
            'Home Improvement': 2,
            'Housing': 1,
            'Income': 1,
            'Interest': 2,
            'Investment': 1,
            'Loan Payment': 1,
            'Online Shopping': 5,
            'Personal care': 3,
            'Shopping': 4,
            'Transfers': 10,
            'Transportation': 3,
            'Utilities': 4
            }


# Get the first and last days of the month 
def get_month_frame(start):
    month_dic = dict()
    month_dic['start'] = datetime.strptime(start, '%Y-%m-%d').date()
    first = month_dic['start'].replace(day=1)
    first_next = (first + timedelta(days = 35)).replace(day=1)
    month_dic['end'] = first_next - timedelta(days=1)
    return month_dic

# Get a few months
def get_all_months(n_months):
    start_date = initial_date 
    # Initialize the timeranges dictionary
    timeranges1 = get_month_frame(str(initial_date))
    # Initialize an empty list to store the month frames
    month_frames = []
    # Loop to generate new time ranges and add them to the list
    for i in range(n_months):
        start = initial_date if i == 0 else timeranges1['end'] + timedelta(days=1)
        timeranges1 = get_month_frame(str(start))
        month_frames.append(timeranges1)
    return month_frames


# Load empty frames to use 
bills = pd.DataFrame(columns = ['Descriptions', 'Amount', 'Category', 'Subcategory'])

# To get one month data, 
# - we go through categories from the dictionary that stores the number of occurences of each category 
# - we load the descriptions for this category only
# - we sample number of transactions as defined in the dictionary sample_dic

def get_one_month_data():
    month_data_list = []

    for this_Cat in categories:
        n_cat = sample_dic[this_Cat]
        dat_Cat = des_df.loc[des_df.Category == this_Cat, :]

        if this_Cat not in ['Income', 'Housing'] and n_cat > 0:
            n_cat_r = np.random.randint(n_cat - 1, n_cat)
        elif this_Cat in ['Income', 'Housing']:
            n_cat_r = 1
        else:
            n_cat_r = 0

        if n_cat_r > dat_Cat.shape[0]:
            selected_cat_data = dat_Cat.sample(n_cat_r, replace=True)
        else:
            selected_cat_data = dat_Cat.sample(n_cat_r, replace=False)

        month_data_list.append(selected_cat_data)

    month_data = pd.concat(month_data_list, ignore_index=True)
    return month_data

get_one_month_data()


# The following lines to the above category sampling for the number of months defined
# Increase months and add customers 
def generate_simulation_data(n_months):
    month_frames = get_all_months(n_months)  
    # empty data frame to be used as placeholder
    full_data = pd.DataFrame()

    for kk in range(n_customers_no_Fraud):
        # some empty data frames 
        out_data = pd.DataFrame()

        # This is the random number that will change the transactions to be variable in size (Amount)
        # we do not want all transactions picked up from the descriptions dataset to be of the same size 
        perc_from = random.uniform(0.1, 0.5)

        for _ in range(n_months):
            # take each month and add data
            this_dates = month_frames[_]
            dates = [str(this_dates['start']), str(this_dates['end'])]

            One_month_data = get_one_month_data() 

                # for each person we want their income and utilities to be the same every month 
                # so we will store them here 
            if _ == 0:
                # Store the income and utilities rows to use for future months
                store_constant = One_month_data.loc[One_month_data.Category == 'Income', :].drop_duplicates()
                util = One_month_data.loc[One_month_data.Category == 'Utilities', :]
                house = One_month_data.loc[One_month_data.Category == 'Housing', :].drop_duplicates()
                bills = pd.concat([util, store_constant, house])
            else:
                # once one month data is created, the same income and utilities will be added to the future months 
                # Remove these categoires and substitute with the already save data for income, utilities, housing
                categories_to_remove = ['Income', 'Utilities', 'Housing']
                One_month_data = One_month_data[~One_month_data['Category'].isin(categories_to_remove)]
                One_month_data = pd.concat([One_month_data, bills])
                
            # For the current month, generate a list of dates with random skew
            date_range = pd.date_range(start=dates[0], end=dates[1])
            date_weights = np.random.rand(len(date_range))
            dates = np.random.choice(date_range, size=One_month_data.shape[0], replace=True,
                                      p=date_weights / date_weights.sum())

            One_month_data['date'] = dates

            # Put utilities income and housing to be on the first day or another day that is in the beginning of the month 
            first = date_range[0].replace(day=1)  # First day of the selected month
            utilities = One_month_data.loc[One_month_data.Category == 'Utilities', 'date']
            utilities_len = len(utilities)
            first_few = pd.date_range(first, first + timedelta(days=utilities_len - 1))
            mask = One_month_data['Category'].isin(['Income', 'Housing'])
            One_month_data.loc[mask, 'date'] = first
            One_month_data.loc[One_month_data.Category == 'Utilities', 'date'] = first_few

            One_month_data = One_month_data.sort_values(by='date')

            # record the sum of spendings (exclude income)
            if _ == 0:
                spendings = One_month_data.loc[One_month_data.Category != 'Income', :]['Amount'].sum()
                
            # add a variable amount for each amount (so it does not repeat from starting dataset)
            One_month_data.loc[One_month_data.Category == 'Income', 'Amount'] = spendings + perc_from * spendings

            # append this one month data to the full data for this person 
            out_data = pd.concat([out_data, One_month_data])

        # add customer id 
        out_data['customer_id'] = kk + 1
        full_data = pd.concat([full_data, out_data])

    return full_data

# This will take a while as it generates the full dataset 
full_data = generate_simulation_data(n_months = 2)

# Add customer id 
ids = pd.DataFrame(full_data.customer_id.drop_duplicates()).reset_index(drop = True)

# prepare a sequence
sequence = [i for i in range(10000, 20000)]
ids_2 = pd.DataFrame(random.sample(sequence, n_customers_no_Fraud))
ids['customer_id2'] = ids_2[0]
full_data = pd.merge(full_data,ids,on='customer_id',how='left')
full_data = full_data.drop('customer_id', axis = 1)
full_data.rename(columns = {'customer_id2': 'customer_id'}, inplace = True)

len(ids.customer_id2.drop_duplicates())

# Add Category income vs spendings    
full_data['type'] = full_data['Category'].apply(lambda x: 'income' if x=='Income' else 'spending')
# Add date 
full_data['month'] = full_data['date'].dt.strftime('%B')

# get some counts 
stat = full_data.groupby(['customer_id','type', 'month'])['Amount'].sum()
cc = full_data.groupby(['customer_id','month'])['month'].count()

# save the non-fraud dataset 
full_data.to_csv(save_path + '/final_logical_non_fraud.csv')

print("--- %s seconds ---" % (time.time() - start_time))
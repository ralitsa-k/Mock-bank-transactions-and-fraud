
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
perc_scammes = 30/100
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
for cust in np.array(scammed_sample_data.select(pl.col('customer_id')).unique()):
    # get the data per customer 
    curr_data = scammed_sample_data.filter(pl.col('customer_id') == cust)
    # add column fraud type 
    curr_data = curr_data.with_columns(pl.lit('none').alias('fraud_type')).with_row_index()
    # get the index of the non-fraud data to be removed 

    # If the income of this person is in a high quartile, assign Romance to them 
    # If not, assign other types of scam 
    np.array(curr_data.filter(pl.col('Category') == 'Income').select(pl.col('Amount')))
    if curr_data.filter(pl.col('Category') == 'Income').select(pl.col('Amount'))[0].item() > quartiles[2]:
       nn = random.randint(romance_per_month-2, romance_per_month)
       fraud_d = romance_only.sample(nn)
    else:
       nn = random.randint(noromance_per_month-2, noromance_per_month)
       fraud_d = no_romance.sample(nn)
       
    sample_dates = curr_data.select(pl.col('date', 'month', 'customer_id')).sample(nn)  
    # fraud is always spending (for now), so add this column   
    fraud_d = fraud_d.with_columns(type = pl.lit('spending'))
    # add the dates to fraud transactions 
    fraud_d = fraud_d.with_columns(date = sample_dates['date'],
                         month = sample_dates['month'],
                         customer_id = sample_dates['customer_id']
                         )
    
    # the dates used for the fraud transactions have to be used to remove some regular transactions 
    # (so that the number of transactions are still the same per customer per month)
    # but making sure income housing and utilities don't get removed 
    sdates = sample_dates.select(pl.col("date"))
    shorten_data = curr_data.filter((~pl.col('Category').is_in(['Income', 'Utilities', 'Housing'])) & pl.col('date').is_in(sdates))
    remove_d = shorten_data.sample(fraud_d.shape[0], with_replacement = True).select(pl.col('index'))
    # from the full data remove the random non-fraud transactions given their defined index
    curr_data = curr_data.join(remove_d, on = 'index', how = 'anti')    
    fraud_d = fraud_d.rename({"Descriptions":"transactions_description",
                    "subcategory":"Subcategory",
                    }).drop(["Unnamed: 0", "case_id", "number_of_transactions","bank","data_ver", "case_value"])
    curr_data = pl.concat([curr_data, fraud_d], how = "diagonal_relaxed")
   
   
    # add the fraud transactions to the non-fraud data of this customer 
    fraud_and_no = pl.concat([fraud_and_no, curr_data])
    
# add a flag indicated that all of the customers in this data set have been scammed at least once     
fraud_and_no = fraud_and_no.with_columns(customer_scammed = 1)



#%% Add the non-fraud data to the customers who were scammed 

# remove the scammed customers from the original data
non_fraud = large_non_fraud_data.filter(~pl.col('customer_id').is_in(scammed_ids))
# make sure ordinary transaction are of fraud_type none 
non_fraud = non_fraud.with_columns(fraud_type = pl.lit('none'))
# all of these customers are not scammed, so the flag is 0
non_fraud= non_fraud.with_columns(customer_scammed = 0)

# combine the two data sets 
fraud_and_no = fraud_and_no.select(non_fraud.columns)
data_enriched = pl.concat([fraud_and_no,non_fraud])

#%%  Assign banks to tranasactions 

list_banks_uk = ["bank_A", "bank_B", "bank_C", "bank_D", "bank_E"]
to_b = pl.DataFrame({"bank_to":random.choices(list_banks_uk, k = data_enriched.shape[0])})
from_b = pl.DataFrame({"bank_from":random.choices(list_banks_uk, k = data_enriched.shape[0])})
data_enriched = pl.concat([data_enriched,to_b], how = "horizontal")
data_enriched = pl.concat([data_enriched,from_b], how = "horizontal")
   
deposits_descr = ['Deposit', 'Interest', 'from Savings' ]

# assigning banks to customers. 
# income is always going into the customers bank
# all spending always going out from customers bank    
def assign_banks(customer_data, own_bank):

        # income always to own bank, spending always from own bank

    customer_data = customer_data.with_columns(
        pl.when(((pl.col("type") == "income") | (pl.col("Category")=="Income")))
        .then(pl.lit(own_bank))
        .otherwise(pl.col('bank_to'))
        .alias('bank_to'),
        pl.when((pl.col("type") == "spending"))
        .then(pl.lit(own_bank))
        .otherwise(pl.col('bank_from'))
        .alias('bank_from'))
    # income always from the same bank

    banks_income = customer_data.filter(pl.col("Category")=="Income")[0].select(['bank_to', 'bank_from'])
    customer_data = customer_data.with_columns(pl.when(pl.col("Category")=="Income")
                .then(pl.lit(banks_income.select('bank_from')).alias('bank_from'))
                .otherwise(pl.col('bank_from')))

    # If a description contains 'from Savings or to Savings, it means the transaction
        #   should be going between the bank of the customer 
    descriptions = customer_data.filter(pl.col("transaction_description").is_in(['Deposit', 'Interest', 'from Savings']))
    if len(descriptions) > 0:
        customer_data = customer_data.with_columns(
            pl.when(pl.col("transaction_description").is_in(['Deposit', 'Interest', 'from Savings']))
            .then(pl.lit(own_bank))
            .otherwise(pl.col('bank_to'))
            .alias('bank_to'),
            pl.when(pl.col("transaction_description").is_in(['Deposit', 'Interest', 'from Savings']))
            .then(pl.lit(own_bank))
            .otherwise(pl.col('bank_from'))
            .alias('bank_from'))
        
    utils = customer_data.filter((pl.col("Category")=="Utilities") & (pl.col("fraud_type") == "none"))
    months = len(utils['date'].unique())
    util_banks = random.choices(list_banks_uk,k=int(len(utils)/months))*months
    utils = utils.with_columns(
        pl.lit(banks_income.select('bank_from')).alias('bank_from'),
        pl.Series("bank_to", values = util_banks)
    )
    
    customer_data = customer_data.filter((pl.col("Category")!="Utilities"))
    customer_data = pl.concat([customer_data, utils])

    return customer_data

# Use the function assign_banks to add 'bank to' and 'bank from' to each customer
data_with_banks = pl.DataFrame()
for cust in data_enriched['customer_id'].unique():
    own_bank = random.choice(list_banks_uk)
    curr_data = data_enriched.filter(pl.col("customer_id")==cust)
    curr_data_with_banks = assign_banks(customer_data = curr_data, own_bank = own_bank)
    data_with_banks = pl.concat([data_with_banks, curr_data_with_banks])

# no bank to for withdrawal 
data_with_banks = data_with_banks.with_columns(
        pl.when(((pl.col("Category") == "Cash Withdrawal")))
        .then(pl.lit('nan'))
        .otherwise(pl.col('bank_to'))
        .alias('bank_to'))


#%% Make variable incomes and spendings per customer 
import matplotlib.pyplot as plt
# Increasing spendings and income for some customers 
cust_r = data_with_banks['customer_id'].unique()
# factor by which the amounts will be increased (customer-wise)
factor_increase = np.random.uniform(1, 2, size = len(cust_r))

# Create a dictionary to map customer IDs to their corresponding factors
factor_dict = pl.DataFrame({"customer_id": cust_r, 
                            "factor_increase": factor_increase})

# Multiply the amount of each participant by a factor, so that amounts are more variable
data_with_banks_amounts = data_with_banks.join(factor_dict, on = 'customer_id', how = 'inner')
data_with_banks_amounts = data_with_banks_amounts.with_columns(
                                                               pl.when(pl.col("Amount") > 2)
                                                               .then(pl.col('Amount')*pl.col('factor_increase'))
                                                               .otherwise(pl.col('Amount')+4)
                                                               .alias("newAmount"))


# Check the average spend per customer
plot_d = data_with_banks_amounts.group_by('customer_id').agg(pl.col('newAmount').mean())
plt.hist(plot_d.select(pl.col('newAmount')), bins = 20)


# Save the data
data_with_banks_amounts.write_csv(save_path+'final_bank_data.csv')
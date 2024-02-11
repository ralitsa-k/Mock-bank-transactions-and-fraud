
import pandas as pd 
import os

curr_path = os.path.abspath(__file__)
curr_abs_path = curr_path.split('Plotting')[0]

full_df = pd.read_csv(curr_abs_path + '/OutputData/data_with_classified_scam.csv')

full_df['date'] = pd.to_datetime(full_df['date'], format='%d/%m/%Y')
frr1 = full_df.loc[full_df.is_scam_transaction == 1, ['Descriptions', 'is_scam_transaction', 'fraud_type']].reset_index(drop=True).drop_duplicates()
frr = full_df.loc[(full_df['date'] < '10/10/2023') & (full_df.is_scam_transaction == 1), ['Descriptions', 'is_scam_transaction', 'fraud_type']].reset_index(drop=True).drop_duplicates()


full_df.shape

full_df.loc[:,['customer_id']].drop_duplicates().nunique()
full_df.loc[:,['customer_id', 'customer_scammed']].drop_duplicates().value_counts('customer_scammed')

full_df.groupby('customer_id')['customer_scammed'].max().value_counts()

full_df.head()

#full_df.loc[full_df.Category == 'Interest', 'bank_from'] = full_df.loc[full_df.Category == 'Interest', 'bank_to']


# Fix multiple transactions per customer 
cc = full_df.groupby('customer_id').count()/10
cust_id_too_much = cc.loc[cc.Category > 70].index

nn = pd.DataFrame(cust_id_too_much)
remove = full_df.loc[full_df.customer_id.isin(cust_id_too_much)].sort_values(['customer_id'])

unique_rows = remove.loc[:,['Descriptions', 'customer_id','Category']].drop_duplicates()

remove.loc[:,['customer_id']].drop_duplicates().nunique()
remove.loc[:,['customer_id', 'customer_scammed']].drop_duplicates().value_counts('customer_scammed')

remove.groupby('customer_id')['customer_scammed'].max().value_counts()



import seaborn as sns
import matplotlib.pyplot as plt
colors = ['#a7ba42','#95ccba','#ffdede','#f94f8a',
          '#fff0cb', '#f2cc84','#d1b2e0', '#660099','#079999']
sns.set_palette(sns.color_palette(colors))

plot_fraud = full_df.loc[~full_df.customer_id.isin(cust_id_too_much) &
                         (full_df.fraud_type != 'none') &
                         (full_df.Category != 'Investment'), :].groupby(['Category'])['Amount'].mean().reset_index()

plot_fraud = plot_fraud.sort_values('Amount', ascending=False)
# import textwrap
# def wrap_labels(ax, width, break_long_words=True):
#     labels = []
#     for label in ax.get_xticklabels():
#         text = label.get_text()
#         labels.append(textwrap.fill(text, width=width,
#                       break_long_words=break_long_words))
#     ax.set_xticklabels(labels, rotation=0)
    
# wrap_labels(fig, 5)
fig = plt.figure()
ax = sns.barplot(plot_fraud,
            x = 'Category',
            y = 'Amount',
            palette = colors)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
fig.tight_layout()
plt.show()




# OLD STATS, will depend on current parameters 
# months data 
len(full_df.month.unique())

# n of customers
len(full_df.customer_id.unique())

# n scammed customers
len(full_df.loc[full_df.customer_scammed == 1, :].customer_id.unique())

# average transaactions per person for the whole period (should be around 63)
clean_masters_data.groupby('customer_id').count()['Descriptions'].mean()/len(full_df.month.unique())

scammed_data = full_df.loc[full_df.customer_scammed == 1, :]
# number of scams per person scammed : on average  for the whole period per person (e.g. 26 out of 625)
scammed_data.groupby(['customer_id', 'is_scam_transaction']).count()


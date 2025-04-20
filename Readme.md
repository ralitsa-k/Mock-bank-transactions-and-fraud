# This project created mock bank transaction data with fraudulent and regular transactions in Python

The data is completely made up

Results plotted in Tableau
![image](https://github.com/ralitsa-k/Mock-bank-transactions-and-fraud/assets/44056292/50e26d5c-ddd9-45dc-8bab-b568e808d56d)

### Note: Generating bank transactions will take some time. Average 4 minutes if selection is 100 customers for 3 months. 

1) Generating_bank_transactions.py
2) Embed_fraud.py

## Banks Data Generations

This project aims to generate realistic bank transaction with fraudulent transactions within. 
Transactions happen between 5 banks and one international. 
The scripts generate realistic normal transactions and then adds fradulent ones to a percent of all customers. 

## Considerations 

- CEO type of fraud is removed from the original fraud types. 
- Maybe need to change the way non-fraud categories are bound to fraud ones. E.g. right now romance can be 'Romance':['Personal care', 'Dining Out', 'Entertainment', 'Home Improvement','Holiday'].

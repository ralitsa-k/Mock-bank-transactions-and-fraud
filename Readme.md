# This project created mock bank transaction data with fraudulent and regular transactions in Python

Results plotted in Tableau
![image](https://github.com/ralitsa-k/Mock-bank-transactions-and-fraud/assets/44056292/50e26d5c-ddd9-45dc-8bab-b568e808d56d)

## To generate your own data run the scripts in the following order:

1) Generating_bank_transactions.py
2) Embed_fraud.py

('Descriptions additions.py' can be run but is not necessary
       as the non-fraud data it creates is already provided)
('Generating_fraud_data.py' is also unnecesary, as the data it creates
      is provided: 'case_transaction.csv')

## Banks Data Generations

This project aims to generate realistic bank transaction with fraudulent transactions within. 
Transactions happen between 5 banks and one international. 
The scripts generate realistic normal transactions and then adds fradulent ones to a percent of all customers. 

## To generate your own data: 

The structure of your folders and files should be: 

    -- Banks-Data-Gen (or name of repo)
       - OutputData 
         |- 
       - SourceData 
         |- no_fraud_transactions.csv               (created manually)
         |- fraud_transaction_descriptions.csv      (created manually)
         |- case_transaction.csv                    (created by Julian and Kathy, but also: Generating_fraud_data.py)

         |- fraud_with_descriptions.csv             (creted with Descriptions_additions.csv)
         |- longer_non_fraud_data.csv               (created with Descriptions_additions.csv)





## Considerations 

- CEO type of fraud is removed from the original fraud types. 
- The values of fraud are increased from the original fraud data set, because they were small in relation to 
regular transactions (maybe reconsider). 
- Maybe need to change the way non-fraud categories are bound to fraud ones. E.g. right now romance can be 'Romance':['Personal care', 'Dining Out', 'Entertainment', 'Home Improvement','Holiday'].

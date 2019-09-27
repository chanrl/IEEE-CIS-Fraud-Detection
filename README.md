## IEEE-CIS-Fraud-Detection

# Background
Exploring a financial dataset from Vesta Corporation. This (Kaggle competition)[kaggle.com/c/ieee-fraud-detection/overview] is seeking the best solutions for fraud prevention.

# Objectives
- Clean e-commerce transactions dataset provided by Vesta
- Engineer features for Random Forest and Logistic Regression to train and validate on
- Benchmark machine learning models on the Vesta e-commerce transactions dataset

# Thoughts
I was able to feature engineer most of the data columns. I did not use time as a feature, although I do understand that might have predictive value. I have not experienced converting time deltas to date time, or simply converting it to a different value that could have some relationship to fraudulent transactions or not.

For a lot of the numerical features, such as distance and days between transactions, I filled the nan (missing) values with the median value, and then created a different column to indicate whether or not the row was originally nan and replaced with a median value.

Address columns that were already entered as numerical codes were pretty easy to convert into categorical codes with Pandas.
I did not use the Vespa or id features (over 300 V# features) yet as I was seeking to create a baseline model and start adding the V/id features in later. It is hard to know what these codes are when there's no input about the data from Vespa. E.g. is 0 for the V1 column supposed to represent a false or none type? Or is it just a categorical code? Knowing these answers would help me assign nan values to reflect the intent of the V columns.

Another aspect I was not too sure about was the meaning of 'credit or debit' as an entry for the card used. They had 'credit', 'debit', 'charge card', and nan for values, but I just replaced 'credit or debit' as nan since that seemed like the logical replacement as the transaction has no idea what the card type is.

When I have more time, I will look into revisiting the Vespa ID columns, as there's about 30 columns with strings, so I would need to use some clever regex to extract the proper values. There's one column with the browser used, and it would have versions along with the browser used. It might be predictive if for example, fraudulent transactions occur under older versions of internet explorer or AOL browser.




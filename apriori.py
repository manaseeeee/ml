import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the dataset
basket = pd.read_csv("d:\Downloads\Groceries_dataset.csv\Groceries_dataset.csv")
display(basket.head())

# Transform itemDescription to a list for each entry (prepare for groupby aggregation)
basket['itemDescription'] = basket['itemDescription'].transform(lambda x: [x])

# Group by Member_number and Date, aggregating all items in each transaction
basket = basket.groupby(['Member_number', 'Date'])['itemDescription'].sum().reset_index()
display(basket.head())

# Convert the data into the format needed for TransactionEncoder
transactions = basket['itemDescription'].tolist()
encoder = TransactionEncoder()
transactions = pd.DataFrame(encoder.fit(transactions).transform(transactions), columns=encoder.columns_)

# Generate frequent itemsets with minimum support
frequent_itemsets = apriori(transactions, min_support=6/len(basket), use_colnames=True)

# Generate association rules with minimum lift threshold
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
display(rules.head())
print("Rules identified:", len(rules))

# Visualize the distribution of association rules in a 3D plot
sns.set(style="whitegrid")
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
x = rules['support']
y = rules['confidence']
z = rules['lift']
ax.set_xlabel("Support")
ax.set_ylabel("Confidence")
ax.set_zlabel("Lift")
ax.scatter(x, y, z)
ax.set_title("3D Distribution of Association Rules")
plt.show()

# Filter and display rules with 'whole milk' as the consequent
milk_rules = rules[rules['consequents'].astype(str).str.contains('whole milk')]
milk_rules = milk_rules.sort_values(by=['lift'], ascending=False).reset_index(drop=True)
display(milk_rules.head())

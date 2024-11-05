import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

basket = pd.read_csv("d:\Downloads\Groceries_dataset.csv\Groceries_dataset.csv")
display(basket.head())

basket['itemDescription'] = basket['itemDescription'].transform(lambda x: [x])

basket = basket.groupby(['Member_number', 'Date'])['itemDescription'].sum().reset_index()
display(basket.head())

transactions = basket['itemDescription'].tolist()
encoder = TransactionEncoder()
transactions = pd.DataFrame(encoder.fit(transactions).transform(transactions), columns=encoder.columns_)

frequent_itemsets = apriori(transactions, min_support=6/len(basket), use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
display(rules.head())
print("Rules identified:", len(rules))

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

milk_rules = rules[rules['consequents'].astype(str).str.contains('whole milk')]
milk_rules = milk_rules.sort_values(by=['lift'], ascending=False).reset_index(drop=True)
display(milk_rules.head())

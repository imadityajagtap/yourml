import pandas as pd
import numpy as np
import plotly.express as px 
import plotly.graph_objects as go
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
plt.style.use("default")
#!pip install --upgrade plotly
mrk=pd.read_csv("Market_Basket_Optimisation.csv")
mrk.shape
mrk.head()
mrk.describe()
transaction = []
for i in range(0, mrk.shape[0]): 
    for j in range(0, mrk.shape[1]):
        transaction.append(mrk.values[i,j])
transaction = np.array(transaction)
df = pd.DataFrame(transaction, columns=['items'])
df["incident_count"] = 1
indexNames = df[df["items"] == "nan"].index
df.drop(indexNames , inplace=True)
df_table = df.groupby("items").sum().sort_values("incident_count",
ascending=False).reset_index()
df_table.head(10).style.background_gradient(cmap="Blues")
df_table["all"] = "all"
fig = px.treemap(df_table.head(30), path=['all', "items"], values='incident_count',color=df_table["incident_count"].head(30), hover_data=['items'],color_continuous_scale='ice')
fig.show()
transaction = []
for i in range(mrk.shape[0]):
    transaction.append([str(mrk.values[i,j]) for j in range(mrk.shape[1])])
transaction = np.array(transaction)
top20 = df_table["items"].head(20).values
array = []
df_top20_multiple_record_check = pd.DataFrame(columns=top20)
for i in range(0, len(top20)):
    array = []
    for j in range(0,transaction.shape[0]):
        array.append(np.count_nonzero(transaction[j]==top20[i]))
    if len(array) == len(mrk):
        df_top20_multiple_record_check[top20[i]] = array
    else:
     continue
df_top20_multiple_record_check.head(10)
df_top20_multiple_record_check.describe()
transaction = []
for i in range(0, mrk.shape[0]):
    transaction.append(mrk.values[i,0])
transaction = np.array(transaction)
df_first = pd.DataFrame(transaction, columns=["items"])
df_first["incident_count"] = 1
indexNames = df_first[df_first['items'] == "nan" ].index
df_first.drop(indexNames , inplace=True)
df_table_first = df_first.groupby("items").sum().sort_values("incident_count",
ascending=False).reset_index()

df_table_first["food"] = "food"
df_table_first = df_table_first.truncate(before=-1, after=15)
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (20, 20)
first_choice = nx.from_pandas_edgelist(df_table_first, source = 'food', target
= "items", edge_attr = True)
pos = nx.spring_layout(first_choice)
nx.draw_networkx_nodes(first_choice, pos, node_size = 12500, node_color =
"lavender")
nx.draw_networkx_edges(first_choice, pos, width = 3, alpha = 0.6, edge_color =
'black')
nx.draw_networkx_labels(first_choice, pos, font_size = 18, font_family = 'sans-serif')
plt.axis('off')
plt.grid()
plt.title('Top Choices', fontsize = 25)
plt.show()
transaction = []
for i in range(mrk.shape[0]):
    transaction.append([str(mrk.values[i,j]) for j in range(mrk.shape[1])])
transaction = np.array(transaction)
transaction
te = TransactionEncoder()
te_ary = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_ary,columns=te.columns_) 
datasetfirst50 = df_table["items"].head(50).values
dataset = dataset.loc[datasetfirst50(50)]
dataset
def encode_units(x):
    if x == False:
        return 0 
    if x == True:
        return 1
dataset = dataset.applymap(encode_units)
dataset.head(10)
frequent_itemsets = apriori(dataset, min_support=0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
frequent_itemsets[ (frequent_itemsets['length'] == 2) &
(frequent_itemsets['support'] >= 0.05) ]
frequent_itemsets[ (frequent_itemsets['length'] == 3) ].head()
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules["antecedents_length"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_length"] = rules["consequents"].apply(lambda x: len(x))
rules.sort_values("lift",ascending=False)
rules.sort_values("confidence",ascending=False)
rules[~rules["consequents"].str.contains("mineral water", regex=False) &
~rules["antecedents"].str.contains("mineral water", regex=False)].sort_values("confidence", ascending=False).head(10)
rules[rules["antecedents"].str.contains("ground beef", regex=False) & rules["antecedents_length"] == 1].sort_values("confidence", ascending=False).head(10)
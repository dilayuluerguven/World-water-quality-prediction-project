#keşifsel veri analizi (EDA) 
import numpy as np #linear algebra(matematik işlemleri)
import pandas as pd #data processing(cvs dosya okuması)
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px

import missingno as msno #missing value 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,RepeatedStratifiedKFold,train_test_split
from sklearn.metrics import precision_score,confusion_matrix

from sklearn import tree

df= pd.read_csv("water_potability.csv")

describe = df.describe()
info = df.info()
print(info)

#dependent variable analysis (bağımlı değişken analizi)
d = pd.DataFrame(df["Potability"].value_counts()).reset_index()
d.columns = ["Label", "count"]
d["Label"] = d["Label"].replace({0: "Not Potable", 1: "Potable"})
fig = px.pie(
    d,
    values="count",
    names="Label",
    hole=0.35,
    opacity=0.8,
    labels={"Label": "Potability", "count": "Number of Samples"}
)
fig.update_layout(title=dict(text="Pie Chart of Potability Feature"))
fig.update_traces(textposition="outside", textinfo="percent+label")
fig.show()
fig.write_html("potability_pie_chart.html")
#korelasyon analizi
sns.clustermap(df.corr(), dendrogram_ratio=(0.1, 0.2), cmap="vlag",annot=True,linewidths=0.8,figsize=(10,10))
plt.show()

#distrubution of features

non_potable = df.query("Potability==0")
potable = df.query("Potability==1")
plt.figure()
for ax,col in enumerate(df.columns[:9]):
    plt.subplot(3,3,ax+1)
    plt.title(col)
    sns.kdeplot(x=non_potable[col],label="Non Potable")
    sns.kdeplot(x=potable[col],label="Potable")
    plt.legend()
plt.tight_layout()
plt.show()

#missing value analysis
msno.matrix(df)
plt.show()


#preprocessing:missing value problem,train-test split,normalization,encoding

#modelleme: decision tree, random forest

#evaluation: decision tree visualization

#hyperparameter tuning:  random forest

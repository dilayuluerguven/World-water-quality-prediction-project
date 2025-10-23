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


#preprocessing:missing value problem,train-test split,normalization
print(df.isnull().sum())
df["ph"].fillna(value=df["ph"].mean(),inplace=True)
df["Sulfate"].fillna(value=df["Sulfate"].mean(),inplace=True)
df["Trihalomethanes"].fillna(value=df["Trihalomethanes"].mean(),inplace=True)
print(df.isnull().sum())

#train  test split
X=df.drop("Potability",axis=1).values
Y=df["Potability"].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42,)

#min-max normalization
x_train_max = np.max(X_train)
x_train_min = np.min(X_train)
X_train = (X_train - x_train_min) / (x_train_max - x_train_min)
X_test = (X_test - x_train_min) / (x_train_max - x_train_min)

#modelleme: decision tree, random forest
models=[{"DIC":DecisionTreeClassifier(max_depth=3)},
        {"RFC":RandomForestClassifier()}]
finalResults=[]#scoreları tutacak
cmList=[]#matriksleri tutacak
for model_dict in models:
    for name, model in model_dict.items():
        model.fit(X_train, Y_train)
        model_result = model.predict(X_test)
        score = precision_score(Y_test, model_result)

        finalResults.append({name: score})
        cm = confusion_matrix(Y_test, model_result)
        cmList.append({name: cm})
print("Final Results:",finalResults)

for cm_dict in cmList:
    for name, cm in cm_dict.items():
        plt.figure()
        sns.heatmap(cm, annot=True, linewidths=0.8, fmt=".0f")
        plt.title(name)
        plt.show()
#evaluation: decision tree visualization

df_clf = models[0]["DIC"]
plt.figure()
tree.plot_tree(df_clf,feature_names=df.columns.tolist()[:-1],class_names=["0","1"],filled=True,precision=5)
plt.show()
#hyperparameter tuning:  random forest
